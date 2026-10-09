"""Compute-bound calibration target.

Each /work request does a fixed amount of CPU work (a tight arithmetic loop of
`units * 100k` iterations), so a lower CPU limit throttles the request and its
latency grows in proportion. That is the relationship the simulator's
service-time model needs, and it is what the stock factorizator (a fixed 2 s
wall-clock loop) cannot provide. Work size is tunable per request so sweeps stay
fast and can be driven up to the overload knee.

Multi-tier chain: if DOWNSTREAM_URL is set, /work does this tier's CPU work and
then forwards to the next tier, returning both this tier's compute time and the
downstream time. A request entering the frontend therefore traverses every tier
once and its end-to-end latency is the sum of the per-tier sojourns -- exactly
the open-chain topology the simulator assumes. With DOWNSTREAM_URL
unset the app is the single-tier target, unchanged.

Runs on the public factorizator image (Python 3.9 + Flask); this file is mounted
over /app/app.py via a ConfigMap, so no image build is needed.
"""
import json
import os
import time
import urllib.request

from flask import Flask, jsonify, request

app = Flask(__name__)

DEFAULT_UNITS = int(os.environ.get("WORK_UNITS", "40"))
TIER = os.environ.get("TIER", "target")
# Next hop in the chain (e.g. http://logic.blackbox.svc:8080); unset = leaf tier.
DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_URL", "").rstrip("/")
DOWNSTREAM_TIMEOUT = float(os.environ.get("DOWNSTREAM_TIMEOUT", "30"))

# Resident memory footprint (working set). Held for the pod's lifetime and
# touched so the pages are actually resident, so a memory limit below this size
# makes Kubernetes OOM-kill the pod. This turns the memory limit into a real
# feasibility constraint the benchmark can calibrate.
_MEM_MB = int(os.environ.get("WORK_MEM_MB", "64"))
_ballast = bytearray(_MEM_MB * 1024 * 1024)
for _i in range(0, len(_ballast), 4096):
    _ballast[_i] = 1


def cpu_work(units: int) -> int:
    # Pure-Python LCG loop: CPU-bound and GIL-held, so CFS throttling stretches
    # wall time roughly in proportion to 1 / cpu_limit.
    x = 0
    for _ in range(units * 100_000):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x


def call_downstream():
    """Forward to the next tier, returning its parsed JSON body and wall time."""
    t0 = time.perf_counter()
    with urllib.request.urlopen(DOWNSTREAM_URL + "/work",
                                timeout=DOWNSTREAM_TIMEOUT) as r:
        body = json.loads(r.read())
    return body, (time.perf_counter() - t0) * 1000.0


@app.route("/work", methods=["GET"])
def work():
    units = int(request.args.get("units", DEFAULT_UNITS))
    t0 = time.perf_counter()
    val = cpu_work(units)
    compute_ms = (time.perf_counter() - t0) * 1000.0

    # Leaf tier: return this tier's own sojourn.
    if not DOWNSTREAM_URL:
        return jsonify(tier=TIER, units=units, compute_ms=compute_ms,
                       downstream_ms=0.0, total_ms=compute_ms,
                       chain=[TIER], val=val), 200

    # Non-leaf: do this tier's work, then traverse the rest of the chain. The
    # returned total is this tier's compute plus the measured downstream time, so
    # the frontend's total_ms is the end-to-end sum over the whole chain.
    try:
        body, downstream_ms = call_downstream()
    except Exception as e:  # downstream unreachable/timed out -> chain failure
        return jsonify(tier=TIER, units=units, compute_ms=compute_ms,
                       error=f"downstream: {e}"), 502
    return jsonify(tier=TIER, units=units, compute_ms=compute_ms,
                   downstream_ms=downstream_ms,
                   total_ms=compute_ms + downstream_ms,
                   chain=[TIER] + body.get("chain", []), val=val), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="healthy"), 200


if __name__ == "__main__":
    # Threaded so concurrent requests share the throttled CPU (matches an
    # M/M/1-per-replica station once the CPU limit saturates).
    app.run(host="0.0.0.0", port=8080, threaded=True)
