"""In-cluster load generator (stdlib only, runs on the target image).

Fires closed-loop load at the target service with a fixed concurrency for a set
duration and prints a JSON summary (throughput and latency percentiles). Uses
only the standard library so it runs on the plain Python image with no extra
installs. Invoked via `kubectl exec` from the sweep controller.

    python loadgen.py --url http://factorizator.blackbox.svc:8080/work \
        --units 40 --concurrency 8 --duration 20
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one_request(url: str, timeout: float) -> float:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000.0
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--units", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    url = f"{args.url}?units={args.units}"
    lat, errors = [], 0
    stop = time.perf_counter() + args.duration

    def worker():
        nonlocal errors
        while time.perf_counter() < stop:
            ms = one_request(url, args.timeout)
            if ms < 0:
                errors += 1
            else:
                lat.append(ms)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for _ in range(args.concurrency):
            ex.submit(worker)

    lat.sort()
    n = len(lat)

    def pct(p):
        return lat[min(n - 1, int(p * n))] if n else -1.0

    out = {
        "concurrency": args.concurrency,
        "units": args.units,
        "duration_s": args.duration,
        "completed": n,
        "errors": errors,
        "throughput_rps": n / args.duration if args.duration else 0.0,
        "lat_ms_mean": sum(lat) / n if n else -1.0,
        "lat_ms_p50": pct(0.50),
        "lat_ms_p90": pct(0.90),
        "lat_ms_p99": pct(0.99),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
