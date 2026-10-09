"""Validate the additive WAN-latency assumption on real hardware: the model
adds `wan_rtt_ms` once per cross-class tier boundary (`topology.py Topology.wan_rtt`,
default 30 ms edge<->cloud). r0/r1/r2 sit on one flat, sub-millisecond LAN
(measured: ~0.5-0.8 ms all pairs), so there is no natural cross-segment RTT to
measure -- this injects a real `tc netem` delay on the specific r1<->r2 path
(targeted by peer IP, so it only touches pod-to-pod traffic between those two
nodes, not each node's link to the k3s API server) and checks whether forcing
one tier boundary across it adds the expected latency, the same additive-sum
methodology as the chain confirmation (calibration/measure_chain.py).

Uses small WORK_UNITS (fast per-tier compute, ~tens of ms) so the ~30 ms
injected delay is a clearly visible fraction of end-to-end latency rather than
buried in the chain's normal (much larger, ~1s-scale) per-tier compute time.

Three deployments, same chain, only placement differs:
  same-node:   frontend, logic, backend all pinned to r1 (no cross-node hop)
  cross-node:  frontend+logic on r1, backend on r2 (one hop), no netem yet
  cross+netem: same cross-node placement, with tc netem delay injected

    python -m calibration.measure_wan_latency
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from viz import COLOR, INK_2, SINGLE, new_fig, style_axes, savefig

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
TIERS = ["frontend", "logic", "backend"]
ENV = {"frontend": "FE", "logic": "LOGIC", "backend": "BE"}
NS = "blackbox"
DEPLOY = os.path.join(os.path.dirname(__file__), "cluster", "deploy_chain.sh")

# Determined via `ip -d link show flannel.1` + `bridge fdb show dev flannel.1`
# on each node: the VXLAN tunnel (pod-to-pod traffic) binds to `eno1`, NOT the
# separate isolated "cluster LAN" (enp1s0f3/enp1s0f0) that first looked like
# the obvious candidate -- that network carries k3s API server traffic
# (confirmed via tcpdump: port 6443 between r0 and r1), not pod data-plane
# traffic. A tcpdump trace during a real pod-to-pod request showed ZERO packets
# on enp1s0f3/enp1s0f0, and flannel.1's own `local`/fdb entries point at each
# node's eno1 address instead. This is why the first netem attempt (correctly
# delaying raw ICMP to the cluster-LAN IP) had no reliable effect on HTTP
# latency between pods: it was delaying a link the pod traffic never used.
# Set these for your cluster: each node's host name, the interface flannel's
# VXLAN tunnel uses, and that interface's address.
NODE = {
    "r1": {"fqdn": "r1.testbed.example",
           "iface": "eno1", "cluster_ip": "192.0.2.1"},
    "r2": {"fqdn": "r2.testbed.example",
           "iface": "eno1", "cluster_ip": "192.0.2.2"},
}
DELAY_MS = 15  # each direction; r1->r2 delayed + r2->r1 delayed = 30ms round trip
WORK_UNITS = 2  # small: keeps per-tier compute a few tens of ms, not ~1s


def ssh(alias, cmd, timeout=30, check=True):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", alias, cmd],
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh {alias} {cmd!r}: {r.stderr.strip()}")
    return r.stdout.strip()


def inject_delay(alias, peer_ip, delay_ms):
    iface = NODE[alias]["iface"]
    remove_delay(alias, check=False)  # idempotent: clear any stale qdisc first
    ssh(alias, f"sudo -n tc qdisc add dev {iface} root handle 1: prio")
    ssh(alias, f"sudo -n tc qdisc add dev {iface} parent 1:3 handle 30: "
              f"netem delay {delay_ms}ms")
    ssh(alias, f"sudo -n tc filter add dev {iface} protocol ip parent 1:0 "
              f"prio 3 u32 match ip dst {peer_ip} flowid 1:3")


def remove_delay(alias, check=True):
    iface = NODE[alias]["iface"]
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", alias,
                        f"sudo -n tc qdisc del dev {iface} root"],
                       capture_output=True, text=True, timeout=30)
    if check and r.returncode != 0 and "No such" not in r.stderr:
        print(f"  warning: cleanup on {alias} said: {r.stderr.strip()}", file=sys.stderr)


def kubectl(*a, timeout=60, check=True):
    r = subprocess.run(["kubectl", "-n", NS, *a], capture_output=True, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout.strip()


def deploy(fe_node, logic_node, be_node):
    env = dict(os.environ)
    env["FE_UNITS"] = env["LOGIC_UNITS"] = env["BE_UNITS"] = str(WORK_UNITS)
    env["FE_NODE"] = fe_node; env["LOGIC_NODE"] = logic_node; env["BE_NODE"] = be_node
    r = subprocess.run(["bash", DEPLOY], env=env, capture_output=True, text=True,
                       timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"deploy failed: {r.stderr.strip()[-500:]}")


def measure(duration=15.0, concurrency=1):
    pod = kubectl("get", "pod", "-l", "app=loadgen", "-o",
                  "jsonpath={.items[0].metadata.name}")
    url = f"http://frontend.{NS}.svc:8080/work"
    out = kubectl("exec", pod, "--", "python", "/load/loadgen.py", "--url", url,
                  "--units", str(WORK_UNITS), "--concurrency", str(concurrency),
                  "--duration", str(duration), timeout=int(duration) + 60)
    return json.loads(out.splitlines()[-1])


def pod_nodes():
    out = kubectl("get", "pod", "-l", "chain=blackbox", "-o",
                  "jsonpath={range .items[*]}{.metadata.labels.app}{\" \"}"
                  "{.spec.nodeName}{\"\\n\"}{end}")
    return dict(line.split() for line in out.splitlines() if len(line.split()) == 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    r1, r2 = NODE["r1"]["fqdn"], NODE["r2"]["fqdn"]
    results = {}
    warmup_s = min(args.duration, 10.0)

    print("[1/3] same-node baseline: all tiers on r1 ...")
    deploy(r1, r1, r1)
    print("  placement:", pod_nodes())
    print(f"  warming up ({warmup_s}s, discarded)...")
    measure(warmup_s)
    results["same_node"] = measure(args.duration)
    print("  ", results["same_node"])

    print("\n[2/3] cross-node, no netem: frontend+logic on r1, backend on r2 ...")
    deploy(r1, r1, r2)
    print("  placement:", pod_nodes())
    print(f"  warming up ({warmup_s}s, discarded)...")
    measure(warmup_s)
    results["cross_node_no_delay"] = measure(args.duration)
    print("  ", results["cross_node_no_delay"])

    print(f"\n[3/3] cross-node WITH netem ({DELAY_MS}ms each direction, "
          f"targeted r1<->r2) ...")
    try:
        inject_delay("r1", NODE["r2"]["cluster_ip"], DELAY_MS)
        inject_delay("r2", NODE["r1"]["cluster_ip"], DELAY_MS)
        # same pods as step 2 (no redeploy), but warm up again: the delay
        # itself can disrupt in-flight keep-alive connections, and we want the
        # netem-vs-no-netem comparison clean of any reconnection cost too.
        print(f"  warming up ({warmup_s}s, discarded)...")
        measure(warmup_s)
        results["cross_node_with_delay"] = measure(args.duration)
        print("  ", results["cross_node_with_delay"])
    finally:
        remove_delay("r1")
        remove_delay("r2")
        print("  netem removed from r1 and r2")

    same = results["same_node"]["lat_ms_p50"]
    cross_no_delay = results["cross_node_no_delay"]["lat_ms_p50"]
    cross_delay = results["cross_node_with_delay"]["lat_ms_p50"]
    real_lan_delta = cross_no_delay - same
    netem_delta = cross_delay - cross_no_delay
    expected = 2 * DELAY_MS

    print(f"\nsame-node p50:              {same:8.2f} ms")
    print(f"cross-node p50 (no netem):  {cross_no_delay:8.2f} ms  "
          f"(delta vs same-node: {real_lan_delta:+.2f} ms -- real flat-LAN cost)")
    print(f"cross-node p50 (netem):     {cross_delay:8.2f} ms  "
          f"(delta vs no-netem: {netem_delta:+.2f} ms, expected ~{expected} ms)")
    if expected:
        print(f"measured/expected ratio: {netem_delta / expected:.3f}")

    plot_wan(same, cross_no_delay, cross_delay, expected,
            os.path.join(FIG_DIR, "wan_latency.png"))


def plot_wan(same, cross_no_delay, cross_delay, expected_delay_ms, path):
    labels = ["same node", "cross node", f"cross node\n+{expected_delay_ms:.0f} ms RTT"]
    vals = [same, cross_no_delay, cross_delay]
    fig, ax = new_fig(SINGLE, 2.4)
    bars = ax.bar(labels, vals, width=0.55, color=COLOR["measured"])
    ax.bar_label(bars, labels=[f"{v:.0f} ms" for v in vals], padding=2,
                 fontsize=7, color=INK_2)
    exp = cross_no_delay + expected_delay_ms
    ax.hlines(exp, 1.72, 2.28, color=COLOR["fit"], lw=1.4, ls="--")
    ax.annotate(f"additive model\n{exp:.0f} ms", (1.72, exp), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=7, color=INK_2)
    ax.set_ylabel("p50 chain latency (ms)")
    ax.set_title("Injected WAN delay adds to chain latency")
    ax.margins(y=0.15)
    style_axes(ax)
    savefig(fig, path)


if __name__ == "__main__":
    main()
