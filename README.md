# Black-box multi-objective autoscaling benchmark: paper artifact

This branch (`paper-artifact`) holds the code, testbed data and stored
results behind the paper *Black-Box Multi-Objective Problem Modeling and
Benchmarking for Cloud Resource Management* (under review). The stored
results are attached to the `paper-artifact-v1` release.

The `main` branch of this repository is a separate deliverable: the benchmark
as produced during a Short-Term Scientific Mission of COST Action CA22137
(ROAR-NET), tagged `v0.1`. This branch has its own history and extends that
work with the edge-cloud and dynamic variants, the tuned HPA baseline, the
held-out day rotation and the six-node testbed check.

## Contents

| Path | What it is |
|---|---|
| `blackbox/` | The oracle: latency, cost and energy for the static, edge-cloud and dynamic variants, and the tuned HPA rule |
| `experiments/` | NSGA-II, MOEA/D, MORL, random search and tuned HPA runners, the held-out day rotation sweeps, HV, IGD+, GD+ and Wilcoxon tests, and the analysis scripts |
| `calibration/` | Testbed scripts and measured data: node power curves, three-tier chain, memory limit sweep, six-node front deployment, network delay |
| `tests/` | Unit tests |

## Setup

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
make test
```

The searches use the Azure Functions 2021 invocation trace. Download
`AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar` from
https://github.com/Azure/AzurePublicDataset and extract the `.txt` file into
`data/`.

## Stored results

`paper-artifact-results.tar.gz` on the `paper-artifact-v1` release holds the
test-day front of every run. Extract it in the repository root:

| Folder | Content |
|---|---|
| `results_dayrot/test<d>_<variant>/` | Held-out day rotation: test days 3 to 13, variants `raw`, `sla`, `edge`, `dynamic`, one `<method>_seed<s>.npz` per run (10 seeds) |
| `results_2000/` | Static raw latency on test day 3 with 2000 oracle calls |
| `results_mechanism_check/` | Decision-space coherence for the static, dynamic and HPA spaces |
| `results_placement_distinctness/` | Edge-cloud placement check, 10 seeds |

Each `.npz` file stores `F`, the run's front scored on the test day (latency
in ms, cost, energy in W; under the SLA framing the latency column is the
hinge max(0, latency - 20)), and `F_train`, the same set on the search days.
`experiments/aggregate.py` reads a folder and reports HV, IGD+, GD+ and
Wilcoxon tests.

## Rerunning the searches

```bash
make sweep-static  NODE=0 NODES=1   # raw and SLA framings
make sweep-dynamic NODE=0 NODES=1   # dynamic, edge-cloud, random search
make mechanism                      # decision-space coherence
make placement                      # edge-cloud placement check
make budget2000                     # 2000 calls on test day 3
make noise                          # CV of the objectives vs K
make timing                         # oracle call time
```

`NODE` and `NODES` split the jobs across machines. The full rerun is on the
order of 100 CPU hours, mostly the dynamic variant. All runs are seeded.

## Testbed

The testbed scripts need a k3s cluster with RAPL readable on every node
(`/sys/class/powercap/intel-rapl:0/energy_uj`). The paper used six bare-metal
nodes with an Intel Core i5-9500 (6 cores) each.

| Measurement | Script | Data |
|---|---|---|
| Node power curve | `calibration/cluster/measure_edge_node.py --edge-cores 6` | `calibration/data/edge_node_measured_curve_r1.json`, `_r2.json` |
| Edge power curve (4 of 6 cores) | `calibration/cluster/measure_edge_node.py` | `calibration/data/edge_node_sweep.csv`, `edge_node_measured_curve.json` |
| Three-tier chain | `calibration/measure_chain.py`, `fit_chain.py` | `calibration/data/chain_sweep.csv` |
| Memory limit sweep | `calibration/run_sweep.py`, `fit_cluster.py` | `calibration/data/cluster_sweep.csv` |
| Six-node front deployment | `experiments/dump_front.py`, `calibration/measure_front_multinode.py` | `calibration/data/front_multinode.csv` |
| Network delay | `calibration/measure_wan_latency.py` | printed by the script |

`calibration/cluster/` holds the chain application, the load generator and
the deploy scripts. Host names, interfaces and addresses in the scripts are
those of our testbed and must be changed for another cluster.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
