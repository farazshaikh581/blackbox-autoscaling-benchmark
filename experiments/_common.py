"""Shared helpers for the algorithm runners."""
from __future__ import annotations

import os

import numpy as np


def add_workload_args(ap) -> None:
    """Register the shared workload-selection flags on an ArgumentParser.

    The offline benchmark defaults to the synthetic diurnal profile so it runs
    out of the box; `--workload azure` swaps in the Azure Functions 2021
    invocation trace. The trace path may also come from the AZURE_TRACE
    environment variable.

    `--train-days` / `--test-day` turn on the held-out-day protocol: every
    algorithm spends its search budget on the concatenated train days, then its
    final configurations are re-scored once on the held-out test day so that the
    fronts being compared measure generalization rather than fit to a single
    realization. Both families see exactly the same train and test traces.
    """
    ap.add_argument("--workload", choices=["synthetic", "azure"], default="synthetic",
                    help="arrival-rate source (default: synthetic diurnal)")
    ap.add_argument("--trace", type=str, default=os.environ.get("AZURE_TRACE"),
                    help="Azure trace path for --workload azure (or $AZURE_TRACE)")
    ap.add_argument("--day", type=int, default=0, help="day index of the Azure trace")
    ap.add_argument("--train-days", type=str, default=None,
                    help="comma-separated Azure day indices to train on (held-out "
                         "protocol), e.g. 0,1,2")
    ap.add_argument("--test-day", type=int, default=None,
                    help="held-out Azure day index the final front is scored on")
    ap.add_argument("--sla-ms", type=float, default=None,
                    help="optional SLA-hinge framing: replace the latency objective "
                         "with max(0, latency_ms - SLA) so the search stops chasing "
                         "sub-SLA latency and trades cost/energy within the feasible "
                         "region (e.g. --sla-ms 20). Off by default (raw latency).")


def add_topology_args(ap) -> None:
    """Register the topology-selection flag shared by the runners.

    `--edge` swaps the single-location default chain for the 2-class edge/cloud
    topology, which adds a per-tier placement decision variable. Off by default,
    so the standard benchmark is unchanged.
    """
    ap.add_argument("--edge", action="store_true",
                    help="use the 2-class edge/cloud topology (adds a per-tier "
                         "placement variable); default is single-location")


def build_topology(args):
    """Build the topology selected by `add_topology_args` (and `--tiers`)."""
    from blackbox import default_topology, edge_cloud_topology
    tiers = getattr(args, "tiers", 3)
    topo = (edge_cloud_topology(tiers) if getattr(args, "edge", False)
            else default_topology(tiers))
    return topo


def _parse_days(spec: str):
    return [int(s) for s in str(spec).replace(" ", "").split(",") if s != ""]


def build_workload(args):
    """Construct the (single) Workload selected by `add_workload_args` flags.

    This is the search workload. For the held-out protocol it is the concatenated
    train days; otherwise it is the single day (or the synthetic profile).
    """
    train, _ = build_train_test(args)
    return train


def build_train_test(args):
    """Return `(train_workload, test_workload)` for the selected source.

    `test_workload` is None unless the held-out-day protocol is requested
    (`--workload azure` with `--test-day`, and optionally `--train-days`). Both
    families call this with identical args, so they train and test on exactly the
    same traces.
    """
    from blackbox.workload import synthetic_diurnal, from_azure_trace
    if getattr(args, "workload", "synthetic") != "azure":
        return synthetic_diurnal(), None
    if not args.trace:
        raise SystemExit(
            "--workload azure requires --trace <path> (or $AZURE_TRACE). "
            "Download AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt; "
            "see README."
        )
    test_day = getattr(args, "test_day", None)
    if test_day is None:
        # No held-out day: single-day search workload, backward compatible.
        return from_azure_trace(args.trace, day_index=args.day), None
    train_days_spec = getattr(args, "train_days", None)
    train_days = _parse_days(train_days_spec) if train_days_spec else [args.day]
    if test_day in train_days:
        raise SystemExit(
            f"--test-day {test_day} must be held out of --train-days {train_days}"
        )
    train = from_azure_trace(args.trace, days=train_days)
    test = from_azure_trace(args.trace, day_index=test_day)
    return train, test


def bounded_front(F, cap: int):
    """Non-dominated set of `F`, capped to `cap` points by crowding distance.

    The reported-front fairness rule shared by every algorithm: keep only the
    non-dominated rows, and if there are more than `cap` (the EA population size),
    drop the most crowded ones (NSGA-II crowding distance) until `cap` remain, so
    the compared fronts have comparable cardinality regardless of how many
    configurations a method produced.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    F = np.atleast_2d(np.asarray(F, float))
    F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    cap = int(cap)
    while len(F) > cap:
        F = np.delete(F, int(np.argmin(_crowding(F))), axis=0)
    return F


def _crowding(F):
    n, m = F.shape
    cd = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k])
        cd[order[0]] = cd[order[-1]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span == 0:
            continue
        cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return cd


def holdout_front(problem_test, X, cap: int):
    """Re-score a run's produced decision vectors on the held-out test problem.

    `problem_test` is a problem built with the *test* workload; `X` is the final
    non-dominated decision set from the search (on train). Returns the capped
    non-dominated front of the test-day objectives -- the object compared across
    algorithms under the held-out protocol.
    """
    F = problem_test.evaluate(X, return_values_of=["F"])
    return bounded_front(F, cap)


def save_history(path, F, hist_n_evals, hist_F, algo: str, seed: int,
                 F_train=None) -> None:
    """Save a final front and its anytime history in the aggregator's format.

    `F` is the final non-dominated set that `aggregate.py` compares (rows are
    objective vectors). Under the held-out protocol `F` is the test-day front and
    `F_train` carries the train-day front for reference; the anytime history
    (`hist_n_evals`, `hist_F`) is always the train-side search progress, since the
    test day is never seen during search. Object arrays require pickle on load.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        F=np.atleast_2d(F),
        hist_n_evals=np.asarray(hist_n_evals, dtype=int),
        hist_F=np.array([np.atleast_2d(f) for f in hist_F], dtype=object),
        algo=algo,
        seed=seed,
    )
    if F_train is not None:
        payload["F_train"] = np.atleast_2d(F_train)
    np.savez(path, **payload)
    print(f"saved -> {path}")


def plot_run_from_history(hist_n, hist_F, F, algo: str, path: str,
                          label: str | None = None) -> None:
    """Convergence (front size vs evaluations) + final Pareto-front overlay for
    one algorithm run, saved standalone (no cross-algorithm normalization, so
    this is available immediately after a single `run_*.py` call, unlike
    `aggregate.py`'s hypervolume convergence which needs a multi-seed,
    multi-algorithm reference set)."""
    from viz import SINGLE, method_style, new_fig, plot_single_front, savefig, style_axes

    F = np.atleast_2d(F)
    st = method_style(algo)
    label = label or st["label"]

    if hist_n:
        sizes = [len(np.atleast_2d(f)) for f in hist_F]
        fig, ax = new_fig(SINGLE, 2.2)
        ax.plot(hist_n, sizes, color=st["color"])
        ax.set_xlabel("evaluations")
        ax.set_ylabel("non-dominated points")
        ax.set_title(f"{label}: front size during search")
        ax.set_ylim(bottom=0)
        style_axes(ax)
        savefig(fig, path.replace(".png", "_convergence.png"))

    # Same convention as aggregate.py's front overlay: drop timed-out/infeasible
    # points so one outlier doesn't collapse the axes for the rest of the front.
    F_plot = F[F[:, 0] < 5000] if (F[:, 0] >= 5000).any() else F
    plot_single_front(F_plot, algo, path.replace(".png", "_front.png"), label=label)


def plot_single_run(res, F, algo: str, path: str, label: str | None = None) -> None:
    """`plot_run_from_history` for a pymoo `res` (NSGA-II/MOEA-D), reading the
    anytime history off `res.history` (requires `minimize(..., save_history=True)`)."""
    hist_n, hist_F = [], []
    if getattr(res, "history", None):
        hist_n = [a.evaluator.n_eval for a in res.history]
        hist_F = [a.opt.get("F") for a in res.history]
    plot_run_from_history(hist_n, hist_F, F, algo, path, label=label)


def save_run(path: str, res, algo: str, seed: int, F_test=None) -> None:
    """Save the final front and the anytime history of a pymoo run.

    Stores, per generation, the cumulative evaluation count and the current
    non-dominated set, so `aggregate.py` can compute hypervolume against
    evaluations. When `F_test` is given (held-out protocol) it becomes the
    compared front `F`, and the train front `res.F` is kept as `F_train`.
    """
    hist_n = [a.evaluator.n_eval for a in res.history]
    hist_F = [a.opt.get("F") for a in res.history]
    if F_test is not None:
        save_history(path, F_test, hist_n, hist_F, algo, seed, F_train=res.F)
    else:
        save_history(path, res.F, hist_n, hist_F, algo, seed)
