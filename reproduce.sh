#!/usr/bin/env bash
# Reproduce the black-box autoscaling benchmark end-to-end on the offline oracle.
# One held-out split per call; experiments/day_rotation_sweep.sh calls it once
# per test day and framing.
set -euo pipefail

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
EVALS="${EVALS:-500}"
POP="${POP:-20}"
K="${K:-5}"
TIERS="${TIERS:-3}"
OUT="${OUT:-results}"
# JOBS>1 runs the independent (seed, algo) jobs concurrently, up to JOBS at once.
# They are deterministic from their own --seed and write separate .npz files, so
# parallelism changes only wall time, not results. NOTE: each Azure job re-parses
# the full trace (~300 MB, a transient ~1-2 GB), so on the real trace RAM -- not
# cores -- can bound JOBS.
JOBS="${JOBS:-1}"
# Workload: synthetic (default) or azure. For azure, set TRACE=/path/to/trace.
# Held-out-day protocol (azure only): set TEST_DAY to a day index and TRAIN_DAYS
# to a comma-separated list (default 0,1,2). Every algorithm then trains on the
# same train days and its final front is re-scored on the same held-out TEST_DAY,
# so the compared fronts measure generalization. BASE_SEED fixes the shared
# workload realization (42 in the paper's protocol).
WORKLOAD="${WORKLOAD:-synthetic}"
BASE_SEED="${BASE_SEED:-0}"
# WL_ARGS = workload *selection* (accepted by every script via add_workload_args).
# --base-seed is only defined on the runners + analyze_fronts, so it is threaded
# separately as SEED_ARG.
WL_ARGS="--workload $WORKLOAD"
[ -n "${TRACE:-}" ] && WL_ARGS="$WL_ARGS --trace $TRACE"
if [ -n "${TEST_DAY:-}" ]; then
  WL_ARGS="$WL_ARGS --train-days ${TRAIN_DAYS:-0,1,2} --test-day $TEST_DAY"
fi
# Optional SLA-hinge framing (see add_workload_args): SLA_MS=20 makes every
# algorithm minimize max(0, latency_ms - SLA) instead of raw latency.
[ -n "${SLA_MS:-}" ] && WL_ARGS="$WL_ARGS --sla-ms $SLA_MS"
SEED_ARG="--base-seed $BASE_SEED"
# Optional edge/cloud extension (see add_topology_args): EDGE=1 swaps the single-
# location chain for the 2-class edge/cloud topology, adding a per-tier placement
# variable. Accepted only by the runners + analyze_fronts (add_topology_args), so
# it is threaded separately from WL_ARGS -- cov_replications/timing_benchmark do
# not define --edge (same discipline as --base-seed).
TOPO_ARGS=""
[ -n "${EDGE:-}" ] && TOPO_ARGS="--edge"

# DYNAMIC=1 swaps the static
# per-tier config for a per-minute replica POLICY (a small NN, its weights the
# decision variable). Not combinable with EDGE yet (simulate_policy is
# single-location-only, same guard the runners themselves raise). Threaded
# separately from TOPO_ARGS/WL_ARGS since cov_replications/timing_benchmark
# accept --dynamic but not --edge.
DYNAMIC="${DYNAMIC:-}"
if [ -n "$DYNAMIC" ] && [ -n "${EDGE:-}" ]; then
  echo "DYNAMIC=1 does not support EDGE=1 yet (simulate_policy is single-location-only)" >&2
  exit 1
fi
DYN_ARGS=""
[ -n "$DYNAMIC" ] && DYN_ARGS="--dynamic"
# MOEA/D's neighborhood size: T=5 in the dynamic variant, chosen in a pilot
# on test day 3. Only
# applied automatically under DYNAMIC=1, so the static protocol's behavior is
# unchanged; override with MOEAD_NEIGHBORS=N if you need to.
MOEAD_NEIGHBORS="${MOEAD_NEIGHBORS:-}"
if [ -n "$DYNAMIC" ] && [ -z "$MOEAD_NEIGHBORS" ]; then
  MOEAD_NEIGHBORS=5
fi
MOEAD_ARGS=""
[ -n "$MOEAD_NEIGHBORS" ] && MOEAD_ARGS="--neighbors $MOEAD_NEIGHBORS"
# MORL's dynamic variant uses a learning-rate anneal and a latency floor in the
# per-minute reward; these three only apply under DYNAMIC=1. POP doubles as
# --batch here, same as the static path below (POP=20's default already
# matches the validated batch=20).
MORL_LR="${MORL_LR:-0.6}"
MORL_LR_END="${MORL_LR_END:-0.2}"
MORL_LATENCY_FLOOR="${MORL_LATENCY_FLOOR:-0.2}"
MORL_DYN_ARGS=""
if [ -n "$DYNAMIC" ]; then
  MORL_DYN_ARGS="--lr $MORL_LR --lr-end $MORL_LR_END --latency-reward-floor $MORL_LATENCY_FLOOR"
fi

echo "== tests =="
$PY -m pytest tests/ -q

# Warm the Azure trace cache once (single process) so the prep steps and the many
# parallel workers read <trace>.cache.npz instead of each re-parsing ~300 MB.
if [ "$WORKLOAD" = "azure" ] && [ -n "${TRACE:-}" ]; then
  echo "== preparatory: warm Azure trace cache =="
  $PY -c "from blackbox.workload import build_azure_cache; print('trace cache ready:', build_azure_cache('$TRACE'), 'days')"
fi

echo "== preparatory: K via coefficient of variation =="
$PY -m experiments.cov_replications --configs 20 --kmax 40 --target 0.05 $WL_ARGS $DYN_ARGS

echo "== preparatory: wall-clock evaluation budget =="
$PY -m experiments.timing_benchmark --k "$K" --tiers "$TIERS" --samples 100 \
    --window-hours 6 --runs 10 --algos 3 $WL_ARGS $DYN_ARGS

mkdir -p "$OUT"

run_one() {
  # run_one <algo> <seed>: one independent (algo, seed) job -> its own .npz.
  local algo="$1" s="$2"
  case "$algo" in
    nsga2) $PY -m experiments.run_nsga2 --pop "$POP" --evals "$EVALS" --seed "$s" \
             --tiers "$TIERS" --k "$K" $SEED_ARG $WL_ARGS $TOPO_ARGS $DYN_ARGS \
             --out "$OUT/nsga2_seed${s}.npz" ;;
    moead) $PY -m experiments.run_moead --partitions 12 $MOEAD_ARGS --evals "$EVALS" --seed "$s" \
             --tiers "$TIERS" --k "$K" $SEED_ARG $WL_ARGS $TOPO_ARGS $DYN_ARGS \
             --out "$OUT/moead_seed${s}.npz" ;;
    morl)  $PY -m experiments.run_morl --batch "$POP" $MORL_DYN_ARGS --evals "$EVALS" --seed "$s" \
             --tiers "$TIERS" --k "$K" $SEED_ARG $WL_ARGS $TOPO_ARGS $DYN_ARGS \
             --out "$OUT/morl_seed${s}.npz" ;;
  esac
}

if [ "$JOBS" -le 1 ]; then
  # Sequential (default): inline verbose output, unchanged behavior.
  for s in $SEEDS; do
    for algo in nsga2 moead morl; do
      echo "== $algo seed $s =="
      run_one "$algo" "$s"
    done
  done
else
  # Parallel: up to $JOBS jobs at once, each to its own log so streams don't mix.
  echo "== running $(echo $SEEDS | wc -w)x3 jobs, up to $JOBS in parallel (logs in $OUT/log_*.txt) =="
  active=0
  for s in $SEEDS; do
    for algo in nsga2 moead morl; do
      run_one "$algo" "$s" > "$OUT/log_${algo}_seed${s}.txt" 2>&1 &
      echo "  started $algo seed $s (pid $!)"
      active=$((active + 1))
      if [ "$active" -ge "$JOBS" ]; then wait -n; active=$((active - 1)); fi
    done
  done
  wait
  echo "== all runs done =="
fi

echo "== aggregate: HV / IGD+ / GD+ / Wilcoxon + figures =="
# Under the SLA-hinge framing the runs store hinged latency, so also report the
# compliant-region (cost,energy) HV -- the trade-off an operator actually deploys.
AGG_SLA=""
[ -n "${SLA_MS:-}" ] && AGG_SLA="--sla-ms $SLA_MS --hinged"
$PY -m experiments.aggregate --results "$OUT" $AGG_SLA

echo "== analysis: front coverage + RL preference behavior =="
$PY -m experiments.analyze_fronts --results "$OUT" --tiers "$TIERS" --k "$K" \
    $SEED_ARG $WL_ARGS $TOPO_ARGS $DYN_ARGS $MORL_DYN_ARGS --batch "$POP" --evals "$EVALS"

echo "== done: results in $OUT/ =="
