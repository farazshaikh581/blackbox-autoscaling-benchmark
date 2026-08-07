#!/usr/bin/env bash
# Reproduce the black-box autoscaling benchmark end-to-end on the offline oracle.
# Uses documented placeholder physics (see topology.py CALIBRATE markers); swap
# in host-calibrated constants via calibration/ before reporting final numbers.
set -euo pipefail

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
EVALS="${EVALS:-500}"
POP="${POP:-20}"
K="${K:-5}"
TIERS="${TIERS:-3}"
OUT="${OUT:-results}"

echo "== tests =="
$PY -m pytest tests/ -q

echo "== preparatory: K via coefficient of variation =="
$PY -m experiments.cov_replications --configs 20 --kmax 40 --target 0.05

echo "== preparatory: wall-clock evaluation budget =="
$PY -m experiments.timing_benchmark --k "$K" --tiers "$TIERS" --samples 100 \
    --window-hours 6 --runs 10 --algos 3

mkdir -p "$OUT"
for s in $SEEDS; do
  echo "== NSGA-II seed $s =="
  $PY -m experiments.run_nsga2 --pop "$POP" --evals "$EVALS" --seed "$s" \
      --tiers "$TIERS" --k "$K" --out "$OUT/nsga2_seed${s}.npz"
  echo "== MOEA/D seed $s =="
  $PY -m experiments.run_moead --partitions 12 --evals "$EVALS" --seed "$s" \
      --tiers "$TIERS" --k "$K" --out "$OUT/moead_seed${s}.npz"
  echo "== MORL seed $s =="
  $PY -m experiments.run_morl --evals "$EVALS" --seed "$s" \
      --tiers "$TIERS" --k "$K" --out "$OUT/morl_seed${s}.npz"
done

echo "== RL weights vs MOEA/D reference directions, front analysis =="
$PY -m experiments.analyze_fronts --results "$OUT" --partitions 6 --seed 1 \
    --tiers "$TIERS" --k "$K"

echo "== done: results in $OUT/ =="
