#!/usr/bin/env bash
# Held-out day rotation for the runs day_rotation_sweep.sh does not cover:
#   dynamic: NSGA-II, MOEA/D, MORL, random search and tuned HPA 
#   edge:    NSGA-II, MOEA/D, MORL with the edge/cloud placement axis
#   random:  static random search, raw and SLA_MS=20, into the existing
#            test<t>_raw and test<t>_sla dirs
# Train days d..d+2, test day d+3, test days 3..13, base seed 42, seeds 1..10,
# 500 evals, pop 20, K=5. Same flags as reproduce.sh (DYNAMIC=1 applies MOEA/D
# neighbors 5 and the MORL lr anneal and latency reward floor).
# One job per (day, method, seed), so the load spreads evenly over nodes.
#
# Usage:  bash experiments/dynamic_day_rotation_sweep.sh <NODE_INDEX> <NUM_NODES>
# FILTER=<regex> runs only the matching jobs (e.g. a timing pilot).
set -uo pipefail
cd "$(dirname "$0")/.."

NODE=${1:?node index}; NN=${2:?num nodes}
PY=venv/bin/python
TR=data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
OUT=results_dayrot; mkdir -p "$OUT/logs"
LOG="$OUT/dyn_node${NODE}.log"
DAYS="3 4 5 6 7 8 9 10 11 12 13"
MORL_DYN="--batch 20 --lr 0.6 --lr-end 0.2 --latency-reward-floor 0.2"

jobs_list() {
  for t in $DAYS; do
    for s in $(seq 1 10); do
      d="test${t}_dynamic"
      echo "$t $d/nsga2_seed$s.npz experiments.run_nsga2 --dynamic --pop 20 --no-plot --seed $s"
      echo "$t $d/moead_seed$s.npz experiments.run_moead --dynamic --partitions 12 --neighbors 5 --no-plot --seed $s"
      echo "$t $d/morl_seed$s.npz experiments.run_morl --dynamic $MORL_DYN --no-plot --seed $s"
      echo "$t $d/random_seed$s.npz experiments.run_random --dynamic --pop 20 --seed $s"
      echo "$t $d/hpa_seed$s.npz experiments.run_hpa --pop 20 --seed $s"
    done
  done
  for t in $DAYS; do
    for s in $(seq 1 10); do
      d="test${t}_edge"
      echo "$t $d/nsga2_seed$s.npz experiments.run_nsga2 --edge --pop 20 --no-plot --seed $s"
      echo "$t $d/moead_seed$s.npz experiments.run_moead --edge --partitions 12 --no-plot --seed $s"
      echo "$t $d/morl_seed$s.npz experiments.run_morl --edge --batch 20 --no-plot --seed $s"
      echo "$t test${t}_raw/random_seed$s.npz experiments.run_random --pop 20 --seed $s"
      echo "$t test${t}_sla/random_seed$s.npz experiments.run_random --sla-ms 20 --pop 20 --seed $s"
    done
  done
}

run_one() {
  t=$1; out="$OUT/$2"; shift 2
  [ -f "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  tag=$(echo "${out#$OUT/}" | tr '/' '_' | sed 's/\.npz$//')
  if $PY -m "$@" --workload azure --trace "$TR" \
      --train-days "$((t - 3)),$((t - 2)),$((t - 1))" --test-day "$t" \
      --evals 500 --k 5 --base-seed 42 --out "$out" > "$OUT/logs/$tag.txt" 2>&1; then
    echo "[$(date '+%F %T')] DONE $out"
  else
    echo "[$(date '+%F %T')] FAIL $out (see $OUT/logs/$tag.txt)"
  fi
}
export -f run_one; export PY OUT TR

echo "[$(date '+%F %T')] node$NODE of $NN start" >> "$LOG"
jobs_list | grep -E "${FILTER:-.}" | awk -v n="$NN" -v k="$NODE" '(NR - 1) % n == k' \
  | xargs -P "$(nproc)" -I{} bash -c 'run_one {}' >> "$LOG" 2>&1
echo "[$(date '+%F %T')] node$NODE of $NN finished" >> "$LOG"
