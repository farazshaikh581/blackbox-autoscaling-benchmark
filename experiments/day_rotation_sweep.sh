#!/usr/bin/env bash
# Held-out day rotation: slide the train/test window across the two-week trace
# (train days d..d+2, test day d+3) so no result rests on test day 3 alone.
# Raw and SLA_MS=20 at every test day 3..13, with NSGA-II, MOEA/D and MORL.
# Random search for these two framings runs in dynamic_day_rotation_sweep.sh.
# Base seed 42 throughout, matching the canonical day-3 protocol.
#
# Usage:  bash experiments/day_rotation_sweep.sh <NODE_INDEX> <NUM_NODES>
set -uo pipefail
cd "$(dirname "$0")/.."

NODE=${1:?node index}; NN=${2:?num nodes}
PY=venv/bin/python
TR=data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
JPN=$(nproc)
LOGDIR=results_dayrot; mkdir -p "$LOGDIR"
LOG="$LOGDIR/node${NODE}.log"
log(){ echo "[$(date '+%F %T')] node$NODE: $*" | tee -a "$LOG"; }

JOBS_LIST=()
for t in 3 4 5 6 7 8 9 10 11 12 13; do JOBS_LIST+=("$t raw" "$t sla"); done

log "=== day rotation start ($NODE of $NN) ==="
k=0
for job in "${JOBS_LIST[@]}"; do
  if [ $((k % NN)) -eq "$NODE" ]; then
    read -r t mode <<<"$job"
    train="$((t - 3)),$((t - 2)),$((t - 1))"
    out="$LOGDIR/test${t}_${mode}"
    extra=()
    case "$mode" in
      sla) extra=(SLA_MS=20) ;;
    esac
    if [ -f "$out/COMPLETE" ]; then
      log "skip $out (COMPLETE)"
    else
      log "START $out  (train $train, test $t, $mode)"
      env WORKLOAD=azure PYTHON="$PY" TRACE="$TR" OUT="$out" JOBS="$JPN" \
          BASE_SEED=42 TRAIN_DAYS="$train" TEST_DAY="$t" "${extra[@]}" \
          bash reproduce.sh >>"$LOG" 2>&1 \
        && touch "$out/COMPLETE" && log "DONE  $out" \
        || log "FAIL  $out (see $LOG)"
    fi
  fi
  k=$((k + 1))
done
log "=== day rotation finished ($NODE of $NN) ==="
