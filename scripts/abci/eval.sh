#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
# Evaluate a checkpoint on a fixed init-state set. Vars: RUN, INIT (npz under data/init),
# CKPT (default last), WORKERS, VIDEOS (per worker), TAG, SET (space-separated KEY=VALUE policy overrides).
set -euo pipefail
source "$PAI_ROOT/relayout-adapt/scripts/abci/job_env.sh"
CKPT="${CKPT:-last}"
POLICY="$PAI_ROOT/runs/$RUN/checkpoints/$CKPT/pretrained_model"
OUT="$PAI_ROOT/runs/$RUN/eval_${TAG:-${INIT%.npz}}_$CKPT"
W="${WORKERS:-12}"
SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done
mkdir -p "$OUT"
for k in $(seq 0 $((W - 1))); do
  python -m relayout.rollout --policy "$POLICY" --init "$PAI_ROOT/data/init/$INIT" --out "$OUT" \
    --shard "$k/$W" --videos "${VIDEOS:-1}" "${SET_ARGS[@]}" > "$OUT/worker_$k.log" 2>&1 &
done
wait
python -m relayout.summarize "$OUT"
