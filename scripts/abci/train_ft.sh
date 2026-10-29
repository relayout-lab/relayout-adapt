#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
# Fine-tune a trained ACT checkpoint on another dataset. Vars: RUN, DATASET, BASE (run name of the
# initial policy), BASE_CKPT (default 010000), SEED, STEPS.
set -euo pipefail
source "$PAI_ROOT/relayout-adapt/scripts/abci/job_env.sh"
OUT="$PAI_ROOT/runs/$RUN"
test ! -e "$OUT"
lerobot-train --policy.path="$PAI_ROOT/runs/$BASE/checkpoints/${BASE_CKPT:-010000}/pretrained_model" \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="local/$DATASET" --dataset.root="$PAI_ROOT/data/lerobot/$DATASET" \
  --output_dir="$OUT" --job_name="$RUN" --seed="${SEED:-1}" --steps="${STEPS:-5000}" \
  --batch_size=32 --num_workers=12 --save_freq="${STEPS:-5000}" --log_freq=500 \
  --env_eval_freq=0 --wandb.enable=false
