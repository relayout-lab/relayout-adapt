#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
# Train ACT with LeRobot. Vars: RUN, DATASET (dir under data/lerobot), SEED, STEPS (default 10000;
# 30000 overfit the 37 source demos in G0).
set -euo pipefail
source "$PAI_ROOT/relayout-adapt/scripts/abci/job_env.sh"
OUT="$PAI_ROOT/runs/$RUN"
test ! -e "$OUT"
lerobot-train --dataset.repo_id="local/$DATASET" --dataset.root="$PAI_ROOT/data/lerobot/$DATASET" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=10 \
  --output_dir="$OUT" --job_name="$RUN" --seed="${SEED:-1}" --steps="${STEPS:-10000}" \
  --batch_size=32 --num_workers=12 --save_freq=10000 --log_freq=500 \
  --env_eval_freq=0 --wandb.enable=false
