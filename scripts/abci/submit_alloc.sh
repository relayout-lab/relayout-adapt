#!/bin/bash
# Second round: fine-tune pi0 on round-1 demos + the units each strategy allocated, then evaluate
# on the final shifted grid and the standard layout. Same steps and seeds as G2.
#   ABCI_GROUP=<group> PAI_ROOT=<root> [PRIORITY=20] [STEPS=5000] [STRATEGIES="uniform failure agent"] bash scripts/abci/submit_alloc.sh
set -euo pipefail
: "${ABCI_GROUP:?}" "${PAI_ROOT:?}"
D="$(cd "$(dirname "$0")" && pwd)"
BASE=g0_act_std_s3
BASE_CKPT=010000
q() { /opt/pbs/bin/qsub -P "$ABCI_GROUP" -p "${PRIORITY:-20}" -o "$PAI_ROOT/logs/" "$@"; }
for st in ${STRATEGIES:-uniform failure agent}; do
  for seed in 1 2 3; do
    run="g3_${st}_s$seed"
    tr=$(q -N "$run" -v PAI_ROOT="$PAI_ROOT",RUN="$run",DATASET="redmug_$st",BASE="$BASE",BASE_CKPT="$BASE_CKPT",SEED="$seed",STEPS="${STEPS:-5000}" "$D/train_ft.sh")
    e1=$(q -N "ev_${run}_shift" -W depend=afterok:"$tr" -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=shift_grid.npz,TAG=shift "$D/eval.sh")
    e2=$(q -N "ev_${run}_std" -W depend=afterok:"$tr" -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=std_200.npz,TAG=std200 "$D/eval.sh")
    echo "$run train $tr eval $e1 $e2"
  done
done
