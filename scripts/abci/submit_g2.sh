#!/bin/bash
# G2: from the same initial policy, fine-tune on the source demos (control) and on converted demos,
# with the same number of steps and 3 seeds, then evaluate every policy on the shifted grid and
# the standard layout. The initial policy itself is evaluated on the shifted grid too.
#   ABCI_GROUP=<group> PAI_ROOT=<root> [PRIORITY=20] [STEPS=5000] bash scripts/abci/submit_g2.sh
set -euo pipefail
: "${ABCI_GROUP:?}" "${PAI_ROOT:?}"
D="$(cd "$(dirname "$0")" && pwd)"
BASE=g0_act_std_s3
BASE_CKPT=010000
q() { /opt/pbs/bin/qsub -P "$ABCI_GROUP" -p "${PRIORITY:-20}" -o "$PAI_ROOT/logs/" "$@"; }
echo "pi0 shift $(q -N ev_pi0_shift -v PAI_ROOT="$PAI_ROOT",RUN="$BASE",CKPT="$BASE_CKPT",INIT=shift_grid.npz,TAG=shift "$D/eval.sh")"
for ds in redmug_std redmug_gen; do
  for seed in 1 2 3; do
    run="g2_${ds#redmug_}_s$seed"
    tr=$(q -N "$run" -v PAI_ROOT="$PAI_ROOT",RUN="$run",DATASET="$ds",BASE="$BASE",BASE_CKPT="$BASE_CKPT",SEED="$seed",STEPS="${STEPS:-5000}" "$D/train_ft.sh")
    e1=$(q -N "ev_${run}_shift" -W depend=afterok:"$tr" -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=shift_grid.npz,TAG=shift "$D/eval.sh")
    e2=$(q -N "ev_${run}_std" -W depend=afterok:"$tr" -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=std_200.npz,TAG=std200 "$D/eval.sh")
    echo "$run train $tr eval $e1 $e2"
  done
done
