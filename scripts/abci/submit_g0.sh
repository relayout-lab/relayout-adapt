#!/bin/bash
# G0: prepare data -> train ACT with 3 seeds -> evaluate each on the standard layout.
#   ABCI_GROUP=<group> PAI_ROOT=<root> [PRIORITY=20] bash scripts/abci/submit_g0.sh
set -euo pipefail
: "${ABCI_GROUP:?}" "${PAI_ROOT:?}"
D="$(cd "$(dirname "$0")" && pwd)"
q() { /opt/pbs/bin/qsub -P "$ABCI_GROUP" -p "${PRIORITY:-20}" -o "$PAI_ROOT/logs/" "$@"; }
prep=$(q -N g0_prepare -v PAI_ROOT="$PAI_ROOT" "$D/prepare.sh")
echo "prepare $prep"
for seed in 1 2 3; do
  run="g0_act_std_s$seed"
  tr=$(q -N "$run" -W depend=afterok:"$prep" \
       -v PAI_ROOT="$PAI_ROOT",RUN="$run",DATASET=redmug_std,SEED="$seed",STEPS=30000 "$D/train.sh")
  ev=$(q -N "ev_$run" -W depend=afterok:"$tr" \
       -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=std_200.npz,TAG=std200 "$D/eval.sh")
  echo "$run train $tr eval $ev"
done
