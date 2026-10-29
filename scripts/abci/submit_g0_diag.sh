#!/bin/bash
# G0 follow-up evals on the standard layout: temporal ensembling on the last checkpoint,
# and the intermediate checkpoints with the default chunked execution.
#   ABCI_GROUP=<group> PAI_ROOT=<root> [PRIORITY=20] bash scripts/abci/submit_g0_diag.sh
set -euo pipefail
: "${ABCI_GROUP:?}" "${PAI_ROOT:?}"
D="$(cd "$(dirname "$0")" && pwd)"
q() { /opt/pbs/bin/qsub -P "$ABCI_GROUP" -p "${PRIORITY:-20}" -o "$PAI_ROOT/logs/" "$@"; }
for seed in 1 2 3; do
  run="g0_act_std_s$seed"
  echo "$run te $(q -N "ev_${run}_te" \
    -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=std_200.npz,TAG=std200_te,SET="n_action_steps=1 temporal_ensemble_coeff=0.01" "$D/eval.sh")"
  for ckpt in 010000 020000; do
    echo "$run $ckpt $(q -N "ev_${run}_$ckpt" \
      -v PAI_ROOT="$PAI_ROOT",RUN="$run",INIT=std_200.npz,TAG=std200,CKPT="$ckpt" "$D/eval.sh")"
  done
done
