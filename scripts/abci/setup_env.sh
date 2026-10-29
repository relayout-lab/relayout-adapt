#!/bin/bash
# Build the Python environment on an ABCI 3.0 login node (no GPU needed).
# Jobs run on rt_HG and render with EGL, the same backend as a local GPU machine.
#   PAI_ROOT=/groups/<group>/<you>/pai_final bash scripts/abci/setup_env.sh
# Layout under $PAI_ROOT: relayout-adapt/ (this repo), envs/, data/, runs/, logs/
set -euo pipefail
: "${PAI_ROOT:?set PAI_ROOT}"
ENV_DIR="$PAI_ROOT/envs/py312"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
mkdir -p "$PAI_ROOT"/{envs,data/source,runs,logs}

# shellcheck disable=SC1090
source "$CONDA_SH"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.12
fi
PIP="$ENV_DIR/bin/pip"
$PIP install -q --upgrade pip
# cu128 build: the CUDA 13 wheels on PyPI need a newer driver than ABCI may have.
$PIP install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
# egl-probe (via robomimic) uses an old CMakeLists that CMake 4 rejects.
CMAKE_POLICY_VERSION_MINIMUM=3.5 $PIP install -q "lerobot[libero,training]==0.6.1"
"$ENV_DIR/bin/python" -c "import torch, lerobot; print(torch.__version__, torch.version.cuda)"
# Pre-fetch what jobs would otherwise download: LIBERO assets (~400MB, ~/.cache/libero)
# and the ImageNet ResNet18 weights used by ACT (~/.cache/torch).
(cd "$(dirname "$0")/../.." && "$ENV_DIR/bin/python" -c "import relayout, libero.libero as L; L.get_assets_path()" < /dev/null)
"$ENV_DIR/bin/python" -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

bash "$(dirname "$0")/../fetch_source.sh" "$PAI_ROOT/data/source"
echo "setup done"
