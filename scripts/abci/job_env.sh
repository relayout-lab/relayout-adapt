# Sourced by the PBS job scripts. Requires PAI_ROOT (passed with qsub -v).
: "${PAI_ROOT:?set PAI_ROOT}"
export PATH="$PAI_ROOT/envs/py312/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# PBS sets CUDA_VISIBLE_DEVICES to a GPU UUID, which robosuite's EGL helper parses as an int.
# The job only sees its own GPU, so index 0 is that GPU.
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PAI_ROOT/relayout-adapt"
echo "host=$(hostname) job=${PBS_JOBID:-local} start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
