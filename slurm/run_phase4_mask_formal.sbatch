#!/bin/bash -l
#SBATCH --job-name=s2i-p4-formal
#SBATCH --account=p201211
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/project/home/p201211/lkh/intent_update_lab/outputs/slurm/%x-%j.out
#SBATCH --error=/project/home/p201211/lkh/intent_update_lab/outputs/slurm/%x-%j.err

set -euo pipefail

ROOT="/project/home/p201211/lkh/intent_update_lab"
CONFIG="${1:-${ROOT}/configs/phase4_mask_real_math_qwen.yaml}"
OUTPUT_DIR="${2:-}"

mkdir -p "${ROOT}/outputs/slurm"
cd "${ROOT}"

. "/project/home/p201211/lkh/.miniconda/etc/profile.d/conda.sh"
conda activate s2i

export HF_HOME="${HF_HOME:-/project/home/p201211/lkh/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${WANDB_DIR:-${ROOT}/outputs/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-intent-update-mask-formal}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}"

echo "[INFO] host=$(hostname)"
echo "[INFO] root=${ROOT}"
echo "[INFO] config=${CONFIG}"
echo "[INFO] output_dir=${OUTPUT_DIR:-<config default>}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
PY
nvidia-smi

ARGS=(--config "${CONFIG}")
if [ -n "${OUTPUT_DIR}" ]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/run_phase4_mask_mvp.py "${ARGS[@]}"
