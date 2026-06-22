#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/oceanstor/home/e1553870/ECHO}"
export TMPDIR="${TMPDIR:-/tmp}"
export PBS_CONF_FILE="${PBS_CONF_FILE:-/etc/pbs.conf}"
if [ -r "${PBS_CONF_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${PBS_CONF_FILE}"
  set +a
fi
export PBS_DEFAULT="${PBS_DEFAULT:-${PBS_SERVER:-cbis-pbs}}"

mkdir -p "${ROOT}/outputs/pbs"

submit() {
  local script="$1"
  shift
  qsub "$@" < "${script}"
}

stage1_echo=$(submit \
  "${ROOT}/pbs/run_phase4_mask_v100_no_retain_4gpu.pbs" \
  -v "CONFIG=${ROOT}/configs/claim1_stage1_echo_reasoning_raw_qwen_v100.yaml")
echo "stage1_echo=${stage1_echo}"

stage2_echo=$(submit \
  "${ROOT}/pbs/run_stage2_masked_sft_v100_4gpu.pbs" \
  -W "depend=afterok:${stage1_echo}" \
  -v "CONFIG=${ROOT}/configs/claim1_stage2_echo_math_qwen_v100.yaml")
echo "stage2_echo=${stage2_echo}"

stage1_sft=$(submit \
  "${ROOT}/pbs/run_stage1_sft_real_math_v100_4gpu.pbs" \
  -W "depend=afterok:${stage2_echo}" \
  -v "CONFIG=${ROOT}/configs/claim1_stage1_sft_reasoning_matched_qwen_v100.yaml")
echo "stage1_sft=${stage1_sft}"

stage2_sft=$(submit \
  "${ROOT}/pbs/run_stage2_masked_sft_v100_4gpu.pbs" \
  -W "depend=afterok:${stage1_sft}" \
  -v "CONFIG=${ROOT}/configs/claim1_stage2_full_sft_matched_qwen_v100.yaml")
echo "stage2_sft=${stage2_sft}"

quick_eval=$(submit \
  "${ROOT}/pbs/run_eval_generation_claim1_quick_v100_4gpu.pbs" \
  -W "depend=afterok:${stage2_sft}" \
  -v "CONFIG=${ROOT}/configs/eval_generation_claim1_quick_qwen_v100.yaml")
echo "quick_eval=${quick_eval}"

{
  echo "submitted_at=$(date -Is)"
  echo "stage1_echo=${stage1_echo}"
  echo "stage2_echo=${stage2_echo}"
  echo "stage1_sft=${stage1_sft}"
  echo "stage2_sft=${stage2_sft}"
  echo "quick_eval=${quick_eval}"
} > "${ROOT}/outputs/pbs/claim1_train_eval_chain_latest.txt"
