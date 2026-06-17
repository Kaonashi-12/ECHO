# Intent-Guided Token Masking

This repository contains the current implementation of an online
model-conditioned token mask generator for LLM adaptation.

The goal is to train a lightweight mask head that decides which completion
tokens should drive parameter updates. The head is conditioned on the current
LLM state, so its decisions can depend on what the model already knows. The
training objective asks the mask to choose support tokens whose induced update
improves another same-domain dataset while preserving behavior on unrelated
retain data.

## Current Route

The active experiment is a cross-dataset meta-learning setup:

1. Sample a support batch from one math-reasoning dataset.
2. Let the current LLM and mask head score completion tokens only.
3. Build a virtual LoRA update from the masked support loss.
4. Evaluate the virtually updated model on a different math dataset.
5. Penalize drift on an unrelated multiple-choice QA retain dataset.
6. Update the mask head, and update the live LoRA with support gradients plus a
   scaled meta-gradient signal.

The current implementation deliberately avoids prompt-token training. Prompts
are context only; labels, losses, KL, and saved mask positions are defined on
completion tokens.

## Data

Same-domain math datasets:

- `openai/gsm8k`
- `ChilleD/SVAMP`
- `EleutherAI/asdiv`
- `garrethlee/MAWPS`
- `ChilleD/MultiArith`

Unrelated retain datasets:

- `allenai/ai2_arc` (`ARC-Challenge`)
- `allenai/openbookqa`
- `tau/commonsense_qa`

The data stream forms episodes where support and target come from different
math datasets. Retain batches come from a different QA domain.

## Model And Objective

The default formal config uses `Qwen/Qwen2.5-0.5B` with functional LoRA on
attention projection modules.

Key implementation details:

- Answer-only tokenization masks prompt labels with `-100`.
- The language-model head is applied only to valid completion positions.
- Target and retain losses backpropagate through the virtual masked LoRA update.
- Retain KL is computed only on retain completion tokens.
- Gradient checkpointing and support micro-batching reduce the higher-order
  graph memory footprint.
- Mask traces can be saved as compressed JSONL files for later visualization.

## Important Files

```text
configs/phase4_mask_real_math_qwen.yaml
scripts/run_phase4_mask_mvp.py
scripts/export_mask_trace_pdf.py
slurm/run_phase4_mask_formal.sbatch
src/s2i/data/real_math_cross.py
src/s2i/eval/mask_metrics.py
src/s2i/methods/capability_mask.py
```

The script name still contains `mvp` for continuity with earlier local runs,
but the active config and code path are the formal real-dataset route described
above.

## Environment

Create the environment from the project root:

```bash
conda env create -f environment.yml
conda activate s2i
pip install -e .
```

Recommended cache locations on a shared cluster:

```bash
export HF_HOME=/path/to/project/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export WANDB_DIR=/path/to/project/outputs/wandb
export TOKENIZERS_PARALLELISM=false
```

## Run

For a Slurm cluster:

```bash
sbatch slurm/run_phase4_mask_formal.sbatch \
  configs/phase4_mask_real_math_qwen.yaml
```

For direct execution on one node with four visible GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/run_phase4_mask_mvp.py \
  --config configs/phase4_mask_real_math_qwen.yaml
```

## Outputs

Runtime artifacts are intentionally ignored by git:

```text
outputs/<run-name>/config.json
outputs/<run-name>/metrics.jsonl
outputs/<run-name>/mask_trace_rank*.jsonl.gz
outputs/<run-name>/checkpoint_step_*.pt
outputs/slurm/*.out
outputs/slurm/*.err
```

Use `scripts/export_mask_trace_pdf.py` to render saved mask traces into a PDF
for manual inspection.
