# Current Research Route

## Background

Standard supervised fine-tuning treats every labeled completion token as a
training signal. That is a poor fit for a setting where the model may already
know parts of the answer, where some tokens only encode dataset style, and
where a useful update should improve transferable capability without damaging
unrelated behavior.

This project trains a token-level update policy instead of manually defining
which tokens are useful. The policy is implemented as a lightweight mask head on
top of the current LLM. It reads model states and token statistics, then gates
which completion tokens are allowed to contribute to a LoRA update.

The intended end state is a mask generator that can be frozen or reused as a
training-time skill: given new data and a current model, it selects tokens that
still have positive marginal training value for that model.

## Core Idea

The mask head should not be a standalone data classifier. It should be
conditioned on the current LLM, because the same token can be useful for a weak
model and redundant for a stronger one.

The training signal is therefore defined by the effect of a masked update:

```text
support batch -> mask -> virtual LoRA update
virtual LoRA update -> target same-domain loss
virtual LoRA update -> unrelated retain KL
```

The mask is good when the support tokens it selects produce a virtual update
that lowers loss on a different dataset from the same domain, while avoiding
drift on unrelated data.

## Formal Episode

Each training step samples three batches:

- `support`: math-reasoning examples from one dataset.
- `target`: math-reasoning examples from a different dataset.
- `retain`: unrelated multiple-choice QA examples.

The configured math datasets are GSM8K, SVAMP, ASDiv, MAWPS, and MultiArith.
The retain datasets are ARC-Challenge, OpenBookQA, and CommonsenseQA.

The support-target split is cross-dataset by construction. This makes it harder
for the mask to succeed by selecting dataset-specific formatting or length
artifacts; the selected tokens must produce an update that transfers across
datasets inside the same broad capability domain.

## Model Path

The active model is `Qwen/Qwen2.5-0.5B` with functional LoRA modules inserted
into attention projections. Base model weights are frozen. The live LoRA weights
represent the model's trainable adaptation state.

The mask head consumes:

- final hidden states for next-token prediction positions,
- token loss,
- predictive entropy,
- top-logit margin,
- normalized token position,
- a validity mask that is true only for completion tokens.

Prompt tokens are never supervised. They are used only as context.

## Inner And Outer Objectives

The inner objective is the mask-weighted support loss. It is used to construct a
virtual LoRA update from detached shadow LoRA tensors. This keeps the target and
retain losses from directly supervising the persistent LoRA weights.

The outer objective is:

```text
target_loss_after_virtual_update
+ retain_kl_weight * retain_kl_after_virtual_update
+ mask_cost_weight * mask_rate
+ optional mask-budget penalty
```

The target loss measures whether the selected support tokens produce useful
same-domain transfer. The retain KL measures whether the same update preserves
unrelated behavior. Both losses affect the mask through the virtual update path.

The live LoRA update combines:

- ordinary masked support-learning gradients,
- a scaled meta-gradient contribution from the target/retain objective.

This keeps the live model improving while still training the mask toward
cross-dataset marginal utility.

## Memory-Safe Formal Implementation

The current implementation includes several memory controls required for
higher-order training on A100 40GB-class GPUs:

- compute LM logits only at valid completion positions,
- compute retain KL only at retain completion positions,
- use non-reentrant gradient checkpointing,
- micro-batch target and retain evaluation,
- micro-batch support shadow-gradient construction.

The active config is:

```text
support_batch_size: 64
target_batch_size: 64
retain_batch_size: 32
support_micro_batch_size: 8
target_micro_batch_size: 8
retain_micro_batch_size: 4
```

## Metrics To Watch

Primary metrics:

- `target_loss_before`: target loss before the virtual update.
- `target_loss_after`: target loss after the virtual update.
- `future_gain`: `target_loss_before - target_loss_after`.
- `retain_kl`: unrelated-domain drift after the virtual update.
- `mask_rate`: fraction of completion tokens selected by the mask.

Diagnostics:

- `valid_tokens`,
- `inner_grad_norm`,
- `outer_mask_grad_norm`,
- `outer_lora_grad_norm`,
- `support_lora_grad_norm`,
- `combined_lora_grad_norm`,
- `gpu_mem_max_gb`,
- `step_seconds`,
- support/target/retain domain names.

Mask trace files save per-example token masks for later inspection. They should
be rendered to PDF only for analysis; raw trace files and rendered reports are
runtime artifacts and are not tracked by git.

## Migration Notes

Runtime outputs, checkpoints, W&B files, cache directories, and Slurm logs are
ignored. A new cluster needs only:

- repository source files,
- `environment.yml` or `requirements.txt`,
- Hugging Face access if datasets/models require it,
- W&B configuration if online logging is desired,
- a cluster-specific Slurm wrapper adapted from
  `slurm/run_phase4_mask_formal.sbatch`.
