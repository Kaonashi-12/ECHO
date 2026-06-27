#!/usr/bin/env python
"""Stage 2 masked SFT with a frozen token mask generator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import (  # noqa: E402
    all_reduce_grads,
    forward_for_mask,
    init_distributed,
    is_rank0,
    load_model_and_tokenizer,
    mask_type_summary,
    mask_loss_denominator,
    make_random_mask,
    make_top_loss_mask,
    model_hidden_states,
    parameter_grad_norm,
    reduce_metrics,
    setup_wandb,
    slice_batch,
    tokenize,
    token_loss_from_hidden,
    write_jsonl,
)
from s2i.eval.mask_metrics import mask_summary, masked_mean  # noqa: E402
from s2i.methods.capability_mask import (  # noqa: E402
    MaskHeadConfig,
    TokenMaskHead,
    copy_lora_state,
    iter_lora_parameters,
    load_lora_state,
    lora_parameter_norm,
)
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


class PromptCompletionStream:
    """Small deterministic-with-replacement HF prompt/completion sampler."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        from datasets import load_dataset

        if config.get("mode") != "hf_prompt_completion":
            raise ValueError(f"Unsupported data.mode={config.get('mode')!r}")
        path = config["path"]
        name = config.get("config")
        split = config.get("split", "train")
        dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
        max_examples = config.get("max_examples")
        prompt_template = config.get("prompt_template", "{problem}")
        completion_template = config.get("completion_template", " {solution}")
        metadata_keys = list(config.get("metadata_keys", []))

        examples: list[dict[str, Any]] = []
        for index, row in enumerate(dataset):
            if max_examples is not None and len(examples) >= int(max_examples):
                break
            try:
                prompt = prompt_template.format(**row)
                completion = completion_template.format(**row)
            except KeyError as exc:
                raise KeyError(f"Missing field {exc} in row {index} from {path}") from exc
            prompt = _normalize_text(prompt)
            completion = str(completion).strip()
            if not prompt or not completion:
                continue
            if not completion.startswith((" ", "\n")):
                completion = " " + completion
            metadata = {key: row.get(key) for key in metadata_keys}
            examples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": metadata,
                    "source_id": str(row.get("unique_id", index)),
                }
            )
        if not examples:
            raise ValueError(f"No examples loaded from {path}")
        self.examples = examples
        self.rng = random.Random(seed)

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        return [self.rng.choice(self.examples) for _ in range(batch_size)]


def _normalize_text(text: Any) -> str:
    return "\n".join(" ".join(line.strip().split()) for line in str(text).strip().splitlines())


def _resolve_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def load_stage1_checkpoint(
    config: dict[str, Any],
    model,
    mask_head: TokenMaskHead,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_config = config["checkpoint"]
    path = _resolve_path(checkpoint_config["path"])
    checkpoint = torch.load(path, map_location="cpu")
    mask_strategy = str(config.get("training", {}).get("mask_strategy", "learned"))
    require_mask_head = checkpoint_config.get("require_mask_head", mask_strategy != "full")
    if "mask_head" in checkpoint:
        mask_head.load_state_dict(checkpoint["mask_head"], strict=True)
    elif require_mask_head:
        raise ValueError(f"Checkpoint has no mask_head state: {path}")
    if checkpoint_config.get("load_lora", True):
        if "lora" not in checkpoint:
            raise ValueError(f"Checkpoint has no LoRA state: {path}")
        load_lora_state(
            model,
            checkpoint["lora"],
            device=device,
            strict=checkpoint_config.get("strict_lora", True),
        )
    return {
        "path": str(path),
        "step": int(checkpoint.get("step", -1)),
    }


def make_mask(
    mask_head: TokenMaskHead,
    stats: dict[str, torch.Tensor],
    strategy: str,
    mode: str,
    threshold: float,
    calibration: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_f = stats["valid"].float()
    if strategy == "full":
        return valid_f, valid_f, torch.ones((), device=valid_f.device, dtype=valid_f.dtype)
    soft_mask = mask_head(
        stats["hidden"],
        stats["scalars"],
        stats["valid"],
    ).detach()
    learned_rate = (soft_mask * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    if strategy == "learned":
        if mode == "soft":
            mask = soft_mask
        if mode in {"hard", "threshold"}:
            mask = (soft_mask >= threshold).float() * valid_f
        elif mode != "soft":
            raise ValueError(f"Unsupported mask_mode={mode!r}")
        return calibrate_mask(mask, soft_mask, stats["valid"], calibration), soft_mask, learned_rate
    if strategy in {"random_same_rate", "random"}:
        mask = make_random_mask(stats["valid"], float(learned_rate.detach().cpu()))
        return mask, soft_mask, learned_rate
    if strategy in {"top_loss_same_rate", "top_loss"}:
        mask = make_top_loss_mask(
            stats["token_loss"].detach(),
            stats["valid"],
            float(learned_rate.detach().cpu()),
        )
        return mask, soft_mask, learned_rate
    raise ValueError(f"Unsupported mask_strategy={strategy!r}")


def calibrate_mask(
    mask: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    calibration: dict[str, Any] | None,
) -> torch.Tensor:
    if not calibration or calibration.get("mode", "none") in {"none", None}:
        return mask
    mode = calibration["mode"]
    valid_f = valid.float()
    if mode in {"mean", "target_mean"}:
        target = float(calibration["target"])
        current = (mask * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        return (mask * (target / current.clamp_min(1e-6))).clamp(max=1.0) * valid_f
    if mode in {"sample_top_fraction", "per_sample_top_fraction"}:
        fraction = float(calibration["fraction"])
        calibrated = torch.zeros_like(mask)
        for row_idx in range(mask.shape[0]):
            valid_positions = valid[row_idx].bool().nonzero(as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            k = max(1, min(int(valid_positions.numel()), int(round(fraction * int(valid_positions.numel())))))
            row_scores = scores[row_idx, valid_positions]
            top_offsets = torch.topk(row_scores, k=k).indices
            calibrated[row_idx, valid_positions[top_offsets]] = 1.0
        return calibrated * valid_f
    raise ValueError(f"Unsupported mask_calibration.mode={mode!r}")


def masked_sft_backward(
    model,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    normalization: str,
    budget_floor_rate: float | None,
    micro_batch_size: int | None,
) -> torch.Tensor:
    valid = batch["labels"][:, 1:].ne(-100)
    denom = mask_loss_denominator(
        mask,
        valid,
        normalization=normalization,
        budget_floor_rate=budget_floor_rate,
    ).detach()
    batch_size = int(batch["input_ids"].shape[0])
    micro_batch_size = micro_batch_size or batch_size
    loss_value = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        sub_batch = slice_batch(batch, start, end)
        hidden_states = model_hidden_states(
            model,
            sub_batch["input_ids"],
            sub_batch["attention_mask"],
        )
        token_loss, sub_valid = token_loss_from_hidden(
            model,
            hidden_states,
            sub_batch["labels"],
        )
        if not bool(sub_valid.any()):
            del hidden_states, token_loss
            continue
        sub_mask = mask[start:end]
        weighted_sum = (token_loss * sub_mask * sub_valid.float()).sum()
        if not weighted_sum.requires_grad:
            del hidden_states, token_loss
            continue
        sub_loss = weighted_sum / denom.clamp_min(1.0)
        sub_loss.backward()
        loss_value = loss_value + sub_loss.detach()
        del hidden_states, token_loss, sub_loss
    return loss_value


def save_stage2_checkpoint(
    path: Path,
    step: int,
    model,
    mask_head: TokenMaskHead,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    source_checkpoint: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "mask_head": mask_head.state_dict(),
            "lora": copy_lora_state(model, device=torch.device("cpu")),
            "lora_optimizer": optimizer.state_dict(),
            "config": config,
            "source_checkpoint": source_checkpoint,
        },
        path,
    )


def _metadata_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    subjects = sorted({str(example["metadata"].get("subject")) for example in examples if example["metadata"].get("subject") is not None})
    levels = []
    for example in examples:
        value = example["metadata"].get("level")
        if value is None:
            continue
        try:
            levels.append(float(value))
        except (TypeError, ValueError):
            text = str(value)
            digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
            if digits:
                levels.append(float(digits))
    summary: dict[str, Any] = {}
    if subjects:
        summary["subjects"] = ",".join(subjects[:4])
    if levels:
        summary["avg_level"] = sum(levels) / len(levels)
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rank, local_rank, world_size, device = init_distributed()
    seed = int(config.get("seed", 0))
    set_seed(seed + rank)

    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir = (ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    stream = PromptCompletionStream(config["data"], seed=seed + 1009 * rank)
    model, tokenizer, replaced_modules = load_model_and_tokenizer(config, device)
    hidden_size = int(model.config.hidden_size)
    mask_head = TokenMaskHead(MaskHeadConfig(hidden_size=hidden_size, **config["mask_head"])).to(device)
    source_checkpoint = load_stage1_checkpoint(config, model, mask_head, device)
    mask_head.eval()
    for param in mask_head.parameters():
        param.requires_grad_(False)

    lora_params = list(iter_lora_parameters(model))
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config["optim"]["lora_lr"],
        weight_decay=config["optim"].get("lora_weight_decay", 0.0),
    )
    run = setup_wandb(config, output_dir, rank)
    if is_rank0(rank):
        print(
            f"[stage2] rank0 starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} examples={len(stream.examples)} "
            f"source_step={source_checkpoint['step']} output_dir={output_dir}",
            flush=True,
        )

    training = config["training"]
    max_length = int(config["model"]["max_length"])
    batch_size = int(training["batch_size"])
    micro_batch_size = training.get("micro_batch_size")
    mask_strategy = training.get("mask_strategy", "learned")
    mask_mode = training.get("mask_mode", "soft")
    mask_threshold = float(training.get("mask_threshold", 0.5))
    mask_calibration = training.get("mask_calibration", {"mode": "none"})
    normalization = training.get("mask_loss_normalization", "mask_sum")
    budget_floor_rate = training.get("mask_loss_budget_floor")
    log_every = int(training.get("log_every", 1))
    save_every = int(training.get("save_every", 0))
    metrics_path = output_dir / "metrics.jsonl"

    for step in range(1, int(training["steps"]) + 1):
        step_start = time.time()
        examples = stream.sample_batch(batch_size)
        batch = tokenize(
            tokenizer,
            [example["prompt"] for example in examples],
            [example["completion"] for example in examples],
            max_length,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            mask_stats = forward_for_mask(model, batch)
            mask, learned_mask, learned_rate = make_mask(
                mask_head,
                mask_stats,
                mask_strategy,
                mask_mode,
                mask_threshold,
                mask_calibration,
            )
            full_loss = masked_mean(mask_stats["token_loss"], mask_stats["valid"])
            summary_tensors = mask_summary(
                mask,
                mask_stats["valid"],
                mask_stats["token_loss"],
                mask_stats["entropy"],
                mask_stats["margin"],
            )
            summary_tensors.update(
                mask_type_summary(
                    mask,
                    mask_stats["valid"],
                    mask_stats.get("token_types"),
                    mask_stats.get("loss_weights"),
                )
            )
            hard_rate = (
                ((mask >= mask_threshold).float() * mask_stats["valid"].float()).sum()
                / mask_stats["valid"].float().sum().clamp_min(1.0)
            )
        train_loss = masked_sft_backward(
            model,
            batch,
            mask,
            normalization=normalization,
            budget_floor_rate=budget_floor_rate,
            micro_batch_size=micro_batch_size,
        )
        grad_norm_before_clip = parameter_grad_norm(lora_params)
        all_reduce_grads(lora_params, world_size)
        torch.nn.utils.clip_grad_norm_(
            lora_params,
            config["optim"].get("max_lora_grad_norm", 1.0),
        )
        optimizer.step()

        elapsed = time.time() - step_start
        metrics = {
            "step": float(step),
            "train_loss": float(train_loss.detach().cpu()),
            "full_loss": float(full_loss.detach().cpu()),
            "hard_mask_rate": float(hard_rate.detach().cpu()),
            "learned_mask_rate": float(learned_rate.detach().cpu()),
            "lora_grad_norm": float(grad_norm_before_clip.detach().cpu()),
            "lora_param_norm": float(lora_parameter_norm(model).detach().cpu()),
            "source_checkpoint_step": float(source_checkpoint["step"]),
            "step_seconds": elapsed,
            "tokens_per_second": float(
                batch["attention_mask"].sum().detach().cpu() / max(elapsed, 1e-6)
            ),
        }
        metrics.update({key: float(value.detach().cpu()) for key, value in summary_tensors.items()})
        if device.type == "cuda":
            metrics["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            metrics["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        reduced = reduce_metrics(metrics, device, world_size)
        if is_rank0(rank) and (step % log_every == 0 or step == 1):
            row = dict(reduced)
            row.update(_metadata_summary(examples))
            row["training_mode"] = "stage2_masked_sft"
            row["mask_strategy"] = mask_strategy
            row["mask_mode"] = mask_mode
            row["mask_calibration"] = mask_calibration
            row["mask_loss_normalization"] = normalization
            row["source_checkpoint"] = source_checkpoint["path"]
            write_jsonl(metrics_path, row)
            if run is not None:
                run.log(row, step=step)
            print(
                "[stage2] "
                f"step={step} loss={row['train_loss']:.4f} full={row['full_loss']:.4f} "
                f"strategy={mask_strategy} mask={row['mask_rate']:.3f} "
                f"learned={row['learned_mask_rate']:.3f} hard={row['hard_mask_rate']:.3f} "
                f"grad={row['lora_grad_norm']:.3f} sec={row['step_seconds']:.2f}",
                flush=True,
            )

        if is_rank0(rank) and save_every and step % save_every == 0:
            save_stage2_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                step,
                model,
                mask_head,
                optimizer,
                config,
                source_checkpoint,
            )

    if is_rank0(rank):
        save_stage2_checkpoint(
            output_dir / "checkpoint_final.pt",
            int(training["steps"]),
            model,
            mask_head,
            optimizer,
            config,
            source_checkpoint,
        )
        if run is not None:
            run.finish()
        print(f"[stage2] done. Metrics: {metrics_path}", flush=True)

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
