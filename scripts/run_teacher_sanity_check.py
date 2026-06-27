#!/usr/bin/env python
"""Teacher-mask sanity check against simple one-step adaptation baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import (  # noqa: E402
    build_fast_lora_state_from_mask,
    forward_for_mask,
    init_distributed,
    is_rank0,
    load_model_and_tokenizer,
    make_random_mask,
    make_stream,
    make_teacher_mask,
    mask_type_summary,
    outer_loss_from_fast_state,
    reduce_metrics,
    sequence_loss,
    tokenize,
    write_jsonl,
)
from s2i.data.real_math_cross import TextEpisode  # noqa: E402
from s2i.eval.mask_metrics import mask_summary  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def make_extreme_loss_mask(
    token_loss: torch.Tensor,
    valid: torch.Tensor,
    mask_rate: float,
    *,
    high: bool,
    per_sample: bool = False,
) -> torch.Tensor:
    valid_bool = valid.bool()
    mask = torch.zeros_like(token_loss)
    if per_sample:
        for row_idx in range(token_loss.shape[0]):
            row_valid = valid_bool[row_idx]
            row_count = int(row_valid.sum().item())
            if row_count == 0:
                continue
            k = max(1, min(row_count, int(round(mask_rate * row_count))))
            row_values = token_loss[row_idx, row_valid]
            row_scores = row_values if high else -row_values
            threshold = torch.topk(row_scores, k=k).values.min()
            mask[row_idx, row_valid] = (row_scores >= threshold).float()
        return mask
    valid_count = int(valid_bool.sum().item())
    if valid_count == 0:
        return mask
    k = max(1, min(valid_count, int(round(mask_rate * valid_count))))
    values = token_loss[valid_bool]
    scores = values if high else -values
    threshold = torch.topk(scores, k=k).values.min()
    mask[valid_bool] = (scores >= threshold).float()
    return mask


def evaluate_mask(
    model,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    support_stats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_config = config["training"]
    outer_config = config.get("outer", {})
    fast_state, _inner_grad_norm, inner_loss = build_fast_lora_state_from_mask(
        model,
        support_batch,
        support_stats,
        mask,
        float(config["inner"]["lr"]),
        train_config.get("mask_loss_normalization", "valid_count"),
        train_config.get("mask_loss_budget_floor", outer_config.get("mask_budget_target")),
        support_micro_batch_size=train_config.get("support_micro_batch_size"),
        create_graph=False,
    )
    (
        _outer_loss,
        _summary_tensors,
        target_loss_before,
        target_loss_after,
        _kl_loss,
    ) = outer_loss_from_fast_state(
        model,
        support_stats,
        mask,
        target_batch,
        retain_batch,
        fast_state,
        retain_kl_weight=float(outer_config.get("retain_kl_weight", 0.0)),
        mask_cost_weight=float(outer_config.get("mask_cost_weight", 0.0)),
        mask_budget_target=outer_config.get("mask_budget_target"),
        mask_budget_weight=float(outer_config.get("mask_budget_weight", 0.0)),
        mask_budget_mode=outer_config.get("mask_budget_mode", "symmetric"),
        target_micro_batch_size=train_config.get("target_micro_batch_size"),
        retain_micro_batch_size=train_config.get("retain_micro_batch_size"),
    )
    return inner_loss.detach(), target_loss_before.detach(), target_loss_after.detach()


def summarize_rows(rows: list[dict[str, Any]], last_n: int) -> dict[str, Any]:
    tail = rows[-last_n:] if last_n > 0 else rows
    summary: dict[str, Any] = {
        "rows": len(rows),
        "tail_rows": len(tail),
        "last_step": rows[-1].get("step") if rows else None,
    }
    numeric_keys = sorted(
        key
        for row in tail
        for key, value in row.items()
        if isinstance(value, (float, int))
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in tail if isinstance(row.get(key), (float, int))]
        if values:
            summary[f"{key}_mean"] = statistics.mean(values)
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rank, _local_rank, world_size, device = init_distributed()
    seed = int(config.get("seed", 0))
    set_seed(seed + rank)

    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir = (ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
    if world_size > 1:
        dist.barrier()

    stream = make_stream(config, seed=seed, rank=rank)
    model, tokenizer, replaced_modules = load_model_and_tokenizer(config, device)
    model.train()

    train_config = config["training"]
    outer_config = config.get("outer", {})
    max_length = int(config["model"]["max_length"])
    target_loss_focus = config.get("target_loss", {"mode": "all"})
    support_loss_focus = config.get("support_loss", {"mode": "all"})
    retain_batch_size = int(train_config.get("retain_batch_size", 0))
    retain_kl_weight = float(outer_config.get("retain_kl_weight", 0.0))
    if retain_kl_weight and retain_batch_size <= 0:
        raise ValueError("retain_kl_weight > 0 requires retain_batch_size > 0")

    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    local_rows: list[dict[str, Any]] = []

    if is_rank0(rank):
        print(
            f"[teacher-sanity] starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} output_dir={output_dir}",
            flush=True,
        )

    for step in range(1, int(train_config["steps"]) + 1):
        step_start = time.time()
        episode: TextEpisode = stream.sample_episode(
            support_size=int(train_config["support_batch_size"]),
            target_size=int(train_config["target_batch_size"]),
            retain_size=retain_batch_size,
        )
        support_batch = tokenize(
            tokenizer,
            episode.support_prompts,
            episode.support_completions,
            max_length,
            device,
            loss_focus=support_loss_focus,
        )
        target_batch = tokenize(
            tokenizer,
            episode.target_prompts,
            episode.target_completions,
            max_length,
            device,
            loss_focus=target_loss_focus,
        )
        if retain_batch_size > 0:
            retain_batch = tokenize(
                tokenizer,
                episode.retain_prompts,
                episode.retain_completions,
                max_length,
                device,
            )
        else:
            retain_batch = {"attention_mask": torch.zeros((), dtype=torch.long, device=device)}

        support_stats = forward_for_mask(model, support_batch, detach_token_loss=True)
        valid = support_stats["valid"]
        valid_f = valid.float()
        with torch.no_grad():
            base_target_loss = sequence_loss(
                model,
                target_batch,
                micro_batch_size=train_config.get("target_micro_batch_size"),
            ).detach()

        (
            teacher_mask,
            teacher_inner_loss,
            teacher_inner_grad_norm,
            _teacher_before,
            teacher_after,
            _teacher_kl,
            teacher_metrics,
        ) = make_teacher_mask(
            model,
            support_batch,
            target_batch,
            retain_batch,
            support_stats,
            inner_lr=float(config["inner"]["lr"]),
            retain_kl_weight=retain_kl_weight,
            mask_cost_weight=float(outer_config.get("mask_cost_weight", 0.0)),
            mask_budget_target=outer_config.get("mask_budget_target"),
            mask_budget_weight=float(outer_config.get("mask_budget_weight", 0.0)),
            mask_budget_mode=outer_config.get("mask_budget_mode", "symmetric"),
            mask_loss_normalization=train_config.get("mask_loss_normalization", "valid_count"),
            mask_loss_budget_floor=train_config.get(
                "mask_loss_budget_floor",
                outer_config.get("mask_budget_target"),
            ),
            teacher_config=config.get("teacher", {}),
            support_micro_batch_size=train_config.get("support_micro_batch_size"),
            target_micro_batch_size=train_config.get("target_micro_batch_size"),
            retain_micro_batch_size=train_config.get("retain_micro_batch_size"),
        )
        teacher_rate = float(
            ((teacher_mask * valid_f).sum() / valid_f.sum().clamp_min(1.0)).detach().cpu()
        )
        masks = {
            "teacher": teacher_mask.detach(),
            "full": valid_f,
            "random": make_random_mask(valid, teacher_rate),
            "top_loss": make_extreme_loss_mask(
                support_stats["token_loss"].detach(),
                valid,
                teacher_rate,
                high=True,
            ),
            "top_loss_per_sample": make_extreme_loss_mask(
                support_stats["token_loss"].detach(),
                valid,
                teacher_rate,
                high=True,
                per_sample=True,
            ),
            "low_loss": make_extreme_loss_mask(
                support_stats["token_loss"].detach(),
                valid,
                teacher_rate,
                high=False,
            ),
        }
        row_metrics: dict[str, float] = {
            "step": float(step),
            "target_loss_before": float(base_target_loss.detach().cpu()),
            "teacher_inner_loss": float(teacher_inner_loss.detach().cpu()),
            "teacher_inner_grad_norm": float(teacher_inner_grad_norm.detach().cpu()),
            "teacher_mask_rate": teacher_rate,
            "target_loss_focus_tokens": float(
                target_batch["loss_weights"][:, 1:].sum().detach().cpu()
            ),
            "target_loss_focus_fraction": float(
                (
                    target_batch["loss_weights"][:, 1:].sum()
                    / target_batch["labels"][:, 1:].ne(-100).float().sum().clamp_min(1.0)
                )
                .detach()
                .cpu()
            ),
        }
        teacher_summary = mask_summary(
            teacher_mask.detach(),
            valid,
            support_stats["token_loss"],
            support_stats["entropy"],
            support_stats["margin"],
        )
        teacher_summary.update(
            mask_type_summary(
                teacher_mask.detach(),
                valid,
                support_stats.get("token_types"),
                support_stats.get("loss_weights"),
            )
        )
        row_metrics.update(
            {f"teacher_{key}": float(value.detach().cpu()) for key, value in teacher_summary.items()}
        )
        row_metrics.update(
            {key: float(value.detach().cpu()) for key, value in teacher_metrics.items()}
        )

        losses_after: dict[str, float] = {
            "teacher": float(teacher_after.detach().cpu()),
        }
        gains: dict[str, float] = {
            "teacher": row_metrics["target_loss_before"] - losses_after["teacher"],
        }
        for name, mask in masks.items():
            if name == "teacher":
                continue
            inner_loss, _before, after = evaluate_mask(
                model,
                support_batch,
                target_batch,
                retain_batch,
                support_stats,
                mask,
                config,
            )
            losses_after[name] = float(after.detach().cpu())
            gains[name] = row_metrics["target_loss_before"] - losses_after[name]
            row_metrics[f"{name}_inner_loss"] = float(inner_loss.detach().cpu())

        for name, value in losses_after.items():
            row_metrics[f"{name}_target_loss_after"] = value
            row_metrics[f"{name}_gain"] = gains[name]
        for name in ("full", "random", "top_loss", "top_loss_per_sample", "low_loss"):
            row_metrics[f"teacher_minus_{name}_gain"] = gains["teacher"] - gains[name]
            row_metrics[f"teacher_win_{name}"] = float(gains["teacher"] > gains[name])
        row_metrics["step_seconds"] = time.time() - step_start
        if device.type == "cuda":
            row_metrics["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            row_metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            row_metrics["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        reduced = reduce_metrics(row_metrics, device, world_size)
        if is_rank0(rank):
            row = dict(reduced)
            row["support_domain"] = episode.support_domain
            row["target_domain"] = episode.target_domain
            row["retain_domain"] = episode.retain_domain
            write_jsonl(metrics_path, row)
            local_rows.append(row)
            if step % int(train_config.get("log_every", 1)) == 0 or step == 1:
                print(
                    "[teacher-sanity] "
                    f"step={step} "
                    f"gain_teacher={row['teacher_gain']:.4f} "
                    f"gain_full={row['full_gain']:.4f} "
                    f"gain_random={row['random_gain']:.4f} "
                    f"gain_top={row['top_loss_gain']:.4f} "
                    f"gain_top_ps={row['top_loss_per_sample_gain']:.4f} "
                    f"gain_low={row['low_loss_gain']:.4f} "
                    f"win_full={row['teacher_win_full']:.2f} "
                    f"win_top={row['teacher_win_top_loss']:.2f} "
                    f"win_top_ps={row['teacher_win_top_loss_per_sample']:.2f} "
                    f"sec={row['step_seconds']:.2f}",
                    flush=True,
                )

    if is_rank0(rank):
        summary = summarize_rows(local_rows, int(config.get("summary", {}).get("last_n", 100)))
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[teacher-sanity] done. Metrics: {metrics_path}", flush=True)
        print(f"[teacher-sanity] summary: {summary_path}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
