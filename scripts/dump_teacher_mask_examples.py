#!/usr/bin/env python
"""Dump concrete teacher-mask token examples for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import (  # noqa: E402
    TOKEN_TYPE_NAMES,
    forward_for_mask,
    load_model_and_tokenizer,
    make_stream,
    make_teacher_mask,
    sequence_loss,
    tokenize,
)
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples-per-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args()


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    return text.replace("\n", "\\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 0)) + int(args.seed_offset)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("This dump is intended to run on a GPU node.")

    stream = make_stream(config, seed=seed, rank=0)
    model, tokenizer, _replaced = load_model_and_tokenizer(config, device)
    model.train()

    train_config = config["training"]
    outer_config = config.get("outer", {})
    max_length = int(config["model"]["max_length"])
    target_loss_focus = config.get("target_loss", {"mode": "all"})
    support_loss_focus = config.get("support_loss", {"mode": "all"})
    retain_batch_size = int(train_config.get("retain_batch_size", 0))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for step in range(1, args.steps + 1):
            episode = stream.sample_episode(
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
            with torch.no_grad():
                target_loss_before = sequence_loss(
                    model,
                    target_batch,
                    micro_batch_size=train_config.get("target_micro_batch_size"),
                ).detach()

            (
                teacher_mask,
                inner_loss,
                _inner_grad_norm,
                _teacher_before,
                target_loss_after,
                _teacher_kl,
                teacher_metrics,
            ) = make_teacher_mask(
                model,
                support_batch,
                target_batch,
                retain_batch,
                support_stats,
                inner_lr=float(config["inner"]["lr"]),
                retain_kl_weight=float(outer_config.get("retain_kl_weight", 0.0)),
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

            valid = support_stats["valid"].detach().bool().cpu()
            mask = teacher_mask.detach().float().cpu()
            token_loss = support_stats["token_loss"].detach().float().cpu()
            token_types = support_stats["token_types"].detach().cpu()
            input_ids = support_batch["input_ids"].detach().cpu()
            loss_weights = support_stats["loss_weights"].detach().float().cpu()
            target_focus = target_batch["loss_weights"][:, 1:].detach().float().cpu()
            target_valid = target_batch["labels"][:, 1:].ne(-100).detach().cpu()

            sample_scores: list[tuple[float, int]] = []
            for sample_idx in range(mask.shape[0]):
                valid_i = valid[sample_idx]
                if int(valid_i.sum().item()) <= 1:
                    sample_scores.append((0.0, sample_idx))
                    continue
                sample_scores.append(
                    (float(mask[sample_idx, valid_i].std(unbiased=False).item()), sample_idx)
                )
            selected_samples = [
                sample_idx
                for _std, sample_idx in sorted(sample_scores, reverse=True)[: args.samples_per_step]
            ]

            samples: list[dict[str, Any]] = []
            for sample_idx in selected_samples:
                valid_positions = valid[sample_idx].nonzero(as_tuple=False).flatten()
                values = mask[sample_idx, valid_positions]
                k = min(args.top_k, int(valid_positions.numel()))
                top_offsets = torch.topk(values, k=k).indices
                bottom_offsets = torch.topk(-values, k=k).indices

                def collect(offsets: torch.Tensor) -> list[dict[str, Any]]:
                    rows: list[dict[str, Any]] = []
                    for offset in offsets.tolist():
                        context_pos = int(valid_positions[offset].item())
                        target_pos = context_pos + 1
                        token_id = int(input_ids[sample_idx, target_pos].item())
                        tok_type = int(token_types[sample_idx, context_pos].item())
                        rows.append(
                            {
                                "context_pos": context_pos,
                                "target_pos": target_pos,
                                "token": token_text(tokenizer, token_id),
                                "token_id": token_id,
                                "mask": round(float(mask[sample_idx, context_pos].item()), 6),
                                "token_loss": round(float(token_loss[sample_idx, context_pos].item()), 6),
                                "token_type": TOKEN_TYPE_NAMES.get(tok_type, str(tok_type)),
                                "support_loss_focus": round(
                                    float(loss_weights[sample_idx, context_pos].item()), 6
                                ),
                            }
                        )
                    return rows

                samples.append(
                    {
                        "sample_index": sample_idx,
                        "prompt": episode.support_prompts[sample_idx],
                        "completion": episode.support_completions[sample_idx],
                        "valid_tokens": int(valid_positions.numel()),
                        "mask_mean": round(float(mask[sample_idx, valid_positions].mean().item()), 6),
                        "mask_std": round(float(mask[sample_idx, valid_positions].std(unbiased=False).item()), 6),
                        "top_tokens": collect(top_offsets),
                        "bottom_tokens": collect(bottom_offsets),
                    }
                )

            row = {
                "step": step,
                "support_domain": episode.support_domain,
                "target_domain": episode.target_domain,
                "target_loss_before": round(float(target_loss_before.detach().cpu().item()), 6),
                "target_loss_after": round(float(target_loss_after.detach().cpu().item()), 6),
                "teacher_gain": round(
                    float((target_loss_before - target_loss_after).detach().cpu().item()),
                    6,
                ),
                "inner_loss": round(float(inner_loss.detach().cpu().item()), 6),
                "teacher_mask_rate": round(float(mask[valid].mean().item()), 6),
                "teacher_mask_std": round(float(mask[valid].std(unbiased=False).item()), 6),
                "target_focus_fraction": round(
                    float(target_focus.sum().item() / max(float(target_valid.float().sum().item()), 1.0)),
                    6,
                ),
                "teacher_metrics": {
                    key: round(float(value.detach().cpu().item()), 6)
                    for key, value in teacher_metrics.items()
                    if value.numel() == 1
                },
                "samples": samples,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[dump-teacher] step={step} gain={row['teacher_gain']:.4f} "
                f"{episode.support_domain}->{episode.target_domain}",
                flush=True,
            )


if __name__ == "__main__":
    main()
