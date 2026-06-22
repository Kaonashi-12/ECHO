#!/usr/bin/env python
"""Direct first-stage SFT baseline on the five real math datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import (  # noqa: E402
    all_reduce_grads,
    init_distributed,
    is_rank0,
    load_model_and_tokenizer,
    model_hidden_states,
    parameter_grad_norm,
    reduce_metrics,
    setup_wandb,
    slice_batch,
    tokenize,
    token_loss_from_hidden,
    write_jsonl,
)
from s2i.data.real_math_cross import PromptCompletionExample, _load_examples  # noqa: E402
from s2i.methods.capability_mask import (  # noqa: E402
    copy_lora_state,
    iter_lora_parameters,
    lora_parameter_norm,
)
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


class RealMathMixtureStream:
    """Sample prompt/completion examples from configured math domains."""

    def __init__(self, domains: list[dict[str, Any]], seed: int, sampling: str) -> None:
        self.rng = random.Random(seed)
        self.sampling = sampling
        self.domains = {
            spec["name"]: _load_examples(spec)
            for spec in domains
        }
        self.domain_names = [name for name, examples in self.domains.items() if examples]
        if not self.domain_names:
            raise ValueError("No non-empty SFT domains configured")
        self.flat_examples: list[tuple[str, PromptCompletionExample]] = [
            (name, example)
            for name in self.domain_names
            for example in self.domains[name]
        ]

    def sample_batch(self, batch_size: int) -> tuple[list[str], list[str], Counter[str]]:
        prompts: list[str] = []
        completions: list[str] = []
        counts: Counter[str] = Counter()
        for _ in range(batch_size):
            if self.sampling in {"uniform_domain", "domain_uniform"}:
                domain = self.rng.choice(self.domain_names)
                example = self.rng.choice(self.domains[domain])
            elif self.sampling in {"proportional", "example_uniform"}:
                domain, example = self.rng.choice(self.flat_examples)
            else:
                raise ValueError(f"Unsupported sampling={self.sampling!r}")
            prompts.append(example.prompt)
            completions.append(example.completion)
            counts[domain] += 1
        return prompts, completions, counts


def sft_backward(
    model,
    batch: dict[str, torch.Tensor],
    micro_batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch["labels"][:, 1:].ne(-100)
    denom = valid.float().sum().clamp_min(1.0).detach()
    batch_size = int(batch["input_ids"].shape[0])
    micro_batch_size = micro_batch_size or batch_size
    total_loss = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)
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
        loss = (token_loss * sub_valid.float()).sum() / denom
        loss.backward()
        total_loss = total_loss + loss.detach()
        del hidden_states, token_loss, loss
    return total_loss, valid.float().sum()


def save_checkpoint(
    path: Path,
    step: int,
    model,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "lora": copy_lora_state(model, device=torch.device("cpu")),
            "lora_optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


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

    data_config = config["data"]
    stream = RealMathMixtureStream(
        data_config["domains"],
        seed=seed + 1009 * rank,
        sampling=data_config.get("sampling", "uniform_domain"),
    )
    model, tokenizer, replaced_modules = load_model_and_tokenizer(config, device)
    lora_params = list(iter_lora_parameters(model))
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config["optim"]["lora_lr"],
        weight_decay=config["optim"].get("lora_weight_decay", 0.0),
    )
    run = setup_wandb(config, output_dir, rank)

    if is_rank0(rank):
        domain_sizes = {name: len(stream.domains[name]) for name in stream.domain_names}
        print(
            f"[stage1-sft] rank0 starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} domains={domain_sizes} "
            f"output_dir={output_dir}",
            flush=True,
        )

    training = config["training"]
    max_length = int(config["model"]["max_length"])
    batch_size = int(training["batch_size"])
    micro_batch_size = training.get("micro_batch_size")
    log_every = int(training.get("log_every", 1))
    save_every = int(training.get("save_every", 0))
    metrics_path = output_dir / "metrics.jsonl"

    for step in range(1, int(training["steps"]) + 1):
        step_start = time.time()
        prompts, completions, domain_counts = stream.sample_batch(batch_size)
        batch = tokenize(tokenizer, prompts, completions, max_length, device)

        optimizer.zero_grad(set_to_none=True)
        loss, valid_tokens = sft_backward(model, batch, micro_batch_size)
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
            "train_loss": float(loss.detach().cpu()),
            "valid_tokens": float(valid_tokens.detach().cpu()),
            "lora_grad_norm": float(grad_norm_before_clip.detach().cpu()),
            "lora_param_norm": float(lora_parameter_norm(model).detach().cpu()),
            "step_seconds": elapsed,
            "tokens_per_second": float(
                batch["attention_mask"].sum().detach().cpu() / max(elapsed, 1e-6)
            ),
        }
        for domain in stream.domain_names:
            metrics[f"domain_count_{domain}"] = float(domain_counts.get(domain, 0))
        if device.type == "cuda":
            metrics["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            metrics["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        reduced = reduce_metrics(metrics, device, world_size)
        if is_rank0(rank) and (step % log_every == 0 or step == 1):
            row = dict(reduced)
            row["training_mode"] = "stage1_sft_real_math_mix"
            row["sampling"] = stream.sampling
            write_jsonl(metrics_path, row)
            if run is not None:
                run.log(row, step=step)
            print(
                "[stage1-sft] "
                f"step={step} loss={row['train_loss']:.4f} "
                f"grad={row['lora_grad_norm']:.3f} "
                f"tokens={row['valid_tokens']:.1f} sec={row['step_seconds']:.2f}",
                flush=True,
            )

        if is_rank0(rank) and save_every and step % save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                step,
                model,
                optimizer,
                config,
            )

    if is_rank0(rank):
        save_checkpoint(
            output_dir / "checkpoint_final.pt",
            int(training["steps"]),
            model,
            optimizer,
            config,
        )
        if run is not None:
            run.finish()
        print(f"[stage1-sft] done. Metrics: {metrics_path}", flush=True)

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
