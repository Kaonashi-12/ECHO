#!/usr/bin/env python
"""Fast proxy evaluation for ECHO exploration runs.

This is a deliberately small, fixed-sample evaluation loop for iteration speed.
It keeps generation prompts/parsers shared with the diagnostic eval while adding
multiple-choice loglikelihood scoring, which is much faster than free-form
generation for ARC/OpenBookQA/CommonsenseQA-style tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_generation import (  # noqa: E402
    CHOICE_KINDS,
    EvalExample,
    answer_correct,
    batched,
    extract_answer,
    format_hf_prompt_completion,
    generate_batch,
    load_checkpoint_lora,
    normalize_text,
    resolve_path,
)
from run_phase4_mask_mvp import init_distributed, is_rank0, load_model_and_tokenizer, write_jsonl  # noqa: E402
from s2i.data.real_math_cross import _format_example  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


FAST_CHOICE_KINDS = CHOICE_KINDS | {"aqua", "aqua_rat", "mmlu", "mmlu_math"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def choice_options(row: dict[str, Any]) -> list[dict[str, str]]:
    choices = row.get("choices")
    if isinstance(choices, dict):
        labels = [str(label) for label in choices.get("label", [])]
        texts = [str(text) for text in choices.get("text", [])]
    elif isinstance(choices, list):
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        texts = []
        for index, choice in enumerate(choices):
            text = str(choice).strip()
            label = labels[index]
            for prefix in (f"{label})", f"{label}.", f"{label}:"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    break
            texts.append(text)
    else:
        options = row.get("options")
        if not isinstance(options, list):
            return []
        labels = [chr(ord("A") + index) for index in range(len(options))]
        texts = []
        for index, option in enumerate(options):
            text = str(option).strip()
            label = labels[index]
            for prefix in (f"{label})", f"{label}.", f"{label}:"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    break
            texts.append(text)
    return [
        {"label": label, "text": " ".join(text.strip().split())}
        for label, text in zip(labels, texts)
        if str(label).strip() and str(text).strip()
    ]


def choice_gold_label(row: dict[str, Any], options: list[dict[str, str]]) -> str:
    for key in ("answerKey", "correct", "answer"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, int):
            if 0 <= value < len(options):
                return options[value]["label"]
        text = str(value).strip()
        if not text:
            continue
        upper = text.upper()
        labels = {option["label"].upper(): option["label"] for option in options}
        if upper in labels:
            return labels[upper]
        if upper.isdigit():
            index = int(upper)
            if 0 <= index < len(options):
                return options[index]["label"]
            if 1 <= index <= len(options):
                return options[index - 1]["label"]
    return ""


def format_choice_list_example(spec: dict[str, Any], row: dict[str, Any], index: int) -> EvalExample:
    options = choice_options(row)
    if not options:
        raise ValueError(f"No choices found for {spec['name']}:{index}")
    gold = choice_gold_label(row, options)
    if not gold:
        raise ValueError(f"No gold choice found for {spec['name']}:{index}")
    question = row.get("question") or row.get("problem") or row.get("input")
    if question is None:
        raise ValueError(f"No question field found for {spec['name']}:{index}")
    question = " ".join(str(question).strip().split())
    choice_lines = "\n".join(f"{option['label']}. {option['text']}" for option in options)
    prompt = f"Question: {question}\n{choice_lines}\nAnswer:"
    return EvalExample(
        prompt=prompt,
        gold_text=f" {gold}",
        gold_answer=gold,
        source_id=str(row.get("id", row.get("unique_id", index))),
        metadata={"choices": options, "answerKey": gold},
    )


def format_fast_example(spec: dict[str, Any], row: dict[str, Any], index: int) -> EvalExample:
    kind = spec.get("kind", spec["name"]).lower()
    if kind in {"aqua", "aqua_rat", "mmlu", "mmlu_math"}:
        example = format_choice_list_example(spec, row, index)
    elif spec.get("prompt_template") or kind in {"math", "hendrycks_math", "math_benchmark"}:
        example = format_hf_prompt_completion(spec, row, index)
    else:
        formatted = _format_example(kind, row, f"{spec['name']}:{index}")
        gold_text = formatted.completion.strip()
        example = EvalExample(
            prompt=formatted.prompt,
            gold_text=gold_text,
            gold_answer=extract_answer(gold_text, kind),
            source_id=formatted.source_id,
            metadata={},
        )

    if kind in FAST_CHOICE_KINDS:
        metadata = dict(example.metadata)
        options = metadata.get("choices") or choice_options(row)
        metadata["choices"] = options
        metadata["answerKey"] = metadata.get("answerKey") or choice_gold_label(row, options)
        example = EvalExample(
            prompt=example.prompt,
            gold_text=example.gold_text,
            gold_answer=metadata["answerKey"] or example.gold_answer,
            source_id=example.source_id,
            metadata=metadata,
        )
    return example


def load_eval_examples(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import load_dataset

    path = spec["path"]
    name = spec.get("config")
    split = spec.get("split", "test")
    dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)

    examples: list[EvalExample] = []
    for index, row in enumerate(dataset):
        try:
            example = format_fast_example(spec, row, index)
        except (KeyError, TypeError, ValueError):
            continue
        if example.prompt.strip() and example.gold_text.strip():
            examples.append(example)

    sample_size = spec.get("sample_size", spec.get("max_examples"))
    if sample_size is not None and len(examples) > int(sample_size):
        seed = int(spec.get("sample_seed", 0))
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(examples)), int(sample_size)))
        examples = [examples[index] for index in indices]

    if not examples:
        raise ValueError(f"No evaluation examples loaded for {spec['name']!r}")
    return examples


def continuation_for_choice(option: dict[str, str], mode: str) -> str:
    label = option["label"]
    text = option["text"]
    if mode == "label":
        return f" {label}"
    if mode == "text":
        return f" {text}"
    return f" {label}. {text}"


def encode_candidate(
    tokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not continuation_ids:
        continuation_ids = [tokenizer.eos_token_id]
    if len(prompt_ids) + len(continuation_ids) > max_length:
        keep_prompt = max(max_length - len(continuation_ids), 1)
        prompt_ids = prompt_ids[-keep_prompt:]
    return prompt_ids + continuation_ids, len(prompt_ids)


def score_candidate_batch(
    model,
    tokenizer,
    candidates: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> None:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch in batched(candidates, batch_size):
        encoded = [
            encode_candidate(tokenizer, item["prompt"], item["continuation"], max_length)
            for item in batch
        ]
        max_len = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        prompt_lens: list[int] = []
        lengths: list[int] = []
        for row, (ids, prompt_len) in enumerate(encoded):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row, :length] = 1
            prompt_lens.append(prompt_len)
            lengths.append(length)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            labels = input_ids[:, 1:]

        for row, item in enumerate(batch):
            start = max(prompt_lens[row] - 1, 0)
            end = max(lengths[row] - 1, start + 1)
            token_log_probs = log_probs[row, start:end].gather(
                dim=-1,
                index=labels[row, start:end].unsqueeze(-1),
            ).squeeze(-1)
            score_sum = float(token_log_probs.sum().detach().cpu())
            token_count = max(int(end - start), 1)
            item["score_sum"] = score_sum
            item["score_norm"] = score_sum / token_count
            item["tokens"] = token_count


def evaluate_choice_loglikelihood(
    model,
    tokenizer,
    checkpoint_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    examples: list[EvalExample],
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    eval_config: dict[str, Any],
) -> dict[str, float | str]:
    checkpoint_step = load_checkpoint_lora(model, checkpoint_spec, device)
    model.eval()
    shard = examples[rank::world_size]
    prediction_path = output_dir / f"{checkpoint_spec['name']}__{dataset_spec['name']}__rank{rank}.jsonl"
    batch_size = int(eval_config.get("choice_batch_size", 32))
    max_length = int(eval_config.get("max_input_length", 768))
    continuation_mode = str(dataset_spec.get("choice_continuation", eval_config.get("choice_continuation", "label_text")))
    primary_score = str(dataset_spec.get("choice_score", eval_config.get("choice_score", "norm")))

    correct_sum = 0
    correct_norm = 0
    total = 0
    candidate_items: list[dict[str, Any]] = []
    start_time = time.time()

    for local_index, example in enumerate(shard):
        options = example.metadata.get("choices", [])
        if not options:
            continue
        group_id = f"{rank}:{local_index}"
        for option in options:
            candidate_items.append(
                {
                    "group_id": group_id,
                    "example": example,
                    "label": option["label"],
                    "text": option["text"],
                    "prompt": example.prompt,
                    "continuation": continuation_for_choice(option, continuation_mode),
                }
            )

    score_candidate_batch(model, tokenizer, candidate_items, device, batch_size, max_length)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidate_items:
        grouped.setdefault(item["group_id"], []).append(item)

    for items in grouped.values():
        example = items[0]["example"]
        gold = example.gold_answer
        pred_sum = max(items, key=lambda item: item["score_sum"])
        pred_norm = max(items, key=lambda item: item["score_norm"])
        correct_sum += int(pred_sum["label"] == gold)
        correct_norm += int(pred_norm["label"] == gold)
        total += 1
        write_jsonl(
            prediction_path,
            {
                "checkpoint": checkpoint_spec["name"],
                "checkpoint_path": str(resolve_path(checkpoint_spec["path"])) if checkpoint_spec.get("path") else "base",
                "checkpoint_step": checkpoint_step,
                "dataset": dataset_spec["name"],
                "source_id": example.source_id,
                "gold_answer": gold,
                "pred_sum": pred_sum["label"],
                "pred_norm": pred_norm["label"],
                "correct_sum": bool(pred_sum["label"] == gold),
                "correct_norm": bool(pred_norm["label"] == gold),
                "primary_correct": bool((pred_norm if primary_score == "norm" else pred_sum)["label"] == gold),
                "scores": [
                    {
                        "label": item["label"],
                        "text": item["text"],
                        "score_sum": item["score_sum"],
                        "score_norm": item["score_norm"],
                        "tokens": item["tokens"],
                    }
                    for item in items
                ],
            },
        )

    counts = torch.tensor([correct_sum, correct_norm, total], dtype=torch.float64, device=device)
    if world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    global_sum, global_norm, global_total = [float(value) for value in counts.detach().cpu()]
    primary_correct = global_norm if primary_score == "norm" else global_sum
    return {
        "checkpoint": checkpoint_spec["name"],
        "checkpoint_path": str(resolve_path(checkpoint_spec["path"])) if checkpoint_spec.get("path") else "base",
        "checkpoint_step": float(checkpoint_step),
        "dataset": dataset_spec["name"],
        "mode": "choice_loglikelihood",
        "kind": dataset_spec.get("kind", dataset_spec["name"]),
        "total": global_total,
        "correct": primary_correct,
        "accuracy": primary_correct / max(global_total, 1.0),
        "accuracy_sum": global_sum / max(global_total, 1.0),
        "accuracy_norm": global_norm / max(global_total, 1.0),
        "rank_elapsed_seconds": time.time() - start_time,
        "world_size": float(world_size),
    }


def evaluate_generation_short(
    model,
    tokenizer,
    checkpoint_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    examples: list[EvalExample],
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    eval_config: dict[str, Any],
) -> dict[str, float | str]:
    checkpoint_step = load_checkpoint_lora(model, checkpoint_spec, device)
    model.eval()
    shard = examples[rank::world_size]
    prediction_path = output_dir / f"{checkpoint_spec['name']}__{dataset_spec['name']}__rank{rank}.jsonl"
    batch_size = int(eval_config.get("generation_batch_size", eval_config.get("batch_size", 4)))
    max_chars = int(eval_config.get("prediction_max_chars", 2048))
    max_new_tokens = int(eval_config.get("max_new_tokens", 256))
    kind = dataset_spec.get("kind", dataset_spec["name"]).lower()

    correct = 0
    total = 0
    no_answer = 0
    truncated = 0
    generated_tokens = 0
    start_time = time.time()

    for batch in batched(shard, batch_size):
        predictions = generate_batch(model, tokenizer, [example.prompt for example in batch], device, eval_config)
        for example, prediction in zip(batch, predictions):
            is_correct, pred_answer, gold_answer = answer_correct(prediction, example.gold_text, kind)
            token_count = len(tokenizer(prediction, add_special_tokens=False)["input_ids"])
            correct += int(is_correct)
            total += 1
            no_answer += int(not normalize_text(pred_answer))
            truncated += int(token_count >= max_new_tokens)
            generated_tokens += token_count
            write_jsonl(
                prediction_path,
                {
                    "checkpoint": checkpoint_spec["name"],
                    "checkpoint_path": str(resolve_path(checkpoint_spec["path"])) if checkpoint_spec.get("path") else "base",
                    "checkpoint_step": checkpoint_step,
                    "dataset": dataset_spec["name"],
                    "source_id": example.source_id,
                    "correct": bool(is_correct),
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "prediction": prediction[:max_chars],
                    "gold_text": example.gold_text[:max_chars],
                    "generated_tokens": token_count,
                    "truncated": bool(token_count >= max_new_tokens),
                    "metadata": example.metadata,
                },
            )

    counts = torch.tensor(
        [correct, total, no_answer, truncated, generated_tokens],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    global_correct, global_total, global_no_answer, global_truncated, global_tokens = [
        float(value) for value in counts.detach().cpu()
    ]
    return {
        "checkpoint": checkpoint_spec["name"],
        "checkpoint_path": str(resolve_path(checkpoint_spec["path"])) if checkpoint_spec.get("path") else "base",
        "checkpoint_step": float(checkpoint_step),
        "dataset": dataset_spec["name"],
        "mode": "generation_short",
        "kind": kind,
        "total": global_total,
        "correct": global_correct,
        "accuracy": global_correct / max(global_total, 1.0),
        "no_answer_rate": global_no_answer / max(global_total, 1.0),
        "truncation_rate": global_truncated / max(global_total, 1.0),
        "avg_generated_tokens": global_tokens / max(global_total, 1.0),
        "rank_elapsed_seconds": time.time() - start_time,
        "world_size": float(world_size),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rank, _local_rank, world_size, device = init_distributed()
    set_seed(int(config.get("seed", 0)) + rank)

    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    model, tokenizer, replaced_modules = load_model_and_tokenizer(config, device)
    tokenizer.padding_side = config.get("generation", {}).get("padding_side", "left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for param in model.parameters():
        param.requires_grad_(False)

    checkpoints = list(config["checkpoints"])
    datasets = list(config["datasets"])
    if args.checkpoint_name:
        checkpoints = [spec for spec in checkpoints if spec["name"] == args.checkpoint_name]
    if args.dataset_name:
        datasets = [spec for spec in datasets if spec["name"] == args.dataset_name]
    if not checkpoints:
        raise ValueError("No checkpoints selected")
    if not datasets:
        raise ValueError("No datasets selected")

    if is_rank0(rank):
        print(
            f"[fast-eval] starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} output_dir={output_dir}",
            flush=True,
        )

    loaded_datasets = {spec["name"]: load_eval_examples(spec) for spec in datasets}
    metrics_path = output_dir / "metrics.jsonl"
    for checkpoint_spec in checkpoints:
        for dataset_spec in datasets:
            examples = loaded_datasets[dataset_spec["name"]]
            mode = dataset_spec.get("mode", "choice_loglikelihood" if dataset_spec.get("kind", "").lower() in FAST_CHOICE_KINDS else "generation_short")
            if is_rank0(rank):
                print(
                    f"[fast-eval] checkpoint={checkpoint_spec['name']} "
                    f"dataset={dataset_spec['name']} mode={mode} examples={len(examples)}",
                    flush=True,
                )
            if mode == "choice_loglikelihood":
                row = evaluate_choice_loglikelihood(
                    model,
                    tokenizer,
                    checkpoint_spec,
                    dataset_spec,
                    examples,
                    output_dir,
                    rank,
                    world_size,
                    device,
                    config.get("generation", {}),
                )
            elif mode == "generation_short":
                row = evaluate_generation_short(
                    model,
                    tokenizer,
                    checkpoint_spec,
                    dataset_spec,
                    examples,
                    output_dir,
                    rank,
                    world_size,
                    device,
                    config.get("generation", {}),
                )
            else:
                raise ValueError(f"Unsupported dataset mode={mode!r}")
            if is_rank0(rank):
                write_jsonl(metrics_path, row)
                print(
                    f"[fast-eval] done checkpoint={row['checkpoint']} dataset={row['dataset']} "
                    f"mode={row['mode']} acc={row['accuracy']:.4f} "
                    f"correct={int(row['correct'])}/{int(row['total'])}",
                    flush=True,
                )

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
