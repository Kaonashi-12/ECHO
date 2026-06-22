#!/usr/bin/env python
"""Diagnostic generation eval for ECHO experiments.

This script uses local prompts and answer parsing. It is useful for debugging
model behavior, but it is not an official benchmark entry point. Formal tables
should use scripts/run_lm_eval_official.py with lm-evaluation-harness tasks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import init_distributed, is_rank0, load_model_and_tokenizer, write_jsonl  # noqa: E402
from s2i.data.real_math_cross import _format_example  # noqa: E402
from s2i.methods.capability_mask import iter_lora_parameters, load_lora_state  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


CHOICE_KINDS = {"arc", "ai2_arc", "arc_challenge", "arc_easy", "openbookqa", "openbook", "commonsenseqa", "commonsense_qa", "csqa"}
MATH_DATASET_KINDS = {
    "math",
    "math500",
    "hendrycks_math",
    "math_benchmark",
    "olympiad_math",
    "omni_math",
}
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?")
CHOICE_RE = re.compile(r"(?<![A-Za-z0-9])([A-E])(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class EvalExample:
    prompt: str
    gold_text: str
    gold_answer: str
    source_id: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def resolve_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def normalize_text(text: Any) -> str:
    return " ".join(str(text).replace("\u2212", "-").strip().split())


def resolve_field(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in str(field).split("."):
        if isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(field)
    return value


def scalar_text(value: Any, *, list_sep: str = "; ") -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [scalar_text(item, list_sep=list_sep) for item in value]
        return list_sep.join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_text(value)


def row_template_values(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): scalar_text(value) for key, value in row.items()}


def normalize_prompt(text: Any) -> str:
    return "\n".join(
        " ".join(line.strip().split())
        for line in str(text).replace("\u2212", "-").strip().splitlines()
    )


def strip_latex_noise(text: str) -> str:
    text = normalize_text(text)
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\ ": "",
        "$": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("\\frac", "frac")
    return normalize_text(text)


def find_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    marker = "\\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        pos = idx + len(marker)
        depth = 1
        chars: list[str] = []
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        if chars:
            answers.append("".join(chars).strip())
        start = max(pos, idx + len(marker))
    return answers


def final_answer_text(text: str) -> str:
    text = normalize_text(text)
    if "####" in text:
        text = text.split("####")[-1].strip()
    boxed = find_boxed_answers(text)
    if boxed:
        return boxed[-1]
    lowered = text.lower()
    for marker in ("final answer is", "the answer is", "answer is", "answer:"):
        idx = lowered.rfind(marker)
        if idx >= 0:
            candidate = text[idx + len(marker):].strip()
            if candidate:
                return candidate
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return text.strip()


def normalize_answer(answer: str) -> str:
    answer = strip_latex_noise(answer)
    answer = re.sub(r"^[:=\s]+", "", answer)
    answer = re.sub(r"[\s\.;,]+$", "", answer)
    answer = answer.replace("{,}", ",")
    answer = answer.replace(",", "")
    return answer.lower()


def parse_number(answer: str) -> float | None:
    answer = normalize_answer(answer)
    if not answer:
        return None
    if "/" in answer and answer.count("/") == 1:
        left, right = answer.split("/")
        try:
            denom = float(right)
            if denom == 0.0:
                return None
            return float(left) / denom
        except ValueError:
            return None
    try:
        return float(answer)
    except ValueError:
        match = NUMBER_RE.search(answer)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None


def extract_choice_answer(text: str) -> str:
    text = normalize_text(text)
    match = CHOICE_RE.search(text)
    return match.group(1).upper() if match else ""


def extract_answer(text: str, kind: str) -> str:
    kind = kind.lower()
    if kind in CHOICE_KINDS:
        return extract_choice_answer(text)
    return final_answer_text(text)


def answer_correct(prediction: str, gold: str, kind: str) -> tuple[bool, str, str]:
    pred_answer = extract_answer(prediction, kind)
    gold_answer = extract_answer(gold, kind)
    if kind.lower() in CHOICE_KINDS:
        return bool(pred_answer and pred_answer == gold_answer), pred_answer, gold_answer
    if kind.lower() in MATH_DATASET_KINDS:
        try:
            from lm_eval.tasks.hendrycks_math.utils import is_equiv

            pred_math = re.sub(r"[\s\.;,]+$", "", pred_answer.strip())
            gold_math = re.sub(r"[\s\.;,]+$", "", gold_answer.strip())
            if pred_math and is_equiv(pred_math, gold_math):
                return True, pred_answer, gold_answer
        except Exception:
            pass

    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)
    pred_num = parse_number(pred_norm)
    gold_num = parse_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=1e-4, abs_tol=1e-4), pred_answer, gold_answer
    return bool(pred_norm and pred_norm == gold_norm), pred_answer, gold_answer


def format_hf_prompt_completion(spec: dict[str, Any], row: dict[str, Any], index: int) -> EvalExample:
    prompt_template = spec.get("prompt_template", "{problem}")
    answer_template = spec.get("answer_template")
    metadata_keys = list(spec.get("metadata_keys", []))
    template_values = row_template_values(row)
    prompt = normalize_prompt(prompt_template.format(**template_values))
    if answer_template:
        gold_text = normalize_text(answer_template.format(**template_values))
    else:
        answer_field = spec.get("answer_field") or spec.get("completion_field")
        if answer_field:
            gold_text = scalar_text(
                resolve_field(row, answer_field),
                list_sep=str(spec.get("answer_list_sep", "; ")),
            )
        else:
            raise ValueError(f"Dataset {spec['name']} needs answer_template or answer_field")
    metadata = {}
    for key in metadata_keys:
        try:
            metadata[key] = resolve_field(row, key)
        except KeyError:
            metadata[key] = None
    source_field = spec.get("source_id_field", "unique_id")
    try:
        source_id = scalar_text(resolve_field(row, source_field))
    except KeyError:
        source_id = str(index)
    return EvalExample(
        prompt=prompt,
        gold_text=gold_text,
        gold_answer=extract_answer(gold_text, spec.get("kind", "")),
        source_id=source_id,
        metadata=metadata,
    )


def format_example(spec: dict[str, Any], row: dict[str, Any], index: int) -> EvalExample:
    kind = spec.get("kind", spec["name"]).lower()
    if kind in MATH_DATASET_KINDS:
        return format_hf_prompt_completion(spec, row, index)
    if spec.get("prompt_template"):
        return format_hf_prompt_completion(spec, row, index)

    formatted = _format_example(kind, row, f"{spec['name']}:{index}")
    gold_text = formatted.completion.strip()
    return EvalExample(
        prompt=formatted.prompt,
        gold_text=gold_text,
        gold_answer=extract_answer(gold_text, kind),
        source_id=formatted.source_id,
        metadata={},
    )


def row_passes_filters(row: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    for item in filters:
        field = str(item["field"])
        op = str(item.get("op", "eq")).lower()
        try:
            raw_value = resolve_field(row, field)
        except KeyError:
            raw_value = None
        value = scalar_text(raw_value)
        expected = item.get("value")
        expected_values = item.get("values")
        if expected_values is None and expected is not None:
            expected_values = [expected]
        expected_text = scalar_text(expected) if expected is not None else ""
        expected_texts = [scalar_text(entry) for entry in expected_values or []]

        if op == "eq" and value != expected_text:
            return False
        if op == "ne" and value == expected_text:
            return False
        if op == "in" and value not in expected_texts:
            return False
        if op == "not_in" and value in expected_texts:
            return False
        if op == "not_empty" and not value:
            return False
        if op == "empty" and value:
            return False
        if op == "contains" and expected_text not in value:
            return False
        if op == "not_contains" and expected_text in value:
            return False
        if op == "regex" and not re.search(expected_text, value):
            return False
        if op == "not_regex" and re.search(expected_text, value):
            return False
    return True


def load_eval_examples(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import load_dataset

    path = spec["path"]
    name = spec.get("config")
    split = spec.get("split", "test")
    dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    max_examples = spec.get("max_examples")
    examples: list[EvalExample] = []
    for index, row in enumerate(dataset):
        if max_examples is not None and len(examples) >= int(max_examples):
            break
        if not row_passes_filters(row, spec.get("filters", [])):
            continue
        try:
            example = format_example(spec, row, index)
        except (KeyError, TypeError, ValueError):
            continue
        if example.prompt.strip() and example.gold_text.strip():
            examples.append(example)
    sample_size = spec.get("sample_size")
    if sample_size is not None and len(examples) > int(sample_size):
        seed = int(spec.get("sample_seed", 0))
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(examples)), int(sample_size)))
        examples = [examples[index] for index in indices]
    if not examples:
        raise ValueError(f"No evaluation examples loaded for {spec['name']!r}")
    return examples


def batched(items: list[EvalExample], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def load_checkpoint_lora(model, checkpoint_spec: dict[str, Any], device: torch.device) -> int:
    if checkpoint_spec.get("type") == "base" or checkpoint_spec.get("path") in {None, ""}:
        with torch.no_grad():
            for param in iter_lora_parameters(model):
                param.zero_()
        return 0
    path = resolve_path(checkpoint_spec["path"])
    checkpoint = torch.load(path, map_location="cpu")
    if "lora" not in checkpoint:
        raise ValueError(f"Checkpoint has no LoRA state: {path}")
    load_lora_state(
        model,
        checkpoint["lora"],
        device=device,
        strict=checkpoint_spec.get("strict_lora", True),
    )
    return int(checkpoint.get("step", -1))


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    generation: dict[str, Any],
) -> list[str]:
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(generation.get("max_input_length", 768)),
        add_special_tokens=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(generation.get("max_new_tokens", 128)),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=float(generation.get("temperature", 1.0)),
            top_p=float(generation.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation_ids = output_ids[:, encoded["input_ids"].shape[1]:]
    return tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)


def evaluate_one(
    model,
    tokenizer,
    checkpoint_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    examples: list[EvalExample],
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    generation: dict[str, Any],
) -> dict[str, float | str]:
    checkpoint_step = load_checkpoint_lora(model, checkpoint_spec, device)
    model.eval()
    shard = examples[rank::world_size]
    prediction_path = output_dir / f"predictions_rank{rank}.jsonl"
    batch_size = int(generation.get("batch_size", 4))
    correct = 0
    total = 0
    generated_tokens = 0
    start_time = time.time()
    max_chars = int(generation.get("prediction_max_chars", 2048))
    kind = dataset_spec.get("kind", dataset_spec["name"]).lower()

    for batch in batched(shard, batch_size):
        prompts = [example.prompt for example in batch]
        predictions = generate_batch(model, tokenizer, prompts, device, generation)
        for example, prediction in zip(batch, predictions):
            is_correct, pred_answer, gold_answer = answer_correct(prediction, example.gold_text, kind)
            correct += int(is_correct)
            total += 1
            generated_tokens += len(tokenizer(prediction, add_special_tokens=False)["input_ids"])
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
                    "metadata": example.metadata,
                },
            )

    counts = torch.tensor([correct, total, generated_tokens], dtype=torch.float64, device=device)
    if world_size > 1:
        import torch.distributed as dist

        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    elapsed = time.time() - start_time
    global_correct, global_total, global_tokens = [float(value) for value in counts.detach().cpu()]
    return {
        "checkpoint": checkpoint_spec["name"],
        "checkpoint_path": str(resolve_path(checkpoint_spec["path"])) if checkpoint_spec.get("path") else "base",
        "checkpoint_step": float(checkpoint_step),
        "dataset": dataset_spec["name"],
        "dataset_path": dataset_spec["path"],
        "dataset_split": dataset_spec.get("split", "test"),
        "kind": kind,
        "total": global_total,
        "correct": global_correct,
        "accuracy": global_correct / max(global_total, 1.0),
        "avg_generated_tokens": global_tokens / max(global_total, 1.0),
        "rank_elapsed_seconds": elapsed,
        "world_size": float(world_size),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rank, local_rank, world_size, device = init_distributed()
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
            f"[eval] starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} output_dir={output_dir}",
            flush=True,
        )

    loaded_datasets = {spec["name"]: load_eval_examples(spec) for spec in datasets}
    metrics_path = output_dir / "metrics.jsonl"
    for checkpoint_spec in checkpoints:
        for dataset_spec in datasets:
            examples = loaded_datasets[dataset_spec["name"]]
            if is_rank0(rank):
                print(
                    f"[eval] checkpoint={checkpoint_spec['name']} "
                    f"dataset={dataset_spec['name']} examples={len(examples)}",
                    flush=True,
                )
            row = evaluate_one(
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
            if is_rank0(rank):
                write_jsonl(metrics_path, row)
                print(
                    f"[eval] done checkpoint={row['checkpoint']} dataset={row['dataset']} "
                    f"acc={row['accuracy']:.4f} correct={int(row['correct'])}/{int(row['total'])}",
                    flush=True,
                )

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
