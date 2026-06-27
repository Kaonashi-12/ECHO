#!/usr/bin/env python
"""Train the Phase 4 capability-relative token mask MVP."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2i.data.real_math_cross import RealMathCrossDatasetStream, TextEpisode
from s2i.eval.mask_metrics import mask_summary, masked_mean
from s2i.methods.capability_mask import (
    MaskHeadConfig,
    TokenMaskHead,
    copy_lora_state,
    install_functional_lora,
    iter_lora_modules,
    iter_lora_parameters,
    lora_parameter_norm,
    load_lora_state,
    make_fast_lora_state,
    use_fast_lora,
)
from s2i.utils.config import load_yaml
from s2i.utils.seed import set_seed


TOKEN_TYPE_OTHER = 0
TOKEN_TYPE_FORMAT = 1
TOKEN_TYPE_PUNCT = 2
TOKEN_TYPE_NUMBER = 3
TOKEN_TYPE_OPERATOR = 4
TOKEN_TYPE_WORD = 5
TOKEN_TYPE_ANSWER = 6
TOKEN_TYPE_NAMES = {
    TOKEN_TYPE_OTHER: "other",
    TOKEN_TYPE_FORMAT: "format",
    TOKEN_TYPE_PUNCT: "punct",
    TOKEN_TYPE_NUMBER: "number",
    TOKEN_TYPE_OPERATOR: "operator",
    TOKEN_TYPE_WORD: "word",
    TOKEN_TYPE_ANSWER: "answer",
}
FORMAT_RE = re.compile(
    r"\b("
    r"question|answer|solution|final|therefore|thus|hence|boxed|problem|choices"
    r")\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d")
OPERATOR_RE = re.compile(r"[=+\-*/^<>≤≥≈]|\\frac|\\sqrt|\\times|\\cdot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_mask_mvp_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        init_kwargs: dict[str, Any] = {"backend": "nccl"}
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def is_rank0(rank: int) -> bool:
    return rank == 0


def all_reduce_grads(parameters, world_size: int) -> None:
    if world_size == 1:
        return
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(world_size)
    return {key: float(value) for key, value in zip(keys, values.detach().cpu())}


def setup_wandb(config: dict, output_dir: Path, rank: int):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False) or not is_rank0(rank):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] package is not installed; continuing without wandb", flush=True)
        return None

    mode = wandb_config.get("mode", "online")
    try:
        return wandb.init(
            project=wandb_config.get("project", "intent-update-mask-mvp"),
            entity=wandb_config.get("entity") or None,
            name=wandb_config.get("name"),
            mode=mode,
            dir=str(output_dir),
            config=config,
        )
    except Exception as exc:
        if not wandb_config.get("allow_fallback_offline", True):
            raise
        print(f"[wandb] init failed in mode={mode!r}: {exc}", flush=True)
        print("[wandb] falling back to offline logging", flush=True)
        return wandb.init(
            project=wandb_config.get("project", "intent-update-mask-mvp"),
            name=wandb_config.get("name"),
            mode="offline",
            dir=str(output_dir),
            config=config,
        )


def make_stream(config: dict, seed: int, rank: int):
    data_config = config["data"]
    mode = data_config["mode"]
    if mode in {"real_math_cross", "hf_math_cross"}:
        return RealMathCrossDatasetStream(
            domains=data_config["domains"],
            retain_domains=data_config.get("retain_domains", []),
            seed=seed + 1009 * rank,
        )
    raise ValueError(f"Unsupported data.mode={mode!r}")


def load_model_and_tokenizer(config: dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    if device.type == "cuda":
        # The outer objective differentiates through an inner backward pass.
        # Flash/mem-efficient SDPA kernels do not currently expose the needed
        # higher-order derivatives, so force math/eager attention.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dtype_name = model_config.get("dtype", "bf16")
    if device.type == "cuda" and dtype_name == "bf16":
        dtype = torch.bfloat16
    elif device.type == "cuda" and dtype_name == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_config.get("padding_side", "right")

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": model_config.get("trust_remote_code", False),
    }
    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **model_kwargs)
    model.config.use_cache = False
    if model_config.get("gradient_checkpointing", False):
        use_reentrant = bool(model_config.get("gradient_checkpointing_use_reentrant", False))
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    replaced = install_functional_lora(
        model,
        target_modules=config["lora"]["target_modules"],
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"].get("dropout", 0.0),
    )
    model.train()
    return model, tokenizer, replaced


def load_state_bank(config: dict, model, device: torch.device) -> list[dict[str, Any]]:
    state_config = config.get("state_bank", {})
    states: list[dict[str, Any]] = []
    if state_config.get("include_initial", False):
        states.append(
            {
                "name": "initial",
                "source": "current_init",
                "step": 0,
                "lora": copy_lora_state(model, device=torch.device("cpu")),
            }
        )

    for raw_path in state_config.get("paths", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        checkpoint = torch.load(path, map_location="cpu")
        if "lora" not in checkpoint:
            raise ValueError(f"Checkpoint has no LoRA state: {path}")
        states.append(
            {
                "name": path.stem,
                "source": str(path),
                "step": int(checkpoint.get("step", -1)),
                "lora": checkpoint["lora"],
            }
        )

    if not states:
        states.append(
            {
                "name": "live",
                "source": "live_model",
                "step": -1,
                "lora": copy_lora_state(model, device=torch.device("cpu")),
            }
        )
    return states


def sample_state(
    state_bank: list[dict[str, Any]],
    rng: random.Random,
    step: int,
    schedule: dict | None = None,
) -> tuple[int, dict[str, Any]]:
    if not schedule or schedule.get("type", "uniform") == "uniform":
        idx = rng.randrange(len(state_bank))
        return idx, state_bank[idx]
    if schedule["type"] == "cycle":
        idx = (step - 1) % len(state_bank)
        return idx, state_bank[idx]
    raise ValueError(f"Unsupported state-bank sampling schedule: {schedule}")


def _find_boxed_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    marker = "\\boxed{"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        pos = idx + len(marker)
        depth = 1
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if pos > idx + len(marker):
            spans.append((idx, pos))
        start = max(pos, idx + len(marker))
    return spans


def _answer_char_span(completion: str) -> tuple[int, int] | None:
    text = str(completion)
    boxed_spans = _find_boxed_spans(text)
    if boxed_spans:
        return boxed_spans[-1]
    if "####" in text:
        start = text.rfind("####") + len("####")
        while start < len(text) and text[start].isspace():
            start += 1
        return (start, len(text)) if start < len(text) else None
    lowered = text.lower()
    markers = (
        "final answer is",
        "the final answer is",
        "the answer is",
        "answer is",
        "answer:",
        "final answer:",
    )
    marker_pos: tuple[int, int] | None = None
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx >= 0 and (marker_pos is None or idx > marker_pos[0]):
            marker_pos = (idx, idx + len(marker))
    if marker_pos is not None:
        start = marker_pos[1]
        while start < len(text) and text[start] in " :=\t\n\r":
            start += 1
        return (start, len(text)) if start < len(text) else marker_pos
    number_matches = list(re.finditer(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text))
    if number_matches:
        match = number_matches[-1]
        return match.span()
    stripped = text.rstrip()
    if not stripped:
        return None
    start = max(0, len(stripped) - 80)
    return (start, len(stripped))


def _classify_token_text(token_text: str, is_answer_span: bool) -> int:
    stripped = token_text.strip()
    if is_answer_span:
        return TOKEN_TYPE_ANSWER
    if not stripped:
        return TOKEN_TYPE_PUNCT
    if FORMAT_RE.search(stripped):
        return TOKEN_TYPE_FORMAT
    if NUMBER_RE.search(stripped):
        return TOKEN_TYPE_NUMBER
    if OPERATOR_RE.search(stripped):
        return TOKEN_TYPE_OPERATOR
    if all(not ch.isalnum() for ch in stripped):
        return TOKEN_TYPE_PUNCT
    if any(ch.isalpha() for ch in stripped):
        return TOKEN_TYPE_WORD
    return TOKEN_TYPE_OTHER


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _expand_positions(
    positions: set[int],
    valid_positions: list[int],
    window_tokens: int,
) -> set[int]:
    if not positions or window_tokens <= 0:
        return set(positions)
    valid_set = set(valid_positions)
    expanded = set(positions)
    for pos in list(positions):
        for delta in range(-window_tokens, window_tokens + 1):
            candidate = pos + delta
            if candidate in valid_set:
                expanded.add(candidate)
    return expanded


def _loss_weight_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {"mode": "all"}
    return dict(config)


def tokenize(
    tokenizer,
    prompts: list[str],
    completions: list[str],
    max_length: int,
    device: torch.device,
    loss_focus: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    focus_config = _loss_weight_config(loss_focus)
    focus_mode = str(focus_config.get("mode", "all")).lower()
    texts = [
        prompt + completion
        for prompt, completion in zip(prompts, completions)
    ]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    loss_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    token_type_ids = torch.full_like(input_ids, TOKEN_TYPE_OTHER, dtype=torch.long)
    prompt_lengths = []
    for prompt in prompts:
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_length = min(len(prompt_ids), max_length)
        prompt_lengths.append(prompt_length)
    for row_idx, prompt_length in enumerate(prompt_lengths):
        labels[row_idx, :prompt_length] = -100
        valid_positions = labels[row_idx].ne(-100).nonzero(as_tuple=False).flatten().tolist()
        if not valid_positions:
            continue

        prompt_char_len = len(prompts[row_idx])
        answer_span_rel = _answer_char_span(completions[row_idx])
        answer_span_abs = (
            (prompt_char_len + answer_span_rel[0], prompt_char_len + answer_span_rel[1])
            if answer_span_rel is not None
            else None
        )
        answer_positions: set[int] = set()
        numeric_positions: set[int] = set()
        operator_positions: set[int] = set()
        for pos in valid_positions:
            start = int(offsets[row_idx, pos, 0].item())
            end = int(offsets[row_idx, pos, 1].item())
            if end <= start:
                token_text = tokenizer.decode([int(input_ids[row_idx, pos].detach().cpu())])
                start = prompt_char_len
                end = prompt_char_len + len(token_text)
            else:
                token_text = texts[row_idx][start:end]
            in_answer_span = answer_span_abs is not None and _overlaps((start, end), answer_span_abs)
            token_type = _classify_token_text(token_text, in_answer_span)
            token_type_ids[row_idx, pos] = token_type
            if in_answer_span:
                answer_positions.add(pos)
            if token_type == TOKEN_TYPE_NUMBER:
                numeric_positions.add(pos)
            if token_type == TOKEN_TYPE_OPERATOR:
                operator_positions.add(pos)

        selected = set(valid_positions)
        if focus_mode in {"answer_focus", "final_answer", "final_answer_window"}:
            selected = set(answer_positions)
            selected = _expand_positions(
                selected,
                valid_positions,
                int(focus_config.get("answer_window_tokens", 0)),
            )
            if focus_config.get("include_numeric_tokens", False):
                selected.update(numeric_positions)
            if focus_config.get("include_operator_tokens", False):
                selected.update(operator_positions)
            min_tokens = int(focus_config.get("min_tokens_per_sample", 1))
            if len(selected) < min_tokens:
                selected.update(valid_positions[-min_tokens:])
            max_tokens = focus_config.get("max_tokens_per_sample")
            if max_tokens is not None and len(selected) > int(max_tokens):
                anchor = max(answer_positions) if answer_positions else valid_positions[-1]
                selected = set(
                    sorted(selected, key=lambda pos: (abs(pos - anchor), pos))[: int(max_tokens)]
                )
        elif focus_mode in {"all", "completion", "full"}:
            selected = set(valid_positions)
        else:
            raise ValueError(f"Unsupported loss_focus.mode={focus_mode!r}")

        background_weight = float(focus_config.get("background_weight", 0.0))
        if background_weight:
            loss_weights[row_idx, valid_positions] = background_weight
        if selected:
            loss_weights[row_idx, sorted(selected)] = float(focus_config.get("focus_weight", 1.0))
        if float(loss_weights[row_idx, valid_positions].sum().item()) <= 0.0:
            loss_weights[row_idx, valid_positions[-1:]] = 1.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights.to(device),
        "token_type_ids": token_type_ids.to(device),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long, device=device),
    }


def detached_entropy_from_logits(logits: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """Compute token entropy without materializing full-vocab probabilities."""

    with torch.no_grad():
        detached = logits.detach()
        max_values = detached.max(dim=-1).values.float()
        exp_sum = torch.zeros_like(max_values)
        weighted_logit_sum = torch.zeros_like(max_values)
        for chunk in torch.split(detached, chunk_size, dim=-1):
            chunk_float = chunk.float()
            exp_values = torch.exp(chunk_float - max_values.unsqueeze(-1))
            exp_sum += exp_values.sum(dim=-1)
            weighted_logit_sum += (exp_values * chunk_float).sum(dim=-1)
        exp_sum = exp_sum.clamp_min(1e-30)
        log_z = max_values + exp_sum.log()
        return log_z - weighted_logit_sum / exp_sum


def model_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the LM backbone without projecting every token through lm_head."""

    if hasattr(model, "model"):
        backbone = model.model
    elif hasattr(model, "transformer"):
        backbone = model.transformer
    else:
        raise AttributeError(
            "The selective-logit path needs a model with `.model` or `.transformer` backbone"
        )
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def token_loss_from_hidden(
    model,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_hidden = hidden_states[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    token_loss = torch.zeros(
        shifted_labels.shape,
        device=shifted_hidden.device,
        dtype=torch.float32,
    )
    if valid.any():
        valid_logits = model.lm_head(shifted_hidden[valid])
        loss_logits = valid_logits.float() if valid_logits.device.type == "cpu" else valid_logits
        token_loss[valid] = F.cross_entropy(
            loss_logits,
            shifted_labels[valid],
            reduction="none",
        ).float()
    return token_loss * valid.float(), valid


def token_stats_from_hidden(
    model,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_hidden = hidden_states[:, :-1, :]
    token_loss, valid = token_loss_from_hidden(model, hidden_states, labels)
    entropy = torch.zeros(
        valid.shape,
        device=shifted_hidden.device,
        dtype=torch.float32,
    )
    margin = torch.zeros_like(entropy)
    with torch.no_grad():
        if valid.any():
            valid_logits = model.lm_head(shifted_hidden[valid])
            entropy[valid] = detached_entropy_from_logits(valid_logits).float()
            top2 = torch.topk(valid_logits.detach(), k=2, dim=-1).values.float()
            margin[valid] = top2[..., 0] - top2[..., 1]
    return token_loss, entropy, margin, valid


def token_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    token_loss = torch.zeros(
        shifted_labels.shape,
        device=shifted_logits.device,
        dtype=torch.float32,
    )
    if valid.any():
        valid_logits = shifted_logits[valid]
        loss_logits = valid_logits.float() if valid_logits.device.type == "cpu" else valid_logits
        token_loss[valid] = F.cross_entropy(
            loss_logits,
            shifted_labels[valid],
            reduction="none",
        ).float()
    return token_loss * valid.float(), valid


def token_stats_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    token_loss, valid = token_loss_from_logits(logits, labels)
    entropy = torch.zeros(
        valid.shape,
        device=shifted_logits.device,
        dtype=torch.float32,
    )
    margin = torch.zeros_like(entropy)
    with torch.no_grad():
        if valid.any():
            valid_logits = shifted_logits[valid]
            entropy[valid] = detached_entropy_from_logits(valid_logits).float()
            top2 = torch.topk(valid_logits.detach(), k=2, dim=-1).values.float()
            margin[valid] = top2[..., 0] - top2[..., 1]
    return token_loss, entropy, margin, valid


def slice_batch(batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in batch.items()}


def shifted_loss_weights(batch: dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    if "loss_weights" not in batch:
        return valid.float()
    weights = batch["loss_weights"][:, 1:].to(device=valid.device, dtype=torch.float32)
    return weights * valid.float()


def sequence_loss(
    model,
    batch: dict[str, torch.Tensor],
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    batch_size = int(batch["input_ids"].shape[0])
    if not micro_batch_size or micro_batch_size >= batch_size:
        hidden_states = model_hidden_states(
            model,
            batch["input_ids"],
            batch["attention_mask"],
        )
        token_loss, valid = token_loss_from_hidden(model, hidden_states, batch["labels"])
        weights = shifted_loss_weights(batch, valid)
        return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    total_loss = None
    total_weight = torch.zeros((), device=batch["input_ids"].device, dtype=torch.float32)
    for start in range(0, batch_size, micro_batch_size):
        sub_batch = slice_batch(batch, start, min(start + micro_batch_size, batch_size))
        hidden_states = model_hidden_states(
            model,
            sub_batch["input_ids"],
            sub_batch["attention_mask"],
        )
        token_loss, valid = token_loss_from_hidden(model, hidden_states, sub_batch["labels"])
        weights = shifted_loss_weights(sub_batch, valid)
        loss_sum = (token_loss * weights).sum()
        total_loss = loss_sum if total_loss is None else total_loss + loss_sum
        total_weight = total_weight + weights.sum()
    if total_loss is None:
        return torch.zeros((), device=batch["input_ids"].device)
    return total_loss / total_weight.clamp_min(1.0)


def sequence_loss_with_stats(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden_states = model_hidden_states(
        model,
        batch["input_ids"],
        batch["attention_mask"],
    )
    token_loss, _, _, valid = token_stats_from_hidden(model, hidden_states, batch["labels"])
    weights = shifted_loss_weights(batch, valid)
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def forward_for_mask(
    model,
    batch: dict[str, torch.Tensor],
    detach_token_loss: bool = False,
) -> dict[str, torch.Tensor]:
    all_hidden = model_hidden_states(
        model,
        batch["input_ids"],
        batch["attention_mask"],
    )
    token_loss, entropy, margin, valid = token_stats_from_hidden(
        model,
        all_hidden,
        batch["labels"],
    )
    if detach_token_loss:
        token_loss = token_loss.detach()
    hidden = all_hidden[:, :-1, :]
    positions = torch.linspace(
        0.0,
        1.0,
        steps=hidden.shape[1],
        device=hidden.device,
        dtype=torch.float32,
    )[None, :, None].expand(hidden.shape[0], hidden.shape[1], 1)
    scalars = torch.stack(
        [
            token_loss.detach().float(),
            entropy.detach().float(),
            margin.detach().float(),
            positions.squeeze(-1),
        ],
        dim=-1,
    )
    return {
        "token_loss": token_loss,
        "entropy": entropy,
        "margin": margin,
        "valid": valid,
        "hidden": hidden,
        "scalars": scalars,
        "loss_weights": shifted_loss_weights(batch, valid),
        "token_types": batch.get(
            "token_type_ids",
            torch.full_like(batch["labels"], TOKEN_TYPE_OTHER),
        )[:, 1:].to(device=valid.device, dtype=torch.long),
    }


def mask_loss_denominator(
    mask: torch.Tensor,
    valid: torch.Tensor,
    normalization: str = "valid_count",
    budget_floor_rate: float | None = None,
) -> torch.Tensor:
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp_min(1.0)
    if normalization in {"valid_count", "valid"}:
        return valid_count
    if normalization in {"mask_sum", "mask_mass"}:
        return (mask * valid_f).sum().clamp_min(1.0)
    if normalization in {"budget_floor", "mask_budget_floor"}:
        if budget_floor_rate is None:
            raise ValueError("mask_loss_normalization=budget_floor requires mask_loss_budget_floor")
        mask_sum = (mask * valid_f).sum()
        floor = valid_count * float(budget_floor_rate)
        return torch.maximum(mask_sum, floor).clamp_min(1.0)
    raise ValueError(f"Unsupported mask loss normalization: {normalization!r}")


def masked_token_loss(
    token_loss: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    normalization: str = "valid_count",
    budget_floor_rate: float | None = None,
) -> torch.Tensor:
    valid_f = valid.float()
    denom = mask_loss_denominator(mask, valid, normalization, budget_floor_rate)
    return (token_loss * mask * valid_f).sum() / denom


def apply_mask_transform(
    mask: torch.Tensor,
    valid: torch.Tensor,
    mode: str = "none",
    target_mean: float | None = None,
) -> torch.Tensor:
    mode = mode.lower()
    if mode in {"none", "identity", ""}:
        return mask
    valid_f = valid.float()
    if mode in {"rescale_mean", "fixed_mean", "mean_rescale"}:
        if target_mean is None:
            raise ValueError(f"mask_transform_mode={mode!r} requires mask_transform_target")
        per_sample_mass = (mask * valid_f).sum(dim=1, keepdim=True).clamp_min(1e-6)
        per_sample_count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = (per_sample_count * float(target_mean)) / per_sample_mass.detach()
        return (mask * scale).clamp(0.0, 1.0) * valid_f
    if mode in {"fixed_mean_exact", "capped_mean", "exact_mean"}:
        if target_mean is None:
            raise ValueError(f"mask_transform_mode={mode!r} requires mask_transform_target")
        target = float(target_mean)
        if target <= 0.0:
            return torch.zeros_like(mask)
        if target >= 1.0:
            return valid_f
        per_sample_count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_mass = per_sample_count * target
        detached_mask = (mask.detach() * valid_f).clamp_min(0.0)
        lo = torch.zeros_like(target_mass)
        hi = torch.ones_like(target_mass)
        for _ in range(32):
            mass = (detached_mask * hi).clamp(0.0, 1.0).sum(dim=1, keepdim=True)
            hi = torch.where(mass < target_mass, hi * 2.0, hi)
        for _ in range(32):
            mid = (lo + hi) * 0.5
            mass = (detached_mask * mid).clamp(0.0, 1.0).sum(dim=1, keepdim=True)
            lo = torch.where(mass < target_mass, mid, lo)
            hi = torch.where(mass < target_mass, hi, mid)
        return (mask * hi.detach()).clamp(0.0, 1.0) * valid_f
    raise ValueError(f"Unsupported mask_transform_mode={mode!r}")


def apply_mask_budget_penalty(
    outer_loss: torch.Tensor,
    summary_tensors: dict[str, torch.Tensor],
    target: float | None,
    weight: float,
    mode: str = "symmetric",
) -> torch.Tensor:
    if target is None or weight == 0.0:
        summary_tensors["mask_budget_loss"] = torch.zeros((), device=outer_loss.device)
        return outer_loss
    budget_target = torch.tensor(
        float(target),
        device=summary_tensors["mask_rate"].device,
        dtype=summary_tensors["mask_rate"].dtype,
    )
    budget_delta = summary_tensors["mask_rate"] - budget_target
    if mode in {"symmetric", "both"}:
        budget_loss = budget_delta.pow(2)
    elif mode in {"upper", "ceiling", "max"}:
        budget_loss = F.relu(budget_delta).pow(2)
    elif mode in {"lower", "floor", "min"}:
        budget_loss = F.relu(-budget_delta).pow(2)
    else:
        raise ValueError(f"Unsupported mask_budget_mode={mode!r}")
    summary_tensors["mask_budget_loss"] = budget_loss
    return outer_loss + weight * budget_loss


def mask_type_summary(
    mask: torch.Tensor,
    valid: torch.Tensor,
    token_types: torch.Tensor | None = None,
    loss_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if token_types is None:
        return {}
    valid_bool = valid.bool()
    valid_f = valid.float()
    mask_mass = (mask * valid_f).sum().clamp_min(1e-6)
    metrics: dict[str, torch.Tensor] = {}
    for type_id, name in TOKEN_TYPE_NAMES.items():
        type_valid = valid_bool & token_types.eq(type_id)
        type_f = type_valid.float()
        count = type_f.sum()
        if count.item() == 0:
            zero = torch.zeros((), device=mask.device)
            metrics[f"token_type_{name}_count"] = zero
            metrics[f"mask_rate_{name}"] = zero
            metrics[f"mask_mass_frac_{name}"] = zero
            continue
        type_mask_mass = (mask * type_f).sum()
        metrics[f"token_type_{name}_count"] = count
        metrics[f"mask_rate_{name}"] = type_mask_mass / count.clamp_min(1.0)
        metrics[f"mask_mass_frac_{name}"] = type_mask_mass / mask_mass
    if loss_weights is not None:
        focus = (loss_weights > 0).float() * valid_f
        focus_count = focus.sum()
        metrics["loss_focus_tokens"] = focus_count
        metrics["loss_focus_fraction"] = focus_count / valid_f.sum().clamp_min(1.0)
        metrics["mask_rate_loss_focus"] = (mask * focus).sum() / focus_count.clamp_min(1.0)
    return metrics


def add_mask_budget_metric(
    summary_tensors: dict[str, torch.Tensor],
    target: float | None,
    mode: str = "symmetric",
) -> None:
    if "mask_budget_loss" in summary_tensors:
        return
    if target is None:
        summary_tensors["mask_budget_loss"] = torch.zeros(
            (),
            device=summary_tensors["mask_rate"].device,
            dtype=summary_tensors["mask_rate"].dtype,
        )
        return
    budget_target = torch.tensor(
        float(target),
        device=summary_tensors["mask_rate"].device,
        dtype=summary_tensors["mask_rate"].dtype,
    )
    budget_delta = summary_tensors["mask_rate"] - budget_target
    if mode in {"symmetric", "both"}:
        summary_tensors["mask_budget_loss"] = budget_delta.pow(2)
    elif mode in {"upper", "ceiling", "max"}:
        summary_tensors["mask_budget_loss"] = F.relu(budget_delta).pow(2)
    elif mode in {"lower", "floor", "min"}:
        summary_tensors["mask_budget_loss"] = F.relu(-budget_delta).pow(2)
    else:
        raise ValueError(f"Unsupported mask_budget_mode={mode!r}")


def make_fast_state_from_loss(
    model,
    loss: torch.Tensor,
    inner_lr: float,
    create_graph: bool,
) -> tuple[dict, torch.Tensor]:
    params = list(iter_lora_parameters(model))
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    return make_fast_lora_state(model, list(grads), inner_lr)


def parameter_grad_norm(parameters) -> torch.Tensor:
    total = None
    device = None
    for param in parameters:
        device = param.device
        if param.grad is None:
            continue
        value = param.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), device=device or torch.device("cpu"))
    return total.sqrt()


def clone_parameter_grads(parameters, scale: float = 1.0) -> list[torch.Tensor | None]:
    grads: list[torch.Tensor | None] = []
    for param in parameters:
        if param.grad is None:
            grads.append(None)
        else:
            grads.append(param.grad.detach().clone().mul_(scale))
    return grads


def clear_parameter_grads(parameters) -> None:
    for param in parameters:
        param.grad = None


def add_parameter_grads(parameters, grads: list[torch.Tensor | None]) -> None:
    for param, grad in zip(parameters, grads):
        if grad is None:
            continue
        if param.grad is None:
            param.grad = grad.to(device=param.device, dtype=param.dtype)
        else:
            param.grad.add_(grad.to(device=param.device, dtype=param.dtype))


def make_shadow_lora_state(model) -> tuple[dict, list[torch.Tensor]]:
    state = {}
    params: list[torch.Tensor] = []
    for module in iter_lora_modules(model):
        shadow_A = module.lora_A.detach().clone().requires_grad_(True)
        shadow_B = module.lora_B.detach().clone().requires_grad_(True)
        state[module] = {"A": shadow_A, "B": shadow_B}
        params.extend([shadow_A, shadow_B])
    return state, params


def make_fast_lora_state_from_shadow(
    shadow_state: dict,
    grads: list[torch.Tensor | None],
    lr: float,
) -> tuple[dict, torch.Tensor]:
    fast_state = {}
    grad_norm_sq = None
    grad_iter = iter(grads)
    for module, values in shadow_state.items():
        grad_A = next(grad_iter)
        grad_B = next(grad_iter)
        base_A = values["A"].detach()
        base_B = values["B"].detach()
        fast_A = base_A if grad_A is None else base_A - lr * grad_A
        fast_B = base_B if grad_B is None else base_B - lr * grad_B
        fast_state[module] = {"A": fast_A, "B": fast_B}
        for grad in (grad_A, grad_B):
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            grad_norm_sq = value if grad_norm_sq is None else grad_norm_sq + value
    if grad_norm_sq is None:
        first = next(iter(shadow_state.values()))
        grad_norm = torch.zeros((), device=first["A"].device)
    else:
        grad_norm = grad_norm_sq.sqrt()
    return fast_state, grad_norm


def retain_kl(
    model,
    retain_batch: dict[str, torch.Tensor],
    fast_state: dict,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    batch_size = int(retain_batch["input_ids"].shape[0])
    if micro_batch_size and micro_batch_size < batch_size:
        total_kl = None
        total_valid = torch.zeros((), device=retain_batch["input_ids"].device, dtype=torch.float32)
        for start in range(0, batch_size, micro_batch_size):
            sub_batch = slice_batch(retain_batch, start, min(start + micro_batch_size, batch_size))
            sub_kl = retain_kl(model, sub_batch, fast_state, micro_batch_size=None)
            valid_count = sub_batch["labels"][:, 1:].ne(-100).float().sum()
            weighted_kl = sub_kl * valid_count
            total_kl = weighted_kl if total_kl is None else total_kl + weighted_kl
            total_valid = total_valid + valid_count
        if total_kl is None:
            return torch.zeros((), device=retain_batch["input_ids"].device)
        return total_kl / total_valid.clamp_min(1.0)

    valid = retain_batch["labels"][:, 1:].ne(-100)
    if not bool(valid.any()):
        return torch.zeros((), device=retain_batch["input_ids"].device)

    with torch.no_grad():
        base_hidden = model_hidden_states(
            model,
            retain_batch["input_ids"],
            retain_batch["attention_mask"],
        )
        base_logits = model.lm_head(base_hidden[:, :-1, :][valid]).float()
    with use_fast_lora(fast_state):
        updated_hidden = model_hidden_states(
            model,
            retain_batch["input_ids"],
            retain_batch["attention_mask"],
        )
        updated_logits = model.lm_head(updated_hidden[:, :-1, :][valid]).float()
    base_probs = F.softmax(base_logits, dim=-1)
    updated_log_probs = F.log_softmax(updated_logits, dim=-1)
    return F.kl_div(updated_log_probs, base_probs, reduction="batchmean")


def baseline_adapt_loss(
    model,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    inner_lr: float,
    mask_loss_normalization: str = "valid_count",
    mask_loss_budget_floor: float | None = None,
    target_micro_batch_size: int | None = None,
) -> float:
    support_stats = forward_for_mask(model, support_batch)
    inner_loss = masked_token_loss(
        support_stats["token_loss"],
        fixed_mask.to(support_stats["token_loss"].device),
        support_stats["valid"],
        normalization=mask_loss_normalization,
        budget_floor_rate=mask_loss_budget_floor,
    )
    fast_state, _ = make_fast_state_from_loss(model, inner_loss, inner_lr, create_graph=False)
    with torch.no_grad(), use_fast_lora(fast_state):
        loss = sequence_loss(model, target_batch, micro_batch_size=target_micro_batch_size)
    return float(loss.detach().cpu())


def make_top_loss_mask(
    token_loss: torch.Tensor,
    valid: torch.Tensor,
    mask_rate: float,
    *,
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
            row_loss = token_loss[row_idx, row_valid]
            threshold = torch.topk(row_loss, k=k).values.min()
            mask[row_idx, row_valid] = (row_loss >= threshold).float()
        return mask
    valid_count = int(valid_bool.sum().item())
    if valid_count == 0:
        return mask
    k = max(1, min(valid_count, int(round(mask_rate * valid_count))))
    flat_loss = token_loss[valid_bool]
    threshold = torch.topk(flat_loss, k=k).values.min()
    mask[valid_bool] = (token_loss[valid_bool] >= threshold).float()
    return mask


def make_random_mask(valid: torch.Tensor, mask_rate: float) -> torch.Tensor:
    random_values = torch.rand_like(valid.float())
    return (random_values < mask_rate).float() * valid.float()


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


class MaskTraceWriter:
    """Write per-step, per-sample token mask traces without extra forwards."""

    def __init__(self, output_dir: Path, config: dict[str, Any], rank: int) -> None:
        self.config = config
        self.rank = rank
        self.enabled = bool(config.get("enabled", False))
        self.every_n_steps = int(config.get("every_n_steps", 1))
        self.threshold = float(config.get("threshold", 0.5))
        self.top_k = int(config.get("top_k", 32))
        self.round_digits = int(config.get("round_digits", 6))
        self.include_text = bool(config.get("include_text", True))
        self.include_token_ids = bool(config.get("include_token_ids", True))
        self.include_token_loss = bool(config.get("include_token_loss", True))
        self.include_all_values = bool(config.get("include_all_values", False))
        self.flush_every_n_steps = int(config.get("flush_every_n_steps", 1))
        self.handle = None
        if not self.enabled:
            return
        filename = config.get("filename", "mask_trace_rank{rank}.jsonl.gz").format(rank=rank)
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(
            path,
            "at",
            encoding="utf-8",
            compresslevel=int(config.get("compresslevel", 3)),
        )

    def _round_list(self, values: torch.Tensor) -> list[float]:
        return [round(float(value), self.round_digits) for value in values.tolist()]

    def write_step(
        self,
        *,
        step: int,
        episode: TextEpisode,
        sampled_state_idx: int,
        sampled_state: dict[str, Any],
        support_batch: dict[str, torch.Tensor],
        support_stats: dict[str, torch.Tensor],
        mask: torch.Tensor,
        target_loss_before: torch.Tensor,
        target_loss_after: torch.Tensor,
    ) -> None:
        if self.handle is None or step % self.every_n_steps != 0:
            return

        mask_cpu = mask.detach().float().cpu()
        valid_cpu = support_stats["valid"].detach().cpu().bool()
        input_ids_cpu = support_batch["input_ids"].detach().cpu()
        attention_cpu = support_batch["attention_mask"].detach().cpu()
        prompt_lengths_cpu = support_batch["prompt_lengths"].detach().cpu()
        token_loss_cpu = support_stats["token_loss"].detach().float().cpu()

        samples = []
        for sample_idx in range(mask_cpu.shape[0]):
            mask_i = mask_cpu[sample_idx]
            valid_i = valid_cpu[sample_idx]
            valid_context_positions = valid_i.nonzero(as_tuple=False).flatten()
            selected_context_positions = (valid_i & (mask_i >= self.threshold)).nonzero(
                as_tuple=False
            ).flatten()
            selected_target_positions = selected_context_positions + 1
            selected_mask_values = mask_i[selected_context_positions]
            input_length = int(attention_cpu[sample_idx].sum().item())

            sample: dict[str, Any] = {
                "sample_index": sample_idx,
                "input_length": input_length,
                "prompt_length": int(prompt_lengths_cpu[sample_idx].item()),
                "valid_count": int(valid_context_positions.numel()),
                "selected_count": int(selected_context_positions.numel()),
                "selected_fraction": round(
                    float(selected_context_positions.numel()) / max(int(valid_context_positions.numel()), 1),
                    self.round_digits,
                ),
                "soft_mask_sum": round(
                    float(mask_i[valid_i].sum().item()) if valid_context_positions.numel() else 0.0,
                    self.round_digits,
                ),
                "soft_mask_mean": round(
                    float(mask_i[valid_i].mean().item()) if valid_context_positions.numel() else 0.0,
                    self.round_digits,
                ),
                "selected_context_positions": selected_context_positions.tolist(),
                "selected_target_positions": selected_target_positions.tolist(),
                "selected_mask_values": self._round_list(selected_mask_values),
            }
            if self.include_text:
                sample["text"] = episode.support_texts[sample_idx]
                sample["prompt"] = episode.support_prompts[sample_idx]
                sample["completion"] = episode.support_completions[sample_idx]
            if self.include_token_ids:
                sample["selected_token_ids"] = input_ids_cpu[
                    sample_idx, selected_target_positions
                ].tolist()
            if self.include_token_loss:
                sample["selected_token_loss"] = self._round_list(
                    token_loss_cpu[sample_idx, selected_context_positions]
                )

            if self.top_k > 0 and valid_context_positions.numel():
                k = min(self.top_k, int(valid_context_positions.numel()))
                top_offsets = torch.topk(mask_i[valid_context_positions], k=k).indices
                top_context_positions = valid_context_positions[top_offsets]
                top_target_positions = top_context_positions + 1
                sample["top_context_positions"] = top_context_positions.tolist()
                sample["top_target_positions"] = top_target_positions.tolist()
                sample["top_mask_values"] = self._round_list(mask_i[top_context_positions])
                if self.include_token_ids:
                    sample["top_token_ids"] = input_ids_cpu[
                        sample_idx, top_target_positions
                    ].tolist()
                if self.include_token_loss:
                    sample["top_token_loss"] = self._round_list(
                        token_loss_cpu[sample_idx, top_context_positions]
                    )

            if self.include_all_values:
                sample["valid_context_positions"] = valid_context_positions.tolist()
                sample["valid_target_positions"] = (valid_context_positions + 1).tolist()
                sample["valid_mask_values"] = self._round_list(mask_i[valid_context_positions])

            samples.append(sample)

        row = {
            "step": step,
            "rank": self.rank,
            "support_domain": episode.support_domain,
            "target_domain": episode.target_domain,
            "retain_domain": episode.retain_domain,
            "state_bank_index": sampled_state_idx,
            "state_bank_step": int(sampled_state.get("step", -1)),
            "state_name": sampled_state["name"],
            "state_source": sampled_state["source"],
            "threshold": self.threshold,
            "target_loss_before": round(float(target_loss_before.detach().cpu()), self.round_digits),
            "target_loss_after": round(float(target_loss_after.detach().cpu()), self.round_digits),
            "future_gain": round(
                float((target_loss_before - target_loss_after).detach().cpu()),
                self.round_digits,
            ),
            "samples": samples,
        }
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        if self.flush_every_n_steps and step % self.flush_every_n_steps == 0:
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def save_checkpoint(
    path: Path,
    step: int,
    mask_head: TokenMaskHead,
    model,
    mask_optimizer: torch.optim.Optimizer,
    lora_optimizer: torch.optim.Optimizer,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "mask_head": mask_head.state_dict(),
            "lora": {name: value.detach().cpu() for name, value in model.named_parameters() if "lora_" in name},
            "mask_optimizer": mask_optimizer.state_dict(),
            "lora_optimizer": lora_optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def run_mask_only_meta_step(
    model,
    mask_head: TokenMaskHead,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    inner_lr: float,
    retain_kl_weight: float,
    mask_cost_weight: float,
    mask_budget_target: float | None,
    mask_budget_weight: float,
    mask_budget_mode: str,
    target_micro_batch_size: int | None = None,
    retain_micro_batch_size: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    support_stats = forward_for_mask(model, support_batch)
    mask = mask_head(
        support_stats["hidden"].detach(),
        support_stats["scalars"].detach(),
        support_stats["valid"],
    )
    inner_loss = masked_token_loss(
        support_stats["token_loss"],
        mask,
        support_stats["valid"],
    )
    fast_state, inner_grad_norm = make_fast_state_from_loss(
        model,
        inner_loss,
        inner_lr=inner_lr,
        create_graph=True,
    )

    with torch.no_grad():
        target_loss_before = sequence_loss(
            model,
            target_batch,
            micro_batch_size=target_micro_batch_size,
        )
    with use_fast_lora(fast_state):
        target_loss_after = sequence_loss(
            model,
            target_batch,
            micro_batch_size=target_micro_batch_size,
        )
    kl_loss = (
        retain_kl(
            model,
            retain_batch,
            fast_state,
            micro_batch_size=retain_micro_batch_size,
        )
        if retain_kl_weight
        else torch.zeros((), device=target_loss_after.device)
    )
    summary_tensors = mask_summary(
        mask,
        support_stats["valid"],
        support_stats["token_loss"],
        support_stats["entropy"],
        support_stats["margin"],
    )
    outer_loss = (
        target_loss_after
        + retain_kl_weight * kl_loss
        + mask_cost_weight * summary_tensors["mask_rate"]
    )
    outer_loss = apply_mask_budget_penalty(
        outer_loss,
        summary_tensors,
        mask_budget_target,
        mask_budget_weight,
        mask_budget_mode,
    )
    return (
        outer_loss,
        support_stats,
        mask,
        inner_loss,
        inner_grad_norm,
        target_loss_before,
        target_loss_after,
        kl_loss,
    )


def build_fast_lora_state_from_mask(
    model,
    support_batch: dict[str, torch.Tensor],
    support_stats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    inner_lr: float,
    mask_loss_normalization: str,
    mask_loss_budget_floor: float | None,
    support_micro_batch_size: int | None = None,
    create_graph: bool = True,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    shadow_state, shadow_params = make_shadow_lora_state(model)
    denom = mask_loss_denominator(
        mask,
        support_stats["valid"],
        mask_loss_normalization,
        mask_loss_budget_floor,
    )
    batch_size = int(support_batch["input_ids"].shape[0])
    support_micro_batch_size = support_micro_batch_size or batch_size
    shadow_grads: list[torch.Tensor | None] | None = None
    inner_loss_sum = torch.zeros((), device=support_batch["input_ids"].device)
    with use_fast_lora(shadow_state):
        for start in range(0, batch_size, support_micro_batch_size):
            end = min(start + support_micro_batch_size, batch_size)
            sub_batch = slice_batch(support_batch, start, end)
            shadow_hidden = model_hidden_states(
                model,
                sub_batch["input_ids"],
                sub_batch["attention_mask"],
            )
            shadow_token_loss, shadow_valid = token_loss_from_hidden(
                model,
                shadow_hidden,
                sub_batch["labels"],
            )
            sub_mask = mask[start:end]
            sub_valid_f = shadow_valid.float()
            gate_weights = sub_mask * sub_valid_f / denom
            sub_grads = torch.autograd.grad(
                shadow_token_loss,
                shadow_params,
                grad_outputs=gate_weights,
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            if shadow_grads is None:
                shadow_grads = list(sub_grads)
            else:
                shadow_grads = [
                    old if new is None else new if old is None else old + new
                    for old, new in zip(shadow_grads, sub_grads)
                ]
            inner_loss_sum = inner_loss_sum + (
                shadow_token_loss.detach() * sub_mask * sub_valid_f
            ).sum()
            del shadow_hidden, shadow_token_loss, shadow_valid, sub_grads
    if shadow_grads is None:
        shadow_grads = [None for _ in shadow_params]
    fast_state, inner_grad_norm = make_fast_lora_state_from_shadow(
        shadow_state,
        shadow_grads,
        lr=inner_lr,
    )
    inner_loss = inner_loss_sum / denom.detach().clamp_min(1.0)
    return fast_state, inner_grad_norm, inner_loss


def outer_loss_from_fast_state(
    model,
    support_stats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    fast_state: dict,
    retain_kl_weight: float,
    mask_cost_weight: float,
    mask_budget_target: float | None,
    mask_budget_weight: float,
    mask_budget_mode: str,
    target_micro_batch_size: int | None = None,
    retain_micro_batch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target_loss_before = sequence_loss(
            model,
            target_batch,
            micro_batch_size=target_micro_batch_size,
        )
    with use_fast_lora(fast_state):
        target_loss_after = sequence_loss(
            model,
            target_batch,
            micro_batch_size=target_micro_batch_size,
        )
    kl_loss = (
        retain_kl(
            model,
            retain_batch,
            fast_state,
            micro_batch_size=retain_micro_batch_size,
        )
        if retain_kl_weight
        else torch.zeros((), device=target_loss_after.device)
    )
    summary_tensors = mask_summary(
        mask,
        support_stats["valid"],
        support_stats["token_loss"],
        support_stats["entropy"],
        support_stats["margin"],
    )
    summary_tensors.update(
        mask_type_summary(
            mask,
            support_stats["valid"],
            support_stats.get("token_types"),
            support_stats.get("loss_weights"),
        )
    )
    outer_loss = (
        target_loss_after
        + retain_kl_weight * kl_loss
        + mask_cost_weight * summary_tensors["mask_rate"]
    )
    outer_loss = apply_mask_budget_penalty(
        outer_loss,
        summary_tensors,
        mask_budget_target,
        mask_budget_weight,
        mask_budget_mode,
    )
    return outer_loss, summary_tensors, target_loss_before, target_loss_after, kl_loss


def teacher_mix_value(config: dict[str, Any], step: int, total_steps: int) -> float:
    start = float(config.get("teacher_mix_start", 1.0))
    end = float(config.get("teacher_mix_end", 0.0))
    warmup = int(config.get("teacher_warmup_steps", 0))
    anneal_steps = int(config.get("teacher_anneal_steps", max(total_steps - warmup, 1)))
    if step <= warmup:
        return start
    progress = min(1.0, max(0.0, (step - warmup) / max(anneal_steps, 1)))
    return start + progress * (end - start)


def distillation_loss(
    student_mask: torch.Tensor,
    teacher_mask: torch.Tensor,
    valid: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp_min(1.0)
    mse = ((student_mask - teacher_mask.detach()).pow(2) * valid_f).sum() / valid_count
    loss = float(config.get("mse_weight", 1.0)) * mse
    metrics: dict[str, torch.Tensor] = {"student_teacher_mse": mse}
    budget_target = config.get("student_budget_target")
    budget_weight = float(config.get("student_budget_weight", 0.0))
    if budget_target is not None and budget_weight:
        student_rate = (student_mask * valid_f).sum() / valid_count
        budget_loss = (student_rate - float(budget_target)).pow(2)
        loss = loss + budget_weight * budget_loss
        metrics["student_budget_loss"] = budget_loss
    corr = torch.zeros((), device=student_mask.device)
    mask = valid.bool()
    if mask.sum() >= 2:
        s = student_mask[mask].float()
        t = teacher_mask.detach()[mask].float()
        s = s - s.mean()
        t = t - t.mean()
        denom = s.norm() * t.norm()
        if denom.item() > 0.0:
            corr = (s * t).sum() / denom
    metrics["student_teacher_corr"] = corr
    metrics["student_mask_rate"] = (student_mask * valid_f).sum() / valid_count
    metrics["teacher_mask_rate"] = (teacher_mask.detach() * valid_f).sum() / valid_count
    return loss, metrics


def make_teacher_mask(
    model,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    support_stats: dict[str, torch.Tensor],
    inner_lr: float,
    retain_kl_weight: float,
    mask_cost_weight: float,
    mask_budget_target: float | None,
    mask_budget_weight: float,
    mask_budget_mode: str,
    mask_loss_normalization: str,
    mask_loss_budget_floor: float | None,
    teacher_config: dict[str, Any],
    support_micro_batch_size: int | None = None,
    target_micro_batch_size: int | None = None,
    retain_micro_batch_size: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    valid = support_stats["valid"]
    teacher_lr = float(teacher_config.get("lr", 5.0))
    teacher_steps = int(teacher_config.get("steps", 1))
    temperature = float(teacher_config.get("temperature", 1.0))
    transform_mode = teacher_config.get("mask_transform_mode", "fixed_mean_exact")
    transform_target = teacher_config.get("mask_transform_target", mask_budget_target)
    init_mode = str(teacher_config.get("init_mode", "zeros")).lower()
    if init_mode in {"zeros", "uniform", "flat"}:
        logits = torch.zeros_like(support_stats["token_loss"], requires_grad=True)
    elif init_mode in {"top_loss", "top_loss_mask", "loss_topk"}:
        init_rate = float(
            teacher_config.get(
                "init_mask_rate",
                transform_target if transform_target is not None else mask_budget_target or 0.35,
            )
        )
        init_scale = float(teacher_config.get("init_scale", 4.0))
        init_per_sample = bool(teacher_config.get("init_per_sample", True))
        init_mask = make_top_loss_mask(
            support_stats["token_loss"].detach(),
            valid,
            init_rate,
            per_sample=init_per_sample,
        )
        logits = torch.where(
            init_mask.bool(),
            torch.full_like(support_stats["token_loss"], init_scale),
            torch.full_like(support_stats["token_loss"], -init_scale),
        )
        logits = (logits * valid.float()).detach().requires_grad_(True)
    else:
        raise ValueError(f"Unsupported teacher.init_mode={init_mode!r}")
    last_metrics: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        target_loss_before = sequence_loss(
            model,
            target_batch,
            micro_batch_size=target_micro_batch_size,
        ).detach()
    target_loss_after = target_loss_before
    kl_loss = torch.zeros((), device=target_loss_before.device)
    inner_loss = torch.zeros((), device=target_loss_before.device)
    inner_grad_norm = torch.zeros((), device=target_loss_before.device)
    teacher_grad_norm = torch.zeros((), device=target_loss_before.device)

    for _ in range(max(teacher_steps, 0)):
        raw_mask = torch.sigmoid(logits / temperature) * valid.float()
        teacher_mask = apply_mask_transform(
            raw_mask,
            valid,
            mode=transform_mode,
            target_mean=transform_target,
        )
        fast_state, inner_grad_norm, inner_loss = build_fast_lora_state_from_mask(
            model,
            support_batch,
            support_stats,
            teacher_mask,
            inner_lr,
            mask_loss_normalization,
            mask_loss_budget_floor,
            support_micro_batch_size=support_micro_batch_size,
            create_graph=True,
        )
        (
            outer_loss,
            last_metrics,
            _,
            target_loss_after,
            kl_loss,
        ) = outer_loss_from_fast_state(
            model,
            support_stats,
            teacher_mask,
            target_batch,
            retain_batch,
            fast_state,
            retain_kl_weight,
            mask_cost_weight,
            mask_budget_target,
            mask_budget_weight,
            mask_budget_mode,
            target_micro_batch_size,
            retain_micro_batch_size,
        )
        grad = torch.autograd.grad(
            outer_loss,
            logits,
            retain_graph=False,
            create_graph=False,
        )[0]
        teacher_grad_norm = grad.detach().float().norm()
        logits = (logits - teacher_lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        raw_mask = torch.sigmoid(logits / temperature) * valid.float()
        teacher_mask = apply_mask_transform(
            raw_mask,
            valid,
            mode=transform_mode,
            target_mean=transform_target,
        )
        valid_bool = valid.bool()
        if valid_bool.any():
            last_metrics["logit_std"] = logits.detach()[valid_bool].float().std(unbiased=False)
            last_metrics["raw_mask_std"] = raw_mask.detach()[valid_bool].float().std(unbiased=False)
        else:
            last_metrics["logit_std"] = torch.zeros((), device=target_loss_before.device)
            last_metrics["raw_mask_std"] = torch.zeros((), device=target_loss_before.device)
        last_metrics["logit_grad_norm"] = teacher_grad_norm
        last_metrics["steps"] = torch.tensor(float(teacher_steps), device=target_loss_before.device)

    if bool(teacher_config.get("eval_final", True)):
        fast_state, inner_grad_norm, inner_loss = build_fast_lora_state_from_mask(
            model,
            support_batch,
            support_stats,
            teacher_mask,
            inner_lr,
            mask_loss_normalization,
            mask_loss_budget_floor,
            support_micro_batch_size=support_micro_batch_size,
            create_graph=False,
        )
        (
            _final_outer_loss,
            last_metrics,
            _,
            target_loss_after,
            kl_loss,
        ) = outer_loss_from_fast_state(
            model,
            support_stats,
            teacher_mask,
            target_batch,
            retain_batch,
            fast_state,
            retain_kl_weight,
            mask_cost_weight,
            mask_budget_target,
            mask_budget_weight,
            mask_budget_mode,
            target_micro_batch_size,
            retain_micro_batch_size,
        )
        with torch.no_grad():
            valid_bool = valid.bool()
            if valid_bool.any():
                last_metrics["logit_std"] = logits.detach()[valid_bool].float().std(unbiased=False)
                last_metrics["raw_mask_std"] = raw_mask.detach()[valid_bool].float().std(unbiased=False)
            else:
                last_metrics["logit_std"] = torch.zeros((), device=target_loss_before.device)
                last_metrics["raw_mask_std"] = torch.zeros((), device=target_loss_before.device)
            last_metrics["logit_grad_norm"] = teacher_grad_norm
            last_metrics["steps"] = torch.tensor(float(teacher_steps), device=target_loss_before.device)
            last_metrics["final_eval"] = torch.ones((), device=target_loss_before.device)
    else:
        last_metrics["final_eval"] = torch.zeros((), device=target_loss_before.device)
    return (
        teacher_mask.detach(),
        inner_loss.detach(),
        inner_grad_norm.detach(),
        target_loss_before.detach(),
        target_loss_after.detach(),
        kl_loss.detach(),
        {f"teacher_{key}": value.detach() for key, value in last_metrics.items()},
    )


def run_teacher_student_meta_step(
    model,
    mask_head: TokenMaskHead,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    step: int,
    total_steps: int,
    inner_lr: float,
    retain_kl_weight: float,
    mask_cost_weight: float,
    mask_budget_target: float | None,
    mask_budget_weight: float,
    mask_budget_mode: str,
    mask_loss_normalization: str,
    mask_loss_budget_floor: float | None,
    teacher_config: dict[str, Any],
    distill_config: dict[str, Any],
    student_mask_transform_mode: str = "none",
    student_mask_transform_target: float | None = None,
    support_micro_batch_size: int | None = None,
    target_micro_batch_size: int | None = None,
    retain_micro_batch_size: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    support_stats = forward_for_mask(model, support_batch, detach_token_loss=True)
    (
        teacher_mask,
        inner_loss,
        inner_grad_norm,
        target_loss_before,
        target_loss_after,
        kl_loss,
        teacher_metrics,
    ) = make_teacher_mask(
        model,
        support_batch,
        target_batch,
        retain_batch,
        support_stats,
        inner_lr,
        retain_kl_weight,
        mask_cost_weight,
        mask_budget_target,
        mask_budget_weight,
        mask_budget_mode,
        mask_loss_normalization,
        mask_loss_budget_floor,
        teacher_config,
        support_micro_batch_size,
        target_micro_batch_size,
        retain_micro_batch_size,
    )
    student_mask = mask_head(
        support_stats["hidden"].detach(),
        support_stats["scalars"].detach(),
        support_stats["valid"],
    )
    student_mask = apply_mask_transform(
        student_mask,
        support_stats["valid"],
        mode=student_mask_transform_mode,
        target_mean=student_mask_transform_target,
    )
    distill, distill_metrics = distillation_loss(
        student_mask,
        teacher_mask,
        support_stats["valid"],
        distill_config,
    )
    mix = teacher_mix_value(distill_config, step, total_steps)
    train_mask = (mix * teacher_mask + (1.0 - mix) * student_mask.detach()) * support_stats[
        "valid"
    ].float()
    summary_tensors = mask_summary(
        student_mask.detach(),
        support_stats["valid"],
        support_stats["token_loss"],
        support_stats["entropy"],
        support_stats["margin"],
    )
    summary_tensors.update(
        mask_type_summary(
            student_mask.detach(),
            support_stats["valid"],
            support_stats.get("token_types"),
            support_stats.get("loss_weights"),
        )
    )
    summary_tensors.update(teacher_metrics)
    summary_tensors.update(distill_metrics)
    summary_tensors["teacher_mix"] = torch.tensor(mix, device=student_mask.device)
    return (
        distill,
        support_stats,
        summary_tensors,
        student_mask.detach(),
        train_mask.detach(),
        inner_loss,
        inner_grad_norm,
        target_loss_before,
        target_loss_after,
        kl_loss,
    )


def run_mask_to_lora_meta_step(
    model,
    mask_head: TokenMaskHead,
    support_batch: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor],
    inner_lr: float,
    retain_kl_weight: float,
    mask_cost_weight: float,
    mask_budget_target: float | None,
    mask_budget_weight: float,
    mask_budget_mode: str,
    mask_loss_normalization: str,
    mask_loss_budget_floor: float | None,
    mask_transform_mode: str = "none",
    mask_transform_target: float | None = None,
    support_micro_batch_size: int | None = None,
    target_micro_batch_size: int | None = None,
    retain_micro_batch_size: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # The real model produces the mask, so target/retain meta-gradients can
    # improve the model's mask-producing features. The virtual inner update is
    # built on detached shadow LoRA tensors, which blocks direct target/retain
    # supervision of the persistent LoRA weights.
    support_stats = forward_for_mask(model, support_batch, detach_token_loss=True)
    mask = mask_head(
        support_stats["hidden"],
        support_stats["scalars"].detach(),
        support_stats["valid"],
    )
    mask = apply_mask_transform(
        mask,
        support_stats["valid"],
        mode=mask_transform_mode,
        target_mean=mask_transform_target,
    )

    fast_state, inner_grad_norm, inner_loss = build_fast_lora_state_from_mask(
        model,
        support_batch,
        support_stats,
        mask,
        inner_lr,
        mask_loss_normalization,
        mask_loss_budget_floor,
        support_micro_batch_size=support_micro_batch_size,
        create_graph=True,
    )
    (
        outer_loss,
        summary_tensors,
        target_loss_before,
        target_loss_after,
        kl_loss,
    ) = outer_loss_from_fast_state(
        model,
        support_stats,
        mask,
        target_batch,
        retain_batch,
        fast_state,
        retain_kl_weight,
        mask_cost_weight,
        mask_budget_target,
        mask_budget_weight,
        mask_budget_mode,
        target_micro_batch_size,
        retain_micro_batch_size,
    )
    return (
        outer_loss,
        support_stats,
        mask,
        inner_loss,
        inner_grad_norm,
        target_loss_before,
        target_loss_after,
        kl_loss,
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
        dist.barrier()

    stream = make_stream(config, seed=seed, rank=rank)
    model, tokenizer, replaced_modules = load_model_and_tokenizer(config, device)
    hidden_size = int(model.config.hidden_size)
    mask_head = TokenMaskHead(MaskHeadConfig(hidden_size=hidden_size, **config["mask_head"])).to(device)
    state_bank = load_state_bank(config, model, device)
    state_rng = random.Random(seed + 7919 * (rank + 1))

    mask_optimizer = torch.optim.AdamW(
        mask_head.parameters(),
        lr=config["optim"]["mask_lr"],
        weight_decay=config["optim"].get("mask_weight_decay", 0.0),
    )
    lora_params = list(iter_lora_parameters(model))
    lora_optimizer = torch.optim.AdamW(
        lora_params,
        lr=config["optim"]["lora_lr"],
        weight_decay=config["optim"].get("lora_weight_decay", 0.0),
    )

    run = setup_wandb(config, output_dir, rank)
    if is_rank0(rank):
        print(
            f"[phase4] rank0 starting world_size={world_size} device={device} "
            f"replaced_lora_modules={replaced_modules} state_bank={len(state_bank)} "
            f"output_dir={output_dir}",
            flush=True,
        )

    max_length = config["model"]["max_length"]
    train_config = config["training"]
    inner_lr = config["inner"]["lr"]
    retain_kl_weight = config["outer"].get("retain_kl_weight", 0.0)
    mask_cost_weight = config["outer"].get("mask_cost_weight", 0.0)
    mask_budget_target = config["outer"].get("mask_budget_target")
    mask_budget_weight = float(config["outer"].get("mask_budget_weight", 0.0))
    mask_budget_mode = config["outer"].get("mask_budget_mode", "symmetric")
    baseline_every = train_config.get("baseline_every", 0)
    log_every = train_config.get("log_every", 1)
    save_every = train_config.get("save_every", 0)
    support_micro_batch_size = train_config.get("support_micro_batch_size")
    target_micro_batch_size = train_config.get("target_micro_batch_size")
    retain_micro_batch_size = train_config.get("retain_micro_batch_size")
    mask_loss_normalization = train_config.get("mask_loss_normalization", "valid_count")
    mask_loss_budget_floor = train_config.get("mask_loss_budget_floor", mask_budget_target)
    mask_transform_mode = train_config.get("mask_transform_mode", "none")
    mask_transform_target = train_config.get("mask_transform_target")
    metrics_path = output_dir / "metrics.jsonl"
    trace_writer = MaskTraceWriter(output_dir, config.get("mask_trace", {}), rank)
    training_mode = train_config.get("mode", "online_lora")
    if training_mode not in {
        "online_lora",
        "state_bank_mask_only",
        "online_mask_to_lora",
        "teacher_student_mask_to_lora",
    }:
        raise ValueError(f"Unsupported training.mode={training_mode!r}")
    lora_meta_grad_weight = float(config["outer"].get("lora_meta_grad_weight", 1.0))
    target_loss_focus = config.get("target_loss", {"mode": "all"})
    support_loss_focus = config.get("support_loss", {"mode": "all"})
    teacher_config = config.get("teacher", {})
    distill_config = config.get("distillation", {})
    student_mask_transform_mode = train_config.get("student_mask_transform_mode", mask_transform_mode)
    student_mask_transform_target = train_config.get(
        "student_mask_transform_target",
        mask_transform_target,
    )

    for step in range(1, train_config["steps"] + 1):
        step_start = time.time()
        sampled_state_idx, sampled_state = sample_state(
            state_bank,
            state_rng,
            step,
            config.get("state_bank", {}).get("sampling"),
        )
        if training_mode == "state_bank_mask_only":
            load_lora_state(model, sampled_state["lora"], device=device)

        retain_batch_size = int(train_config.get("retain_batch_size", 0))
        if retain_kl_weight and retain_batch_size <= 0:
            raise ValueError("retain_kl_weight > 0 requires retain_batch_size > 0")

        episode: TextEpisode = stream.sample_episode(
            support_size=train_config["support_batch_size"],
            target_size=train_config["target_batch_size"],
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
            retain_batch = {
                "attention_mask": torch.zeros((), dtype=torch.long, device=device),
            }

        mask_optimizer.zero_grad(set_to_none=True)
        lora_optimizer.zero_grad(set_to_none=True)

        extra_metrics: dict[str, float] = {}
        if training_mode == "online_mask_to_lora":
            (
                outer_loss,
                support_stats,
                mask,
                inner_loss,
                inner_grad_norm,
                target_loss_before,
                target_loss_after,
                kl_loss,
            ) = run_mask_to_lora_meta_step(
                model,
                mask_head,
                support_batch,
                target_batch,
                retain_batch,
                inner_lr,
                retain_kl_weight,
                mask_cost_weight,
                mask_budget_target,
                mask_budget_weight,
                mask_budget_mode,
                mask_loss_normalization,
                mask_loss_budget_floor,
                mask_transform_mode,
                mask_transform_target,
                support_micro_batch_size,
                target_micro_batch_size,
                retain_micro_batch_size,
            )
            summary_tensors = mask_summary(
                mask,
                support_stats["valid"],
                support_stats["token_loss"],
                support_stats["entropy"],
                support_stats["margin"],
            )
            summary_tensors.update(
                mask_type_summary(
                    mask,
                    support_stats["valid"],
                    support_stats.get("token_types"),
                    support_stats.get("loss_weights"),
                )
            )
            add_mask_budget_metric(summary_tensors, mask_budget_target, mask_budget_mode)
            outer_loss.backward()
            extra_metrics["outer_lora_grad_norm"] = float(
                parameter_grad_norm(lora_params).detach().cpu()
            )
            extra_metrics["outer_mask_grad_norm"] = float(
                parameter_grad_norm(mask_head.parameters()).detach().cpu()
            )
            outer_lora_grads = clone_parameter_grads(
                lora_params,
                scale=lora_meta_grad_weight,
            )
            clear_parameter_grads(lora_params)

            support_stats_for_model = forward_for_mask(model, support_batch)
            with torch.no_grad():
                detached_mask = mask_head(
                    support_stats_for_model["hidden"].detach(),
                    support_stats_for_model["scalars"].detach(),
                    support_stats_for_model["valid"],
                )
            model_loss = masked_token_loss(
                support_stats_for_model["token_loss"],
                detached_mask,
                support_stats_for_model["valid"],
                normalization=mask_loss_normalization,
                budget_floor_rate=mask_loss_budget_floor,
            )
            model_loss.backward()
            extra_metrics["support_lora_grad_norm"] = float(
                parameter_grad_norm(lora_params).detach().cpu()
            )
            add_parameter_grads(lora_params, outer_lora_grads)
            extra_metrics["combined_lora_grad_norm"] = float(
                parameter_grad_norm(lora_params).detach().cpu()
            )
            all_reduce_grads(mask_head.parameters(), world_size)
            all_reduce_grads(lora_params, world_size)
            torch.nn.utils.clip_grad_norm_(
                mask_head.parameters(),
                config["optim"].get("max_grad_norm", 1.0),
            )
            torch.nn.utils.clip_grad_norm_(lora_params, config["optim"].get("max_lora_grad_norm", 1.0))
            mask_optimizer.step()
            lora_optimizer.step()
        elif training_mode == "teacher_student_mask_to_lora":
            (
                outer_loss,
                support_stats,
                summary_tensors,
                mask,
                train_mask,
                inner_loss,
                inner_grad_norm,
                target_loss_before,
                target_loss_after,
                kl_loss,
            ) = run_teacher_student_meta_step(
                model,
                mask_head,
                support_batch,
                target_batch,
                retain_batch,
                step,
                int(train_config["steps"]),
                inner_lr,
                retain_kl_weight,
                mask_cost_weight,
                mask_budget_target,
                mask_budget_weight,
                mask_budget_mode,
                mask_loss_normalization,
                mask_loss_budget_floor,
                teacher_config,
                distill_config,
                student_mask_transform_mode,
                student_mask_transform_target,
                support_micro_batch_size,
                target_micro_batch_size,
                retain_micro_batch_size,
            )
            add_mask_budget_metric(summary_tensors, mask_budget_target, mask_budget_mode)
            outer_loss.backward()
            extra_metrics["outer_mask_grad_norm"] = float(
                parameter_grad_norm(mask_head.parameters()).detach().cpu()
            )
            all_reduce_grads(mask_head.parameters(), world_size)
            torch.nn.utils.clip_grad_norm_(
                mask_head.parameters(),
                config["optim"].get("max_grad_norm", 1.0),
            )
            mask_optimizer.step()

            lora_optimizer.zero_grad(set_to_none=True)
            support_stats_for_model = forward_for_mask(model, support_batch)
            model_loss = masked_token_loss(
                support_stats_for_model["token_loss"],
                train_mask,
                support_stats_for_model["valid"],
                normalization=mask_loss_normalization,
                budget_floor_rate=mask_loss_budget_floor,
            )
            model_loss.backward()
            extra_metrics["support_lora_grad_norm"] = float(
                parameter_grad_norm(lora_params).detach().cpu()
            )
            all_reduce_grads(lora_params, world_size)
            torch.nn.utils.clip_grad_norm_(
                lora_params,
                config["optim"].get("max_lora_grad_norm", 1.0),
            )
            lora_optimizer.step()
        else:
            (
                outer_loss,
                support_stats,
                mask,
                inner_loss,
                inner_grad_norm,
                target_loss_before,
                target_loss_after,
                kl_loss,
            ) = run_mask_only_meta_step(
                model,
                mask_head,
                support_batch,
                target_batch,
                retain_batch,
                inner_lr,
                retain_kl_weight,
                mask_cost_weight,
                mask_budget_target,
                mask_budget_weight,
                mask_budget_mode,
                target_micro_batch_size,
                retain_micro_batch_size,
            )
            summary_tensors = mask_summary(
                mask,
                support_stats["valid"],
                support_stats["token_loss"],
                support_stats["entropy"],
                support_stats["margin"],
            )
            summary_tensors.update(
                mask_type_summary(
                    mask,
                    support_stats["valid"],
                    support_stats.get("token_types"),
                    support_stats.get("loss_weights"),
                )
            )
            add_mask_budget_metric(summary_tensors, mask_budget_target, mask_budget_mode)
            outer_loss.backward()
            all_reduce_grads(mask_head.parameters(), world_size)
            torch.nn.utils.clip_grad_norm_(mask_head.parameters(), config["optim"].get("max_grad_norm", 1.0))
            mask_optimizer.step()
            lora_optimizer.zero_grad(set_to_none=True)

            if training_mode == "online_lora":
                support_stats_for_model = forward_for_mask(model, support_batch)
                with torch.no_grad():
                    detached_mask = mask_head(
                        support_stats_for_model["hidden"].detach(),
                        support_stats_for_model["scalars"].detach(),
                        support_stats_for_model["valid"],
                    )
                model_loss = masked_token_loss(
                    support_stats_for_model["token_loss"],
                    detached_mask,
                    support_stats_for_model["valid"],
                )
                model_loss.backward()
                all_reduce_grads(lora_params, world_size)
                torch.nn.utils.clip_grad_norm_(lora_params, config["optim"].get("max_lora_grad_norm", 1.0))
                lora_optimizer.step()
            else:
                model_loss = torch.zeros((), device=device)

        elapsed = time.time() - step_start
        metrics = {
            "step": float(step),
            "outer_loss": float(outer_loss.detach().cpu()),
            "inner_loss": float(inner_loss.detach().cpu()),
            "model_loss": float(model_loss.detach().cpu()),
            "target_loss_before": float(target_loss_before.detach().cpu()),
            "target_loss_after": float(target_loss_after.detach().cpu()),
            "future_gain": float((target_loss_before - target_loss_after).detach().cpu()),
            "retain_kl": float(kl_loss.detach().cpu()),
            "inner_grad_norm": float(inner_grad_norm.detach().cpu()),
            "lora_param_norm": float(lora_parameter_norm(model).detach().cpu()),
            "state_bank_index": float(sampled_state_idx),
            "state_bank_step": float(sampled_state.get("step", -1)),
            "step_seconds": elapsed,
            "tokens_per_second": float(
                (
                    support_batch["attention_mask"].sum()
                    + target_batch["attention_mask"].sum()
                    + retain_batch["attention_mask"].sum()
                ).detach().cpu()
                / max(elapsed, 1e-6)
            ),
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
        metrics.update(extra_metrics)
        metrics.update({key: float(value.detach().cpu()) for key, value in summary_tensors.items()})
        if device.type == "cuda":
            metrics["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
            metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            metrics["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        trace_writer.write_step(
            step=step,
            episode=episode,
            sampled_state_idx=sampled_state_idx,
            sampled_state=sampled_state,
            support_batch=support_batch,
            support_stats=support_stats,
            mask=mask,
            target_loss_before=target_loss_before,
            target_loss_after=target_loss_after,
        )

        if baseline_every and step % baseline_every == 0:
            current_rate = max(0.0, min(1.0, metrics["mask_rate"]))
            full_mask = support_stats["valid"].float()
            top_mask = make_top_loss_mask(
                support_stats["token_loss"].detach(),
                support_stats["valid"],
                current_rate,
            )
            random_mask = make_random_mask(support_stats["valid"], current_rate)
            full_loss = baseline_adapt_loss(
                model,
                support_batch,
                target_batch,
                full_mask,
                inner_lr,
                mask_loss_normalization=mask_loss_normalization,
                mask_loss_budget_floor=mask_loss_budget_floor,
                target_micro_batch_size=target_micro_batch_size,
            )
            top_loss = baseline_adapt_loss(
                model,
                support_batch,
                target_batch,
                top_mask,
                inner_lr,
                mask_loss_normalization=mask_loss_normalization,
                mask_loss_budget_floor=mask_loss_budget_floor,
                target_micro_batch_size=target_micro_batch_size,
            )
            random_loss = baseline_adapt_loss(
                model,
                support_batch,
                target_batch,
                random_mask,
                inner_lr,
                mask_loss_normalization=mask_loss_normalization,
                mask_loss_budget_floor=mask_loss_budget_floor,
                target_micro_batch_size=target_micro_batch_size,
            )
            before = metrics["target_loss_before"]
            metrics.update(
                {
                    "baseline_full_target_loss": full_loss,
                    "baseline_top_loss_target_loss": top_loss,
                    "baseline_random_target_loss": random_loss,
                    "baseline_full_gain": before - full_loss,
                    "baseline_top_loss_gain": before - top_loss,
                    "baseline_random_gain": before - random_loss,
                }
            )

        reduced = reduce_metrics(
            {key: value for key, value in metrics.items() if isinstance(value, float)},
            device,
            world_size,
        )
        if is_rank0(rank) and (step % log_every == 0 or step == 1):
            row = dict(reduced)
            row["support_domain"] = episode.support_domain
            row["target_domain"] = episode.target_domain
            row["retain_domain"] = episode.retain_domain
            row["training_mode"] = training_mode
            row["state_name"] = sampled_state["name"]
            row["state_source"] = sampled_state["source"]
            write_jsonl(metrics_path, row)
            if run is not None:
                run.log(row, step=step)
            print(
                "[phase4] "
                f"step={step} gain={row['future_gain']:.4f} "
                f"target={row['target_loss_before']:.4f}->{row['target_loss_after']:.4f} "
                f"mask={row['mask_rate']:.3f} kl={row['retain_kl']:.5f} "
                f"state={row['state_name']} "
                f"domains={episode.support_domain}->{episode.target_domain} "
                f"sec={row['step_seconds']:.2f}",
                flush=True,
            )

        if is_rank0(rank) and save_every and step % save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                step,
                mask_head,
                model,
                mask_optimizer,
                lora_optimizer,
                config,
            )

    if is_rank0(rank):
        save_checkpoint(
            output_dir / "checkpoint_final.pt",
            train_config["steps"],
            mask_head,
            model,
            mask_optimizer,
            lora_optimizer,
            config,
        )
        if run is not None:
            run.finish()
        print(f"[phase4] done. Metrics: {metrics_path}", flush=True)

    trace_writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
