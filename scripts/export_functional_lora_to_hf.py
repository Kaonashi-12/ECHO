#!/usr/bin/env python
"""Export ECHO FunctionalLoRA checkpoints as plain HuggingFace models.

The official lm-evaluation-harness HF backend cannot load this repo's custom
FunctionalLoRA wrappers directly. This script merges the saved LoRA deltas into
the frozen base weights and writes a normal AutoModelForCausalLM directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2i.methods.capability_mask import (  # noqa: E402
    FunctionalLoRAConv1D,
    FunctionalLoRALinear,
    install_functional_lora,
    load_lora_state,
)
from s2i.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/lm_eval_qwen_v100.yaml"))
    parser.add_argument("--model-name", default=None, help="Export one named model from the config.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def save_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def merge_linear(module: FunctionalLoRALinear, save_dtype: torch.dtype) -> nn.Linear:
    merged = nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        dtype=save_dtype,
        device=module.weight.device,
    )
    delta = (module.lora_B.float() @ module.lora_A.float()) * float(module.scaling)
    merged_weight = module.weight.float() + delta
    with torch.no_grad():
        merged.weight.copy_(merged_weight.to(dtype=save_dtype))
        if module.bias is not None and merged.bias is not None:
            merged.bias.copy_(module.bias.to(dtype=save_dtype))
    return merged


def merge_conv1d(module: FunctionalLoRAConv1D, save_dtype: torch.dtype):
    from transformers.pytorch_utils import Conv1D

    merged = Conv1D(module.out_features, module.in_features)
    merged.weight = nn.Parameter(merged.weight.to(dtype=save_dtype, device=module.weight.device))
    merged.bias = nn.Parameter(merged.bias.to(dtype=save_dtype, device=module.bias.device))
    delta_linear_layout = (module.lora_B.float() @ module.lora_A.float()) * float(module.scaling)
    merged_weight = module.weight.float() + delta_linear_layout.T
    with torch.no_grad():
        merged.weight.copy_(merged_weight.to(dtype=save_dtype))
        merged.bias.copy_(module.bias.to(dtype=save_dtype))
    return merged


def merge_functional_lora_modules(model: nn.Module, save_dtype: torch.dtype) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, FunctionalLoRALinear):
            parent, child_name = resolve_parent(model, name)
            setattr(parent, child_name, merge_linear(module, save_dtype))
            replaced += 1
        elif isinstance(module, FunctionalLoRAConv1D):
            parent, child_name = resolve_parent(model, name)
            setattr(parent, child_name, merge_conv1d(module, save_dtype))
            replaced += 1
    if replaced == 0:
        raise ValueError("No FunctionalLoRA modules were found to merge")
    return replaced


def torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def selected_specs(config: dict[str, Any], model_name: str | None) -> list[dict[str, Any]]:
    specs = list(config["models"])
    if model_name:
        specs = [spec for spec in specs if spec["name"] == model_name]
    specs = [spec for spec in specs if spec.get("type", "lora") != "base"]
    if not specs:
        raise ValueError(f"No exportable LoRA models selected: {model_name or '<all>'}")
    return specs


def checkpoint_lora_config(
    checkpoint: dict[str, Any],
    spec: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint.get("config") or {}
    lora = dict(fallback)
    lora.update(config.get("lora") or {})
    lora.update(spec.get("lora") or {})
    return lora


def export_one(
    spec: dict[str, Any],
    config: dict[str, Any],
    overwrite: bool,
    dry_run: bool,
) -> None:
    checkpoint_path = resolve_path(spec["checkpoint"])
    output_dir = resolve_path(spec["hf_export_dir"])
    if dry_run:
        print(f"[export:dry-run] {spec['name']}: {checkpoint_path} -> {output_dir}")
        return
    if save_ready(output_dir) and not overwrite:
        print(f"[export] skip existing {spec['name']}: {output_dir}")
        return
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists but is incomplete: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "lora" not in checkpoint:
        raise ValueError(f"Checkpoint has no lora state: {checkpoint_path}")

    base_model = spec.get("base_model") or config["base_model"]["name"]
    trust_remote_code = bool(config["base_model"].get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    lora_config = checkpoint_lora_config(checkpoint, spec, config.get("lora", {}))
    replaced = install_functional_lora(
        model,
        target_modules=list(lora_config["target_modules"]),
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=0.0,
    )
    load_lora_state(
        model,
        checkpoint["lora"],
        device=torch.device("cpu"),
        strict=bool(spec.get("strict_lora", True)),
    )
    save_dtype = torch_dtype(str(config.get("export", {}).get("save_dtype", "fp16")))
    merged = merge_functional_lora_modules(
        model,
        save_dtype=save_dtype,
    )
    if merged != replaced:
        raise RuntimeError(f"LoRA merge count mismatch: installed={replaced} merged={merged}")
    model.to(dtype=save_dtype)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=str(config.get("export", {}).get("max_shard_size", "2GB")),
    )
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "name": spec["name"],
        "base_model": base_model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "lora": lora_config,
        "saved_dtype": str(config.get("export", {}).get("save_dtype", "fp16")),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "echo_export_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"[export] wrote {spec['name']}: {output_dir}")


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    for spec in selected_specs(config, args.model_name):
        export_one(spec, config, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
