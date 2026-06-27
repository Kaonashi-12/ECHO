#!/usr/bin/env python
"""Summarize route-decision metrics from ECHO training and eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage-1 output dir or metrics.jsonl")
    parser.add_argument("--stage2", default=None, help="Stage-2 output dir or metrics.jsonl")
    parser.add_argument("--eval", default=None, help="Generation eval output dir or metrics.jsonl")
    parser.add_argument("--last-n", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metrics_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return mean(values) if values else None


def summarize_stage(rows: list[dict[str, Any]], last_n: int, prefix: str) -> dict[str, Any]:
    tail = rows[-last_n:] if last_n > 0 else rows
    keys = [
        "future_gain",
        "baseline_full_gain",
        "baseline_random_gain",
        "baseline_top_loss_gain",
        "student_teacher_corr",
        "student_teacher_mse",
        "mask_rate",
        "mask_std",
        "teacher_mask_rate",
        "teacher_mask_std",
        "teacher_logit_grad_norm",
        "teacher_logit_std",
        "teacher_raw_mask_std",
        "teacher_final_eval",
        "teacher_mix",
        "target_loss_focus_tokens",
        "target_loss_focus_fraction",
        "mask_mass_frac_format",
        "mask_mass_frac_answer",
        "mask_mass_frac_number",
        "mask_mass_frac_operator",
        "teacher_mask_mass_frac_format",
        "teacher_mask_mass_frac_answer",
        "teacher_mask_mass_frac_number",
        "teacher_mask_mass_frac_operator",
        "step_seconds",
        "gpu_mem_max_gb",
    ]
    out: dict[str, Any] = {
        f"{prefix}_rows": len(rows),
        f"{prefix}_tail_rows": len(tail),
    }
    for key in keys:
        value = avg(tail, key)
        if value is not None:
            out[f"{prefix}_{key}_mean"] = value
    if tail:
        out[f"{prefix}_last_step"] = tail[-1].get("step")
        out[f"{prefix}_last_training_mode"] = tail[-1].get("training_mode")
    return out


def summarize_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if "checkpoint" in row and "dataset" in row:
            latest[(str(row["checkpoint"]), str(row["dataset"]))] = row
    for (checkpoint, dataset), row in sorted(latest.items()):
        if "accuracy" in row:
            out[f"eval_accuracy__{checkpoint}__{dataset}"] = float(row["accuracy"])
    return out


def main() -> None:
    args = parse_args()
    stage1_rows = load_jsonl(metrics_path(args.stage1))
    stage2_rows = load_jsonl(metrics_path(args.stage2))
    eval_rows = load_jsonl(metrics_path(args.eval))
    summary: dict[str, Any] = {}
    summary.update(summarize_stage(stage1_rows, args.last_n, "stage1"))
    if stage2_rows:
        summary.update(summarize_stage(stage2_rows, args.last_n, "stage2"))
    if eval_rows:
        summary.update(summarize_eval(eval_rows))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
