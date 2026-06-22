#!/usr/bin/env python
"""Export concrete ECHO mask examples for prompt/response pairs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import forward_for_mask, load_model_and_tokenizer, tokenize  # noqa: E402
from s2i.methods.capability_mask import MaskHeadConfig, TokenMaskHead, load_lora_state  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/stage2_masked_sft_math_qwen_v100_from_2690_ckpt500.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/mask_cases/echo_2690_ckpt500_math_train"))
    parser.add_argument("--dataset-path", default="nlile/hendrycks-MATH-benchmark")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-solution-chars", type=int, default=2200)
    parser.add_argument("--top-k", type=int, default=24)
    return parser.parse_args()


def normalize_prompt(text: Any) -> str:
    return "\n".join(" ".join(line.strip().split()) for line in str(text).strip().splitlines())


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_path, args.dataset_config, split=args.split)
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        problem = row.get("problem")
        solution = row.get("solution")
        if not problem or not solution:
            continue
        solution = str(solution)
        if len(solution) > args.max_solution_chars:
            continue
        candidates.append(
            {
                "index": index,
                "prompt": normalize_prompt(f"Question: {problem}\nSolution:"),
                "completion": " " + solution.strip(),
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
        )
    if len(candidates) < args.num_cases:
        raise ValueError(f"Only {len(candidates)} suitable cases found")
    rng = random.Random(args.seed)
    return rng.sample(candidates, args.num_cases)


def load_echo_components(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    model, tokenizer, replaced = load_model_and_tokenizer(config, device)
    hidden_size = int(model.config.hidden_size)
    mask_head = TokenMaskHead(MaskHeadConfig(hidden_size=hidden_size, **config["mask_head"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mask_head.load_state_dict(checkpoint["mask_head"], strict=True)
    load_lora_state(model, checkpoint["lora"], device=device, strict=True)
    model.eval()
    mask_head.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, mask_head, int(checkpoint.get("step", -1)), replaced


def token_rows_for_case(
    model,
    tokenizer,
    mask_head: TokenMaskHead,
    case: dict[str, Any],
    max_length: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    batch = tokenize(
        tokenizer,
        [case["prompt"]],
        [case["completion"]],
        max_length=max_length,
        device=device,
    )
    with torch.inference_mode():
        stats = forward_for_mask(model, batch, detach_token_loss=True)
        mask = mask_head(stats["hidden"], stats["scalars"], stats["valid"])

    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    valid = stats["valid"][0].detach().cpu().bool()
    mask_values = mask[0].detach().cpu()
    token_loss = stats["token_loss"][0].detach().cpu()
    entropy = stats["entropy"][0].detach().cpu()
    margin = stats["margin"][0].detach().cpu()

    rows: list[dict[str, Any]] = []
    for shifted_pos, is_valid in enumerate(valid.tolist()):
        if not is_valid:
            continue
        token_pos = shifted_pos + 1
        token_id = input_ids[token_pos]
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        rows.append(
            {
                "token_index": token_pos,
                "token": token,
                "token_repr": repr(token),
                "mask": float(mask_values[shifted_pos]),
                "token_loss": float(token_loss[shifted_pos]),
                "entropy": float(entropy[shifted_pos]),
                "margin": float(margin[shifted_pos]),
            }
        )

    masks = [row["mask"] for row in rows]
    losses = [row["token_loss"] for row in rows]
    summary = {
        "num_response_tokens": float(len(rows)),
        "mask_mean": float(sum(masks) / max(len(masks), 1)),
        "mask_max": float(max(masks) if masks else 0.0),
        "mask_min": float(min(masks) if masks else 0.0),
        "loss_mean": float(sum(losses) / max(len(losses), 1)),
    }
    return rows, summary


def color_for_mask(value: float) -> str:
    value = max(0.0, min(1.0, value))
    alpha = 0.08 + 0.72 * value
    return f"rgba(235, 90, 45, {alpha:.3f})"


def write_outputs(
    output_dir: Path,
    cases: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_step: int,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "mask_cases.jsonl"
    md_path = output_dir / "mask_cases.md"
    html_path = output_dir / "mask_cases.html"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    md_lines = [
        "# ECHO Mask Cases",
        "",
        f"checkpoint: `{checkpoint_path}`",
        f"checkpoint_step: `{checkpoint_step}`",
        "",
        "Mask values are the frozen stage-1 ECHO mask head scores on response tokens.",
        "",
    ]
    for case_id, case in enumerate(cases, start=1):
        rows = case["tokens"]
        top_rows = sorted(rows, key=lambda row: row["mask"], reverse=True)[:top_k]
        low_rows = sorted(rows, key=lambda row: row["mask"])[: min(12, len(rows))]
        md_lines.extend(
            [
                f"## Case {case_id}: index={case['index']} subject={case.get('subject')} level={case.get('level')}",
                "",
                f"- mask_mean: `{case['summary']['mask_mean']:.4f}`",
                f"- mask_max: `{case['summary']['mask_max']:.4f}`",
                f"- response_tokens: `{int(case['summary']['num_response_tokens'])}`",
                "",
                "### Prompt",
                "",
                "```text",
                case["prompt"],
                "```",
                "",
                "### Response Excerpt",
                "",
                "```text",
                case["completion"][:1200],
                "```",
                "",
                "### Top Masked Tokens",
                "",
                "|rank|token|mask|loss|",
                "|---:|---|---:|---:|",
            ]
        )
        for rank, row in enumerate(top_rows, start=1):
            token = row["token_repr"].replace("|", "\\|")
            md_lines.append(f"|{rank}|`{token}`|{row['mask']:.4f}|{row['token_loss']:.3f}|")
        md_lines.extend(["", "### Lowest Mask Tokens", "", "|rank|token|mask|loss|", "|---:|---|---:|---:|"])
        for rank, row in enumerate(low_rows, start=1):
            token = row["token_repr"].replace("|", "\\|")
            md_lines.append(f"|{rank}|`{token}`|{row['mask']:.4f}|{row['token_loss']:.3f}|")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:system-ui,sans-serif;line-height:1.45;max-width:1100px;margin:32px auto;padding:0 16px;} .tok{padding:1px 2px;border-radius:3px;} pre{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:6px;} table{border-collapse:collapse;}td,th{border:1px solid #ddd;padding:4px 8px;}</style>",
        "</head><body>",
        "<h1>ECHO Mask Cases</h1>",
        f"<p><b>checkpoint:</b> {html.escape(str(checkpoint_path))}<br><b>step:</b> {checkpoint_step}</p>",
    ]
    for case_id, case in enumerate(cases, start=1):
        html_parts.append(
            f"<h2>Case {case_id}: index={case['index']} subject={html.escape(str(case.get('subject')))} level={html.escape(str(case.get('level')))}</h2>"
        )
        html_parts.append(
            f"<p>mask_mean={case['summary']['mask_mean']:.4f}, mask_max={case['summary']['mask_max']:.4f}, response_tokens={int(case['summary']['num_response_tokens'])}</p>"
        )
        html_parts.append("<h3>Prompt</h3>")
        html_parts.append(f"<pre>{html.escape(case['prompt'])}</pre>")
        html_parts.append("<h3>Masked Response Tokens</h3>")
        html_parts.append("<p>")
        for row in case["tokens"]:
            title = (
                f"mask={row['mask']:.4f} loss={row['token_loss']:.3f} "
                f"entropy={row['entropy']:.3f} margin={row['margin']:.3f}"
            )
            html_parts.append(
                f"<span class='tok' title='{html.escape(title)}' "
                f"style='background:{color_for_mask(row['mask'])}'>{html.escape(row['token'])}</span>"
            )
        html_parts.append("</p>")
    html_parts.append("</body></html>")
    html_path.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_path(args.checkpoint or config["checkpoint"]["path"])
    output_dir = resolve_path(args.output_dir)
    model, tokenizer, mask_head, checkpoint_step, replaced = load_echo_components(
        config,
        checkpoint_path,
        device,
    )
    print(f"[mask-cases] loaded checkpoint={checkpoint_path} step={checkpoint_step} replaced_lora={replaced}")
    raw_cases = load_cases(args)
    exported: list[dict[str, Any]] = []
    max_length = int(config["model"].get("max_length", 768))
    for case in raw_cases:
        rows, summary = token_rows_for_case(
            model,
            tokenizer,
            mask_head,
            case,
            max_length=max_length,
            device=device,
        )
        exported.append({**case, "summary": summary, "tokens": rows})
        print(
            f"[mask-cases] case index={case['index']} tokens={len(rows)} "
            f"mask_mean={summary['mask_mean']:.4f} mask_max={summary['mask_max']:.4f}",
            flush=True,
        )
    write_outputs(output_dir, exported, checkpoint_path, checkpoint_step, args.top_k)
    print(f"[mask-cases] wrote {output_dir}")


if __name__ == "__main__":
    main()
