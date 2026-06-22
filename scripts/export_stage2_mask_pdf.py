#!/usr/bin/env python
"""Export stage-2 ECHO token-mask visualizations as a PDF report."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase4_mask_mvp import forward_for_mask, load_model_and_tokenizer, tokenize  # noqa: E402
from run_stage2_masked_sft import PromptCompletionStream, _resolve_path  # noqa: E402
from s2i.methods.capability_mask import MaskHeadConfig, TokenMaskHead, load_lora_state  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402
from s2i.utils.seed import set_seed  # noqa: E402


PAGE_W = 612.0
PAGE_H = 792.0
MARGIN_X = 34.0
MARGIN_TOP = 34.0
MARGIN_BOTTOM = 34.0
FONT = 7.2
SMALL_FONT = 6.5
HEADER_FONT = 11.0
LINE_H = 10.2
CHAR_W = FONT * 0.58
MAX_CHARS = int((PAGE_W - 2 * MARGIN_X) / CHAR_W)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/claim1_stage2_echo_math_qwen_v100.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=7319)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-path", default=None, help="Optional local model snapshot path.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Checkpoint specs as label=path. Defaults to stage2 step 500/1000/1500/2000/final.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-question-chars", type=int, default=900)
    return parser.parse_args()


def pdf_escape(text: str) -> str:
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class SimplePdf:
    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.commands: list[str] = []

    def new_page(self) -> None:
        if self.commands:
            self.pages.append(self.commands)
        self.commands = []

    def finish_page(self) -> None:
        if self.commands:
            self.pages.append(self.commands)
            self.commands = []

    def text(self, x: float, y: float, text: str, size: float = FONT, color: tuple[float, float, float] = (0, 0, 0)) -> None:
        r, g, b = color
        self.commands.append(
            f"{r:.3f} {g:.3f} {b:.3f} rg BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET"
        )

    def rect(self, x: float, y: float, w: float, h: float, color: tuple[float, float, float]) -> None:
        r, g, b = color
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def save(self, path: Path) -> None:
        self.finish_page()
        objects: dict[int, bytes] = {}
        page_ids: list[int] = []
        content_ids: list[int] = []
        next_id = 4
        for _ in self.pages:
            page_ids.append(next_id)
            content_ids.append(next_id + 1)
            next_id += 2

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"

        for page_id, content_id, commands in zip(page_ids, content_ids, self.pages):
            content = ("\n".join(commands) + "\n").encode("latin-1", "replace")
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("latin-1")
            objects[content_id] = (
                b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream"
            )

        max_id = max(objects)
        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * (max_id + 1)
        for obj_id in range(1, max_id + 1):
            offsets[obj_id] = len(data)
            data.extend(f"{obj_id} 0 obj\n".encode("ascii"))
            data.extend(objects[obj_id])
            data.extend(b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
        data.extend(b"0000000000 65535 f \n")
        for obj_id in range(1, max_id + 1):
            data.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
        data.extend(f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(data))


def mask_color(value: float | None) -> tuple[float, float, float]:
    if value is None or math.isnan(value):
        return (1.0, 1.0, 1.0)
    value = max(0.0, min(1.0, float(value)))
    if value < 0.10:
        return (0.96, 0.96, 0.96)
    if value < 0.20:
        return (0.90, 0.96, 1.00)
    if value < 0.35:
        return (0.84, 0.93, 1.00)
    if value < 0.50:
        return (1.00, 0.95, 0.62)
    if value < 0.65:
        return (1.00, 0.78, 0.34)
    if value < 0.80:
        return (1.00, 0.55, 0.25)
    return (1.00, 0.34, 0.30)


def wrap_chars(text: str, values: list[float | None], max_chars: int = MAX_CHARS) -> list[tuple[str, list[float | None]]]:
    lines: list[tuple[str, list[float | None]]] = []
    current_chars: list[str] = []
    current_values: list[float | None] = []
    for ch, value in zip(text, values):
        if ch == "\n":
            lines.append(("".join(current_chars), current_values))
            current_chars = []
            current_values = []
            continue
        current_chars.append(ch)
        current_values.append(value)
        if len(current_chars) >= max_chars:
            lines.append(("".join(current_chars), current_values))
            current_chars = []
            current_values = []
    if current_chars or not lines:
        lines.append(("".join(current_chars), current_values))
    return lines


def draw_wrapped(pdf: SimplePdf, text: str, values: list[float | None], y: float, *, size: float = FONT) -> float:
    char_w = size * 0.58
    line_h = size * 1.42
    max_chars = int((PAGE_W - 2 * MARGIN_X) / char_w)
    for line, line_values in wrap_chars(text, values, max_chars=max_chars):
        if y < MARGIN_BOTTOM + line_h:
            pdf.new_page()
            y = PAGE_H - MARGIN_TOP
        start = 0
        while start < len(line_values):
            color = mask_color(line_values[start])
            end = start + 1
            while end < len(line_values) and mask_color(line_values[end]) == color:
                end += 1
            if line_values[start] is not None:
                pdf.rect(MARGIN_X + start * char_w, y - 2.0, (end - start) * char_w, line_h - 1.5, color)
            start = end
        pdf.text(MARGIN_X, y, line, size=size)
        y -= line_h
    return y


def draw_legend(pdf: SimplePdf, y: float) -> float:
    pdf.text(MARGIN_X, y, "Color legend: soft mask weight", size=FONT)
    y -= 12.0
    x = MARGIN_X
    for label, value in [("0-.10", 0.05), (".10-.20", 0.15), (".20-.35", 0.25), (".35-.50", 0.42), (".50-.65", 0.55), (".65-.80", 0.70), (".80-1", 0.90)]:
        pdf.rect(x, y - 2.0, 28.0, 10.0, mask_color(value))
        pdf.text(x + 32.0, y, label, size=SMALL_FONT)
        x += 72.0
    return y - 18.0


def quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * p))]


def histogram(values: list[float], bins: list[float]) -> list[int]:
    counts = [0] * (len(bins) - 1)
    for value in values:
        for idx in range(len(bins) - 1):
            if (bins[idx] <= value < bins[idx + 1]) or (idx == len(bins) - 2 and value <= bins[idx + 1]):
                counts[idx] += 1
                break
    return counts


def load_checkpoint_into(model, mask_head: TokenMaskHead, path: Path, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    load_lora_state(model, checkpoint["lora"], device=device, strict=True)
    mask_head.load_state_dict(checkpoint["mask_head"], strict=True)
    return int(checkpoint.get("step", -1))


def checkpoint_specs(args: argparse.Namespace, run_dir: Path) -> list[tuple[str, Path]]:
    if args.checkpoints:
        specs = []
        for item in args.checkpoints:
            label, raw_path = item.split("=", 1)
            path = Path(raw_path)
            specs.append((label, path if path.is_absolute() else ROOT / path))
        return specs
    raw = [
        ("step500", run_dir / "checkpoint_step_500.pt"),
        ("step1000", run_dir / "checkpoint_step_1000.pt"),
        ("step1500", run_dir / "checkpoint_step_1500.pt"),
        ("step2000", run_dir / "checkpoint_step_2000.pt"),
        ("final", run_dir / "checkpoint_final.pt"),
    ]
    return [(label, path) for label, path in raw if path.exists()]


def sample_examples(config: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    stream = PromptCompletionStream(config["data"], seed=seed)
    rng = random.Random(seed)
    return [rng.choice(stream.examples) for _ in range(count)]


def token_mask_char_values(
    tokenizer,
    text: str,
    prompt: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    max_length: int,
) -> tuple[list[float | None], list[float], bool]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    values: list[float | None] = [None] * len(text)
    token_values: list[float] = []
    valid_cpu = valid.detach().bool().cpu()
    mask_cpu = mask.detach().float().cpu()
    seq_len = min(len(offsets), int(attention_mask.sum().detach().cpu().item()))
    for context_pos in valid_cpu.nonzero(as_tuple=False).flatten().tolist():
        target_pos = context_pos + 1
        if target_pos >= seq_len or target_pos >= len(offsets):
            continue
        start, end = offsets[target_pos]
        if end <= start:
            continue
        value = float(mask_cpu[context_pos].item())
        token_values.append(value)
        for idx in range(max(0, start), min(len(values), end)):
            values[idx] = value
    truncated = bool(len(encoded["input_ids"]) >= max_length and offsets and offsets[-1][1] < len(text))
    return values, token_values, truncated


def summarize_sample(token_values: list[float], threshold: float) -> dict[str, float]:
    if not token_values:
        return {"mean": 0.0, "hard": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(token_values) / len(token_values),
        "hard": sum(1 for value in token_values if value >= threshold) / len(token_values),
        "p95": quantile(token_values, 0.95),
        "max": max(token_values),
    }


def draw_hist_page(pdf: SimplePdf, label: str, path: Path, all_values: list[float], threshold: float) -> None:
    pdf.new_page()
    y = PAGE_H - MARGIN_TOP
    pdf.text(MARGIN_X, y, f"Stage {label} Mask Distribution", size=HEADER_FONT)
    y -= 14.0
    pdf.text(MARGIN_X, y, f"checkpoint: {path}", size=SMALL_FONT)
    y -= 14.0
    y = draw_legend(pdf, y)
    if all_values:
        summary = (
            f"n={len(all_values)} mean={sum(all_values)/len(all_values):.4f} "
            f"p50={quantile(all_values, .50):.4f} p95={quantile(all_values, .95):.4f} "
            f"max={max(all_values):.4f} hard>={threshold:.2f}="
            f"{sum(1 for value in all_values if value >= threshold) / len(all_values):.4f}"
        )
        pdf.text(MARGIN_X, y, summary, size=FONT)
        y -= 16.0
    bins = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
    counts = histogram(all_values, bins)
    max_count = max(counts) if counts else 1
    for idx, count in enumerate(counts):
        if y < MARGIN_BOTTOM + 14:
            pdf.new_page()
            y = PAGE_H - MARGIN_TOP
        label_text = f"[{bins[idx]:.2f},{bins[idx + 1]:.2f}]"
        width = 360.0 * count / max(max_count, 1)
        pdf.text(MARGIN_X, y, f"{label_text:>13} {count:6d}", size=FONT)
        pdf.rect(MARGIN_X + 130.0, y - 2.0, width, 8.0, mask_color((bins[idx] + bins[idx + 1]) / 2.0))
        y -= 12.0


def draw_sample_page(
    pdf: SimplePdf,
    *,
    stage_label: str,
    sample_index: int,
    example: dict[str, Any],
    text: str,
    prompt: str,
    values: list[float | None],
    token_values: list[float],
    truncated: bool,
    threshold: float,
    max_question_chars: int,
) -> None:
    pdf.new_page()
    y = PAGE_H - MARGIN_TOP
    stats = summarize_sample(token_values, threshold)
    meta = example.get("metadata", {})
    pdf.text(MARGIN_X, y, f"{stage_label} sample {sample_index:02d} | source={example.get('source_id')} | subject={meta.get('subject')} | level={meta.get('level')}", size=HEADER_FONT)
    y -= 13.0
    pdf.text(
        MARGIN_X,
        y,
        f"mask mean={stats['mean']:.4f} hard>={threshold:.2f}={stats['hard']:.4f} p95={stats['p95']:.4f} max={stats['max']:.4f}"
        + (" | truncated at model max_length" if truncated else ""),
        size=FONT,
    )
    y -= 13.0
    y = draw_legend(pdf, y)

    question = prompt
    if len(question) > max_question_chars:
        question = question[:max_question_chars] + " ... [question truncated in PDF header]"
    pdf.text(MARGIN_X, y, "Prompt:", size=FONT)
    y -= 11.0
    y = draw_wrapped(pdf, question, [None] * len(question), y, size=SMALL_FONT)
    y -= 8.0

    completion_start = len(prompt)
    response = text[completion_start:]
    response_values = values[completion_start:]
    pdf.text(MARGIN_X, y, "Full response, colored by target-token soft mask:", size=FONT)
    y -= 11.0
    draw_wrapped(pdf, response, response_values, y, size=FONT)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_dir = Path(args.run_dir or config["experiment"]["output_dir"])
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    output = Path(args.output or run_dir / "stage2_mask_cases.pdf")
    output = output if output.is_absolute() else ROOT / output
    max_length = int(args.max_length or config["model"]["max_length"])

    if args.model_path:
        config["model"]["name"] = args.model_path
    config["model"]["gradient_checkpointing"] = False
    config["model"]["padding_side"] = config["model"].get("padding_side", "right")

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = sample_examples(config, args.sample_count, args.sample_seed)
    model, tokenizer, _ = load_model_and_tokenizer(config, device)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    mask_head = TokenMaskHead(MaskHeadConfig(hidden_size=hidden_size, **config["mask_head"])).to(device)
    mask_head.eval()

    specs = checkpoint_specs(args, run_dir)
    if not specs:
        raise SystemExit(f"No checkpoints found under {run_dir}")

    pdf = SimplePdf()
    pdf.new_page()
    y = PAGE_H - MARGIN_TOP
    pdf.text(MARGIN_X, y, "Stage-2 ECHO Token Mask Case Report", size=15.0)
    y -= 18.0
    pdf.text(MARGIN_X, y, f"run_dir: {run_dir}", size=FONT)
    y -= 12.0
    pdf.text(MARGIN_X, y, f"samples: {len(examples)} fixed examples, seed={args.sample_seed}, max_length={max_length}", size=FONT)
    y -= 12.0
    pdf.text(MARGIN_X, y, "Same examples are replayed at every checkpoint; colors show soft mask on target tokens.", size=FONT)
    y -= 18.0
    y = draw_legend(pdf, y)
    for label, path in specs:
        pdf.text(MARGIN_X, y, f"{label}: {path}", size=SMALL_FONT)
        y -= 10.0
        if y < MARGIN_BOTTOM + 16:
            pdf.new_page()
            y = PAGE_H - MARGIN_TOP

    for stage_label, ckpt_path in specs:
        step = load_checkpoint_into(model, mask_head, ckpt_path, device)
        all_stage_values: list[float] = []
        rendered: list[tuple[dict[str, Any], str, str, list[float | None], list[float], bool]] = []
        for start in range(0, len(examples), args.batch_size):
            batch_examples = examples[start : start + args.batch_size]
            prompts = [example["prompt"] for example in batch_examples]
            completions = [example["completion"] for example in batch_examples]
            batch = tokenize(tokenizer, prompts, completions, max_length, device)
            with torch.no_grad():
                stats = forward_for_mask(model, batch)
                soft_mask = mask_head(stats["hidden"], stats["scalars"], stats["valid"]).detach()
            for local_idx, example in enumerate(batch_examples):
                text = prompts[local_idx] + completions[local_idx]
                values, token_values, truncated = token_mask_char_values(
                    tokenizer,
                    text,
                    prompts[local_idx],
                    batch["input_ids"][local_idx],
                    batch["attention_mask"][local_idx],
                    soft_mask[local_idx],
                    stats["valid"][local_idx],
                    max_length,
                )
                all_stage_values.extend(token_values)
                rendered.append((example, text, prompts[local_idx], values, token_values, truncated))
            del batch, stats, soft_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
        draw_hist_page(pdf, f"{stage_label} (ckpt_step={step})", ckpt_path, all_stage_values, args.mask_threshold)
        for sample_idx, item in enumerate(rendered):
            example, text, prompt, values, token_values, truncated = item
            draw_sample_page(
                pdf,
                stage_label=f"{stage_label} ckpt_step={step}",
                sample_index=sample_idx,
                example=example,
                text=text,
                prompt=prompt,
                values=values,
                token_values=token_values,
                truncated=truncated,
                threshold=args.mask_threshold,
                max_question_chars=args.max_question_chars,
            )
        print(
            json.dumps(
                {
                    "stage": stage_label,
                    "checkpoint_step": step,
                    "checkpoint": str(ckpt_path),
                    "tokens": len(all_stage_values),
                    "mean": sum(all_stage_values) / max(len(all_stage_values), 1),
                    "hard_rate": sum(1 for value in all_stage_values if value >= args.mask_threshold)
                    / max(len(all_stage_values), 1),
                    "p50": quantile(all_stage_values, 0.50) if all_stage_values else 0.0,
                    "p95": quantile(all_stage_values, 0.95) if all_stage_values else 0.0,
                    "max": max(all_stage_values) if all_stage_values else 0.0,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    pdf.save(output)
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
