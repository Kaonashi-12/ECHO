#!/usr/bin/env python
"""Export token-mask trace samples to a lightweight PDF report."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PAGE_W = 612.0
PAGE_H = 792.0
MARGIN_X = 36.0
MARGIN_TOP = 36.0
MARGIN_BOTTOM = 34.0
FONT_SIZE = 8.5
LINE_H = 12.0
CHAR_W = FONT_SIZE * 0.60
MAX_CHARS = int((PAGE_W - 2 * MARGIN_X) / CHAR_W)


def safe_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def iter_gzip_jsonl(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            while True:
                try:
                    line = handle.readline()
                except EOFError:
                    break
                if not line:
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except EOFError:
        return


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

    def text(self, x: float, y: float, text: str, size: float = FONT_SIZE) -> None:
        self.commands.append(f"0 0 0 rg BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET")

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
            objects[content_id] = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream"

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


def char_highlights(text: str, offsets: list[tuple[int, int]], selected_positions: list[int]) -> list[bool]:
    flags = [False] * len(text)
    for pos in selected_positions:
        if pos < 0 or pos >= len(offsets):
            continue
        start, end = offsets[pos]
        start = max(0, min(len(text), int(start)))
        end = max(start, min(len(text), int(end)))
        for idx in range(start, end):
            flags[idx] = True
    return flags


def wrap_text(text: str, flags: list[bool]) -> list[tuple[str, list[bool]]]:
    lines: list[tuple[str, list[bool]]] = []
    current_chars: list[str] = []
    current_flags: list[bool] = []
    for ch, flag in zip(text, flags):
        if ch == "\n":
            lines.append(("".join(current_chars), current_flags))
            current_chars = []
            current_flags = []
            continue
        current_chars.append(ch)
        current_flags.append(flag)
        if len(current_chars) >= MAX_CHARS:
            lines.append(("".join(current_chars), current_flags))
            current_chars = []
            current_flags = []
    if current_chars or not lines:
        lines.append(("".join(current_chars), current_flags))
    return lines


def draw_highlighted_lines(pdf: SimplePdf, x: float, y: float, lines: list[tuple[str, list[bool]]]) -> float:
    for line, flags in lines:
        idx = 0
        while idx < len(flags):
            if not flags[idx]:
                idx += 1
                continue
            start = idx
            while idx < len(flags) and flags[idx]:
                idx += 1
            pdf.rect(x + start * CHAR_W, y - 2.0, (idx - start) * CHAR_W, LINE_H - 1.0, (1.0, 0.88, 0.30))
        pdf.text(x, y, line, FONT_SIZE)
        y -= LINE_H
    return y


def choose_samples(samples: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if any(int(sample.get("selected_count", 0)) > 0 for sample in samples):
        return sorted(samples, key=lambda sample: (-int(sample.get("selected_count", 0)), int(sample.get("sample_index", 0))))[:n]
    if len(samples) <= n:
        return samples
    indexes = [round(i * (len(samples) - 1) / (n - 1)) for i in range(n)]
    return [samples[int(idx)] for idx in indexes]


def find_stage_rows(trace_path: Path, stage_steps: list[int]) -> dict[int, dict[str, Any]]:
    wanted = set(stage_steps)
    rows: dict[int, dict[str, Any]] = {}
    last_row: dict[str, Any] | None = None
    for row in iter_gzip_jsonl(trace_path):
        step = int(row["step"])
        last_row = row
        if step in wanted:
            rows[step] = row
    if last_row is not None and stage_steps[-1] not in rows:
        rows[int(last_row["step"])] = last_row
    return rows


def stage_plan(metrics: list[dict[str, Any]]) -> list[tuple[str, int]]:
    last_step = int(metrics[-1]["step"]) if metrics else 1
    raw = [
        ("initial", 1),
        ("early", 50),
        ("collapse-shift", 100),
        ("checkpoint", 250),
        ("latest", last_step),
    ]
    plan: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, step in raw:
        step = min(step, last_step)
        if step in seen:
            continue
        seen.add(step)
        plan.append((label, step))
    return plan


def metric_summary(metrics_by_step: dict[int, dict[str, Any]], step: int) -> str:
    row = metrics_by_step.get(step, {})
    if not row:
        return "metrics unavailable"
    parts = [
        f"mask_rate={row.get('mask_rate', math.nan):.4f}",
        f"future_gain={row.get('future_gain', math.nan):.4f}",
        f"loss_corr={row.get('loss_mask_corr', math.nan):.3f}",
    ]
    return " | ".join(parts)


def build_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    metrics = safe_jsonl(run_dir / "metrics.jsonl")
    metrics_by_step = {int(row["step"]): row for row in metrics if "step" in row}
    plan = stage_plan(metrics)
    desired_steps = [step for _, step in plan]
    trace_path = run_dir / args.trace_file
    stage_rows = find_stage_rows(trace_path, desired_steps)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    pdf = SimplePdf()
    pdf.new_page()
    y = PAGE_H - MARGIN_TOP
    pdf.text(MARGIN_X, y, "Mask Trace Highlight Report", 15.0)
    y -= 18.0
    pdf.text(MARGIN_X, y, f"run_dir: {run_dir}", 8.5)
    y -= 12.0
    pdf.text(MARGIN_X, y, "Yellow highlight = unmasked token (mask >= trace threshold). Samples are from rank0 trace.", 8.5)
    y -= 20.0

    for label, requested_step in plan:
        row = stage_rows.get(requested_step)
        if row is None and label == "latest" and stage_rows:
            row = stage_rows[max(stage_rows)]
        if row is None:
            continue
        step = int(row["step"])
        samples = choose_samples(row.get("samples", []), args.samples_per_stage)
        header = (
            f"Stage: {label} | step={step} | support={row.get('support_domain')} -> target={row.get('target_domain')} | "
            f"{metric_summary(metrics_by_step, step)}"
        )
        needed = 20.0
        if y - needed < MARGIN_BOTTOM:
            pdf.new_page()
            y = PAGE_H - MARGIN_TOP
        pdf.rect(MARGIN_X - 4.0, y - 5.0, PAGE_W - 2 * MARGIN_X + 8.0, 14.0, (0.90, 0.90, 0.90))
        pdf.text(MARGIN_X, y, header, 8.5)
        y -= 20.0

        for sample in samples:
            text = sample.get("text", "")
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
            )
            offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
            flags = char_highlights(text, offsets, sample.get("selected_target_positions", []))
            lines = wrap_text(text, flags)
            block_h = 30.0 + len(lines) * LINE_H
            if y - block_h < MARGIN_BOTTOM:
                pdf.new_page()
                y = PAGE_H - MARGIN_TOP
            meta = (
                f"sample={sample.get('sample_index')} | selected={sample.get('selected_count')}/{sample.get('valid_count')} "
                f"({sample.get('selected_fraction')}) | soft_mean={sample.get('soft_mask_mean')}"
            )
            pdf.text(MARGIN_X, y, meta, 8.5)
            y -= 12.0
            y = draw_highlighted_lines(pdf, MARGIN_X, y, lines)
            y -= 10.0

    pdf.save(Path(args.output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--trace-file", default="mask_trace_rank0.jsonl.gz")
    parser.add_argument("--samples-per-stage", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    build_report(parse_args())
