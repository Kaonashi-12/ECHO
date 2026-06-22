#!/usr/bin/env python
"""Summarize generation eval metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", help="Path to metrics.jsonl or an eval output directory.")
    return parser.parse_args()


def metrics_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_latest(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["checkpoint"]), str(row["dataset"]))
            latest[key] = row
    return list(latest.values())


def main() -> None:
    path = metrics_path(parse_args().metrics)
    rows = load_latest(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    checkpoints = sorted({str(row["checkpoint"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    by_key = {(str(row["checkpoint"]), str(row["dataset"])): row for row in rows}

    header = ["checkpoint", *datasets, "avg"]
    print("\t".join(header))
    for checkpoint in checkpoints:
        values: list[float] = []
        cells = [checkpoint]
        for dataset in datasets:
            row = by_key.get((checkpoint, dataset))
            if row is None:
                cells.append("")
                continue
            acc = float(row["accuracy"])
            values.append(acc)
            cells.append(f"{acc:.4f}")
        avg = sum(values) / len(values) if values else 0.0
        cells.append(f"{avg:.4f}")
        print("\t".join(cells))


if __name__ == "__main__":
    main()
