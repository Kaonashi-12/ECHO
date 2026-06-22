#!/usr/bin/env python
"""CPU preflight checks for generation eval configs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_generation import load_eval_examples, resolve_path  # noqa: E402
from s2i.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-datasets", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    missing = []

    print(f"config\t{args.config}")
    for spec in config.get("checkpoints", []):
        name = spec["name"]
        path = spec.get("path")
        if spec.get("type") == "base" or path in {None, ""}:
            print(f"checkpoint\t{name}\tbase")
            continue
        resolved = resolve_path(path)
        status = "ok" if resolved.exists() else "missing"
        print(f"checkpoint\t{name}\t{status}\t{resolved}")
        if status == "missing":
            missing.append(str(resolved))

    if missing and not args.allow_missing_checkpoints:
        raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))

    if args.load_datasets:
        for spec in config.get("datasets", []):
            examples = load_eval_examples(spec)
            print(f"dataset\t{spec['name']}\tok\t{len(examples)}")


if __name__ == "__main__":
    main()
