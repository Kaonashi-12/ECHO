#!/usr/bin/env python
"""Run official lm-evaluation-harness tasks for exported ECHO models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2i.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/lm_eval_qwen_v100.yaml"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task override.")
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--no-accelerate", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def lm_eval_entrypoint() -> list[str]:
    if command_exists("lm-eval"):
        return ["lm-eval", "run"]
    if command_exists("lm_eval"):
        return ["lm_eval", "run"]
    return [sys.executable, "-m", "lm_eval", "run"]


def import_check() -> None:
    code = "import lm_eval; print(getattr(lm_eval, '__version__', 'unknown'))"
    try:
        subprocess.run([sys.executable, "-c", code], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "lm_eval is not installed in this environment. Install it with "
            "'pip install \"lm_eval[hf]\"' before running official evaluation."
        ) from exc


def selected_models(config: dict[str, Any], model_name: str | None) -> list[dict[str, Any]]:
    specs = list(config["models"])
    if model_name:
        specs = [spec for spec in specs if spec["name"] == model_name]
    if not specs:
        raise ValueError(f"No models selected: {model_name or '<all>'}")
    return specs


def model_pretrained_path(spec: dict[str, Any], config: dict[str, Any]) -> str:
    if spec.get("type") == "base":
        return str(spec.get("pretrained") or config["base_model"]["name"])
    return str(resolve_path(spec["hf_export_dir"]))


def model_args_string(pretrained: str, config: dict[str, Any], spec: dict[str, Any]) -> str:
    hf_args = dict(config.get("lm_eval", {}).get("model_args", {}))
    hf_args.update(spec.get("model_args", {}))
    hf_args["pretrained"] = pretrained
    return ",".join(f"{key}={value}" for key, value in hf_args.items())


def task_string(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.tasks:
        return args.tasks
    tasks = config.get("lm_eval", {}).get("tasks", [])
    if not tasks:
        raise ValueError("No lm_eval.tasks configured")
    return ",".join(str(task) for task in tasks)


def build_eval_command(
    spec: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    lm_cfg = config.get("lm_eval", {})
    output_root = resolve_path(lm_cfg.get("output_root", "outputs/lm_eval_official/qwen05b_v100"))
    output_path = output_root / spec["name"]
    output_path.mkdir(parents=True, exist_ok=True)

    command: list[str]
    use_accelerate = bool(lm_cfg.get("use_accelerate", True)) and not args.no_accelerate
    num_processes = int(args.num_processes or lm_cfg.get("num_processes", 1))
    if use_accelerate and num_processes > 1:
        command = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
            "-m",
            "lm_eval",
            "run",
        ]
    else:
        command = lm_eval_entrypoint()

    command += [
        "--model",
        str(lm_cfg.get("model", "hf")),
        "--model_args",
        model_args_string(model_pretrained_path(spec, config), config, spec),
        "--tasks",
        task_string(args, config),
        "--batch_size",
        str(lm_cfg.get("batch_size", "auto")),
        "--output_path",
        str(output_path),
    ]
    if not (use_accelerate and num_processes > 1):
        device = lm_cfg.get("device")
        if device:
            command += ["--device", str(device)]
    if lm_cfg.get("num_fewshot") is not None:
        command += ["--num_fewshot", str(lm_cfg["num_fewshot"])]
    if lm_cfg.get("limit") is not None:
        command += ["--limit", str(lm_cfg["limit"])]
    if lm_cfg.get("log_samples", False):
        command.append("--log_samples")
    if lm_cfg.get("confirm_run_unsafe_code", False):
        command.append("--confirm_run_unsafe_code")
    return command, output_path


def list_tasks(args: argparse.Namespace) -> int:
    import_check()
    if command_exists("lm-eval"):
        command = ["lm-eval", "ls", "tasks"]
    elif command_exists("lm_eval"):
        command = ["lm_eval", "ls", "tasks"]
    else:
        command = [sys.executable, "-m", "lm_eval", "ls", "tasks"]
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    return subprocess.run(command).returncode


def write_command_record(output_path: Path, spec: dict[str, Any], command: list[str]) -> None:
    record = {
        "model": spec["name"],
        "command": command,
        "command_pretty": " ".join(shlex.quote(part) for part in command),
    }
    with (output_path / "echo_lm_eval_command.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        raise SystemExit(list_tasks(args))
    if not args.dry_run:
        import_check()
    config = load_yaml(resolve_path(args.config))
    for spec in selected_models(config, args.model_name):
        command, output_path = build_eval_command(spec, config, args)
        write_command_record(output_path, spec, command)
        printable = " ".join(shlex.quote(part) for part in command)
        print(f"[lm_eval] {spec['name']}: {printable}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
