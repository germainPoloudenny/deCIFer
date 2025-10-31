#!/usr/bin/env python3

import argparse
import copy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


DEFAULT_WEIGHTS: List[float] = [1, 2, 5, 10, 20, 50]


def _format_weight_for_name(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_path(config: dict, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    dataset_root = config.get("dataset")
    if not dataset_root:
        raise ValueError("The base config must define a 'dataset' entry or pass --dataset-path.")
    return Path(dataset_root) / "serialized" / "test.h5"


def _prepare_run_config(base_config: dict, *, run_dir: Path, weight: float, patience: int) -> dict:
    config = copy.deepcopy(base_config)
    config["out_dir"] = str(run_dir)
    config["tensorboard_log_dir"] = str(run_dir / "tensorboard")
    config["composition_loss_weight"] = float(weight)
    config["early_stopping_patience"] = int(patience)
    config["init_from"] = "resume"
    config["resume_from_best"] = True
    config.setdefault("always_save_checkpoint", True)
    return config


def _invoke(command: Iterable[str]) -> None:
    rendered = " ".join(command)
    print(f"⚙️  Running: {rendered}")
    subprocess.run(list(command), check=True)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {source}")
    _ensure_directory(destination.parent)
    shutil.copy2(source, destination)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep composition_loss_weight values with training + evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/decifer.yaml"), help="Base YAML config.")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=DEFAULT_WEIGHTS,
        help="Values for composition_loss_weight.",
    )
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=Path("runs/deCIFer/ckpt.pt"),
        help="Checkpoint used as the starting point for every run.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/comp"),
        help="Root directory where run folders will be created.",
    )
    parser.add_argument(
        "--train-nproc",
        type=int,
        default=2,
        help="Forwarded to torchrun --nproc_per_node for training.",
    )
    parser.add_argument(
        "--eval-nproc",
        type=int,
        default=2,
        help="Forwarded to torchrun --nproc_per_node for evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Forwarded to evaluate.py --max-samples.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override the dataset path passed to evaluate.py. Defaults to <dataset>/serialized/test.h5.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Patience forwarded to train.py via the generated config.",
    )
    parser.add_argument(
        "--train-extra-args",
        type=str,
        default="",
        help="Extra CLI arguments appended after train.py.",
    )
    parser.add_argument(
        "--evaluate-extra-args",
        type=str,
        default="",
        help="Extra CLI arguments appended after evaluate.py.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs where the output folder already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    base_config = _load_config(args.config)
    dataset_path = _resolve_dataset_path(base_config, args.dataset_path)

    if args.dry_run:
        print("Dry run enabled; commands will only be printed.")

    torch_executable = sys.executable

    for weight in args.weights:
        weight_str = _format_weight_for_name(weight)
        run_dir = args.out_root / weight_str
        eval_dir = run_dir / "eval"
        config_path = run_dir / "config.yaml"
        run_ckpt = run_dir / "ckpt.pt"
        dataset_name = f"comp_weight_{weight_str}"

        if args.skip_existing and run_dir.exists():
            print(f"⏭️  Skipping weight={weight} (existing {run_dir}).")
            continue

        print(f"\n=== Sweep value: composition_loss_weight={weight} ===")

        _ensure_directory(run_dir)

        if not args.dry_run:
            _copy_checkpoint(args.base_ckpt, run_ckpt)
        else:
            print(f"Would copy {args.base_ckpt} -> {run_ckpt}")

        run_config = _prepare_run_config(
            base_config,
            run_dir=run_dir,
            weight=weight,
            patience=args.early_stopping_patience,
        )
        _write_config(run_config, config_path)

        train_cmd: List[str] = [
            "torchrun",
            "--nproc_per_node",
            str(args.train_nproc),
            "bin/train.py",
            "--config",
            str(config_path),
        ]
        if args.train_extra_args:
            train_cmd.extend(shlex.split(args.train_extra_args))

        if args.dry_run:
            print(" ".join(train_cmd))
        else:
            _invoke(train_cmd)

        _ensure_directory(eval_dir)
        model_ckpt = run_ckpt

        evaluate_cmd: List[str] = [
            "torchrun",
            "--nproc_per_node",
            str(args.eval_nproc),
            "bin/eval/evaluate.py",
            "--model-ckpt",
            str(model_ckpt),
            "--dataset-path",
            str(dataset_path),
            "--out-folder",
            str(eval_dir),
            "--dataset-name",
            dataset_name,
            "--max-samples",
            str(args.max_samples),
            "--distributed",
            "--condition",
            "--add-composition",
            "--add-spacegroup",
        ]
        if args.evaluate_extra_args:
            evaluate_cmd.extend(shlex.split(args.evaluate_extra_args))

        if args.dry_run:
            print(" ".join(evaluate_cmd))
        else:
            _invoke(evaluate_cmd)

        eval_files_dir = eval_dir / "eval_files" / dataset_name
        collected_output = eval_dir
        collect_cmd: List[str] = [
            torch_executable,
            "bin/eval/collect_evaluations.py",
            "--eval-folder-paths",
            str(eval_files_dir),
            "--output-folder",
            str(collected_output),
        ]

        if args.dry_run:
            print(" ".join(collect_cmd))
        else:
            _invoke(collect_cmd)

        collected_pickle = collected_output / f"{dataset_name}.pkl.gz"
        metrics_cmd: List[str] = [
            torch_executable,
            "bin/eval/show_eval_metrics.py",
            str(collected_pickle),
        ]
        metrics_output = eval_dir / "metrics.txt"

        if args.dry_run:
            print(" ".join(metrics_cmd))
            print(f"Would write metrics to {metrics_output}")
        else:
            result = subprocess.run(metrics_cmd, check=True, capture_output=True, text=True)
            metrics_output.write_text(result.stdout, encoding="utf-8")
            print(result.stdout)


if __name__ == "__main__":
    main()
