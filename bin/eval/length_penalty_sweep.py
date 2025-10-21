#!/usr/bin/env python3
"""Sweep length penalties for beam search decoding without conditioning."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


Record = Dict[str, object]

DEFAULT_LENGTH_PENALTIES: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.4)


def _run_command(command: Sequence[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess command while echoing it to stdout."""

    printable = " ".join(shlex.quote(part) for part in command)
    print(f"→ Running: {printable}")
    subprocess.run(command, check=True, env=env)


def _ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def _compute_validity_rate(frame: pd.DataFrame) -> Dict[str, object]:
    if "validity" not in frame.columns:
        return {"validity_rate": float("nan"), "valid_count": 0, "valid_total": 0}

    validity = frame["validity"].dropna()
    validity = pd.to_numeric(validity, errors="coerce").dropna()
    if validity.empty:
        return {"validity_rate": float("nan"), "valid_count": 0, "valid_total": 0}

    valid_count = int((validity > 0).sum())
    valid_total = int(validity.count())
    return {
        "validity_rate": float(valid_count / valid_total) if valid_total else float("nan"),
        "valid_count": valid_count,
        "valid_total": valid_total,
    }


def _compute_spacegroup_match(frame: pd.DataFrame) -> Dict[str, object]:
    expected_cols = {"spacegroup_num_sample", "spacegroup_num_gen"}
    if not expected_cols.issubset(frame.columns):
        return {
            "spacegroup_match_rate": float("nan"),
            "spacegroup_match_count": 0,
            "spacegroup_match_total": 0,
        }

    sample = pd.to_numeric(frame["spacegroup_num_sample"], errors="coerce")
    generated = pd.to_numeric(frame["spacegroup_num_gen"], errors="coerce")
    mask = sample.notna() & generated.notna()
    if not mask.any():
        return {
            "spacegroup_match_rate": float("nan"),
            "spacegroup_match_count": 0,
            "spacegroup_match_total": 0,
        }

    matches = sample[mask] == generated[mask]
    match_count = int(matches.sum())
    match_total = int(matches.count())
    rate = float(match_count / match_total) if match_total else float("nan")
    return {
        "spacegroup_match_rate": rate,
        "spacegroup_match_count": match_count,
        "spacegroup_match_total": match_total,
    }


def _count_with_threshold(
    series: pd.Series,
    threshold: Optional[float],
    *,
    lower_is_better: bool = True,
) -> Dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {
            "count": 0,
            "total": 0,
            "rate": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }

    total = int(numeric.count())
    mean = float(numeric.mean())
    median = float(numeric.median())
    std = float(numeric.std(ddof=0))

    if threshold is None:
        count = total
        rate = float(1.0)
    else:
        matches = numeric <= threshold if lower_is_better else numeric >= threshold
        count = int(matches.sum())
        rate = float(count / total) if total else float("nan")

    return {
        "count": count,
        "total": total,
        "rate": rate,
        "mean": mean,
        "median": median,
        "std": std,
    }


def _collect_metrics(
    pickle_path: Path,
    rmsd_threshold: Optional[float],
    rwp_threshold: Optional[float],
) -> Dict[str, object]:
    frame = pd.read_pickle(pickle_path)
    metrics: Dict[str, object] = {"num_rows": int(len(frame))}
    metrics.update(_compute_validity_rate(frame))
    metrics.update(_compute_spacegroup_match(frame))

    if "rmsd" in frame.columns:
        rmsd_stats = _count_with_threshold(frame["rmsd"], rmsd_threshold, lower_is_better=True)
        num_rows = metrics["num_rows"]
        if num_rows:
            coverage_rate = float(rmsd_stats["total"]) / float(num_rows)
        else:
            coverage_rate = float("nan")
        metrics.update(
            {
                "rmsd_match_count": rmsd_stats["count"],
                "rmsd_match_total": rmsd_stats["total"],
                "rmsd_match_rate": coverage_rate,
                "rmsd_mean": rmsd_stats["mean"],
                "rmsd_median": rmsd_stats["median"],
                "rmsd_std": rmsd_stats["std"],
                "rmsd_threshold": rmsd_threshold,
            }
        )

    if "rwp" in frame.columns:
        rwp_stats = _count_with_threshold(frame["rwp"], rwp_threshold, lower_is_better=True)
        metrics.update(
            {
                "rwp_match_count": rwp_stats["count"],
                "rwp_match_total": rwp_stats["total"],
                "rwp_match_rate": rwp_stats["rate"],
                "rwp_mean": rwp_stats["mean"],
                "rwp_median": rwp_stats["median"],
                "rwp_std": rwp_stats["std"],
                "rwp_threshold": rwp_threshold,
            }
        )

    return metrics


def _format_length_penalty_for_name(value: float) -> str:
    text = f"{value:.2f}" if not float(value).is_integer() else f"{int(value)}"
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return text.replace("-", "neg").replace(".", "p")


def _prepare_summary_record(
    *,
    length_penalty: float,
    args: argparse.Namespace,
    metrics: Dict[str, object],
    eval_folder: Path,
    pickle_path: Path,
    dataset_name: str,
) -> Record:
    record: Record = {
        "length_penalty": float(length_penalty),
        "dataset_name": dataset_name,
        "beam_size": args.beam_size,
        "max_samples": args.max_samples,
        "nproc_per_node": args.nproc_per_node,
        "evaluate_out_folder": str(eval_folder.parent),
        "eval_files_folder": str(eval_folder),
        "collect_pickle_path": str(pickle_path),
    }
    record.update(metrics)
    return record


def _parse_length_penalties(values: Iterable[float]) -> List[float]:
    penalties = [float(value) for value in values]
    if not penalties:
        raise ValueError("At least one --length-penalties value must be provided.")
    return penalties


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bin/evaluate.py for a sweep of length penalty values using a fixed beam size."
        )
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        required=True,
        help="Path to the checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset HDF5 file.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root directory where run artefacts will be written.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam size passed to evaluate.py (default: 10).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of samples passed to evaluate.py via --max-samples (default: 100).",
    )
    parser.add_argument(
        "--length-penalties",
        type=float,
        nargs="+",
        default=list(DEFAULT_LENGTH_PENALTIES),
        help=(
            "Sequence of length penalty values to evaluate. "
            "Defaults to a conventional range used for figure preparation (0.6 0.8 1.0 1.2 1.4)."
        ),
    )
    parser.add_argument(
        "--dataset-name-prefix",
        type=str,
        default="length_penalty",
        help="Prefix used for the --dataset-name argument.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Passed to torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--torchrun-extra-args",
        type=str,
        default="",
        help="Additional arguments appended after torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--evaluate-extra-args",
        type=str,
        default="",
        help="Extra arguments appended to the evaluate.py invocation.",
    )
    parser.add_argument(
        "--beam-deterministic",
        action="store_true",
        help="Forward --beam-deterministic to evaluate.py.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the aggregated CSV summary (default: <out-root>/length_penalty_summary.csv).",
    )
    parser.add_argument(
        "--summary-json-path",
        type=Path,
        default=None,
        help="Where to write the aggregated JSON summary (default: <out-root>/length_penalty_summary.json).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing collected pickle files and skip the corresponding commands.",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        help="Optional RMSD threshold used to count structure matches.",
    )
    parser.add_argument(
        "--rwp-threshold",
        type=float,
        default=None,
        help="Optional RWP threshold used to count diffractogram matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    length_penalties = _parse_length_penalties(args.length_penalties)

    summary_path = args.summary_path or (args.out_root / "length_penalty_summary.csv")
    summary_json_path = args.summary_json_path or (args.out_root / "length_penalty_summary.json")

    _ensure_directory(args.out_root)

    torchrun_base: List[str] = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)]
    if args.torchrun_extra_args:
        torchrun_base.extend(shlex.split(args.torchrun_extra_args))

    records: List[Record] = []

    for length_penalty in length_penalties:
        length_penalty_str = _format_length_penalty_for_name(length_penalty)
        dataset_name = f"{args.dataset_name_prefix}_lp_{length_penalty_str}"
        out_folder = args.out_root / f"length_penalty_{length_penalty_str}"
        eval_files_dir = out_folder / "eval_files" / dataset_name
        collect_output_dir = out_folder / "collected"
        pickle_path = collect_output_dir / f"{dataset_name}.pkl.gz"

        _ensure_directory(out_folder)
        _ensure_directory(collect_output_dir)

        if args.skip_existing and pickle_path.exists():
            print(
                "⚠️  Skipping length_penalty=%s (existing %s)."
                % (length_penalty, pickle_path)
            )
        else:
            evaluate_cmd: List[str] = list(torchrun_base)
            evaluate_cmd.extend(
                [
                    "bin/eval/evaluate.py",
                    "--model-ckpt",
                    str(args.model_ckpt),
                    "--dataset-path",
                    str(args.dataset_path),
                    "--out-folder",
                    str(out_folder),
                    "--dataset-name",
                    dataset_name,
                    "--beam-size",
                    str(args.beam_size),
                    "--length-penalty",
                    str(length_penalty),
                    "--max-samples",
                    str(args.max_samples),
                ]
            )

            if args.beam_deterministic:
                evaluate_cmd.append("--beam-deterministic")

            if args.evaluate_extra_args:
                evaluate_cmd.extend(shlex.split(args.evaluate_extra_args))

            _run_command(evaluate_cmd)

            collect_cmd = [
                sys.executable,
                "bin/eval/collect_evaluations.py",
                "--eval-folder-paths",
                str(eval_files_dir),
                "--output-folder",
                str(collect_output_dir),
            ]
            _run_command(collect_cmd)

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Expected collected results at {pickle_path}, but the file does not exist."
            )

        metrics = _collect_metrics(pickle_path, args.rmsd_threshold, args.rwp_threshold)
        record = _prepare_summary_record(
            length_penalty=length_penalty,
            args=args,
            metrics=metrics,
            eval_folder=eval_files_dir,
            pickle_path=pickle_path,
            dataset_name=dataset_name,
        )
        records.append(record)

    if not records:
        print("No records collected; nothing to write.")
        return

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(by="length_penalty").reset_index(drop=True)
    frame.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV written to {summary_path}")

    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(frame.to_dict(orient="records"), fp, indent=2)
    print(f"✅ Summary JSON written to {summary_json_path}")


if __name__ == "__main__":
    main()

