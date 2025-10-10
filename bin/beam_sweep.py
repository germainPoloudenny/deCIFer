#!/usr/bin/env python3
"""
Utility to automate evaluation runs across multiple beam sizes and collect metrics.

The script launches `bin/evaluate.py` twice per beam size: once with the default
`--num-reps` (top-1 generations) and once with `--num-reps` equal to the beam size.
After each evaluation it executes `bin/collect_evaluations.py` to aggregate the
sample-level results and stores compact metrics that can be plotted later.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


Variant = Dict[str, str]
Record = Dict[str, object]


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
    # Convert to numeric to guard against "True"/"False" strings.
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

    matches = (sample[mask] == generated[mask])
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
        # When no threshold is provided, treat every finite value as a match.
        count = total
        rate = float(1.0)
    else:
        if lower_is_better:
            matches = numeric <= threshold
        else:
            matches = numeric >= threshold
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
        metrics.update(
            {
                "rmsd_match_count": rmsd_stats["count"],
                "rmsd_match_total": rmsd_stats["total"],
                "rmsd_match_rate": rmsd_stats["rate"],
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


def _build_variants(beam_size: int) -> List[Variant]:
    """Describe the two runs (top-1 and top-𝑘) carried out per beam size."""
    return [
        {
            "variant": "top1",
            "dataset_suffix": f"beam{beam_size}_top1",
            "description": "num_reps=1 (top-1 generation)",
            "num_reps": None,
            "collect_args": [],
        },
        {
            "variant": "topk",
            "dataset_suffix": f"beam{beam_size}_topk",
            "description": "num_reps equals beam size",
            "num_reps": str(beam_size),
            "collect_args": ["--top-k", str(beam_size)],
        },
    ]


def _add_common_evaluate_args(
    command: List[str],
    args: argparse.Namespace,
    *,
    dataset_name: str,
    out_folder: Path,
    beam_size: int,
    num_reps: Optional[str],
) -> None:
    command.extend(
        [
            "bin/evaluate.py",
            "--model-ckpt",
            str(args.model_ckpt),
            "--dataset-path",
            str(args.dataset_path),
            "--out-folder",
            str(out_folder),
            "--dataset-name",
            dataset_name,
            "--beam-size",
            str(beam_size),
            "--max-samples",
            str(args.max_samples),
        ]
    )

    if num_reps is not None:
        command.extend(["--num-reps", num_reps])

    if args.length_penalty is not None:
        command.extend(["--length-penalty", str(args.length_penalty)])

    if args.beam_deterministic:
        command.append("--beam-deterministic")

    if args.evaluate_extra_args:
        command.extend(shlex.split(args.evaluate_extra_args))


def _prepare_summary_records(
    args: argparse.Namespace,
    *,
    beam_size: int,
    variant: Variant,
    metrics: Dict[str, object],
    eval_folder: Path,
    pickle_path: Path,
) -> Record:
    record: Record = {
        "beam_size": beam_size,
        "variant": variant["variant"],
        "description": variant["description"],
        "evaluate_out_folder": str(eval_folder.parent),
        "eval_files_folder": str(eval_folder),
        "collect_pickle_path": str(pickle_path),
        "max_samples": args.max_samples,
        "nproc_per_node": args.nproc_per_node,
    }
    record.update(metrics)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep beam sizes and collect evaluation metrics.")
    parser.add_argument(
        "--max-beam-size",
        type=int,
        required=True,
        help="Largest beam size to evaluate; the sweep runs from 1 to this value.",
    )
    parser.add_argument("--max-samples", type=int, required=True, help="Number of samples passed to evaluate.py via --max-samples.")
    parser.add_argument("--model-ckpt", type=Path, required=True, help="Path to the checkpoint used for evaluation.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to the dataset HDF5 file.")
    parser.add_argument("--out-root", type=Path, required=True, help="Root directory where run artefacts will be written.")
    parser.add_argument("--dataset-name-prefix", type=str, default="beam_sweep", help="Prefix used for the --dataset-name argument.")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Passed to torchrun --nproc_per_node.")
    parser.add_argument("--torchrun-extra-args", type=str, default="", help="Additional arguments appended after torchrun --nproc_per_node.")
    parser.add_argument("--evaluate-extra-args", type=str, default="", help="Extra arguments appended to the evaluate.py invocation.")
    parser.add_argument("--beam-deterministic", action="store_true", help="Forward --beam-deterministic to evaluate.py.")
    parser.add_argument("--length-penalty", type=float, default=None, help="Override --length-penalty for evaluate.py.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Where to write the aggregated CSV summary (default: <out-root>/beam_sweep_summary.csv).")
    parser.add_argument("--summary-json-path", type=Path, default=None, help="Where to write the aggregated JSON summary (default: <out-root>/beam_sweep_summary.json).")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing collected pickle files and skip the corresponding commands.")
    parser.add_argument("--rmsd-threshold", type=float, default=None, help="Optional RMSD threshold used to count structure matches.")
    parser.add_argument("--rwp-threshold", type=float, default=None, help="Optional RWP threshold used to count diffractogram matches.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.max_beam_size < 1:
        raise ValueError("--max-beam-size must be at least 1.")
    beam_sizes = list(range(1, args.max_beam_size + 1))

    summary_path = args.summary_path or (args.out_root / "beam_sweep_summary.csv")
    summary_json_path = args.summary_json_path or (args.out_root / "beam_sweep_summary.json")

    _ensure_directory(args.out_root)

    torchrun_base = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)]
    if args.torchrun_extra_args:
        torchrun_base.extend(shlex.split(args.torchrun_extra_args))

    records: List[Record] = []

    for beam_size in beam_sizes:
        variants = _build_variants(beam_size)
        for variant in variants:
            dataset_name = f"{args.dataset_name_prefix}_{variant['dataset_suffix']}"
            out_folder = args.out_root / f"beam_{beam_size}" / variant["variant"]
            eval_files_dir = out_folder / "eval_files" / dataset_name
            collect_output_dir = out_folder / "collected"
            pickle_path = collect_output_dir / f"{dataset_name}.pkl.gz"

            _ensure_directory(out_folder)
            _ensure_directory(collect_output_dir)

            if args.skip_existing and pickle_path.exists():
                print(f"⚠️  Skipping beam_size={beam_size} variant={variant['variant']} (existing {pickle_path}).")
            else:
                # Launch evaluation.
                evaluate_cmd: List[str] = list(torchrun_base)
                _add_common_evaluate_args(
                    evaluate_cmd,
                    args,
                    dataset_name=dataset_name,
                    out_folder=out_folder,
                    beam_size=beam_size,
                    num_reps=variant["num_reps"],
                )
                _run_command(evaluate_cmd)

                # Aggregate the outputs.
                collect_cmd = [
                    sys.executable,
                    "bin/collect_evaluations.py",
                    "--eval-folder-paths",
                    str(eval_files_dir),
                    "--output-folder",
                    str(collect_output_dir),
                ]
                collect_cmd.extend(variant["collect_args"])
                _run_command(collect_cmd)

            if not pickle_path.exists():
                raise FileNotFoundError(
                    f"Expected collected results at {pickle_path}, but the file does not exist."
                )

            metrics = _collect_metrics(pickle_path, args.rmsd_threshold, args.rwp_threshold)
            record = _prepare_summary_records(
                args,
                beam_size=beam_size,
                variant=variant,
                metrics=metrics,
                eval_folder=eval_files_dir,
                pickle_path=pickle_path,
            )
            records.append(record)

    if not records:
        print("No records collected; nothing to write.")
        return

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(by=["beam_size", "variant"]).reset_index(drop=True)
    frame.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV written to {summary_path}")

    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(frame.to_dict(orient="records"), fp, indent=2)
    print(f"✅ Summary JSON written to {summary_json_path}")


if __name__ == "__main__":
    main()
