#!/usr/bin/env python3
"""Launch evaluation runs covering conditioning/decoding configurations."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_SCRIPT = SCRIPT_DIR / "evaluate.py"
COLLECT_SCRIPT = SCRIPT_DIR / "collect_evaluations.py"

Record = Dict[str, object]

MAX_SAMPLE_GRID: Sequence[Optional[int]] = (
    1000,
    10_000,
    100_000,
    1_000_000,
    None,
)


def _format_max_samples(max_samples: Optional[int]) -> str:
    return "all" if max_samples is None else f"{max_samples}"


@dataclass(frozen=True)
class ConditionVariant:
    """Configuration describing how prompts are conditioned."""

    key: str
    description: str
    evaluate_args: Sequence[str]


@dataclass(frozen=True)
class DecodingVariant:
    """Configuration describing decoding hyper-parameters."""

    key: str
    description: str
    beam_size: int
    num_reps: Optional[int]
    beam_deterministic: bool
    evaluate_args: Sequence[str]
    collect_args: Sequence[str]


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


def _build_condition_variants() -> List[ConditionVariant]:
    return [
        ConditionVariant("none", "No conditioning", ()),
        ConditionVariant("comp", "Condition on composition", ("--add-composition",)),
        ConditionVariant(
            "comp_sg",
            "Condition on composition and space group",
            ("--add-composition", "--add-spacegroup"),
        ),
    ]


def _build_decoding_variants(
    beam_size: int,
    *,
    sampling_top_k: int,
    collect_top_k: Optional[int],
) -> List[DecodingVariant]:
    return [
        DecodingVariant(
            key="k_sampling",
            description="Top-k sampling with beam size 1",
            beam_size=1,
            num_reps=1,
            beam_deterministic=False,
            evaluate_args=("--temperature", "1.0", "--top-k", str(sampling_top_k)),
            collect_args=(),
        ),
        DecodingVariant(
            key="beam_search",
            description="Deterministic beam search",
            beam_size=beam_size,
            num_reps=1,
            beam_deterministic=True,
            evaluate_args=(),
            collect_args=(),
        ),
        DecodingVariant(
            key="beam_search_num_reps",
            description="Deterministic beam search with multiple repetitions",
            beam_size=beam_size,
            num_reps=beam_size,
            beam_deterministic=True,
            evaluate_args=(),
            collect_args=("--top-k", str(collect_top_k)) if collect_top_k is not None else (),
        ),
    ]


def _prepare_record(
    *,
    args: argparse.Namespace,
    condition: ConditionVariant,
    decoding: DecodingVariant,
    metrics: Dict[str, object],
    dataset_name: str,
    eval_folder: Path,
    pickle_path: Path,
    max_samples: Optional[int],
) -> Record:
    record: Record = {
        "conditioning": condition.key,
        "conditioning_description": condition.description,
        "decoding": decoding.key,
        "decoding_description": decoding.description,
        "dataset_name": dataset_name,
        "evaluate_out_folder": str(eval_folder.parent),
        "eval_files_folder": str(eval_folder),
        "collect_pickle_path": str(pickle_path),
        "max_samples": max_samples,
        "beam_size_used": decoding.beam_size,
        "num_reps": decoding.num_reps,
        "length_penalty": args.length_penalty,
        "nproc_per_node": args.nproc_per_node,
    }
    record.update(metrics)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a grid of conditioning and decoding configurations."
    )
    parser.add_argument("--model-ckpt", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Dataset HDF5 path.")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root directory where run artefacts will be written.",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        type=str,
        default="conditioning_decoding",
        help="Prefix used for the --dataset-name argument.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam size used by beam-search configurations.",
    )
    parser.add_argument(
        "--sampling-top-k",
        type=int,
        default=50,
        help="Top-k value used for the sampling configuration.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty forwarded to evaluate.py.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
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
        help="Extra arguments appended to evaluate.py invocations.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the aggregated CSV summary (default: <out-root>/conditioning_decoding_summary.csv).",
    )
    parser.add_argument(
        "--summary-json-path",
        type=Path,
        default=None,
        help="Where to write the aggregated JSON summary (default: <out-root>/conditioning_decoding_summary.json).",
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
    if args.beam_size < 1:
        raise ValueError("--beam-size must be at least 1.")

    summary_path = args.summary_path or (args.out_root / "conditioning_decoding_summary.csv")
    summary_json_path = args.summary_json_path or (
        args.out_root / "conditioning_decoding_summary.json"
    )

    _ensure_directory(args.out_root)

    torchrun_base = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)]
    if args.torchrun_extra_args:
        torchrun_base.extend(shlex.split(args.torchrun_extra_args))

    records: List[Record] = []
    condition_variants = _build_condition_variants()

    for max_samples in MAX_SAMPLE_GRID:
        decoding_variants = _build_decoding_variants(
            args.beam_size,
            sampling_top_k=args.sampling_top_k,
            collect_top_k=max_samples,
        )
        max_samples_label = _format_max_samples(max_samples)

        for condition in condition_variants:
            for decoding in decoding_variants:
                dataset_name = (
                    f"{args.dataset_name_prefix}_max_{max_samples_label}_{condition.key}_{decoding.key}"
                )
                out_folder = args.out_root / f"max_{max_samples_label}" / condition.key / decoding.key
                eval_files_dir = out_folder / "eval_files" / dataset_name
                collect_output_dir = out_folder / "collected"
                pickle_path = collect_output_dir / f"{dataset_name}.pkl.gz"

                _ensure_directory(out_folder)
                _ensure_directory(collect_output_dir)

                if args.skip_existing and pickle_path.exists():
                    print(
                        "⚠️  Skipping max_samples=%s condition=%s decoding=%s (existing %s)."
                        % (max_samples_label, condition.key, decoding.key, pickle_path)
                    )
                else:
                    evaluate_cmd: List[str] = list(torchrun_base)
                    evaluate_cmd.extend(
                        [
                            str(EVALUATE_SCRIPT),
                            "--model-ckpt",
                            str(args.model_ckpt),
                            "--dataset-path",
                            str(args.dataset_path),
                            "--out-folder",
                            str(out_folder),
                            "--dataset-name",
                            dataset_name,
                            "--beam-size",
                            str(decoding.beam_size),
                            "--length-penalty",
                            str(args.length_penalty),
                        ]
                    )

                    if max_samples is not None:
                        evaluate_cmd.extend(["--max-samples", str(max_samples)])

                    if decoding.num_reps is not None:
                        evaluate_cmd.extend(["--num-reps", str(decoding.num_reps)])

                    if decoding.beam_deterministic:
                        evaluate_cmd.append("--beam-deterministic")

                    evaluate_cmd.extend(condition.evaluate_args)
                    evaluate_cmd.extend(decoding.evaluate_args)

                    if args.evaluate_extra_args:
                        evaluate_cmd.extend(shlex.split(args.evaluate_extra_args))

                    _run_command(evaluate_cmd)

                    collect_cmd: List[str] = [
                        sys.executable,
                        str(COLLECT_SCRIPT),
                        "--eval-folder-paths",
                        str(eval_files_dir),
                        "--output-folder",
                        str(collect_output_dir),
                    ]
                    collect_cmd.extend(decoding.collect_args)
                    _run_command(collect_cmd)

                if not pickle_path.exists():
                    raise FileNotFoundError(
                        f"Expected collected results at {pickle_path}, but the file does not exist."
                    )

                metrics = _collect_metrics(pickle_path, args.rmsd_threshold, args.rwp_threshold)
                record = _prepare_record(
                    args=args,
                    condition=condition,
                    decoding=decoding,
                    metrics=metrics,
                    dataset_name=dataset_name,
                    eval_folder=eval_files_dir,
                    pickle_path=pickle_path,
                    max_samples=max_samples,
                )
                records.append(record)

    if not records:
        print("No records collected; nothing to write.")
        return

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(by=["max_samples", "conditioning", "decoding"], na_position="last").reset_index(drop=True)
    frame.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV written to {summary_path}")

    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(frame.to_dict(orient="records"), fp, indent=2)
    print(f"✅ Summary JSON written to {summary_json_path}")


if __name__ == "__main__":
    main()
