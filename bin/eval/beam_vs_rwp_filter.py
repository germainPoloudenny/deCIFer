#!/usr/bin/env python3
"""Compare standard beam search against beam search with an RWP-based filter."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict

import pandas as pd

from bin.eval.sampling_vs_beam_search import (
    _collect_metrics,
    _ensure_directory,
    _run_command,
)
from bin.eval.plot.beam_vs_rwp_filter import write_beam_vs_rwp_plots


DEFAULT_BEAM_SIZE = 10
DEFAULT_NUM_REPS = 10
DEFAULT_COLLECT_TOP_K = 10


class VariantConfig(TypedDict):
    variant: str
    dataset_suffix: str
    description: str
    num_reps: Optional[str]
    collect_args: Sequence[str]
    collect_top_k: Optional[int]


Record = Dict[str, object]


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


def _prepare_summary_record(
    args: argparse.Namespace,
    *,
    beam_size: int,
    variant: VariantConfig,
    metrics: Dict[str, object],
    eval_folder: Path,
    pickle_path: Path,
) -> Record:
    num_reps_value: Optional[int]
    if variant["num_reps"] is None:
        num_reps_value = 1
    else:
        num_reps_value = int(variant["num_reps"])

    record: Record = {
        "beam_size": beam_size,
        "variant": variant["variant"],
        "description": variant["description"],
        "num_reps": num_reps_value,
        "collect_top_k": variant["collect_top_k"],
        "evaluate_out_folder": str(eval_folder.parent),
        "eval_files_folder": str(eval_folder),
        "collect_pickle_path": str(pickle_path),
        "max_samples": args.max_samples,
        "nproc_per_node": args.nproc_per_node,
    }
    record.update(metrics)
    return record


def _build_variants(args: argparse.Namespace) -> List[VariantConfig]:
    dataset_prefix = args.dataset_name_prefix
    return [
        {
            "variant": "beam_top1",
            "dataset_suffix": f"beam{args.beam_size}_top1",
            "description": "Beam search (top-1)",
            "num_reps": None,
            "collect_args": [],
            "collect_top_k": None,
        },
        {
            "variant": "beam_rwp",
            "dataset_suffix": f"beam{args.beam_size}_rwp_top{args.collect_top_k}",
            "description": (
                "Beam search + RWP filter "
                f"(num_reps={args.num_reps}, top-k={args.collect_top_k})"
            ),
            "num_reps": str(args.num_reps),
            "collect_args": ["--top-k", str(args.collect_top_k)],
            "collect_top_k": int(args.collect_top_k),
        },
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare beam search against beam search filtered by RWP scores.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help="Beam size passed to evaluate.py (default: 10).",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=DEFAULT_NUM_REPS,
        help="num_reps value for the RWP-filtered run (default: 10).",
    )
    parser.add_argument(
        "--collect-top-k",
        type=int,
        default=DEFAULT_COLLECT_TOP_K,
        help="--top-k value forwarded to collect_evaluations.py for the RWP-filtered run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        required=True,
        help="Number of samples passed to evaluate.py via --max-samples.",
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
        "--dataset-name-prefix",
        type=str,
        default="beam_vs_rwp_max_samples",
        help="Prefix used for the --dataset-name argument.",
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
        help="Extra arguments appended to the evaluate.py invocation.",
    )
    parser.add_argument(
        "--beam-deterministic",
        action="store_true",
        help="Forward --beam-deterministic to evaluate.py for beam runs.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Override --length-penalty for the runs.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Where to write the aggregated CSV summary (default: <out-root>/beam_vs_rwp_summary.csv).",
    )
    parser.add_argument(
        "--summary-json-path",
        type=Path,
        default=None,
        help="Where to write the aggregated JSON summary (default: <out-root>/beam_vs_rwp_summary.json).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional directory where comparison plots will be written (default: <out-root>/plots).",
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
    variants = _build_variants(args)

    summary_path = args.summary_path or (args.out_root / "beam_vs_rwp_summary.csv")
    summary_json_path = (
        args.summary_json_path or (args.out_root / "beam_vs_rwp_summary.json")
    )

    _ensure_directory(args.out_root)

    torchrun_base: List[str] = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)]
    if args.torchrun_extra_args:
        torchrun_base.extend(shlex.split(args.torchrun_extra_args))

    records: List[Record] = []

    for variant in variants:
        dataset_name = f"{args.dataset_name_prefix}_{variant['dataset_suffix']}"
        out_folder = args.out_root / variant["variant"]
        eval_files_dir = out_folder / "eval_files" / dataset_name
        collect_output_dir = out_folder / "collected"
        pickle_path = collect_output_dir / f"{dataset_name}.pkl.gz"

        _ensure_directory(out_folder)
        _ensure_directory(collect_output_dir)

        if args.skip_existing and pickle_path.exists():
            print(
                "⚠️  Skipping variant=%s (existing %s)."
                % (variant["variant"], pickle_path)
            )
        else:
            evaluate_cmd: List[str] = list(torchrun_base)
            _add_common_evaluate_args(
                evaluate_cmd,
                args,
                dataset_name=dataset_name,
                out_folder=out_folder,
                beam_size=args.beam_size,
                num_reps=variant["num_reps"],
            )
            _run_command(evaluate_cmd)

            collect_cmd: List[str] = [
                sys.executable,
                "bin/eval/collect_evaluations.py",
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
        record = _prepare_summary_record(
            args,
            beam_size=args.beam_size,
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
    frame = frame.sort_values(by=["variant"]).reset_index(drop=True)
    frame.to_csv(summary_path, index=False)
    print(f"✅ Summary CSV written to {summary_path}")

    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(frame.to_dict(orient="records"), fp, indent=2)
    print(f"✅ Summary JSON written to {summary_json_path}")

    plots_dir = args.plots_dir or (args.out_root / "plots")
    write_beam_vs_rwp_plots(frame, output_dir=plots_dir)


if __name__ == "__main__":
    main()
