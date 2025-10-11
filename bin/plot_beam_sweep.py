#!/usr/bin/env python3
"""
Generate per-metric plots from the artefacts produced by bin/beam_sweep.py.

For each metric collected from the `collected/*.pkl.gz` files, the script
creates two line plots (beam search normal vs beam search filtered by RWP)
with the beam size on the x-axis.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from beam_sweep import _collect_metrics  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "Unable to import beam_sweep._collect_metrics. "
        "Run this script from the repository root with `python bin/plot_beam_sweep.py`."
    ) from exc


@dataclass(frozen=True)
class VariantConfig:
    """Metadata for a beam sweep variant."""

    directory: str
    label: str
    description: str


VARIANTS: Tuple[VariantConfig, ...] = (
    VariantConfig(directory="top1", label="beam_search_normal", description="Beam search (top-1)."),
    VariantConfig(
        directory="topk",
        label="beam_search_rwp_filtered",
        description="Beam search filtered by RWP (top-k from collect_evaluations).",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create beam sweep metric plots from collected evaluation artefacts."
    )
    parser.add_argument(
        "--beam-study-root",
        type=Path,
        default=Path("runs/deCIFer_cifs_v1_model/beam_study"),
        help="Root directory containing the beam_* folders output by bin/beam_sweep.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the plots and summary CSV will be written (default: <beam-study-root>/plots).",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        help="Optional RMSD threshold passed to the metric collector.",
    )
    parser.add_argument(
        "--rwp-threshold",
        type=float,
        default=None,
        help="Optional RWP threshold passed to the metric collector.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Subset of metric names to plot; defaults to "
            "validity_rate, rmsd_match_rate_overall, spacegroup_match_rate, rwp_mean."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them to disk.",
    )
    return parser.parse_args()


def _extract_beam_size(name: str) -> Optional[int]:
    try:
        return int(name.split("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def _expected_pickle_path(beam_dir: Path, variant: VariantConfig) -> Optional[Path]:
    collected_dir = beam_dir / variant.directory / "collected"
    if not collected_dir.exists():
        return None

    beam_size = _extract_beam_size(beam_dir.name)
    if beam_size is not None:
        expected_name = f"beam_sweep_beam{beam_size}_{variant.directory}.pkl.gz"
        expected_path = collected_dir / expected_name
        if expected_path.exists():
            return expected_path

    candidates = sorted(collected_dir.glob("*.pkl.gz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise RuntimeError(
        f"Multiple pickle files found in {collected_dir} but none match the expected naming scheme."
    )


def gather_metrics(
    root: Path,
    *,
    rmsd_threshold: Optional[float],
    rwp_threshold: Optional[float],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for beam_dir in sorted(root.glob("beam_*")):
        beam_size = _extract_beam_size(beam_dir.name)
        if beam_size is None:
            continue

        for variant in VARIANTS:
            pickle_path = _expected_pickle_path(beam_dir, variant)
            if pickle_path is None:
                continue
            metrics = _collect_metrics(pickle_path, rmsd_threshold, rwp_threshold)
            record: Dict[str, object] = {"beam_size": beam_size, "variant": variant.label}
            record.update(metrics)

            match_count = metrics.get("rmsd_match_count")
            total_rows = metrics.get("num_rows")
            if (
                isinstance(match_count, numbers.Real)
                and isinstance(total_rows, numbers.Real)
                and total_rows
            ):
                record["rmsd_match_rate_overall"] = (
                    float(match_count) / float(total_rows)
                ) * 100.0
            else:
                record["rmsd_match_rate_overall"] = float("nan")

            records.append(record)
    if not records:
        raise FileNotFoundError(
            f"No collected metrics found under {root}. "
            "Ensure bin/beam_sweep.py has been executed successfully."
        )
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["variant", "beam_size"]).reset_index(drop=True)
    return frame


def _slugify_metric(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _select_metrics(frame: pd.DataFrame, requested: Optional[Iterable[str]]) -> List[str]:
    default_metrics = [
        "validity_rate",
        "rmsd_match_rate_overall",
        "spacegroup_match_rate",
        "rwp_mean",
    ]

    available_numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    available_numeric = [col for col in available_numeric if col != "beam_size"]

    if requested:
        missing = sorted(set(requested) - set(frame.columns))
        if missing:
            raise ValueError(f"Unknown metric(s): {', '.join(missing)}.")
        return [metric for metric in requested if metric in available_numeric]

    return [metric for metric in default_metrics if metric in available_numeric]


def _plot_metric(
    frame: pd.DataFrame,
    metric: str,
    output_dir: Path,
    *,
    show: bool,
) -> None:
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant.label]
        if subset.empty or metric not in subset.columns:
            continue

        x = subset["beam_size"]
        y = subset[metric]

        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Beam size")

        display_name = metric.replace("_", " ")
        if metric == "rmsd_match_rate_overall":
            display_name += " (%)"

        ax.set_ylabel(display_name.title())
        ax.set_title(f"{display_name.title()} – {variant.description}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        metric_slug = _slugify_metric(metric)
        file_name = f"{metric_slug}_{variant.label}.png"
        fig.savefig(output_dir / file_name, bbox_inches="tight", dpi=200)

        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    args = parse_args()
    beam_root: Path = args.beam_study_root
    if not beam_root.exists():
        raise FileNotFoundError(f"Beam study root {beam_root} does not exist.")

    output_dir = args.output_dir or (beam_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = gather_metrics(
        beam_root,
        rmsd_threshold=args.rmsd_threshold,
        rwp_threshold=args.rwp_threshold,
    )

    summary_path = output_dir / "beam_sweep_metrics.csv"
    frame.to_csv(summary_path, index=False)

    metrics_to_plot = _select_metrics(frame, args.metrics)
    if not metrics_to_plot:
        raise ValueError("No numeric metrics available to plot.")

    for metric in metrics_to_plot:
        _plot_metric(frame, metric, output_dir, show=args.show)

    summary_json_path = output_dir / "beam_sweep_metrics.json"
    summary_json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
