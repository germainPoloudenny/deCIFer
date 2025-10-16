#!/usr/bin/env python3
"""Utilities to generate publication-quality plots for sampling vs beam search sweeps."""

from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


__all__ = ["write_publication_quality_plots"]

# Apply a slightly larger default font suitable for print usage.
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


_AXIS_LABEL_OVERRIDES = {
    "rmsd_match_count": ("Beam size", "RMSD Match Count"),
}

_TITLE_OVERRIDES = {
    "rmsd_match_count": "Impact of Beam Size on Structural Matching",
}

_LEGEND_TITLES = {
    "rmsd_match_count": "Variant (higher is better)",
}

_COLOR_OVERRIDES = {
    "baseline": "#1f77b4",  # blue
    "top1": "#d62728",  # orange/red
}

_MARKER_OVERRIDES = {
    "baseline": "o",
    "top1": "s",
}

_DEFAULT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

_DEFAULT_MARKERS = ["o", "s", "^", "D", "P", "X"]


def _slugify_metric(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _select_numeric_metrics(frame: pd.DataFrame, ignore: Iterable[str]) -> List[str]:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    ignore_set = set(ignore)
    return [column for column in numeric_columns if column not in ignore_set]


def _format_annotation(value: float) -> str:
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _prepare_style_iterables(frame: pd.DataFrame):
    variants = sorted(frame["variant"].dropna().unique().tolist())
    available_colors = cycle(_DEFAULT_COLORS)
    available_markers = cycle(_DEFAULT_MARKERS)

    color_map = {}
    marker_map = {}
    for variant in variants:
        if variant in _COLOR_OVERRIDES:
            color_map[variant] = _COLOR_OVERRIDES[variant]
        else:
            color_map[variant] = next(available_colors)

        if variant in _MARKER_OVERRIDES:
            marker_map[variant] = _MARKER_OVERRIDES[variant]
        else:
            marker_map[variant] = next(available_markers)

    return color_map, marker_map


def write_publication_quality_plots(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: Optional[Iterable[str]] = None,
    annotate_points: bool = True,
) -> None:
    """Write publication-friendly line plots for the provided metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = (
        list(metrics)
        if metrics is not None
        else _select_numeric_metrics(frame, ignore=["beam_size"])
    )

    if not metric_names:
        print("⚠️  No numeric metrics available for plotting; skipping plot generation.")
        return

    color_map, marker_map = _prepare_style_iterables(frame)

    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        has_data = False

        for variant, subset in frame.groupby("variant"):
            subset = subset.sort_values("beam_size")
            if metric not in subset:
                continue

            y = pd.to_numeric(subset[metric], errors="coerce")
            mask = np.isfinite(y.to_numpy())
            if not mask.any():
                continue

            has_data = True
            label_series = subset["description"] if "description" in subset else pd.Series(dtype=str)
            label = (
                label_series.dropna().iloc[0]
                if not label_series.dropna().empty
                else str(variant)
            )

            color = color_map.get(variant)
            marker = marker_map.get(variant, "o")

            ax.plot(
                subset.loc[mask, "beam_size"],
                y[mask],
                marker=marker,
                linewidth=2,
                markersize=7,
                color=color,
                label=label if metric != "rmsd_match_count" else f"{label} (higher is better)",
            )

            if annotate_points:
                for x_val, y_val in zip(subset.loc[mask, "beam_size"], y[mask]):
                    ax.annotate(
                        _format_annotation(float(y_val)),
                        xy=(x_val, y_val),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color=color,
                    )

        if not has_data:
            plt.close(fig)
            continue

        display_name = metric.replace("_", " ")
        xlabel, ylabel = _AXIS_LABEL_OVERRIDES.get(metric, ("Beam size", display_name.title()))
        title = _TITLE_OVERRIDES.get(metric, display_name.title())
        legend_title = _LEGEND_TITLES.get(metric)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(title=legend_title, frameon=False)

        fig.tight_layout()

        file_name = f"{_slugify_metric(metric)}.png"
        fig.savefig(output_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"✅ Metric plots written to {output_dir}")


def _load_summary(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return pd.DataFrame(data)
    if suffix in {".pkl", ".pkl.gz"}:
        return pd.read_pickle(path)
    raise ValueError(
        "Unsupported summary format. Expected CSV, JSON, or pickle file; "
        f"received: {path}"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality plots from the outputs of "
            "bin/eval/sampling_vs_beam_search.py"
        )
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to the summary CSV/JSON/pickle file produced by sampling_vs_beam_search.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plot images will be written. Defaults to <summary_dir>/plots.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of metric columns to plot. Defaults to all numeric metrics.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable point annotations showing metric values.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    frame = _load_summary(args.summary_path)
    if frame.empty:
        print(
            f"⚠️  Summary file {args.summary_path} contains no records; "
            "skipping plot generation."
        )
        return

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.summary_path.with_name("plots")
    )

    write_publication_quality_plots(
        frame,
        output_dir=output_dir,
        metrics=args.metrics,
        annotate_points=not args.no_annotations,
    )


if __name__ == "__main__":
    main()
