#!/usr/bin/env python3
"""Render comparison plots for the beam-vs-RWP-ranking experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bin.eval.plot.sampling_vs_beam import (
    MetricLayout,
    _METRIC_LAYOUT_OVERRIDES,
    _format_annotation,
    _prepare_style_iterables,
    _select_numeric_metrics,
)

__all__ = ["write_beam_vs_rwp_plots"]

_DEFAULT_IGNORE_METRICS = {
    "beam_size",
    "num_reps",
    "collect_top_k",
    "max_samples",
    "nproc_per_node",
    "valid_total",
    "spacegroup_match_total",
    "rmsd_match_total",
    "rwp_match_total",
    "rmsd_match_rate_ranking",
    "rwp_match_rate_ranking",
}


def _resolve_metric_layout(metric: str) -> MetricLayout:
    override = _METRIC_LAYOUT_OVERRIDES.get(metric)
    if override is not None:
        return MetricLayout(
            x_label="Decoding strategy",
            y_label=override.y_label or metric,
            legend_title=override.legend_title,
            prefer_integer_ticks=override.prefer_integer_ticks,
        )
    return MetricLayout(
        x_label="Decoding strategy",
        y_label=metric,
        legend_title="Variant",
    )


def write_beam_vs_rwp_plots(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    metrics: Optional[Iterable[str]] = None,
    annotate_points: bool = True,
) -> None:
    """Write bar plots comparing decoding strategies for the provided metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names: List[str] = (
        list(metrics)
        if metrics is not None
        else _select_numeric_metrics(frame, ignore=_DEFAULT_IGNORE_METRICS)
    )

    if not metric_names:
        print("⚠️  No numeric metrics available for plotting; skipping plot generation.")
        return

    color_map, _ = _prepare_style_iterables(frame)

    ordered = frame.sort_values("variant")
    categories = ordered[["variant", "description"]].drop_duplicates().reset_index(drop=True)

    for metric in metric_names:
        values = pd.to_numeric(ordered[metric], errors="coerce")
        mask = np.isfinite(values.to_numpy())
        if not mask.any():
            continue

        metric_layout = _resolve_metric_layout(metric)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plotted = False
        x_positions: List[int] = []
        heights: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        plotted_variants: List[str] = []

        for idx, row in categories.iterrows():
            variant = row["variant"]
            description = row["description"]
            subset = ordered[ordered["variant"] == variant]
            numeric = pd.to_numeric(subset[metric], errors="coerce")
            numeric = numeric[np.isfinite(numeric)]
            if numeric.empty:
                continue

            height = float(numeric.mean())
            x_positions.append(idx)
            heights.append(height)
            labels.append(description if isinstance(description, str) and description else str(variant))
            colors.append(color_map.get(variant, "#1f77b4"))
            plotted_variants.append(str(variant))
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.bar(x_positions, heights, color=colors, width=0.55)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(metric_layout.y_label or metric)
        ax.set_xlabel(metric_layout.x_label or "Decoding strategy")

        if metric_layout.legend_title and plotted_variants:
            legend_entries = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    color=color_map.get(variant, "#1f77b4"),
                )
                for variant in plotted_variants
            ]
            legend_labels = labels
            ax.legend(legend_entries, legend_labels, title=metric_layout.legend_title)

        if annotate_points:
            for x, height in zip(x_positions, heights):
                ax.annotate(
                    _format_annotation(height),
                    xy=(x, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

        ax.set_title(metric.replace("_", " ").title())
        fig.tight_layout()

        output_path = output_dir / f"beam_vs_rwp_{metric}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"✅ Plot written to {output_path}")


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported summary format: {path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots for the beam vs RWP-ranked comparison summary.",
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to the CSV or JSON summary produced by beam_vs_rwp_filter.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of metric columns to plot.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable value annotations on the bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    frame = _load_frame(args.summary_path)
    write_beam_vs_rwp_plots(
        frame,
        output_dir=args.output_dir,
        metrics=args.metrics,
        annotate_points=not args.no_annotations,
    )


if __name__ == "__main__":
    main()
