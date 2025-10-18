#!/usr/bin/env python3
"""Generate plots for progressive sampling evaluation metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


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


@dataclass(frozen=True)
class MetricLayout:
    """Configuration controlling how a metric should be rendered."""

    y_label: Optional[str] = None
    prefer_integer_ticks: bool = False
    annotate_as_percent: bool = False
    percent_formatter: bool = False


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _humanize_metric(name: str) -> str:
    parts = []
    for part in name.split("_"):
        upper = part.upper()
        if upper in {"RMSD", "RWP", "WD", "L2"}:
            parts.append(upper)
        else:
            parts.append(part.capitalize())
    return " ".join(parts)


def _select_numeric_metrics(frame: pd.DataFrame, ignore: Iterable[str]) -> list[str]:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    ignore_set = set(ignore)
    return [column for column in numeric_columns if column not in ignore_set]


def _layout_for_metric(metric: str) -> MetricLayout:
    if metric.endswith("_rate"):
        return MetricLayout(
            y_label=f"{_humanize_metric(metric)}", annotate_as_percent=True, percent_formatter=True
        )
    if metric.endswith("_count"):
        return MetricLayout(
            y_label=_humanize_metric(metric),
            prefer_integer_ticks=True,
        )
    if metric.endswith("_total"):
        return MetricLayout(
            y_label=_humanize_metric(metric),
            prefer_integer_ticks=True,
        )
    return MetricLayout(y_label=_humanize_metric(metric))


def _format_value(value: float, *, as_percent: bool = False) -> str:
    if as_percent:
        return f"{value * 100:.1f}%"
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_progressive_sampling_plots(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    metrics: Optional[Iterable[str]] = None,
    annotate_points: bool = True,
) -> None:
    """Render a series of plots showing metric evolution over sampling milestones."""

    if frame.empty:
        print("⚠️  Summary frame is empty; skipping plot generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    x_column = None
    for candidate in ("milestone", "samples"):
        if candidate in frame.columns:
            x_column = candidate
            break

    if x_column is not None:
        x_series = pd.to_numeric(frame[x_column], errors="coerce")
        frame = frame.assign(__x=x_series)
        frame = frame.sort_values("__x")
    else:
        frame = frame.assign(__x=np.arange(1, len(frame) + 1, dtype=float))
        x_column = "__x"

    default_ignore = ["__x"]
    for candidate in ("milestone", "samples"):
        if candidate in frame.columns:
            default_ignore.append(candidate)

    metric_names = (
        list(metrics)
        if metrics is not None
        else _select_numeric_metrics(frame, ignore=default_ignore)
    )

    if not metric_names:
        print("⚠️  No numeric metrics available for plotting; skipping plot generation.")
        return

    for metric in metric_names:
        if metric not in frame.columns:
            print(f"⚠️  Metric '{metric}' not found in the summary; skipping.")
            continue

        y_series = pd.to_numeric(frame[metric], errors="coerce")
        plot_df = pd.DataFrame({"x": frame[x_column], "y": y_series}).dropna()
        if plot_df.empty:
            continue

        plot_df = plot_df.sort_values("x")
        x_values = plot_df["x"].to_numpy(dtype=float)
        y_values = plot_df["y"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.plot(x_values, y_values, marker="o", linewidth=2.3, markersize=7, color="#1f77b4")

        layout = _layout_for_metric(metric)
        xlabel = "Samples evaluated" if x_column in {"milestone", "samples", "__x"} else _humanize_metric(x_column)
        ylabel = layout.y_label or _humanize_metric(metric)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if layout.percent_formatter:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if layout.prefer_integer_ticks:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if annotate_points:
            for x_val, y_val in zip(x_values, y_values):
                annotation = _format_value(float(y_val), as_percent=layout.annotate_as_percent)
                ax.annotate(
                    annotation,
                    xy=(x_val, y_val),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#1a1a1a",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.9,
                    },
                )

        fig.tight_layout()

        file_name = f"{_slugify(metric)}.png"
        fig.savefig(output_dir / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    print(f"✅ Metric plots written to {output_dir}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots illustrating the metrics reported by "
            "bin/eval/progressive_sampling_metrics.py"
        )
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        help="Path to the progressive_metrics.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plot images will be saved. Defaults to <summary_dir>/plots.",
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
        help="Disable annotations that display metric values at each milestone.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    if not args.summary_path.exists():
        raise FileNotFoundError(f"Summary file {args.summary_path} does not exist.")

    frame = pd.read_csv(args.summary_path)
    output_dir = args.output_dir or args.summary_path.with_name("plots")

    write_progressive_sampling_plots(
        frame,
        output_dir=output_dir,
        metrics=args.metrics,
        annotate_points=not args.no_annotations,
    )


if __name__ == "__main__":
    main()
