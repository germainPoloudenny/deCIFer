from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_SEARCH_ROOTS: Sequence[Path] = (
    Path("slurm"),
    Path("evaluations"),
)
PICKLE_PATTERNS: Sequence[str] = ("*.pkl", "*.pkl.gz")


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert a Series to boolean dtype while preserving ``NaN`` values."""
    if series.dtype == bool:
        return series

    # ``astype(bool)`` would treat NaNs as ``True``; we need to keep them as
    # missing values.  ``pd.to_numeric`` handles "True"/"False" strings as well.
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().all():
        return coerced.astype(bool)

    return series.astype("boolean")


def _discover_pickles() -> List[Path]:
    """Find pickled evaluation frames using a heuristic search."""
    discovered: List[Path] = []
    for root in DEFAULT_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in PICKLE_PATTERNS:
            discovered.extend(sorted(root.rglob(pattern)))
        if discovered:
            break
    return discovered


def _iter_pickles(paths: Iterable[str]) -> Iterable[Path]:
    """Yield pickled DataFrame paths from the provided arguments."""
    unique_paths = []
    for entry in paths:
        path = Path(entry).expanduser()
        if not path.exists():
            print(f"⚠️  Path not found: {path}", file=sys.stderr)
            continue
        if path.is_dir():
            for pattern in PICKLE_PATTERNS:
                unique_paths.extend(sorted(path.rglob(pattern)))
        else:
            unique_paths.append(path)

    if not unique_paths:
        unique_paths = _discover_pickles()

    seen = set()
    for path in unique_paths:
        if path.suffix not in {".pkl", ".gz"} and not any(
            str(path).endswith(ext) for ext in (".pkl", ".pkl.gz")
        ):
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        yield path


def _load_frames(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in _iter_pickles(paths):
        try:
            frame = pd.read_pickle(path)
        except Exception as exc:  # pragma: no cover - for defensive CLI usage
            print(f"❌ Unable to read {path}: {exc}", file=sys.stderr)
            continue
        frame["__source_file__"] = path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No evaluation pickle files found. Provide explicit paths with "
            "`python bin/show_eval_metrics.py <file1> <file2> ...>` or place "
            "the exported pickles inside `slurm/`."
        )

    return pd.concat(frames, ignore_index=True)


def _format_rate(label: str, values: pd.Series) -> str:
    values = _coerce_bool(values.dropna())
    if values.empty:
        return f"ℹ️  {label:<24}: aucune donnée"

    positives = int(values.sum())
    total = int(values.count())
    rate = positives / total if total else float("nan")
    return f"✅ {label:<24}: {rate:>6.3f} ({positives}/{total})"


def _spacegroup_match_rate(frame: pd.DataFrame) -> str:
    if not {"spacegroup_num_sample", "spacegroup_num_gen"}.issubset(frame.columns):
        return "ℹ️  Spacegroup match rate     : colonne manquante"

    sample = pd.to_numeric(frame["spacegroup_num_sample"], errors="coerce")
    generated = pd.to_numeric(frame["spacegroup_num_gen"], errors="coerce")
    mask = sample.notna() & generated.notna()
    if not mask.any():
        return "ℹ️  Spacegroup match rate     : aucune donnée"

    matches = (sample[mask] == generated[mask]).astype(int)
    positives = int(matches.sum())
    total = int(matches.count())
    rate = positives / total if total else float("nan")
    return f"✅ Spacegroup match rate     : {rate:>6.3f} ({positives}/{total})"


def _describe_numeric(label: str, series: pd.Series) -> List[str]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return [f"ℹ️  {label}: aucune valeur numérique disponible"]

    percentiles = numeric.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).rename(
        {0.25: "25%", 0.5: "50%", 0.75: "75%", 0.9: "90%", 0.95: "95%"}
    )
    summary_lines = [
        f"📊 {label} (n={int(numeric.count())}):",
        f"    mean ± std : {numeric.mean():.3f} ± {numeric.std(ddof=0):.3f}",
        f"    min / max  : {numeric.min():.3f} – {numeric.max():.3f}",
    ]
    quantile_str = ", ".join(f"{idx}={val:.3f}" for idx, val in percentiles.items())
    summary_lines.append(f"    quantiles  : {quantile_str}")
    return summary_lines


def _print_validity_breakdown(frame: pd.DataFrame) -> None:
    validity_columns = [col for col in frame.columns if col.endswith("_validity")]
    if not validity_columns:
        return

    print("\nDétails des validations :")
    for column in sorted(validity_columns):
        label = column.replace("_", " ")
        print(_format_rate(label, frame[column]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Affiche un résumé des métriques d'évaluation générées par deCIFer.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Fichiers .pkl ou .pkl.gz contenant les résultats de `collect_evaluations`. "
            "Vous pouvez également passer des répertoires – ils seront parcourus "
            "récursivement. Lorsque aucun chemin n'est fourni, le script cherche "
            "automatiquement dans `slurm/`."
        ),
        default=["runs/deCIFer_cifs_v1_model/eval_bsm/cifs_v1.pkl.gz"]
    )
    args = parser.parse_args(argv)

    try:
        frame = _load_frames(args.paths)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    total = len(frame)
    print("=== Résultats de l'évaluation ===")
    print(f"Nombre total d'échantillons : {total}")

    if "validity" in frame.columns:
        print(_format_rate("Validity rate", frame["validity"]))
    else:
        print("ℹ️  Validity rate             : colonne manquante")

    print(_spacegroup_match_rate(frame))

    for label, column in (
        ("RMSD", "rmsd"),
        ("Rwp", "rwp"),
        ("L2 distance", "l2_distance"),
        ("Wd", "wd"),
    ):
        if column not in frame.columns:
            print(f"ℹ️  {label}: colonne manquante")
            continue
        for line in _describe_numeric(f"{label}", frame[column]):
            print(line)

    _print_validity_breakdown(frame)

    seq_columns = ["seq_len_sample", "seq_len_gen"]
    available_seq_cols = [col for col in seq_columns if col in frame.columns]
    if available_seq_cols:
        print("\nLongueurs de séquences :")
        for col in available_seq_cols:
            for line in _describe_numeric(col, frame[col]):
                print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
