#!/usr/bin/env python3
"""Utility to display dataset split sizes for serialized deCIFer datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Affiche le nombre d'entrées pour chaque split disponible d'un "
            "dataset HDF5 (train/val/test) et résume la taille totale et celle "
            "du split de test."
        )
    )
    parser.add_argument(
        "dataset",
        help=(
            "Chemin vers le dataset : dossier contenant les fichiers .h5 ou fichier "
            "HDF5 unique."
        ),
    )
    return parser.parse_args()


def _discover_split_files(base_path: Path) -> Dict[str, Path]:
    """Return available split files under ``base_path``."""

    if base_path.is_file():
        return {"test": base_path}

    splits: Dict[str, Path] = {}
    search_order = {
        "train": [
            base_path / "serialized" / "train.h5",
            base_path / "train" / "train.h5",
            base_path / "train.h5",
        ],
        "val": [
            base_path / "serialized" / "val.h5",
            base_path / "val" / "val.h5",
            base_path / "val.h5",
        ],
        "test": [
            base_path / "serialized" / "test.h5",
            base_path / "test" / "test.h5",
            base_path / "test.h5",
        ],
    }

    for split, candidates in search_order.items():
        for candidate in candidates:
            if candidate.is_file():
                splits[split] = candidate
                break

    fallback_dirs: Iterable[Path] = (base_path, base_path / "serialized")
    for directory in fallback_dirs:
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.suffix == ".h5" and entry.is_file():
                split_name = entry.stem.lower()
                splits.setdefault(split_name, entry)

    return splits


def _dataset_length(h5_path: Path) -> int:
    """Determine dataset length using the first suitable dataset in the file."""

    key_priority = (
        "cif_tokenized",
        "cif_tokens",
        "cif_string",
        "xrd.iq",
        "xrd_disc.iq",
    )

    with h5py.File(h5_path, "r") as handle:
        for key in key_priority:
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return len(handle[key])

        for dataset in handle.values():
            if isinstance(dataset, h5py.Dataset):
                if dataset.shape:
                    return dataset.shape[0]
                return len(dataset)

    raise RuntimeError(f"Impossible de déterminer la taille du dataset pour {h5_path}.")


def main() -> None:
    args = parse_args()
    base_path = Path(args.dataset).expanduser().resolve()

    if not base_path.exists():
        sys.exit(f"Chemin introuvable : {base_path}")

    split_files = _discover_split_files(base_path)
    if not split_files:
        sys.exit(
            "Aucun fichier HDF5 détecté. Fournissez un dossier contenant des fichiers .h5 "
            "ou un chemin vers un fichier unique."
        )

    split_lengths: Dict[str, int] = {}
    for split, path in sorted(split_files.items()):
        try:
            length = _dataset_length(path)
        except Exception as exc:  # pragma: no cover - CLI feedback
            sys.exit(f"Erreur lors de la lecture de {path}: {exc}")

        split_lengths[split] = length
        print(f"Split {split:>5s} : {length:,}")

    total = sum(split_lengths.values())
    print("-" * 40)
    print(f"Taille totale du dataset : {total:,}")

    test_size = split_lengths.get("test")
    if test_size is not None:
        print(f"Taille du dataset de test : {test_size:,}")
    else:
        print("Aucun split de test trouvé.")


if __name__ == "__main__":
    main()

