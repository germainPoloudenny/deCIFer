#!/usr/bin/env python3

"""Display the best validation loss stored inside a training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Import the training config so torch.load can reconstruct the pickled object.
try:  # pragma: no cover - defensive import guard
    from decifer.config.train_config import TrainConfig as _ImportedTrainConfig
    globals()["TrainConfig"] = _ImportedTrainConfig
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the best validation loss stored in a training checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint file (e.g. runs/experiment/ckpt.pt) or its parent directory.",
    )
    return parser.parse_args()


def resolve_checkpoint(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_dir():
        candidate = candidate / "ckpt.pt"
    return candidate


def main() -> int:
    args = parse_args()
    ckpt_path = resolve_checkpoint(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    training_metrics = checkpoint.get("training_metrics")
    if not training_metrics:
        print("[ERROR] No training metrics stored in this checkpoint.", file=sys.stderr)
        return 1

    best_val_loss = training_metrics.get("best_val_loss")
    val_losses = training_metrics.get("val_losses") or []
    epochs = training_metrics.get("epochs") or []

    if best_val_loss is None:
        print("[ERROR] Best validation loss is missing from the checkpoint.", file=sys.stderr)
        return 1

    # Determine the iteration where the validation loss reached its minimum.
    iteration_info = ""
    if val_losses:
        best_idx = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        iteration = epochs[best_idx] if best_idx < len(epochs) else None
        if iteration is not None:
            iteration_info = f" (recorded at iteration {iteration})"

    print(f"Best validation loss: {best_val_loss:.6f}{iteration_info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
