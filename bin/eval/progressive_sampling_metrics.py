#!/usr/bin/env python3
"""Progressive evaluation of k-sampling generations on the test set."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

# Ensure the repository root is on the Python path so we can import helper
# functions from the existing evaluation pipeline.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymatgen.analysis.structure_matcher import StructureMatcher

from bin.eval.evaluate import (  # pylint: disable=wrong-import-position
    get_cif_statistics,
    iter_dataset_samples,
    load_model_from_checkpoint,
    prepare_prompt_and_conditioning,
)
from bin.train import setup_distributed
from decifer.decifer_dataset import DeciferDataset
from decifer.decifer_model import Decifer
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    generate_continuous_xrd_from_cif,
    get_rmsd,
    is_sensible,
    replace_symmetry_loop_with_P1,
    reinstate_symmetry_loop,
    extract_space_group_symbol,
    space_group_symbol_to_number,
)


def _rwp(sample: np.ndarray, generated: np.ndarray) -> float:
    """Compute the residual weighted profile (Rwp) metric."""
    numerator = np.sum(np.square(sample - generated), axis=-1)
    denominator = np.sum(np.square(sample), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.sqrt(np.divide(numerator, denominator, where=denominator != 0))
    if np.isscalar(value):
        return float(value)
    return float(value.item())


def _compute_milestones(total: int, base: int = 100) -> List[int]:
    """Compute progressive evaluation milestones."""
    if total <= 0:
        return []

    milestones: List[int] = []
    current = base
    while current < total:
        milestones.append(current)
        current *= 10

    if not milestones or milestones[-1] != total:
        milestones.append(total)

    return milestones


def _numeric_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "count": int(numeric.count()),
    }


def _summarise_records(records: List[Dict[str, object]]) -> Dict[str, float]:
    frame = pd.DataFrame(records)
    summary: Dict[str, float] = {
        "samples": float(len(frame)),
    }

    if "success" in frame:
        successes = frame["success"].dropna().astype(bool)
        total = int(successes.count())
        summary["success_total"] = float(total)
        summary["success_count"] = float(int(successes.sum()))
        summary["success_rate"] = float(int(successes.sum()) / total) if total else float("nan")

    if "validity" in frame:
        validity = frame["validity"].dropna().astype(bool)
        total = int(validity.count())
        summary["valid_total"] = float(total)
        valid_count = int(validity.sum())
        summary["valid_count"] = float(valid_count)
        summary["validity_rate"] = float(valid_count / total) if total else float("nan")

    validity_columns = [col for col in frame.columns if col.endswith("_validity")]
    for column in validity_columns:
        series = frame[column].dropna().astype(bool)
        total = int(series.count())
        count = int(series.sum())
        rate = float(count / total) if total else float("nan")
        summary[f"{column}_count"] = float(count)
        summary[f"{column}_total"] = float(total)
        summary[f"{column}_rate"] = rate

    expected_cols = {"spacegroup_num_sample", "spacegroup_num_gen"}
    if expected_cols.issubset(frame.columns):
        sample = pd.to_numeric(frame["spacegroup_num_sample"], errors="coerce")
        generated = pd.to_numeric(frame["spacegroup_num_gen"], errors="coerce")
        mask = sample.notna() & generated.notna()
        if mask.any():
            matches = sample[mask] == generated[mask]
            match_count = int(matches.sum())
            match_total = int(matches.count())
        else:
            match_count = 0
            match_total = 0
        summary["spacegroup_match_count"] = float(match_count)
        summary["spacegroup_match_total"] = float(match_total)
        summary["spacegroup_match_rate"] = (
            float(match_count / match_total) if match_total else float("nan")
        )

    for column in ("rmsd", "rwp", "l2_distance", "wd"):
        if column not in frame:
            continue
        stats = _numeric_stats(frame[column])
        summary[f"{column}_mean"] = stats["mean"]
        summary[f"{column}_median"] = stats["median"]
        summary[f"{column}_std"] = stats["std"]
        summary[f"{column}_count"] = float(stats["count"])

    return summary


def _compute_progressive_summaries(
    records: List[Dict[str, object]],
    total_samples: int,
    milestone_base: int,
) -> List[Dict[str, float]]:
    """Aggregate progressive summaries from evaluation records."""

    if total_samples <= 0 or not records:
        return []

    milestones = _compute_milestones(total_samples, base=milestone_base)
    if not milestones:
        return []

    summaries: List[Dict[str, float]] = []
    progressive_records: List[Dict[str, object]] = []
    milestone_iter = iter(milestones)
    try:
        next_milestone = next(milestone_iter)
    except StopIteration:
        return []

    for record in sorted(records, key=lambda item: item.get("index", math.inf)):
        progressive_records.append(record)
        while len(progressive_records) >= next_milestone:
            summary = _summarise_records(progressive_records)
            summary["samples"] = float(len(progressive_records))
            summary["milestone"] = float(len(progressive_records))
            summaries.append(summary)
            try:
                next_milestone = next(milestone_iter)
            except StopIteration:
                next_milestone = math.inf

    return summaries


def _evaluate_generation(
    sample: Dict[str, object],
    generated_tokens: np.ndarray,
    tokenizer: Tokenizer,
    matcher: StructureMatcher,
    clean_kwargs: Dict[str, object],
    *,
    debug: bool = False,
) -> Dict[str, object]:
    """Compute evaluation metrics for a single generated structure."""
    sample_tokens: torch.Tensor = sample["cif_tokens"]  # type: ignore[assignment]
    sample_tokens_np = sample_tokens.cpu().numpy()
    padding_id = tokenizer.padding_id
    sample_tokens_np = sample_tokens_np[sample_tokens_np != padding_id]
    cif_name: str = sample["cif_name"]  # type: ignore[assignment]
    cif_string_sample: str = sample["cif_string"]  # type: ignore[assignment]

    result: Dict[str, object] = {
        "seq_len_sample": float(len(sample_tokens_np)),
        "seq_len_gen": float(len(generated_tokens)),
        "success": False,
    }

    sample_spacegroup_symbol = extract_space_group_symbol(cif_string_sample)
    result["spacegroup_sym_sample"] = sample_spacegroup_symbol
    result["spacegroup_num_sample"] = (
        float(space_group_symbol_to_number(sample_spacegroup_symbol))
        if sample_spacegroup_symbol is not None
        else float("nan")
    )

    try:
        cif_string_gen = tokenizer.decode(generated_tokens)
        cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)
        spacegroup_symbol = extract_space_group_symbol(cif_string_gen)
        if spacegroup_symbol != "P 1":
            cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)

        if not is_sensible(cif_string_gen):
            result["error"] = "Generated CIF deemed not sensible"
            return result

        stats = get_cif_statistics(cif_string_gen, None)
        validity = stats.get("validity", {}) if isinstance(stats, dict) else {}
        formula_valid = bool(validity.get("formula", False))
        site_valid = bool(validity.get("site_multiplicity", False))
        bond_valid = bool(validity.get("bond_length", False))
        spacegroup_valid = bool(validity.get("spacegroup", False))
        all_valid = formula_valid and site_valid and bond_valid and spacegroup_valid

        result.update(
            {
                "cif_string_gen": cif_string_gen,
                "spacegroup_sym_gen": stats.get("spacegroup"),
                "spacegroup_num_gen": (
                    float(space_group_symbol_to_number(stats.get("spacegroup")))
                    if stats.get("spacegroup") is not None
                    else float("nan")
                ),
                "formula_validity": formula_valid,
                "site_multiplicity_validity": site_valid,
                "bond_length_validity": bond_valid,
                "spacegroup_validity": spacegroup_valid,
                "validity": all_valid,
            }
        )

        rmsd_value, matcher_mode = get_rmsd(
            cif_string_sample,
            cif_string_gen,
            matcher=matcher,
            supercell_matcher=None,
        )
        result["rmsd"] = float(rmsd_value) if rmsd_value is not None else float("nan")
        result["structure_match_mode"] = matcher_mode

        clean_gen = generate_continuous_xrd_from_cif(
            cif_string_gen,
            structure_name=cif_name,
            debug=debug,
            **clean_kwargs,
        )
        clean_sample = generate_continuous_xrd_from_cif(
            cif_string_sample,
            structure_name=cif_name,
            debug=debug,
            **clean_kwargs,
        )

        iq_sample = np.asarray(clean_sample["iq"], dtype=np.float64)
        iq_gen = np.asarray(clean_gen["iq"], dtype=np.float64)
        result["rwp"] = _rwp(iq_sample, iq_gen)
        result["l2_distance"] = float(np.linalg.norm(iq_sample - iq_gen))

        q_disc_sample = np.asarray(clean_sample.get("q_disc"), dtype=np.float64)
        q_disc_gen = np.asarray(clean_gen.get("q_disc"), dtype=np.float64)
        iq_disc_sample = np.asarray(clean_sample.get("iq_disc"), dtype=np.float64)
        iq_disc_gen = np.asarray(clean_gen.get("iq_disc"), dtype=np.float64)
        if (
            q_disc_sample.size
            and q_disc_gen.size
            and iq_disc_sample.sum() > 0
            and iq_disc_gen.sum() > 0
        ):
            iq_disc_sample_norm = iq_disc_sample / np.sum(iq_disc_sample)
            iq_disc_gen_norm = iq_disc_gen / np.sum(iq_disc_gen)
            result["wd"] = float(
                wasserstein_distance(
                    q_disc_sample,
                    q_disc_gen,
                    u_weights=iq_disc_sample_norm,
                    v_weights=iq_disc_gen_norm,
                )
            )
        else:
            result["wd"] = float("nan")

        result["success"] = True
        return result
    except Exception as exc:  # pragma: no cover - defensive against evaluation errors
        result["error"] = str(exc)
        return result


def _build_conditioning_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    noise_range = (args.add_noise, args.add_noise) if args.add_noise is not None else None
    broadening = (args.add_broadening, args.add_broadening) if args.add_broadening is not None else None
    fwhm_range = broadening or (args.default_fwhm, args.default_fwhm)
    return {
        "qmin": args.qmin,
        "qmax": args.qmax,
        "qstep": args.qstep,
        "wavelength": args.wavelength,
        "fwhm_range": fwhm_range,
        "eta_range": (args.eta, args.eta),
        "noise_range": noise_range,
        "intensity_scale_range": None,
        "mask_prob": None,
    }


def _build_clean_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "qmin": args.qmin,
        "qmax": args.qmax,
        "qstep": args.qstep,
        "wavelength": args.wavelength,
        "fwhm_range": (args.clean_fwhm, args.clean_fwhm),
        "eta_range": (args.eta, args.eta),
        "noise_range": None,
        "intensity_scale_range": None,
        "mask_prob": None,
    }


def _run_progressive_evaluation(
    args: argparse.Namespace,
    milestone_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Optional[pd.DataFrame]:
    requested_device = "cpu" if (args.device == "cuda" and not torch.cuda.is_available()) else args.device

    dist_config = SimpleNamespace(
        device=requested_device,
        distributed=args.distributed,
        dist_backend=args.dist_backend,
        dist_url=args.dist_url,
    )
    use_distributed, world_size, rank, _ = setup_distributed(dist_config)
    device = torch.device(dist_config.device)

    model: Decifer = load_model_from_checkpoint(args.model_ckpt, device)
    model.eval()
    torch.set_grad_enabled(False)

    tokenizer = Tokenizer()
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    dataset = DeciferDataset(
        args.dataset_path,
        ["cif_name", "cif_tokens", "xrd.q", "xrd.iq", "cif_string", "spacegroup"],
    )

    conditioning_kwargs = _build_conditioning_kwargs(args)
    clean_kwargs = _build_clean_kwargs(args)
    top_k_value = args.top_k if args.top_k is None or args.top_k > 0 else None

    total_samples = len(dataset)
    samples_for_rank = (
        max((total_samples - rank + world_size - 1) // world_size, 0)
        if use_distributed
        else total_samples
    )

    records: List[Dict[str, object]] = []
    local_summaries: List[Dict[str, float]] = []

    milestone_iter: Optional[Iterator[int]] = None
    next_milestone: Optional[int] = None
    if not use_distributed:
        milestone_iter = iter(_compute_milestones(len(dataset), base=args.milestone_base))
        try:
            next_milestone = next(milestone_iter)
        except StopIteration:
            next_milestone = None
    progress = tqdm(
        total=samples_for_rank,
        desc="Sampling",
        disable=args.no_progress or samples_for_rank == 0,
    )

    sample_iterator = iter_dataset_samples(
        dataset,
        rank=rank if use_distributed else 0,
        world_size=world_size if use_distributed else 1,
    )

    for index, sample in sample_iterator:
        prompt, cond_vec, _ = prepare_prompt_and_conditioning(
            sample,
            device,
            add_composition=args.add_composition,
            add_spacegroup=args.add_spacegroup,
            conditioning_kwargs=conditioning_kwargs,
        )

        with torch.no_grad():
            generated_tensor = model.generate_batched_reps(
                prompt,
                args.max_new_tokens,
                cond_vec,
                [[0]],
                args.temperature,
                top_k_value,
                disable_pbar=True,
            )
        generated = generated_tensor[0].detach().cpu().numpy()
        generated = generated[generated != tokenizer.padding_id]

        record = _evaluate_generation(
            sample,
            generated,
            tokenizer,
            matcher,
            clean_kwargs,
            debug=args.debug,
        )
        record["index"] = float(index)
        records.append(record)

        if not use_distributed and next_milestone is not None and len(records) == next_milestone:
            summary = _summarise_records(records)
            summary["samples"] = float(len(records))
            summary["milestone"] = float(len(records))
            local_summaries.append(summary)
            if milestone_callback is not None:
                milestone_callback(summary)
            assert milestone_iter is not None
            try:
                next_milestone = next(milestone_iter)
            except StopIteration:
                next_milestone = None

        progress.update(1)

    progress.close()

    if use_distributed:
        gathered_records: Optional[List[List[Dict[str, object]]]]
        gathered_records = [None] * world_size if rank == 0 else None
        dist.gather_object(records, gathered_records, dst=0)
        all_records: List[Dict[str, object]] = []
        if rank == 0 and gathered_records is not None:
            for worker_records in gathered_records:
                all_records.extend(worker_records or [])
    else:
        all_records = records

    if not use_distributed:
        summaries = local_summaries
    else:
        if use_distributed and rank != 0:
            return None

        summaries = _compute_progressive_summaries(
            all_records,
            total_samples=len(dataset),
            milestone_base=args.milestone_base,
        )

        if milestone_callback is not None:
            for summary in summaries:
                milestone_callback(summary)

    return pd.DataFrame(summaries)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run k-sampling inference over the test set and save evaluation metrics "
            "at progressive milestones."
        )
    )
    parser.add_argument("--model-ckpt", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--dataset-path", required=True, help="Path to the test dataset (HDF5).")
    parser.add_argument("--output", default="progressive_metrics.csv", help="Destination CSV file for metrics.")
    parser.add_argument("--device", default="cuda", help="Device to use for inference (cuda or cpu).")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed multi-GPU evaluation.",
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        help="Distributed backend to use when running with --distributed.",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        help="Initialization URL for torch.distributed when running with --distributed.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k value for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--add-composition", action="store_true", help="Include composition in prompts.")
    parser.add_argument("--add-spacegroup", action="store_true", help="Include spacegroup in prompts.")
    parser.add_argument("--milestone-base", type=int, default=100, help="Base milestone size (default: 100).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode when generating XRD patterns.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")

    parser.add_argument("--add-noise", type=float, default=None, help="Noise amplitude for conditioning signals.")
    parser.add_argument("--add-broadening", type=float, default=None, help="FWHM broadening for conditioning signals.")
    parser.add_argument("--default_fwhm", type=float, default=0.05, help="Default FWHM when no broadening is specified.")
    parser.add_argument("--clean_fwhm", type=float, default=0.05, help="FWHM used for clean XRD generation.")
    parser.add_argument("--qmin", type=float, default=0.0, help="Minimum q value for XRD generation.")
    parser.add_argument("--qmax", type=float, default=10.0, help="Maximum q value for XRD generation.")
    parser.add_argument("--qstep", type=float, default=0.01, help="Step size for q grid in XRD generation.")
    parser.add_argument("--wavelength", default="CuKa", help="Wavelength identifier for XRD simulation.")
    parser.add_argument("--eta", type=float, default=0.5, help="Pseudo-Voigt eta parameter for XRD simulation.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    header_written = False
    wrote_to_disk = False

    def milestone_callback(summary: Dict[str, float]) -> None:
        nonlocal header_written, wrote_to_disk
        frame = pd.DataFrame([summary])
        mode = "a" if header_written else "w"
        frame.to_csv(output_path, mode=mode, header=not header_written, index=False)
        header_written = True
        wrote_to_disk = True
        milestone = int(summary.get("milestone", summary.get("samples", 0)))
        print(f"Recorded progressive metrics for {milestone} samples to {output_path}")

    summaries = _run_progressive_evaluation(args, milestone_callback=milestone_callback)
    if summaries is not None and not wrote_to_disk:
        summaries.to_csv(output_path, index=False)
        wrote_to_disk = True
        if not summaries.empty:
            header_written = True

    if wrote_to_disk:
        print(f"Saved progressive metrics to {output_path}")
    elif summaries is not None:
        print("No progressive metrics were generated.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
