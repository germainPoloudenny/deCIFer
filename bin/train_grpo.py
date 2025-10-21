#!/usr/bin/env python3
"""Fine-tune deCIFer with Group Relative Policy Optimization (GRPO)."""

import argparse
import json
import warnings
from typing import Any, Dict

from omegaconf import OmegaConf

from decifer.rl import GRPOConfig, GRPOTrainer
from decifer.config.train_config import TrainConfig as _SupervisedTrainConfig

# Ensure legacy checkpoints referencing ``__main__.TrainConfig`` remain loadable
TrainConfig = _SupervisedTrainConfig


def _load_config(path: str) -> Dict[str, Any]:
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def build_config(args: argparse.Namespace) -> GRPOConfig:
    base = GRPOConfig()

    if args.config is not None:
        cfg_dict = _load_config(args.config)
        if isinstance(cfg_dict, dict) and "grpo" in cfg_dict and isinstance(cfg_dict["grpo"], dict):
            cfg_dict = cfg_dict["grpo"]

        unknown_keys = []
        for key, value in cfg_dict.items():
            if hasattr(base, key):
                setattr(base, key, value)
            else:
                unknown_keys.append(key)

        if unknown_keys:
            joined = ", ".join(sorted(unknown_keys))
            warnings.warn(
                "Ignoring unrecognised configuration keys: "
                f"{joined}. GRPO fine-tuning expects a dedicated configuration file."
            )

    for field_name in (
        "out_dir",
        "dataset",
        "init_checkpoint",
        "reference_checkpoint",
        "batch_size",
        "group_size",
        "max_new_tokens",
        "temperature",
        "top_k",
        "learning_rate",
        "kl_coef",
        "reward_scale",
        "invalid_reward",
        "max_iterations",
        "log_interval",
        "save_interval",
        "conditioning_cache_dir",
        "precompute_conditioning",
        "precompute_conditioning_batch_size",
        "device",
        "dtype",
        "num_workers",
        "seed",
    ):
        value = getattr(args, field_name)
        if value is not None:
            setattr(base, field_name, value)

    if args.conditioning_kwargs is not None:
        base.conditioning_kwargs = json.loads(args.conditioning_kwargs)

    if args.normalize_advantages is not None:
        base.normalize_advantages = args.normalize_advantages

    if args.data_parallel:
        base.data_parallel = True

    if args.add_spacegroup is not None:
        base.add_spacegroup = args.add_spacegroup

    if args.add_composition is not None:
        base.add_composition = args.add_composition

    if args.mini_batch_size is not None:
        base.mini_batch_size = args.mini_batch_size

    if args.update_epochs is not None:
        base.update_epochs = args.update_epochs

    if args.clip_range is not None:
        base.clip_range = args.clip_range

    if args.target_kl is not None:
        base.target_kl = args.target_kl

    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file.")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None, help="Output directory for checkpoints.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the serialized dataset (HDF5).")
    parser.add_argument("--init-checkpoint", dest="init_checkpoint", type=str, default=None, help="Initial model checkpoint.")
    parser.add_argument(
        "--reference-checkpoint",
        dest="reference_checkpoint",
        type=str,
        default=None,
        help="Optional reference checkpoint used for KL regularisation.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Number of prompts per rollout batch.")
    parser.add_argument("--group-size", dest="group_size", type=int, default=None, help="Number of completions sampled per prompt.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None, help="Maximum tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for generation.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Optional top-k sampling cutoff.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None, help="Optimizer learning rate.")
    parser.add_argument("--kl-coef", dest="kl_coef", type=float, default=None, help="KL penalty coefficient.")
    parser.add_argument("--clip-range", dest="clip_range", type=float, default=None, help="PPO clipping range.")
    parser.add_argument(
        "--target-kl",
        dest="target_kl",
        type=float,
        default=None,
        help="Early stopping threshold on average per-token KL divergence.",
    )
    parser.add_argument("--reward-scale", dest="reward_scale", type=float, default=None, help="Scale factor applied to rewards.")
    parser.add_argument("--invalid-reward", dest="invalid_reward", type=float, default=None, help="Reward assigned to invalid structures.")
    parser.add_argument("--normalize-advantages", dest="normalize_advantages", type=lambda x: x.lower() == "true", default=None, help="Whether to normalise advantages (true/false).")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true", help="Enable torch.nn.DataParallel for multi-GPU GRPO training.")
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=None, help="Total GRPO iterations to run.")
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None, help="Logging interval in iterations.")
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=None, help="Checkpoint interval in iterations.")
    parser.add_argument("--mini-batch-size", dest="mini_batch_size", type=int, default=None, help="Mini-batch size for policy updates.")
    parser.add_argument("--update-epochs", dest="update_epochs", type=int, default=None, help="Number of epochs per policy update.")
    parser.add_argument("--device", type=str, default=None, help="Device identifier (e.g. cpu, cuda:0).")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"], help="Computation dtype for the policy model.")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--add-composition", dest="add_composition", type=lambda x: x.lower() == "true", default=None, help="Include composition information in prompts (true/false).")
    parser.add_argument("--add-spacegroup", dest="add_spacegroup", type=lambda x: x.lower() == "true", default=None, help="Include space-group information in prompts (true/false).")
    parser.add_argument(
        "--precompute-conditioning",
        dest="precompute_conditioning",
        action="store_true",
        help="Precompute continuous XRD conditioning vectors and cache them before GRPO training.",
    )
    parser.add_argument(
        "--no-precompute-conditioning",
        dest="precompute_conditioning",
        action="store_false",
        help="Disable conditioning precomputation regardless of the configuration file.",
    )
    parser.set_defaults(precompute_conditioning=None)
    parser.add_argument(
        "--conditioning-cache-dir",
        dest="conditioning_cache_dir",
        type=str,
        default=None,
        help="Directory used to persist PXRD conditioning caches.",
    )
    parser.add_argument(
        "--precompute-conditioning-batch-size",
        dest="precompute_conditioning_batch_size",
        type=int,
        default=None,
        help="Batch size employed when precomputing conditioning vectors.",
    )
    parser.add_argument(
        "--conditioning-kwargs",
        type=str,
        default=None,
        help="JSON-encoded dictionary of parameters passed to the XRD conditioning routine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
