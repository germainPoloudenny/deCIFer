"""Group Relative Policy Optimization utilities for deCIFer."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from pymatgen.analysis.structure_matcher import StructureMatcher

from decifer.decifer_dataset import DeciferDataset
from decifer.decifer_model import Decifer, DeciferConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    discrete_to_continuous_xrd,
    extract_space_group_symbol,
    generate_continuous_xrd_from_cif,
    get_rmsd,
    is_sensible,
    reinstate_symmetry_loop,
    replace_symmetry_loop_with_P1,
)

TOKENIZER = Tokenizer()
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]
SPACEGROUP_ID = TOKENIZER.token_to_id["_symmetry_space_group_name_H-M"]


def _ensure_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def extract_prompt(
    sequence: torch.Tensor,
    *,
    add_composition: bool = True,
    add_spacegroup: bool = False,
) -> torch.Tensor:
    """Extract the model prompt from a tokenized CIF sequence.

    The prompt contains the ``data_`` header (and optionally composition and
    space-group information) that conditions the autoregressive decoding.
    """

    if sequence.dim() != 1:
        sequence = sequence.view(-1)

    start_matches = (sequence == START_ID).nonzero(as_tuple=True)
    if len(start_matches[0]) == 0:
        raise ValueError("'data_' token not found in sequence")

    end_prompt_index = int(start_matches[0][0].item()) + 1

    if add_composition:
        newline_matches = (sequence[end_prompt_index:] == NEWLINE_ID).nonzero(as_tuple=True)
        if len(newline_matches[0]) == 0:
            raise ValueError("Prompt does not contain composition newline terminator")
        end_prompt_index += int(newline_matches[0][0].item())

        if add_spacegroup:
            spacegroup_matches = (sequence[end_prompt_index:] == SPACEGROUP_ID).nonzero(as_tuple=True)
            if len(spacegroup_matches[0]) == 0:
                raise ValueError("Prompt missing space-group identifier")
            end_prompt_index += int(spacegroup_matches[0][0].item())

            newline_matches = (sequence[end_prompt_index:] == NEWLINE_ID).nonzero(as_tuple=True)
            if len(newline_matches[0]) == 0:
                raise ValueError("Prompt missing space-group newline terminator")
            end_prompt_index += int(newline_matches[0][0].item())

        end_prompt_index += 1

    return sequence[:end_prompt_index].clone()


def prepare_prompt_and_conditioning(
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    add_composition: bool,
    add_spacegroup: bool,
    conditioning_kwargs: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare prompt tokens and conditioning vector for a dataset sample."""

    prompt = extract_prompt(
        sample["cif_tokens"],
        add_composition=add_composition,
        add_spacegroup=add_spacegroup,
    ).unsqueeze(0).to(device)

    if "xrd_cont" in sample:
        cond_vec = sample["xrd_cont"].unsqueeze(0).to(device)
    else:
        conditioning_kwargs = conditioning_kwargs or {}
        xrd = discrete_to_continuous_xrd(
            sample["xrd.q"].unsqueeze(0).to(device),
            sample["xrd.iq"].unsqueeze(0).to(device),
            **conditioning_kwargs,
        )
        cond_vec = xrd["iq"].to(device)

    return prompt, cond_vec


@dataclass
class GRPOConfig:
    """Configuration for GRPO fine-tuning."""

    out_dir: str = "out_grpo"
    dataset: str = ""
    dataset_split: str = "train"
    init_checkpoint: str = ""
    reference_checkpoint: Optional[str] = None
    batch_size: int = 2
    group_size: int = 4
    data_parallel: bool = False
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: Optional[int] = None
    learning_rate: float = 5e-6
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    update_epochs: int = 4
    mini_batch_size: Optional[int] = None
    clip_range: float = 0.2
    kl_coef: float = 0.02
    target_kl: float = 0.1
    invalid_reward: float = -5.0
    reward_scale: float = 1.0
    fallback_reward_scale: float = 1.0
    normalize_advantages: bool = True
    max_iterations: int = 1000
    log_interval: int = 10
    save_interval: int = 200
    add_composition: bool = True
    add_spacegroup: bool = False
    conditioning_kwargs: Dict[str, float] = field(default_factory=dict)
    device: str = "cuda"
    dtype: str = "float32"
    num_workers: int = 0
    seed: int = 1337

    def resolved_device(self) -> torch.device:
        return _ensure_device(self.device)

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        return torch.float32

    def effective_mini_batch_size(self) -> int:
        if self.mini_batch_size is not None:
            return self.mini_batch_size
        return max(1, self.batch_size * self.group_size)


@dataclass
class RolloutSample:
    """Single trajectory collected for GRPO updates."""

    sequence: torch.Tensor
    cond_vec: torch.Tensor
    prompt_length: int
    sequence_length: int
    old_logprob: float
    reward: float
    advantage: float
    rmsd: Optional[float]
    is_valid: bool


class RMSDRewardScorer:
    """Compute RMSD-based rewards for generated CIF structures."""

    def __init__(
        self,
        invalid_reward: float = -5.0,
        *,
        fallback_scale: float = 1.0,
        xrd_kwargs: Optional[Dict[str, float]] = None,
    ):
        self.tokenizer = TOKENIZER
        self.invalid_reward = invalid_reward
        self.fallback_scale = fallback_scale
        self.xrd_kwargs = dict(xrd_kwargs or {})
        self.matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        self.supercell_matcher = StructureMatcher(
            stol=0.5,
            angle_tol=10,
            ltol=0.3,
            attempt_supercell=True,
            supercell_size="minimize",
        )

    def score(
        self,
        reference_cif: str,
        generated_tokens: Sequence[int],
        *,
        reference_xrd: Optional[torch.Tensor] = None,
        xrd_kwargs: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, Optional[float]]]:
        """Return a GRPO reward based on RMSD similarity."""

        cif_string_gen = self.tokenizer.decode(list(generated_tokens))
        cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)
        spacegroup_symbol = extract_space_group_symbol(cif_string_gen)
        if spacegroup_symbol != "P 1":
            cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)

        if not is_sensible(cif_string_gen):
            return self.invalid_reward, {"rmsd": None, "mode": None}

        rmsd, mode = get_rmsd(
            reference_cif,
            cif_string_gen,
            matcher=self.matcher,
            supercell_matcher=self.supercell_matcher,
        )
        if rmsd is None:
            fallback = self._score_with_rwp(
                reference_xrd,
                cif_string_gen,
                xrd_kwargs=xrd_kwargs,
            )
            if fallback is not None:
                reward_value, rwp_value = fallback
                return reward_value, {"rmsd": None, "mode": "rwp", "rwp": rwp_value}
            return self.invalid_reward, {"rmsd": None, "mode": mode, "rwp": None}

        reward = -float(rmsd) * 1.0
        return reward, {"rmsd": float(rmsd), "mode": mode, "rwp": None}

    def _score_with_rwp(
        self,
        reference_xrd: Optional[torch.Tensor],
        cif_string_gen: str,
        *,
        xrd_kwargs: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return a reward based on the residual weighted profile metric."""

        if reference_xrd is None:
            return None

        kwargs: Dict[str, object] = {}
        kwargs.update(self.xrd_kwargs)
        if xrd_kwargs:
            kwargs.update(xrd_kwargs)

        deterministic_kwargs: Dict[str, object] = {}
        for key in ("qmin", "qmax", "qstep"):
            if key in kwargs:
                deterministic_kwargs[key] = kwargs[key]

        if "fwhm_range" in kwargs:
            fwhm_range = kwargs["fwhm_range"]
            if isinstance(fwhm_range, (list, tuple)) and len(fwhm_range) == 2:
                midpoint = float(sum(fwhm_range) / 2.0)
                deterministic_kwargs["fwhm_range"] = (midpoint, midpoint)
        if "eta_range" in kwargs:
            eta_range = kwargs["eta_range"]
            if isinstance(eta_range, (list, tuple)) and len(eta_range) == 2:
                midpoint = float(sum(eta_range) / 2.0)
                deterministic_kwargs["eta_range"] = (midpoint, midpoint)

        deterministic_kwargs.setdefault("noise_range", None)
        deterministic_kwargs.setdefault("mask_prob", None)
        deterministic_kwargs.setdefault("intensity_scale_range", (1.0, 1.0))

        xrd_gen = generate_continuous_xrd_from_cif(
            cif_string_gen,
            **deterministic_kwargs,
        )
        if xrd_gen is None:
            return None

        iq_gen = xrd_gen.get("iq")
        if iq_gen is None:
            return None
        if isinstance(iq_gen, torch.Tensor):
            iq_gen = iq_gen.detach().cpu().to(dtype=torch.float32)
        else:
            iq_gen = torch.tensor(iq_gen, dtype=torch.float32)

        reference = reference_xrd.detach().cpu().to(dtype=torch.float32)
        if reference.dim() > 1:
            reference = reference.squeeze(0)
        if iq_gen.dim() > 1:
            iq_gen = iq_gen.squeeze(0)

        if reference.numel() == 0 or iq_gen.numel() == 0:
            return None

        if reference.numel() != iq_gen.numel():
            min_len = min(reference.numel(), iq_gen.numel())
            reference = reference[:min_len]
            iq_gen = iq_gen[:min_len]

        denominator = torch.sum(reference ** 2)
        if denominator <= 0:
            return None

        numerator = torch.sum((reference - iq_gen) ** 2)
        rwp_value = torch.sqrt(numerator / denominator).item()
        reward = -float(rwp_value) * self.fallback_scale
        return reward, float(rwp_value)


def _pad_sequences(sequences: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of 1D tensors and return stacked tensor with lengths."""

    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    if not lengths.numel():
        return torch.empty(0, 0, dtype=torch.long), lengths
    max_len = int(lengths.max().item())
    padded = torch.full((len(sequences), max_len), fill_value=PADDING_ID, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq
    return padded, lengths


def _compute_completion_logprobs(
    model: Decifer,
    sequences: torch.Tensor,
    sequence_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    cond_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute summed log-probabilities for the generated completions."""

    device = sequences.device
    if sequences.size(0) == 0:
        empty = torch.empty(0, device=device)
        return empty, empty

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    logits, _ = model(inputs, cond_vec=cond_vec, start_indices_batch=[[0]] * sequences.size(0))
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    max_positions = targets.size(1)
    position_indices = torch.arange(max_positions, device=device).unsqueeze(0)
    valid_mask = position_indices < (sequence_lengths - 1).unsqueeze(1)
    completion_mask = position_indices >= (prompt_lengths - 1).unsqueeze(1)
    completion_mask = completion_mask & valid_mask
    completion_logprob = (selected * completion_mask).sum(dim=1)
    return completion_logprob, completion_mask


class GRPOTrainer:
    """Run Group Relative Policy Optimization to fine-tune deCIFer."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = config.resolved_device()
        self.use_data_parallel = self._should_use_data_parallel()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        os.makedirs(self.config.out_dir, exist_ok=True)

        dataset_path = self._resolve_dataset_path(self.config.dataset, self.config.dataset_split)

        data_keys = ["cif_tokens", "xrd.q", "xrd.iq", "cif_string", "cif_name"]
        self.dataset = DeciferDataset(
            dataset_path,
            data_keys,
            precompute_conditioning=False,
        )
        sampler = RandomSampler(self.dataset)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        self.data_iterator = iter(self.data_loader)

        self.model, self.model_args = self._load_model(self.config.init_checkpoint)
        self.model.to(self.device)
        if self.use_data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.train()

        ref_checkpoint = self.config.reference_checkpoint or self.config.init_checkpoint
        self.reference_model, _ = self._load_model(ref_checkpoint)
        self.reference_model.to(self.device)
        if self.use_data_parallel:
            self.reference_model = torch.nn.DataParallel(self.reference_model)
        self.reference_model.eval()
        for param in self._unwrap_model(self.reference_model).parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        self.reward_scorer = RMSDRewardScorer(
            self.config.invalid_reward,
            fallback_scale=self.config.fallback_reward_scale,
            xrd_kwargs=self.config.conditioning_kwargs,
        )

        self.iteration = 0
        self.global_step = 0

    def _should_use_data_parallel(self) -> bool:
        return (
            self.config.data_parallel
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        )

    def _resolve_dataset_path(self, dataset: str, split: str) -> str:
        if not dataset:
            raise ValueError("A dataset path must be provided for GRPO training.")

        resolved_path = os.path.abspath(os.path.expanduser(dataset))

        if os.path.isfile(resolved_path):
            return resolved_path

        if os.path.isdir(resolved_path):
            normalized_split = split or "train"
            candidates = [
                os.path.join(resolved_path, "serialized", f"{normalized_split}.h5"),
                os.path.join(resolved_path, normalized_split, f"{normalized_split}.h5"),
                os.path.join(resolved_path, f"{normalized_split}.h5"),
            ]

            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate

            fallback_dirs = [resolved_path, os.path.join(resolved_path, "serialized")]
            fallback_paths = []
            for directory in fallback_dirs:
                if not os.path.isdir(directory):
                    continue
                for entry in Path(directory).iterdir():
                    if entry.suffix == ".h5" and entry.is_file():
                        fallback_paths.append(str(entry))

            if len(fallback_paths) == 1:
                return fallback_paths[0]

            searched = "\n".join(f" - {path}" for path in candidates + fallback_paths)
            raise FileNotFoundError(
                "Unable to locate an HDF5 dataset file for split "
                f"'{normalized_split}' inside '{resolved_path}'.\n"
                "Checked the following locations:\n"
                f"{searched if searched else '  (no .h5 files found)'}"
            )

        raise FileNotFoundError(
            f"Dataset path '{dataset}' does not exist or is not accessible."
        )

    @staticmethod
    def _unwrap_model(model: Decifer) -> Decifer:
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def _load_model(self, checkpoint_path: str) -> Tuple[Decifer, Dict[str, int]]:
        if not checkpoint_path:
            raise ValueError("An initial checkpoint path must be provided for GRPO training.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = checkpoint["model_args"]
        config = DeciferConfig(**model_args)
        model = Decifer(config)
        state_dict = (
            checkpoint.get("best_model_state")
            or checkpoint.get("best_model")
            or checkpoint.get("model")
            or checkpoint.get("current_model")
        )
        if state_dict is None:
            raise KeyError("Checkpoint does not contain a recognizable model state.")
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        dp_prefix = "module."
        for key in list(state_dict.keys()):
            if key.startswith(dp_prefix):
                state_dict[key[len(dp_prefix) :]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model, model_args

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)
        return batch

    def collect_rollout(self) -> Tuple[List[RolloutSample], Dict[str, float]]:
        self.model.eval()
        samples: List[RolloutSample] = []
        batch = self._next_batch()
        batch_rewards: List[float] = []
        batch_rmsd: List[float] = []
        valid_count = 0

        for idx in range(self.config.batch_size):
            sample_dict = {key: batch[key][idx] for key in batch}
            prompt, cond_vec = prepare_prompt_and_conditioning(
                sample_dict,
                self.device,
                add_composition=self.config.add_composition,
                add_spacegroup=self.config.add_spacegroup,
                conditioning_kwargs=self.config.conditioning_kwargs,
            )
            prompt_length = prompt.size(1)
            cond_vec = cond_vec.to(self.device)

            group_rewards: List[float] = []
            group_rmsd: List[Optional[float]] = []
            trajectories: List[torch.Tensor] = []
            logprobs: List[float] = []
            validity: List[bool] = []

            with torch.no_grad():
                for _ in range(self.config.group_size):
                    generated = self.model.generate(
                        prompt.clone(),
                        self.config.max_new_tokens,
                        cond_vec=cond_vec,
                        start_indices_batch=[[0]],
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        disable_pbar=True,
                    )
                    sequence = generated.squeeze(0).detach().cpu()
                    trajectories.append(sequence)

            padded, lengths = _pad_sequences(trajectories)
            if padded.numel() == 0:
                continue
            padded = padded.to(self.device)
            lengths = lengths.to(self.device)
            prompt_lengths = torch.full_like(lengths, fill_value=prompt_length)
            cond_batch = cond_vec.repeat(padded.size(0), 1)
            reference_pattern = cond_vec.squeeze(0).detach().cpu()

            with torch.no_grad():
                logprob_sum, _ = _compute_completion_logprobs(
                    self.model,
                    padded,
                    lengths,
                    prompt_lengths,
                    cond_batch,
                )

            for seq_idx, sequence in enumerate(trajectories):
                seq_len = int(lengths[seq_idx].item())
                sequence_trimmed = sequence[:seq_len]
                reward, details = self.reward_scorer.score(
                    sample_dict["cif_string"],
                    sequence_trimmed.tolist(),
                    reference_xrd=reference_pattern,
                    xrd_kwargs=self.config.conditioning_kwargs,
                )
                is_valid = details["mode"] != "rwp" and details["rmsd"] is not None
                if is_valid:
                    valid_count += 1
                    batch_rmsd.append(details["rmsd"] or 0.0)
                batch_rewards.append(reward)
                group_rewards.append(reward)
                group_rmsd.append(details["rmsd"])
                validity.append(is_valid)
                logprobs.append(float(logprob_sum[seq_idx].item()))

            rewards_tensor = torch.tensor(group_rewards)
            scaled_rewards = rewards_tensor * self.config.reward_scale
            baseline = scaled_rewards.mean()
            advantages = scaled_rewards - baseline
            if self.config.normalize_advantages:
                std = scaled_rewards.std(unbiased=False)
                if torch.isfinite(std) and std > 1e-6:
                    advantages = advantages / std

            for seq_idx, sequence in enumerate(trajectories):
                seq_len = int(lengths[seq_idx].item())
                samples.append(
                    RolloutSample(
                        sequence=sequence[:seq_len],
                        cond_vec=cond_vec.squeeze(0).cpu(),
                        prompt_length=prompt_length,
                        sequence_length=seq_len,
                        old_logprob=logprobs[seq_idx],
                        reward=float(scaled_rewards[seq_idx].item()),
                        advantage=float(advantages[seq_idx].item()),
                        rmsd=group_rmsd[seq_idx],
                        is_valid=validity[seq_idx],
                    )
                )

        self.model.train()
        metrics = {
            "mean_reward": float(torch.tensor(batch_rewards).mean().item()) if batch_rewards else 0.0,
            "valid_fraction": float(valid_count / max(len(batch_rewards), 1)),
            "mean_rmsd": float(torch.tensor(batch_rmsd).mean().item()) if batch_rmsd else float("nan"),
        }
        return samples, metrics

    def _prepare_minibatch(
        self,
        experiences: Sequence[RolloutSample],
        indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = [experiences[i].sequence for i in indices]
        conds = [experiences[i].cond_vec for i in indices]
        prompt_lengths = torch.tensor([experiences[i].prompt_length for i in indices], device=self.device)
        seq_lengths = torch.tensor([experiences[i].sequence_length for i in indices], device=self.device)
        old_logprobs = torch.tensor([experiences[i].old_logprob for i in indices], device=self.device)
        advantages = torch.tensor([experiences[i].advantage for i in indices], device=self.device)
        padded, _ = _pad_sequences(sequences)
        padded = padded.to(self.device)
        cond_batch = torch.stack(conds).to(self.device)
        return padded, cond_batch, prompt_lengths, seq_lengths, old_logprobs, advantages

    def update_policy(self, experiences: Sequence[RolloutSample]) -> Dict[str, float]:
        if not experiences:
            return {"policy_loss": 0.0}

        mini_batch_size = self.config.effective_mini_batch_size()
        all_indices = list(range(len(experiences)))
        losses: List[float] = []
        approx_kls: List[float] = []
        kl_to_ref: List[float] = []

        for epoch in range(self.config.update_epochs):
            rng = torch.randperm(len(all_indices)).tolist()
            for start in range(0, len(all_indices), mini_batch_size):
                end = start + mini_batch_size
                batch_indices = [all_indices[i] for i in rng[start:end]]
                (
                    padded,
                    cond_batch,
                    prompt_lengths,
                    seq_lengths,
                    old_logprobs,
                    advantages,
                ) = self._prepare_minibatch(experiences, batch_indices)

                logprob_sum, completion_mask = _compute_completion_logprobs(
                    self.model,
                    padded,
                    seq_lengths,
                    prompt_lengths,
                    cond_batch,
                )
                with torch.no_grad():
                    old_logprob_sum, _ = _compute_completion_logprobs(
                        self.reference_model,
                        padded,
                        seq_lengths,
                        prompt_lengths,
                        cond_batch,
                    )

                ratios = torch.exp(logprob_sum - old_logprobs)
                clipped = torch.clamp(ratios, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -torch.mean(torch.min(ratios * advantages, clipped * advantages))

                with torch.no_grad():
                    completion_counts = torch.clamp(completion_mask.sum(dim=1), min=1).to(logprob_sum.dtype)
                    approx_kl = torch.mean((old_logprobs - logprob_sum) / completion_counts)
                ref_diff = (logprob_sum - old_logprob_sum)
                kl_penalty = torch.mean(ref_diff / completion_counts)

                loss = policy_loss + self.config.kl_coef * kl_penalty
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                losses.append(float(loss.item()))
                approx_kls.append(float(approx_kl.item()))
                kl_to_ref.append(float(kl_penalty.item()))

            if approx_kls and abs(approx_kls[-1]) > self.config.target_kl:
                break

        return {
            "policy_loss": float(torch.tensor(losses).mean().item()) if losses else 0.0,
            "approx_kl": float(torch.tensor(approx_kls).mean().item()) if approx_kls else 0.0,
            "kl_to_ref": float(torch.tensor(kl_to_ref).mean().item()) if kl_to_ref else 0.0,
        }

    def save_checkpoint(self) -> None:
        checkpoint_path = os.path.join(self.config.out_dir, "grpo_ckpt.pt")
        model_to_save = self._unwrap_model(self.model)
        torch.save(
            {
                "model_args": self.model_args,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "global_step": self.global_step,
                "config": self.config.__dict__,
            },
            checkpoint_path,
        )

    def train(self) -> None:
        for iteration in range(1, self.config.max_iterations + 1):
            self.iteration = iteration
            experiences, rollout_metrics = self.collect_rollout()
            update_metrics = self.update_policy(experiences)
            self.global_step += len(experiences)

            if iteration % self.config.log_interval == 0:
                print(
                    f"[GRPO] iter={iteration} samples={len(experiences)} "
                    f"mean_reward={rollout_metrics['mean_reward']:.4f} "
                    f"mean_rmsd={rollout_metrics['mean_rmsd']:.4f} "
                    f"valid={rollout_metrics['valid_fraction']:.2%} "
                    f"loss={update_metrics['policy_loss']:.4f} "
                    f"kl={update_metrics['approx_kl']:.4f}",
                    flush=True,
                )

            if iteration % self.config.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
