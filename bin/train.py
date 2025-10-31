#!/usr/bin/env python3

"""
Adapted from:
nanoGPT: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
CrystaLLM: https://github.com/lantunes/CrystaLLM/blob/main/bin/train.py
"""
import os
import copy
import math
import time
import yaml
import random
import pickle

from typing import List, Optional
import argparse

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

from dataclasses import dataclass, field
from contextlib import nullcontext
from tqdm.auto import tqdm

from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import discrete_to_continuous_xrd
from decifer.decifer_dataset import DeciferDataset
    
# Tokenizer, get start, padding and newline IDs
TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]

class RandomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Each time __iter__ is called, radomize the batch indices
        batch_indices = list(self.sampler)
        random.shuffle(batch_indices)

        # Return batches of size batch_Size
        for i in range(0, len(batch_indices), self.batch_size):
            yield batch_indices[i:i + self.batch_size]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def setup_distributed(config: "TrainConfig"):
    """Initialise torch.distributed if requested and return env details."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_distributed = (config.distributed or env_world_size > 1) and env_world_size > 1

    if config.distributed and env_world_size == 1 and env_rank == 0:
        print("[WARN] Distributed training requested but only one process detected; running in single-process mode.", flush=True)

    local_rank = env_local_rank
    if use_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        config.device = f"cuda:{local_rank}"
    elif config.device.startswith("cuda") and not torch.cuda.is_available():
        config.device = "cpu"

    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            config.device = f"cuda:{local_rank}"
    else:
        world_size = 1
        rank = env_rank
        local_rank = env_local_rank

    return use_distributed, world_size, rank, local_rank

@dataclass
class TrainConfig:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'
    resume_from_best: bool = False  # when resuming, optionally load the best checkpoint instead of the latest
    resume_from_best: bool = False  # when resuming, optionally load the best checkpoint instead of the latest

    # data
    dataset: str = ""  # Path to the dataset hdf5 files
    dataset_fraction: float = 1.0  # proportion of each split to use during training (0 < fraction <= 1)
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters
    cond_size: int = 1000
    accumulative_pbar: bool = False
    num_workers_dataloader: int = 0 # Default; single process

    # deCIFer model
    block_size: int = 1024
    vocab_size: int = 372 # Excluding conditioning token
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    boundary_masking: bool = True

    # PXRD embedder
    condition: bool = False
    condition_embedder_hidden_layers: List[int] = field(default_factory=lambda: [512])
    # Augmentation at training time
    qmin: float = 0.0
    qmax: float = 10.0
    qstep: float = 0.01
    wavelength: str = "CuKa"
    fwhm_range_min: float = 0.001 
    fwhm_range_max: float = 0.05
    eta_range_min: float = 0.5
    eta_range_max: float = 0.5
    noise_range_min: float = 0.001
    noise_range_max: float = 0.05
    intensity_scale_range_min: float = 1.0
    intensity_scale_range_max: float = 1.0
    mask_prob: float = 0.0

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 50_000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster (Not supported for deCIFer currently)
    validate: bool = False  # whether to evaluate the model using the validation set
    seed: int = 1337
    distributed: bool = False  # enable DistributedDataParallel training
    dist_backend: str = "nccl"  # backend used by torch.distributed
    dist_url: str = "env://"  # init method (default uses torchrun provided env vars)
    use_tensorboard: bool = False
    tensorboard_log_dir: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 50

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to .yaml config file")
    args = parser.parse_args()

    C = OmegaConf.structured(TrainConfig())

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Parse yaml to namespace and merge (DictConfig)
        yaml_dictconfig = OmegaConf.create(yaml_config)
        C = OmegaConf.merge(C, yaml_dictconfig)
    
    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("Using configuration:", flush=True)
        print(OmegaConf.to_yaml(C))
        print(f"Creating {C.out_dir}...", flush=True)
    os.makedirs(C.out_dir, exist_ok=True)
    if C.use_tensorboard and not C.tensorboard_log_dir:
        C.tensorboard_log_dir = os.path.join(C.out_dir, "tensorboard")
    if C.use_tensorboard and rank == 0:
        os.makedirs(C.tensorboard_log_dir, exist_ok=True)

    # Get metadata (vocab size)
    # metadata_path = os.path.join(C.dataset, "metadata.json")
    # with open(metadata_path, "r") as f:
    #     metadata = json.load(f)
    # try:
    #     print(metadata)
    #     C.vocab_size = metadata["vocab_size"]
    #     print(f"Found vocab_size = {C.vocab_size} in {metadata_path}", flush=True)
    # except:
    #     print(f"No metadata for vocab_size found, defaulting to {C.vocab_size}...")
    C.vocab_size = VOCAB_SIZE

    return C

def setup_datasets(C, distributed=False, show_progress=True, *, rank: int = 0, world_size: int = 1):
    
    # Custom collate function
    def collate_fn(batch):
        # batch is a list of dictionaries
        batch_data = {}
        for key in batch[0].keys():
            field_data = [item[key] for item in batch]
            # Pad the sequences to the maximum length in the batch
            if "xrd" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=0.0)
                batch_data[key] = padded_seqs
            elif "cif" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=PADDING_ID)
                batch_data[key] = padded_seqs
            else:
                batch_data[key] = field_data  # Leave 

        return batch_data
    
    # Collect relevant data
    dataset_fields = ["cif_tokens", "xrd.q", "xrd.iq"]

    # Initialise datasets/loaders
    train_h5 = os.path.join(C.dataset, "serialized/train.h5")
    val_h5 = os.path.join(C.dataset, "serialized/val.h5")
    test_h5 = os.path.join(C.dataset, "serialized/test.h5")

    def _build_dataset(split_name: str, h5_path: str, desc: str) -> DeciferDataset:
        return DeciferDataset(
            h5_path,
            dataset_fields,
            progress_desc=desc,
            show_progress=show_progress,
        )

    train_dataset = _build_dataset("train", train_h5, "train dataset")
    val_dataset = _build_dataset("val", val_h5, "validation dataset")
    test_dataset = _build_dataset("test", test_h5, "test dataset")

    dataset_fraction = float(getattr(C, "dataset_fraction", 1.0))
    if not (0.0 < dataset_fraction <= 1.0):
        raise ValueError("dataset_fraction must be in the interval (0, 1].")

    def apply_fraction(dataset):
        if dataset_fraction >= 1.0:
            return dataset
        subset_length = max(1, int(math.ceil(len(dataset) * dataset_fraction)))
        indices = list(range(subset_length))
        return Subset(dataset, indices)

    train_dataset = apply_fraction(train_dataset)
    val_dataset = apply_fraction(val_dataset)
    test_dataset = apply_fraction(test_dataset)
        
    # Random batching sampler, train
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    else:
        train_sampler = SubsetRandomSampler(range(len(train_dataset)))
    train_batch_sampler = RandomBatchSampler(train_sampler, batch_size=C.batch_size, drop_last=False)
    pin_memory = torch.cuda.is_available() and str(C.device).startswith("cuda")
    loader_kwargs = {
        "batch_sampler": train_batch_sampler,
        "num_workers": C.num_workers_dataloader,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if C.num_workers_dataloader > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    train_dataloader = DataLoader(train_dataset, **loader_kwargs)

    # Random batching sampler, val
    if distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        val_sampler = SubsetRandomSampler(range(len(val_dataset)))
    val_batch_sampler = RandomBatchSampler(val_sampler, batch_size=C.batch_size, drop_last=False)
    val_loader_kwargs = {
        "batch_sampler": val_batch_sampler,
        "num_workers": C.num_workers_dataloader,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if C.num_workers_dataloader > 0:
        val_loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)

    # Random batching sampler, test
    if distributed:
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
            drop_last=False,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        test_sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_batch_sampler = RandomBatchSampler(test_sampler, batch_size=C.batch_size, drop_last=False)
    test_loader_kwargs = {
        "batch_sampler": test_batch_sampler,
        "num_workers": C.num_workers_dataloader,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if C.num_workers_dataloader > 0:
        test_loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    test_dataloader = DataLoader(test_dataset, **test_loader_kwargs)

    # Combine loaders for easy access
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }
    samplers = {
        "train": train_sampler if distributed else None,
        "val": val_sampler,
        "test": test_sampler,
    }

    return dataloaders, samplers

if __name__ == "__main__":

    # Parse configuration
    C = parse_config()
    resume_from_best = bool(getattr(C, "resume_from_best", False))
    resume_from_best = bool(getattr(C, "resume_from_best", False))

    # Setup distributed environment (no-op when single process)
    use_distributed, world_size, rank, local_rank = setup_distributed(C)
    is_main_process = rank == 0

    # Set seed (different per rank for data shuffling)
    if C.seed is not None:
        seed = C.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Setup device-specific context
    device_type = "cuda" if C.device.startswith("cuda") else C.device
    if device_type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = torch.cuda.amp.autocast(dtype=ptdtype) if device_type == "cuda" and torch.cuda.is_available() else nullcontext()

    writer = None
    if C.use_tensorboard and is_main_process:
        if SummaryWriter is None:
            print("[WARN] TensorBoard SummaryWriter not available; install the 'tensorboard' package to enable logging.", flush=True)
        else:
            writer = SummaryWriter(log_dir=C.tensorboard_log_dir)
            print(f"TensorBoard logging to {C.tensorboard_log_dir}", flush=True)

    # Setup datasets
    dataloaders, samplers = setup_datasets(
        C,
        distributed=use_distributed,
        show_progress=is_main_process,
        rank=rank,
        world_size=world_size,
    )
    sampler_epochs = {split: 0 for split in dataloaders.keys()}

    # Augmentation kwargs
    augmentation_kwargs = {
        'qmin': C.qmin,
        'qmax': C.qmax,
        'qstep': C.qstep,
        'fwhm_range': (C.fwhm_range_min, C.fwhm_range_max),
        'eta_range': (C.eta_range_min, C.eta_range_max),
        'noise_range': (C.noise_range_min, C.noise_range_max),
        'intensity_scale_range': (C.intensity_scale_range_min, C.intensity_scale_range_max),
        'mask_prob': C.mask_prob,
    }

    # Initialize training metrics
    training_metrics = {
        'iteration_number': 0,
        'patience_counter': 0,
        'best_val_loss': float('inf'),
        'train_losses': [],
        'val_losses': [],
        'epochs': [],
    }

    # Set model arguments
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head,
        n_embd=C.n_embd,
        block_size=C.block_size,
        condition_size=len(np.arange(C.qmin, C.qmax, C.qstep)),
        bias=C.bias,
        vocab_size=C.vocab_size,
        dropout=C.dropout,
        condition=C.condition,
        boundary_masking=C.boundary_masking,
        condition_embedder_hidden_layers=C.condition_embedder_hidden_layers,
    )

    if C.init_from == "scratch":
        if is_main_process:
            print("Initializing a new model from scratch...", flush=True)
        model = Decifer(DeciferConfig(**model_args))

        checkpoint = {
            'model_args': model_args,
            'training_metrics': training_metrics,
            'best_model_state': None,
            'best_optimizer_state': None,
            "local_iteration_number": 0,
            'config': dict(C),
        }

    elif C.init_from == "resume":
        if is_main_process:
            print(f"Resuming training from {C.out_dir}...", flush=True)

        # Find checkpoint
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        try:
            from torch.serialization import add_safe_globals
            from omegaconf.listconfig import ListConfig
            add_safe_globals([ListConfig])
        except Exception as err:
            if is_main_process:
                print(f"[WARN] Could not register safe globals for checkpoint loading: {err}", flush=True)

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except pickle.UnpicklingError as err:
            raise RuntimeError(
                "Failed to load checkpoint. If the file is trusted, delete it or "
                "re-run with a checkpoint generated using PyTorch >=2.6"
            ) from err
        checkpoint_model_args = checkpoint["model_args"]

        # Force these config attributes to be equal otherwise we can't even resume training
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        # Init model and load state dict
        model = Decifer(DeciferConfig(**model_args))
        state_dict = checkpoint.get('current_model')
        if resume_from_best:
            best_state = checkpoint.get('best_model_state')
            if best_state:
                if is_main_process:
                    print("Loading best model weights from checkpoint.", flush=True)
                state_dict = best_state
            elif is_main_process:
                print("Best model weights not found in checkpoint; falling back to latest state.", flush=True)
        if state_dict is None:
            raise RuntimeError("Checkpoint is missing model weights required to resume training.")
        state_dict = copy.deepcopy(state_dict)
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        # Update checkpoint
        for key in ['train_losses', 'val_losses', 'epochs']:
            if key in checkpoint['training_metrics']:
                training_metrics[key] = checkpoint['training_metrics'][key]
                if is_main_process:
                    print(f"Loaded {key}.")
            else:
                if is_main_process:
                    print(f"Could not find {key}, creating empty list")
        training_metrics['iteration_number'] = checkpoint["training_metrics"]["iteration_number"]
        training_metrics['best_val_loss'] = checkpoint["training_metrics"]["best_val_loss"]
    else:
        raise Exception(f"[init_from] '{C.init_from}' not recognized")

    # Send model to device
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and C.dtype == "float16" and torch.cuda.is_available()))

    # Initialize Optimizer
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer_state = checkpoint.get("current_optimizer")
        if resume_from_best:
            best_optimizer_state = checkpoint.get("best_optimizer_state")
            if best_optimizer_state:
                if is_main_process:
                    print("Loading best optimizer state from checkpoint.", flush=True)
                optimizer_state = best_optimizer_state
            elif is_main_process:
                print("Best optimizer state not found in checkpoint; falling back to latest state.", flush=True)
        if optimizer_state is None:
            raise RuntimeError("Checkpoint is missing optimizer state required to resume training.")
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    # Compile model (pytorch 2.0) if specified
    if C.compile:
        if is_main_process:
            print("Compiling the model (takes a ~minute)...", flush=True)
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # Wrap model with DistributedDataParallel if needed
    if use_distributed:
        ddp_kwargs = {}
        if device_type == "cuda" and torch.cuda.is_available():
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DDP(model, **ddp_kwargs)

    # Initialize a dictionary to keep data iterators per split
    data_iters = {}

    def get_batch(split, *, show_progress=False):

        # Retrieve the dataloader and initialize the iterator
        dataloader = dataloaders[split]
        if split not in data_iters:
            if samplers.get(split) is not None and hasattr(samplers[split], "set_epoch"):
                samplers[split].set_epoch(sampler_epochs[split])
            data_iters[split] = iter(dataloader)
        data_iter = data_iters[split]

        use_cuda = device_type == "cuda" and torch.cuda.is_available()
        if use_cuda:
            target_device = torch.device(C.device)
        else:
            target_device = torch.device("cpu")

        def _move_to_target(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device == target_device:
                return tensor
            if use_cuda:
                if tensor.device.type == "cpu":
                    return tensor.pin_memory().to(target_device, non_blocking=True)
                return tensor.to(target_device, non_blocking=True)
            return tensor.to(target_device)

        # Initialize lists to store packed sequences and start indices
        start_indices_list = []
        cond_list = []

        # Collect sequences until we have enough tokens to form at least one full block
        total_sequences = []
        total_token_count = 0
        max_extra_fetches = 5
        extra_fetches = 0
        seq_progress = None
        if show_progress and tqdm is not None:
            seq_progress = tqdm(
                total=C.batch_size,
                desc=f"Preparing first {split} batch",
                leave=False,
            )
        last_progress = -0.1

        def report_progress():
            nonlocal last_progress
            if not show_progress:
                return
            seq_ratio = len(total_sequences) / max(1, C.batch_size)
            token_ratio = total_token_count / max(1, C.block_size)
            progress_ratio = min(1.0, max(seq_ratio, token_ratio))
            if progress_ratio - last_progress >= 0.1 or progress_ratio >= 1.0:
                print(
                    f"[INFO] Collecting {split} batch: sequences {len(total_sequences)}/{C.batch_size}, "
                    f"tokens {total_token_count}/{C.block_size}",
                    flush=True,
                )
                last_progress = progress_ratio
        while total_token_count < C.block_size or len(total_sequences) < C.batch_size:
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler_epochs[split] += 1
                if samplers.get(split) is not None and hasattr(samplers[split], "set_epoch"):
                    samplers[split].set_epoch(sampler_epochs[split])
                data_iter = iter(dataloader)
                data_iters[split] = data_iter
                batch = next(data_iter)

            # Fetch sequences and remove padding tokens
            sequences = batch['cif_tokens']
            sequences = [torch.cat([seq[seq != PADDING_ID], torch.tensor([NEWLINE_ID, NEWLINE_ID], dtype=torch.long)]) for seq in sequences]
            total_sequences.extend(sequences)
            total_token_count += sum(int(seq.numel()) for seq in sequences)
            if seq_progress is not None:
                update_amount = min(len(sequences), C.batch_size - seq_progress.n)
                if update_amount > 0:
                    seq_progress.update(update_amount)
            report_progress()

            # Fetch conditioning and augment to cont signals
            if C.condition:
                xrd_q = batch['xrd.q']
                xrd_iq = batch['xrd.iq']
                if isinstance(xrd_q, torch.Tensor):
                    xrd_q = _move_to_target(xrd_q)
                if isinstance(xrd_iq, torch.Tensor):
                    xrd_iq = _move_to_target(xrd_iq)
                with torch.no_grad():
                    cond_entries = discrete_to_continuous_xrd(xrd_q, xrd_iq, **augmentation_kwargs)['iq']
                if isinstance(cond_entries, torch.Tensor):
                    cond_entries = _move_to_target(cond_entries)
                    cond_list.extend(cond_entries)
                else:
                    for entry in cond_entries:
                        if isinstance(entry, torch.Tensor):
                            cond_list.append(_move_to_target(entry))
                        else:
                            cond_list.append(entry)

        if not total_sequences:
            raise RuntimeError("Failed to collect any sequences for batching.")

        # Attempt to ensure we can form at least one complete block
        while True:
            if not total_sequences:
                raise RuntimeError(f"Unable to collect tokens for split '{split}'.")

            all_tokens = torch.cat(total_sequences)
            if all_tokens.numel() == 0:
                raise RuntimeError(f"Collected tensor for split '{split}' is empty; check dataset integrity.")

            num_full_blocks = all_tokens.size(0) // C.block_size
            num_batches = min(C.batch_size, num_full_blocks)

            if num_batches > 0:
                break

            if extra_fetches < max_extra_fetches:
                extra_fetches += 1
                try:
                    batch = next(data_iter)
                except StopIteration:
                    sampler_epochs[split] += 1
                    if samplers.get(split) is not None and hasattr(samplers[split], "set_epoch"):
                        samplers[split].set_epoch(sampler_epochs[split])
                    data_iter = iter(dataloader)
                    data_iters[split] = data_iter
                    batch = next(data_iter)

                sequences = batch['cif_tokens']
                sequences = [torch.cat([seq[seq != PADDING_ID], torch.tensor([NEWLINE_ID, NEWLINE_ID], dtype=torch.long)]) for seq in sequences]
                total_sequences.extend(sequences)
                total_token_count += sum(int(seq.numel()) for seq in sequences)
                if seq_progress is not None:
                    update_amount = min(len(sequences), C.batch_size - seq_progress.n)
                    if update_amount > 0:
                        seq_progress.update(update_amount)
                report_progress()
                if C.condition:
                    xrd_q = batch['xrd.q']
                    xrd_iq = batch['xrd.iq']
                    if isinstance(xrd_q, torch.Tensor):
                        xrd_q = _move_to_target(xrd_q)
                    if isinstance(xrd_iq, torch.Tensor):
                        xrd_iq = _move_to_target(xrd_iq)
                    with torch.no_grad():
                        cond_entries = discrete_to_continuous_xrd(xrd_q, xrd_iq, **augmentation_kwargs)['iq']
                    if isinstance(cond_entries, torch.Tensor):
                        cond_entries = _move_to_target(cond_entries)
                        cond_list.extend(cond_entries)
                    else:
                        for entry in cond_entries:
                            if isinstance(entry, torch.Tensor):
                                cond_list.append(_move_to_target(entry))
                            else:
                                cond_list.append(entry)
                continue

            # Final fallback: repeat collected sequences to satisfy the required block size
            if is_main_process:
                print(
                    f"[WARN] {split}: insufficient tokens to form block_size={C.block_size} after {max_extra_fetches} extra fetches. "
                    "Repeating collected sequences to proceed.",
                    flush=True,
                )

            tokens_available = all_tokens.size(0)
            if tokens_available == 0:
                raise RuntimeError(
                    f"Unable to form a full block for split '{split}'; no tokens collected after retries."
                )

            repeat_factor = max(1, math.ceil(C.block_size / tokens_available))
            total_sequences = total_sequences * repeat_factor
            if C.condition and cond_list:
                cond_list = cond_list * repeat_factor
            total_token_count = sum(int(seq.numel()) for seq in total_sequences)
            # Loop back to recompute using the expanded sequence list

        # With a sufficient number of tokens, pack sequences into batches without explicit loops
        all_tokens = torch.cat(total_sequences)

        if seq_progress is not None:
            seq_progress.close()

        report_progress()

        # Compute the lengths of sequences
        seq_lengths = torch.tensor([len(seq) for seq in total_sequences])

        # Compute cumulative lengths to find sequence boundaries
        seq_cum_lengths = torch.cumsum(seq_lengths, dim=0)

        if num_batches == 0:
            raise RuntimeError(
                f"Unable to form a full block for split '{split}' even after retries; "
                f"collected_tokens={all_tokens.size(0)}, block_size={C.block_size}."
            )

        # Truncate the tokens to fit into an integer number of blocks
        total_tokens = all_tokens[:num_batches * C.block_size]

        # Reshape the tokens into (num_batches, block_size)
        total_tokens = total_tokens.view(num_batches, C.block_size)

        # Create input (X) and target (Y) sequences
        X_batch = total_tokens[:, :-1]
        Y_batch = total_tokens[:, 1:]

        # Find start indices within each batch
        start_token_mask = X_batch == START_ID
        start_indices = start_token_mask.nonzero(as_tuple=False)

        # Organize start indices per batch item
        start_indices_list = []
        for i in range(num_batches):
            indices = start_indices[start_indices[:, 0] == i][:, 1]
            start_indices_list.append(indices)

        # Handle conditioning data if required
        cond_batch = None
        if C.condition:
            tokens_used = num_batches * C.block_size
            if tokens_used == 0:
                raise RuntimeError("Conditioning requested but no tokens were allocated to the batch.")

            search_value = torch.tensor(tokens_used, device=seq_cum_lengths.device)
            search_index = int(torch.searchsorted(seq_cum_lengths, search_value, right=False).item())
            num_sequences_used = min(search_index + 1, len(cond_list))

            if num_sequences_used == 0:
                raise RuntimeError(
                    "Conditioning requested but no conditioning vectors were gathered;"
                    " verify dataset annotations."
                )

            cond_list = cond_list[:num_sequences_used]
            cond_batch = torch.stack(cond_list)

            required_conditionals = sum(len(indices) for indices in start_indices_list)
            if required_conditionals == 0:
                cond_batch = None
            else:
                if cond_batch.size(0) < required_conditionals:
                    raise RuntimeError(
                        f"Conditioning vectors ({cond_batch.size(0)}) fewer than required insertions ({required_conditionals})."
                    )
                if cond_batch.size(0) > required_conditionals:
                    cond_batch = cond_batch[:required_conditionals]

        # Send to device (CUDA/CPU)
        X_batch = _move_to_target(X_batch)
        Y_batch = _move_to_target(Y_batch)
        if cond_batch is not None:
            cond_batch = _move_to_target(cond_batch)

        return X_batch, Y_batch, cond_batch, start_indices_list

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        loss_device = torch.device(C.device) if device_type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            if eval_iters <= 0:
                out[split] = 0.0
                continue

            total_loss = torch.zeros(1, device=loss_device, dtype=torch.float32)
            for _ in range(eval_iters):
                X, Y, cond, start_indices = get_batch(split)
                with ctx:
                    _, loss = model(X, cond, Y, start_indices)
                total_loss += loss.detach().to(loss_device, dtype=torch.float32)

            total_loss /= eval_iters
            if use_distributed:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                total_loss /= world_size

            out[split] = total_loss.item()
        model.train()
        return out

    def get_lr(it):
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        if it > C.lr_decay_iters:
            return C.min_lr
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    if is_main_process:
        print("Preparing first training batch (may take a moment)...", flush=True)
    X, Y, cond, start_indices = get_batch("train", show_progress=is_main_process)
    if is_main_process:
        print("First training batch ready.", flush=True)
    t0 = time.time()
    local_iteration_number = 0
    stop_training = False
    while not stop_training:
        lr = get_lr(training_metrics['iteration_number']) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if training_metrics['iteration_number'] % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()
                if is_main_process:
                    training_metrics['train_losses'].append(losses['train'])
                    training_metrics['val_losses'].append(losses['val'])
                    training_metrics['epochs'].append(training_metrics['iteration_number'])
                    print(
                        f"step {training_metrics['iteration_number']}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}",
                        flush=True,
                    )
                    if writer is not None:
                        writer.add_scalar("eval/train_loss", losses['train'], training_metrics['iteration_number'])
                        writer.add_scalar("eval/val_loss", losses['val'], training_metrics['iteration_number'])

                    if losses["val"] > training_metrics['best_val_loss'] and local_iteration_number != 0:
                        training_metrics['patience_counter'] += 1
                        print("Patience score increasing to:", training_metrics['patience_counter'])
                    else:
                        training_metrics['best_val_loss'] = losses['val']
                        checkpoint['best_model_state'] = copy.deepcopy(unwrap_model(model).state_dict())
                        checkpoint['best_optimizer_state'] = copy.deepcopy(optimizer.state_dict())
                        if training_metrics['patience_counter'] > 0:
                            print("Patience score resetting.")
                        training_metrics['patience_counter'] = 0

                    if training_metrics['iteration_number'] > 0:
                        checkpoint.update({
                            "local_iteration_number": local_iteration_number,
                            'training_metrics': training_metrics,
                            'current_model': unwrap_model(model).state_dict(),
                            "current_optimizer": optimizer.state_dict(),
                        })
                        print(f"saving checkpoint to {C.out_dir}...", flush=True)
                        torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))

                    if training_metrics['patience_counter'] >= C.early_stopping_patience:
                        print(f"Early stopping triggered after {training_metrics['iteration_number']} iterations")
                        stop_training = True
            else:
                training_metrics['best_val_loss'] = 0.

        if training_metrics['iteration_number'] == 0 and C.eval_only:
            stop_training = True

        if use_distributed:
            stop_tensor = torch.tensor([int(stop_training)], device=torch.device(C.device) if device_type == "cuda" and torch.cuda.is_available() else torch.device("cpu"))
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

        pbar_disabled = not (C.accumulative_pbar and is_main_process)
        small_step_pbar = tqdm(desc='Accumulating losses...', total=C.gradient_accumulation_steps, leave=False, disable=pbar_disabled)
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, cond, Y, start_indices)

            X, Y, cond, start_indices = get_batch("train")
            scaler.scale(loss).backward()
            small_step_pbar.update(1)

        small_step_pbar.close()
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if training_metrics['iteration_number'] % C.log_interval == 0 and is_main_process:
            lossf = loss.item()
            print(f"iter {training_metrics['iteration_number']}: loss {lossf:.4f}, time {dt * 1000:.2f}ms", flush=True)
            if writer is not None:
                writer.add_scalar("train/loss", lossf, training_metrics['iteration_number'])
                writer.add_scalar("train/lr", lr, training_metrics['iteration_number'])
        training_metrics['iteration_number'] += 1
        local_iteration_number += 1

        if training_metrics['iteration_number'] > C.max_iters:
            stop_training = True
            if use_distributed:
                stop_tensor = torch.tensor([1], device=torch.device(C.device) if device_type == "cuda" and torch.cuda.is_available() else torch.device("cpu"))
                dist.broadcast(stop_tensor, src=0)
            break

    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if writer is not None:
        writer.flush()
        writer.close()
