"""Training configuration dataclasses for supervised deCIFer workflows."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """Configuration parameters for supervised pretraining."""

    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # Path to the dataset hdf5 files
    dataset_fraction: float = 1.0  # proportion of each split to use during training (0 < fraction <= 1)
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters
    cond_size: int = 1000
    accumulative_pbar: bool = False
    num_workers_dataloader: int = 0  # Default; single process

    # deCIFer model
    block_size: int = 1024
    vocab_size: int = 372  # Excluding conditioning token
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
