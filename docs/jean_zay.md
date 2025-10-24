# Running deCIFer on Jean Zay

This guide summarizes practical tips for preparing jobs for the [Jean Zay supercomputer](https://www.idris.fr/eng/jean-zay/jean-zay-presentation.html) managed by IDRIS. It assumes that you already have an active account and allocated GPU hours.

## 1. Storage layout

Jean Zay differentiates between several storage areas:

- `$HOME`: small quota, backed up. Keep code and light configuration files here.
- `$WORK`: larger quota, not backed up. Place cloned repositories, pre-trained weights, and Conda environments here. Export a writable path (e.g. `$WORK/deCIFer`) before installing.
- `$SCRATCH`: fastest storage with purge policy (≈30 days). Store temporary data such as intermediate checkpoints, generated CIFs, and temporary datasets.

A convenient layout is:

```bash
# On Jean Zay login node
mkdir -p $WORK/deCIFer $SCRATCH/deCIFer/runs
cd $WORK/deCIFer
git clone https://github.com/XXXX/deCIFer.git
```

## 2. Environment preparation

Jean Zay provides module-based Python stacks. Two common strategies are:

### Option A – Use the PyTorch module

```bash
module purge
module load python/3.11
module load pytorch-gpu/py3/2.3.0
python -m venv $WORK/venvs/decifer
source $WORK/venvs/decifer/bin/activate
pip install --upgrade pip
pip install -e .
```

Adjust the module versions to match the latest recommendations from IDRIS (`module avail pytorch-gpu`).

### Option B – Build a Conda environment

```bash
module purge
module load python/3.11 cuda/12.1
conda create -p $WORK/conda-envs/decifer python=3.9
conda activate $WORK/conda-envs/decifer
pip install -e .
```

> **Tip:** Run all environment creation steps inside an interactive job obtained with `salloc` or `srun --pty` to avoid overloading the login nodes.

## 3. Interactive debugging sessions

To obtain an interactive GPU session:

```bash
salloc --partition=gpu_p2 --time=02:00:00 --ntasks=1 --cpus-per-task=8 \
       --gres=gpu:1 --hint=nomultithread
srun --pty bash
```

Once inside the allocation, activate your environment and launch lightweight tests:

```bash
cd $WORK/deCIFer/deCIFer
source $WORK/venvs/decifer/bin/activate
python bin/prepare_dataset.py --help
```

## 4. Batch jobs with SLURM

The repository ships example batch scripts in [`slurm/`](../slurm). Adapt the resource directives to Jean Zay:

- Replace `#SBATCH -p gpu --gres=gpu:a100:1` with `#SBATCH --partition=gpu_p2 --gres=gpu:1` or another suitable GPU partition (`gpu_p1`, `gpu_p2`, `gpu_a100`).
- Always set an explicit wall time (`--time`), CPU count (`--cpus-per-task`), and memory (`--mem` or `--mem-per-cpu`).
- Route logs to `$WORK` or `$SCRATCH`, e.g. `#SBATCH --output=$WORK/deCIFer/logs/train_%j.out`.

Submit a training job with

```bash
cd $WORK/deCIFer/deCIFer
mkdir -p $WORK/deCIFer/logs
sbatch slurm/train.sh --config configs/train.yaml --out_dir $SCRATCH/deCIFer/runs/exp1
```

The helper script echoes the forwarded arguments before calling the training entrypoint defined in [`slurm/train.sh`](../slurm/train.sh).

### Automating submission scripts

If you prefer to describe your command in a plain text file and auto-generate a submission script, use [`slurm/create_jean_zay_job.py`](../slurm/create_jean_zay_job.py):

```bash
# Write the command you want to run
echo "python -m decifer.train --config configs/train.yaml" > run_command.txt

# Generate a SLURM script that pins the exact git commit
python slurm/create_jean_zay_job.py run_command.txt --gpu-type a100 --gpu-count 2 --job-name decifer-train

# Submit the generated script
sbatch slurm/jean_zay_job.sh
```

The generator captures the current commit hash and checks it out before executing the command, ensuring that updates pushed after submission do not change the code that runs on the cluster. Select the GPU type with `--gpu-type` (`v100`, `a100`, `h100`), request the number of accelerators with `--gpu-count`, and override resources such as wall time or memory if needed. The helper now validates that the account matches the GPU type (for example `nxk@h100` when `--gpu-type h100`) and exits with an explanatory error if they differ. Provide the correct account explicitly with `--account` when necessary.

Monitor progress with `squeue -u $USER` and inspect logs with `tail -f $WORK/deCIFer/logs/train_<jobid>.out`.

## 5. Dataset preparation workflow

1. Stage raw CIF archives to `$WORK` (long-term) or `$SCRATCH` (temporary) depending on size.
2. Launch the preprocessing pipeline via `sbatch slurm/prepare_dataset.sh --data-dir ...` or interactively with `srun` for smaller runs.
3. Keep serialized HDF5 files in `$SCRATCH/deCIFer/datasets` and symlink them into `$WORK/deCIFer/deCIFer/data` if needed.

For large-scale parameter sweeps, study [`slurm/experiment_conditioned.sh`](../slurm/experiment_conditioned.sh), which demonstrates how to use array jobs to explore different noise/broadening settings.

## 6. Checkpointing and artifacts

- Store checkpoints and generated CIFs on `$SCRATCH` during training to benefit from higher throughput.
- Periodically copy important checkpoints back to `$WORK` for persistence.
- Use `rsync` or `scp` from your workstation to fetch results:

```bash
rsync -avz login-jean-zay:/fs/scratch/$USER/deCIFer/runs/exp1/ ./runs/exp1/
```

## 7. Useful monitoring commands

```bash
squeue -u $USER          # list jobs
sacct -j <jobid>         # job accounting after completion
scontrol show job <jobid> # detailed job info
watch -n 60 gpustat      # check GPU utilisation inside an allocation
```

Refer to the [IDRIS documentation](https://docs.idris.fr/) for quota checks (`idqlimit`, `idqinfo`), transferring large datasets (`bbtransfer`), and containerization options if you require Singularity/Apptainer.

## 8. Troubleshooting checklist

- **Module conflicts** – always run `module purge` before loading the recommended stack.
- **Out-of-memory** – decrease `batch_size` in your config or request more GPUs/CPUs.
- **HDF5 errors** – ensure that `$TMPDIR` points to a directory on `$SCRATCH` when launching heavy I/O workloads.
- **Timeouts** – adapt `--time` to the longest expected duration and leverage checkpoints to resume.

Following these conventions should let you reproduce local runs on Jean Zay with minimal modifications.
