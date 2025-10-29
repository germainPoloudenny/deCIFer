#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=v
#SBATCH --partition=gpu_p2
#SBATCH --constraint=v100
#SBATCH --gres=gpu:2
#SBATCH --account=nxk@v100
#SBATCH --time=12:00:00
#SBATCH --output=logs/v.out
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=12


set -euo pipefail

REPO_DIR='/lustre/fswork/projects/rech/nxk/uvv78gt/deCIFer'

cd "$REPO_DIR"

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
module load git

git checkout comp

torchrun --nproc_per_node=2 bin/train.py --config configs/decifer.yaml