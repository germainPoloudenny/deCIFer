#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=h
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/h.out
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=48


set -euo pipefail

REPO_DIR='/lustre/fswork/projects/rech/nxk/uvv78gt/deCIFer'

cd "$REPO_DIR"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
module load git

git checkout grpo

torchrun --nproc_per_node=2  bin/eval/evaluate.py   --model-ckpt runs/deCIFer/ckpt_eval.pt   --dataset-path ../crystallography/data/chili-100k/serialized/test.h5   --out-folder runs/deCIFer/eval_chili-100k --add-composition