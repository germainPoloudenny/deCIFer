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

git checkout grpo

python bin/eval/conditioning_decoding_sweep.py   --model-ckpt runs/deCIFer_cifs_v1_model/ckpt_eval.pt --dataset-path ../crystallography/data/noma-1k/serialized/test.h5 --out-root runs/deCIFer_cifs_v1_model/conditioning_decoding_sweep --max-samples 1000 --collect-top-k-metric rmsd
