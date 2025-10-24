#!/bin/bash
#SBATCH --job-name=train_u_decifer
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_u_decifer.out
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32


set -euo pipefail

REPO_DIR='/home/gpoloudenny/Projects/deCIFer'
COMMIT_HASH='3712400055e8108b2d8209a95c7c29f933aba2c5'
ORIGINAL_REF='grpo'
RUN_COMMAND='torchrun --nproc_per_node=2 bin/train.py --config configs/U-deCIFer_NOMA_training_config.yaml'
GENERATED_AT='20251024_095904'

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"

echo "[Jean Zay helper] Using modules: arch/h100 pytorch-gpu/py3/2.3.1 git"
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
module load git

git checkout $COMMIT_HASH

cleanup() {
    if [ "$ORIGINAL_REF" != "HEAD" ]; then
        git checkout "$ORIGINAL_REF" || true
    fi
}

trap cleanup EXIT

# Active un venv si présent, sinon continue (permet d'utiliser les modules directement)

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
