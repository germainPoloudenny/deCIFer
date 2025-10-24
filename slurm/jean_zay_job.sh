#!/bin/bash
#SBATCH --job-name=task
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:4
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/task.out
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=32


set -euo pipefail

REPO_DIR='/home/gpoloudenny/Projects/deCIFer'
COMMIT_HASH='6a0d761c13bbde4f5fc571b92aadfafa577950da'
ORIGINAL_REF='grpo'
RUN_COMMAND='torchrun --nproc_per_node=2 bin/train.py --config configs/deCIFer_cifs_v1.yaml'
GENERATED_AT='20251025_003530'

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
