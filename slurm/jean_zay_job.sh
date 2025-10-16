#!/bin/bash
#SBATCH --job-name=decifer
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=$WORK/deCIFer/logs/decifer_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread


set -euo pipefail

export OMP_NUM_THREADS=8

REPO_DIR=/home/gpoloudenny/Projects/deCIFer
COMMIT_HASH='8b5eafb7658ae4bfd3492e31b2af465f53b450fa'
RUN_COMMAND='python bin/conditioning_decoding_sweep.py --model-ckp runs/deCIFer_cifs_v1_model/ckpt_eval.pt  --dataset-path ../crystallography/data/structures/cifs_v1/serialized/test.h5   --out-root runs/deCIFer_cifs_v1_model/conditioning_decoding'
GENERATED_AT='20251016_031628'

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"
git fetch --all --prune
if ! git checkout "$COMMIT_HASH"; then
    echo "Failed to checkout commit $COMMIT_HASH" >&2
    exit 1
fi

echo "[Jean Zay helper] Using modules: python/3.11 pytorch-gpu/py3/2.3.0"
module purge
module load python/3.11
module load pytorch-gpu/py3/2.3.0

# Active un venv si présent, sinon continue (permet d'utiliser les modules directement)
source "$WORK/venvs/decifer/bin/activate" 2>/dev/null || true

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
