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

REPO_DIR=/home/gpoloudenny/Projects/deCIFer
COMMIT_HASH='dbc842ca6c3d2495a2915a0348fce930525fdb1a'
RUN_COMMAND='python bin/conditioning_decoding_sweep.py --model-ckp runs/deCIFer_cifs_v1_model/ckpt_eval.pt  --dataset-path ../crystallography/data/structures/cifs_v1/serialized/test.h5   --out-root runs/deCIFer_cifs_v1_model/conditioning_decoding'
GENERATED_AT='20251016_092546'

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
