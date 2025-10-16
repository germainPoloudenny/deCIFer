#!/bin/bash
#SBATCH --job-name=task
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/task.out
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread


set -euo pipefail

REPO_DIR='/home/gpoloudenny/Projects/deCIFer'
COMMIT_HASH='5fed53cb38fdc11b0c2819c252da6def71eb9d5c'
RUN_COMMAND='python bin/conditioning_decoding_sweep.py --model-ckp runs/deCIFer_cifs_v1_model/ckpt_eval.pt  --dataset-path ../crystallography/data/structures/cifs_v1/serialized/test.h5   --out-root runs/deCIFer_cifs_v1_model/conditioning_decoding'
GENERATED_AT='20251016_102844'

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"

module load git
git checkout $COMMIT_HASH

echo "[Jean Zay helper] Using modules:  pytorch-gpu/py3/2.2.0"
module purge
module load  pytorch-gpu/py3/2.2.0

# Active un venv si présent, sinon continue (permet d'utiliser les modules directement)
source "$WORK/venvs/decifer/bin/activate" 2>/dev/null || true

export PYTHONPATH=$PYTHONPATH:/lustre/fswork/projects/rech/nxk/uvv78gt/

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
