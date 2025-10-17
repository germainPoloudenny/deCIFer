#!/bin/bash
#SBATCH --job-name=task
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/task.out
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread


set -euo pipefail

REPO_DIR='/home/gpoloudenny/Projects/deCIFer'
COMMIT_HASH='e1caccf13acb822fbaf22cdc7b597cc77419dc78'
ORIGINAL_REF='beam_stoch'
RUN_COMMAND='torchrun --nproc_per_node 2 /home/gpoloudenny/Projects/deCIFer/bin/eval/evaluate.py --model-ckpt runs/deCIFer_cifs_v1_model/ckpt_eval.pt --dataset-path ../crystallography/data/structures/cifs_v1/serialized/test.h5 --out-folder runs/deCIFer_cifs_v1_model/conditioning_decoding/max_100/none/k_sampling --dataset-name conditioning_decoding_max_100_none_k_sampling --beam-size 1 --length-penalty 1.0 --max-samples 100 --num-reps 1 --temperature 1.0 --top-k 50'
GENERATED_AT='20251016_223303'

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"

echo "[Jean Zay helper] Using modules: arch/h100 pytorch-gpu/py3/2.5.0 git"
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0
module load git

git checkout $COMMIT_HASH

cleanup() {
    if [ "$ORIGINAL_REF" != "HEAD" ]; then
        git checkout "$ORIGINAL_REF" || true
    fi
}

trap cleanup EXIT

# Active un venv si présent, sinon continue (permet d'utiliser les modules directement)
source "$WORK/venvs/decifer/bin/activate" 2>/dev/null || true

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
