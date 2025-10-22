#!/bin/bash
#SBATCH --job-name=task
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --account=nxk@h100
#SBATCH --time=12:00:00
#SBATCH --output=logs/task.out
#SBATCH --ntasks-per-node=2
#SBATCH --hint=nomultithread


set -euo pipefail

REPO_DIR='/home/gpoloudenny/Projects/deCIFer'
COMMIT_HASH='2c2ce0228ebfabbc5f8b5d98b1060b9761e7e9b1'
ORIGINAL_REF='beam_stoch'
RUN_COMMAND='python bin/eval/beam_vs_rwp_filter.py  --model-ckpt runs/deCIFer_cifs_v1_model/ckpt_eval.pt --dataset-path ../crystallography/data/structures/cifs_v1/serialized/test.h5 --out-root runs/deCIFer_cifs_v1_model/beam_search_vs_rwp_filter  --max-samples 1000'
GENERATED_AT='20251019_123329'

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

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

# When launching distributed jobs with torchrun the default master
# port (29500) can easily collide with other users on the same node.
# Pick a free port dynamically unless the caller has already
# specified one.
if [[ "$RUN_COMMAND" == *torchrun* || "$RUN_COMMAND" == *torch.distributed.run* ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
    if [[ -z "${MASTER_PORT:-}" ]]; then
        MASTER_PORT=$(python - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)
        export MASTER_PORT
    fi
    echo "[Jean Zay helper] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
fi

eval "$RUN_COMMAND"
