#!/usr/bin/env python3
"""Generate a Jean Zay ready SLURM script from a command file."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime

GPU_PARTITIONS = {
    "v100": {"partition": "gpu_p2", "gres": "gpu:v100:1"},
    "a100": {"partition": "gpu_p5", "gres": "gpu:a100:1"},
    "h100": {"partition": "gpu_p6", "gres": "gpu:h100:1"},
}

GPU_DEFAULT_ACCOUNTS = {
    "v100": "nxk@v100",
    "a100": "nxk@a100",
    "h100": "nxk@h100",
}


def run_git_command(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.STDOUT)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        print(exc.output.decode(), file=sys.stderr)
        raise SystemExit(
            "Failed to execute git command. Are you inside the repository?"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Jean Zay SLURM batch script from a command stored in a text file."
        )
    )
    parser.add_argument(
        "command_file",
        type=pathlib.Path,
        help="Path to the text file containing the command to run (first line is used).",
    )
    parser.add_argument(
        "--gpu-type",
        choices=sorted(GPU_PARTITIONS),
        default="h100",
        help="GPU type to request on Jean Zay.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help=(
            "Slurm account to charge. Defaults to an account matching the selected GPU "
            "type if available."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("slurm") / "jean_zay_job.sh",
        help="Where to write the generated SLURM script.",
    )
    parser.add_argument(
        "--job-name", default="decifer", help="Job name to use for #SBATCH.")
    parser.add_argument(
        "--time",
        default="12:00:00",
        help="Wall clock limit in HH:MM:SS (default: 12 hours).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="Number of CPUs to request per task.",
    )
    parser.add_argument(
        "--mem",
        default="64G",
        help="Amount of memory to request (e.g. 64G).",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["python/3.11", "pytorch-gpu/py3/2.3.0"],
        help="Module list to load inside the batch job.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.command_file.exists():
        raise SystemExit(f"Command file {args.command_file} does not exist")

    command = args.command_file.read_text().strip()
    if not command:
        raise SystemExit("Command file is empty")

    repo_root = pathlib.Path(run_git_command("rev-parse", "--show-toplevel"))
    commit_hash = run_git_command("rev-parse", "HEAD")

    partition_info = GPU_PARTITIONS[args.gpu_type]
    gres = partition_info["gres"]
    partition = partition_info["partition"]
    account = args.account or GPU_DEFAULT_ACCOUNTS.get(args.gpu_type)

    output_path: pathlib.Path = args.output
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    modules_block = "\n".join(
        f"module load {module}" for module in args.modules if module
    )

    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
    ]
    if account:
        header_lines.append(f"#SBATCH -A {account}")
    header_lines.extend(
        [
            f"#SBATCH --time={args.time}",
            f"#SBATCH --cpus-per-task={args.cpus}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH --output=$WORK/deCIFer/logs/{args.job_name}_%j.out",
            "",
        ]
    )

    job_script = "\n".join(header_lines)

    job_script += f"""

set -euo pipefail

export OMP_NUM_THREADS={args.cpus}

REPO_DIR={shlex.quote(str(repo_root))}
COMMIT_HASH={commit_hash!r}
RUN_COMMAND={shlex.quote(command)}
GENERATED_AT={timestamp!r}

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"
git fetch --all --prune
if ! git checkout "$COMMIT_HASH"; then
    echo "Failed to checkout commit $COMMIT_HASH" >&2
    exit 1
fi

echo "[Jean Zay helper] Using modules: {' '.join(args.modules)}"
module purge
{modules_block}

source "$WORK/venvs/decifer/bin/activate" 2>/dev/null || true

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
"""

    output_path.write_text(job_script)
    output_path.chmod(0o750)

    account_display = account if account else "<none>"
    print(
        f"Generated {output_path} for commit {commit_hash} on partition {partition} "
        f"(account {account_display})."
    )


if __name__ == "__main__":
    main()

