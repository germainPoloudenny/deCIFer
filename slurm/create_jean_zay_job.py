#!/usr/bin/env python3
"""Utility to generate Jean Zay compliant SLURM scripts.

The helper reads a plain text file containing the command to execute and
creates a submission script that can be launched with ``sbatch``. The generated
script pins the current git commit to guarantee reproducibility and exposes a
few knobs tailored to the Jean Zay GPU partitions.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

GPU_CONFIG = {
    "v100": {
        "partition": "gpu_p1",
        "default_mem": "64G",
    },
    "a100": {
        "partition": "gpu_p2",
        "default_mem": "80G",
    },
    "h100": {
        "partition": "gpu_h100",
        "default_mem": "96G",
    },
}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse error handling
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Jean Zay SLURM submission script from a command file.",
    )
    parser.add_argument(
        "command_file",
        type=Path,
        help="Path to a text file containing the command to execute (first non-empty line is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("slurm/jean_zay_job.sh"),
        help="Path to the generated SLURM script (default: slurm/jean_zay_job.sh).",
    )
    parser.add_argument(
        "--job-name",
        default="decifer-job",
        help="Name of the job displayed by squeue (default: decifer-job).",
    )
    parser.add_argument(
        "--gpu-type",
        choices=sorted(GPU_CONFIG),
        default="a100",
        help="GPU type to target (determines the partition).",
    )
    parser.add_argument(
        "--gpu-count",
        type=_positive_int,
        default=1,
        help="Number of GPUs to request with --gres (default: 1).",
    )
    parser.add_argument(
        "--time",
        default="20:00:00",
        help="Maximum wall time requested with --time (default: 20:00:00).",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=_positive_int,
        default=8,
        help="Number of CPUs to allocate per task (default: 8).",
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="Amount of RAM to request (default depends on the GPU type).",
    )
    parser.add_argument(
        "--account",
        default=None,
        help=(
            "Accounting string to charge the job to. Falls back to $IDRIS_ACCOUNT or $SLURM_ACCOUNT when omitted."
        ),
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Partition to use. When omitted the helper infers it from --gpu-type.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory where stdout/stderr logs are written (default: no explicit log path).",
    )
    parser.add_argument(
        "--mail-type",
        default=None,
        help="Optional SLURM --mail-type directive (e.g. ALL, FAIL).",
    )
    parser.add_argument(
        "--mail-user",
        default=None,
        help="Email recipient used together with --mail-type.",
    )
    parser.add_argument(
        "--constraint",
        default=None,
        help="Optional SLURM --constraint directive (advanced users only).",
    )
    return parser.parse_args(argv)


def read_command(command_file: Path) -> str:
    if not command_file.exists():
        raise SystemExit(f"Command file {command_file} does not exist.")
    lines = [line.strip() for line in command_file.read_text().splitlines()]
    for line in lines:
        if line and not line.startswith("#"):
            return line
    raise SystemExit("Command file does not contain a valid command.")


def detect_repo_root() -> Path:
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment guard
        raise SystemExit("This script must be executed from inside a git repository.") from exc
    return Path(repo_root)


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment guard
        raise SystemExit("Failed to obtain the current git commit.") from exc


def resolve_account(gpu_type: str, explicit_account: str | None) -> str | None:
    account = explicit_account or os.getenv("IDRIS_ACCOUNT") or os.getenv("SLURM_ACCOUNT")
    if account is None:
        return None
    suffix = f"@{gpu_type.lower()}"
    if suffix not in account.lower():
        raise SystemExit(
            "Account and GPU type mismatch. Provide an account ending with "
            f"'{suffix}' or override with --account."
        )
    return account


def render_header(args: argparse.Namespace, partition: str, account: str | None) -> list[str]:
    log_dir = args.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    log_path = None
    if log_dir is not None:
        log_path = log_dir / "jean_zay_job_%j.out"

    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres=gpu:{args.gpu_count}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --time={args.time}",
    ]
    memory = args.mem or GPU_CONFIG[args.gpu_type]["default_mem"]
    header.append(f"#SBATCH --mem={memory}")
    if account:
        header.append(f"#SBATCH --account={account}")
    if log_path is not None:
        header.append(f"#SBATCH --output={log_path}")
        header.append(f"#SBATCH --error={log_path}")
    if args.mail_type:
        header.append(f"#SBATCH --mail-type={args.mail_type}")
    if args.mail_user:
        header.append(f"#SBATCH --mail-user={args.mail_user}")
    if args.constraint:
        header.append(f"#SBATCH --constraint={args.constraint}")
    header.append("#SBATCH --hint=nomultithread")
    return header


def write_script(
    output_path: Path,
    header_lines: list[str],
    repo_root: Path,
    commit: str,
    command: str,
) -> None:
    repo_root_str = shlex.quote(str(repo_root))
    header = "\n".join(header_lines)
    body = f"""
set -euo pipefail

WORKDIR={repo_root_str}
echo "==> Hostname: $(hostname)"
echo "==> SLURM job id: $SLURM_JOB_ID"
echo "==> Starting at: $(date)"

echo "==> Working directory: $WORKDIR"
cd "$WORKDIR"

# Ensure we execute the exact same commit that was active when creating the job.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git fetch --all --tags --prune
    git checkout {commit}
fi

COMMAND=$(cat <<'JOB_EOF'
{command}
JOB_EOF
)

echo "==> Launch command"
echo "$COMMAND"

srun --cpu-bind=none bash -lc "$COMMAND"
""".strip("\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{header}\n\n{body}\n")
    output_path.chmod(0o755)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    command = read_command(args.command_file)
    repo_root = detect_repo_root()
    commit = current_commit()

    partition = args.partition or GPU_CONFIG[args.gpu_type]["partition"]
    account = resolve_account(args.gpu_type, args.account)

    header_lines = render_header(args, partition, account)
    write_script(args.output, header_lines, repo_root, commit, command)

    print(f"Wrote SLURM script to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
