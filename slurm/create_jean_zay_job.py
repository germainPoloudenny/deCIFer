#!/usr/bin/env python3
"""Generate a Jean Zay ready SLURM script from a command file."""

from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime

# Mapping des types de GPU -> partition / gres / contrainte
GPU_PARTITIONS = {
    "v100": {"partition": "gpu_p2", "gres": "gpu:2", "constraint": "v100"},
    "a100": {"partition": "gpu_p5", "gres": "gpu:2", "constraint": "a100"},
    "h100": {"partition": "gpu_p6", "gres": "gpu:2", "constraint": "h100"},
}

GPU_DEFAULT_ACCOUNTS = {
    "v100": "nxk@v100",
    "a100": "nxk@a100",
    "h100": "nxk@h100",
}

GPU_DEFAULT_MODULES = {
    "h100": ["pytorch-gpu/py3/2.3.1"],
}

GPU_FALLBACK_MODULES = ["pytorch-gpu/py3/2.3.0"]


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
        default=None,
        help="GPU type to request on Jean Zay (default: h100).",
    )
    parser.add_argument(
        "--account",
        default=None,
        help=(
            "Slurm account to charge. Defaults to an account matching the selected GPU "
            "type if available. You can use a suffix like nxk@v100."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("slurm") / "jean_zay_job.sh",
        help="Where to write the generated SLURM script.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help=(
            "Job name to use for #SBATCH and the log file. Defaults to the command "
            "file name."
        ),
    )
    parser.add_argument(
        "--time",
        default="12:00:00",
        help="Wall clock limit in HH:MM:SS (default: 12 hours).",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=None,
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

    repo_root = pathlib.Path(run_git_command("rev-parse", "--show-toplevel")).resolve()
    commit_hash = run_git_command("rev-parse", "HEAD")
    current_branch = run_git_command("rev-parse", "--abbrev-ref", "HEAD")

    gpu_type = args.gpu_type or "h100"
    account = args.account

    # Si l'account fourni est de la forme "xxx@v100|a100|h100", on en tient compte
    account_gpu_suffix = None
    if account and "@" in account:
        _, _, account_gpu_suffix = account.partition("@")
        if account_gpu_suffix and account_gpu_suffix not in GPU_PARTITIONS:
            raise SystemExit(
                f"Account '{account}' references unsupported GPU type "
                f"'{account_gpu_suffix}'."
            )

    if args.gpu_type is None and account_gpu_suffix:
        gpu_type = account_gpu_suffix

    if account_gpu_suffix and account_gpu_suffix != gpu_type:
        raise SystemExit(
            "The requested GPU type does not match the provided account. "
            f"Account '{account}' cannot be used with GPU type '{gpu_type}'. "
            "Please pass a matching --gpu-type or --account."
        )

    partition_info = GPU_PARTITIONS[gpu_type]
    gres = partition_info["gres"]
    partition = partition_info["partition"]
    constraint = partition_info["constraint"]

    if account is None:
        account = GPU_DEFAULT_ACCOUNTS.get(gpu_type)

    output_path: pathlib.Path = args.output
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    job_name = args.job_name or args.command_file.stem
    job_name = job_name.replace(" ", "_")
    log_file_name = f"logs/{job_name}.out"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.modules is None:
        raw_modules = GPU_DEFAULT_MODULES.get(gpu_type, GPU_FALLBACK_MODULES)
    else:
        raw_modules = args.modules

    modules = [module.strip() for module in raw_modules if module and module.strip()]

    modules_to_load: list[str] = []
    if gpu_type != "v100":
        arch_module = f"arch/{gpu_type}"
        if not any(module.startswith("arch/") for module in modules):
            modules_to_load.append(arch_module)

    modules_to_load.extend(modules)

    if all(module != "git" for module in modules_to_load):
        modules_to_load.append("git")

    modules_block = "\n".join(f"module load {module}" for module in modules_to_load)

    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --constraint={constraint}",   # <<-- s'adapte (v100/a100/h100)
        f"#SBATCH --gres={gres}",               # <<-- on demande bien 2 GPU
    ]
    if account:
        header_lines.append(f"#SBATCH --account={account}")
    header_lines.extend(
        [
            f"#SBATCH --time={args.time}",
            f"#SBATCH --output={log_file_name}",
            f"#SBATCH --ntasks-per-node=2",
            f"#SBATCH --hint=nomultithread",
            f"#SBATCH --cpus-per-task=32",
            "",
        ]
    )

    job_script = "\n".join(header_lines)

    job_script += f"""

set -euo pipefail

REPO_DIR={str(repo_root)!r}
COMMIT_HASH={commit_hash!r}
ORIGINAL_REF={current_branch!r}
RUN_COMMAND={shlex.quote(command)}
GENERATED_AT={timestamp!r}

mkdir -p "$WORK/deCIFer/logs"

cd "$REPO_DIR"
echo "[Jean Zay helper] Restoring commit $COMMIT_HASH"

echo "[Jean Zay helper] Using modules: {' '.join(modules_to_load)}"
module purge
{modules_block}

git checkout $COMMIT_HASH

cleanup() {{
    if [ "$ORIGINAL_REF" != "HEAD" ]; then
        git checkout "$ORIGINAL_REF" || true
    fi
}}

trap cleanup EXIT

# Active un venv si présent, sinon continue (permet d'utiliser les modules directement)

echo "[Jean Zay helper] Generated at $GENERATED_AT"
echo "[Jean Zay helper] Running command: $RUN_COMMAND"

eval "$RUN_COMMAND"
"""

    output_path.write_text(job_script)
    output_path.chmod(0o750)

    account_display = account if account else "<none>"
    print(
        f"Generated {output_path} for commit {commit_hash} on partition {partition} "
        f"with constraint {constraint} (account {account_display})."
    )


if __name__ == "__main__":
    main()
