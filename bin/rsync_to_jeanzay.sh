#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/lustre/fswork/projects/rech/nxk/uvv78gt/deCIFer"
DEFAULT_REMOTE_HOST="jean-zay"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] <relative-path>

Synchronise a path from the local repository to ${REMOTE_ROOT} on Jean Zay.

Options:
  -u, --user USER           Remote SSH user (default: \$JEAN_ZAY_USER or current user)
  -H, --host HOST           Remote SSH host (default: ${DEFAULT_REMOTE_HOST})
  -e, --exclude-file FILE   File with rsync exclude patterns (relative or absolute path)
  -n, --dry-run             Show actions without transferring files
  -h, --help                Display this help message and exit

Environment:
  JEAN_ZAY_USER             Default SSH user if --user is not provided.
EOF
}

REMOTE_HOST="${DEFAULT_REMOTE_HOST}"
REMOTE_USER="${JEAN_ZAY_USER:-}"
EXCLUDE_FILE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -u|--user)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --user requires an argument" >&2
        exit 1
      fi
      REMOTE_USER="$1"
      ;;
    -H|--host)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --host requires an argument" >&2
        exit 1
      fi
      REMOTE_HOST="$1"
      ;;
    -e|--exclude-file)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --exclude-file requires an argument" >&2
        exit 1
      fi
      EXCLUDE_FILE="$1"
      ;;
    -n|--dry-run)
      DRY_RUN=1
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: Unknown option $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
  shift
done

if [[ $# -lt 1 ]]; then
  echo "Error: relative path is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "$REMOTE_USER" ]]; then
  if [[ -n "${USER:-}" ]]; then
    REMOTE_USER="$USER"
  else
    echo "Error: unable to determine SSH user; use --user or set JEAN_ZAY_USER." >&2
    exit 1
  fi
fi

RELATIVE_PATH="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

if [[ ! -e "$RELATIVE_PATH" ]]; then
  echo "Error: ${RELATIVE_PATH} does not exist under ${REPO_ROOT}" >&2
  exit 1
fi

RELATIVE_PATH="${RELATIVE_PATH%/}"
if [[ -z "$RELATIVE_PATH" ]]; then
  RELATIVE_PATH="."
fi

if [[ -n "$EXCLUDE_FILE" ]]; then
  if [[ "$EXCLUDE_FILE" != /* ]]; then
    EXCLUDE_FILE="${REPO_ROOT}/${EXCLUDE_FILE}"
  fi
  if [[ ! -f "$EXCLUDE_FILE" ]]; then
    echo "Error: exclude file ${EXCLUDE_FILE} not found" >&2
    exit 1
  fi
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

if [[ "$RELATIVE_PATH" == "." ]]; then
  RSYNC_SRC="."
  RSYNC_DEST="${REMOTE}:${REMOTE_ROOT}/"
  REMOTE_DIR="${REMOTE_ROOT}"
else
  RSYNC_SRC="$RELATIVE_PATH"
  RSYNC_DEST="${REMOTE}:${REMOTE_ROOT}/${RELATIVE_PATH}"
  RELATIVE_DIR="$(dirname "$RELATIVE_PATH")"
  if [[ "$RELATIVE_DIR" == "." ]]; then
    REMOTE_DIR="${REMOTE_ROOT}"
  else
    REMOTE_DIR="${REMOTE_ROOT}/${RELATIVE_DIR}"
  fi
fi

RSYNC_OPTS=(-avh --info=progress2)
if [[ -n "$EXCLUDE_FILE" ]]; then
  RSYNC_OPTS+=(--exclude-from="$EXCLUDE_FILE")
fi
if (( DRY_RUN )); then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ "$REMOTE_DIR" != "${REMOTE_ROOT}" && "$REMOTE_DIR" != "${REMOTE_ROOT}/" ]]; then
  REMOTE_DIR="${REMOTE_DIR%/}"
fi

if (( DRY_RUN )); then
  echo "[dry-run] Would ensure remote directory ${REMOTE_DIR}"
else
  MKDIR_CMD=$(printf 'mkdir -p %q' "$REMOTE_DIR")
  ssh "$REMOTE" "$MKDIR_CMD"
fi

echo "Syncing ${RSYNC_SRC} -> ${RSYNC_DEST}"
rsync "${RSYNC_OPTS[@]}" "$RSYNC_SRC" "$RSYNC_DEST"
