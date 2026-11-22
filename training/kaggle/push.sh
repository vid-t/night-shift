#!/usr/bin/env bash
# Start a run on Kaggle: upload the run files as a private dataset, then start a script on a GPU.
#
#   bash training/kaggle/push.sh [SCRIPT]        # SCRIPT default: training/kaggle/first_run.py
#
# Needs uv, the exported files (uv run hai-export) and a Kaggle API token, either in
# KAGGLE_API_TOKEN (the token, or a file that holds it) or in ~/.kaggle/access_token.
# Create the token at https://www.kaggle.com/settings/api. The account must be phone-verified,
# or Kaggle gives the run no GPU and no internet.
#
# KAGGLE_TIME_LIMIT    seconds after which Kaggle stops the run (default 5400)
# KAGGLE_SKIP_UPLOAD=1 start the script on the run files already on Kaggle
# KAGGLE_UPLOAD_ONLY=1 upload the run files and start nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SCRIPT="${1:-$HERE/first_run.py}"
TIME_LIMIT="${KAGGLE_TIME_LIMIT:-5400}"
JSONL="$ROOT/data/processed/opentslm"

# kaggle 2.2.5 imports six without declaring it
kaggle() { uvx --quiet --with six --from kaggle==2.2.5 kaggle "$@"; }

test -f "$SCRIPT" || { echo "no such script: $SCRIPT" >&2; exit 1; }
test -f "$JSONL/train.jsonl" || { echo "no exported files in $JSONL: run 'uv run hai-export' first" >&2; exit 1; }
OWNER="$(kaggle config view | sed -n 's/^- username: //p')"
test -n "$OWNER" || { echo "no Kaggle login: set KAGGLE_API_TOKEN or write ~/.kaggle/access_token" >&2; exit 1; }

NAME="$(basename "$SCRIPT" .py)"
DATASET="$OWNER/night-shift-run-files"
KERNEL="$OWNER/night-shift-${NAME//_/-}"
BUILD="$(mktemp -d)"
mkdir -p "$BUILD/files" "$BUILD/kernel"

echo "== 1. run files -> private dataset $DATASET"
if [ "${KAGGLE_SKIP_UPLOAD:-0}" = "1" ]; then
  echo "upload skipped (KAGGLE_SKIP_UPLOAD=1)"
else
  cp "$ROOT"/training/{HAIAttackQADataset.py,add_stage.py,setup.sh,score.py,launch.py} "$HERE"/{gpu_check.py,kaggle_common.py} "$BUILD/files/"
  cp "$JSONL"/{train,validation,test,test_shuffled,validation_shuffled}.jsonl "$BUILD/files/"
  VERSION="$(git -C "$ROOT" rev-parse --short HEAD)"
  test -z "$(git -C "$ROOT" status --porcelain)" || VERSION="$VERSION plus uncommitted changes"
  echo "night-shift $VERSION, uploaded $(date -u +%Y-%m-%dT%H:%MZ)" > "$BUILD/files/VERSION"
  cat > "$BUILD/files/dataset-metadata.json" <<EOF
  {"title": "Night Shift run files", "id": "$DATASET", "licenses": [{"name": "CC-BY-SA-4.0"}]}
EOF
  if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
    kaggle datasets version -p "$BUILD/files" -m "$VERSION"
  else
    kaggle datasets create -p "$BUILD/files"
  fi
  sleep 15
  for _ in $(seq 60); do
    STATUS="$(kaggle datasets status "$DATASET" 2>&1 | tail -1)"
    case "$STATUS" in *ready*) break ;; esac
    sleep 5
  done
  echo "dataset status: $STATUS"
  case "$STATUS" in *ready*) ;; *) echo "the dataset is not ready; try again in a minute" >&2; exit 1 ;; esac
fi

if [ "${KAGGLE_UPLOAD_ONLY:-0}" = "1" ]; then
  echo "upload only (KAGGLE_UPLOAD_ONLY=1): no run started"
  exit 0
fi

echo "== 2. $NAME.py -> $KERNEL on a T4, private, at most $TIME_LIMIT s"
cp "$SCRIPT" "$BUILD/kernel/$NAME.py"
cat > "$BUILD/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "Night Shift ${NAME//_/ }",
  "code_file": "$NAME.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
kaggle kernels push -p "$BUILD/kernel" --accelerator NvidiaTeslaT4 -t "$TIME_LIMIT"

echo
echo "Watch it:      https://www.kaggle.com/code/$KERNEL"
echo "Status:        uvx --with six --from kaggle==2.2.5 kaggle kernels status $KERNEL"
echo "Get results:   uvx --with six --from kaggle==2.2.5 kaggle kernels output $KERNEL -p runs/kaggle/$NAME"
