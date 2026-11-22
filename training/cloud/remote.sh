#!/usr/bin/env bash
# Drive a rented GPU machine from this computer. Run from anywhere inside hai-attack-reader.
#
#   export HAI_REMOTE=hai@203.0.113.7          # user@address of the machine (required)
#   export HAI_REMOTE_KEY=~/.ssh/id_ed25519    # private SSH key (this is the default)
#
#   bash training/cloud/remote.sh sync                  copy the code and the exported files to ~/hai there
#   bash training/cloud/remote.sh setup                 install uv, set up OpenTSLM, report the GPU (once)
#   bash training/cloud/remote.sh start NAME [options]  start run_variant.py NAME [options] in the background
#   bash training/cloud/remote.sh mps                   switch NVIDIA's multi-process service on, so parallel runs share the card
#   bash training/cloud/remote.sh status                GPU load and where every run stands
#   bash training/cloud/remote.sh log NAME [LINES]      the last lines of one run's log
#   bash training/cloud/remote.sh fetch DIR             copy every run's kept files, without checkpoints, to DIR
#   bash training/cloud/remote.sh fetch-model NAME DIR  copy one run's best_model.pt to DIR
#   bash training/cloud/remote.sh fetch-models DIR [NAME...]  copy the best_model.pt of these runs (default: all) to DIR/NAME/kept
#   bash training/cloud/remote.sh run 'COMMAND'         run a shell command in ~/hai there
#
# The machine needs an NVIDIA GPU with drivers, git, curl and tar. Nothing secret is copied to it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
: "${HAI_REMOTE:?set HAI_REMOTE to user@address of the machine}"
KEY="${HAI_REMOTE_KEY:-$HOME/.ssh/id_ed25519}"
LLM_ID="unsloth/Llama-3.2-1B"
FILES=(train validation test test_shuffled validation_shuffled)

remote() {  # run a command in ~/hai on the machine, with uv on the PATH
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$HAI_REMOTE" \
    "export PATH=\"\$HOME/.local/bin:\$PATH\"; mkdir -p ~/hai && cd ~/hai && $1"
}

case "${1:-}" in
  sync)
    for name in "${FILES[@]}"; do
      test -f "$ROOT/data/processed/opentslm/$name.jsonl" || { echo "missing $name.jsonl: run 'uv run hai-export' first" >&2; exit 1; }
    done
    VERSION="$(git -C "$ROOT" rev-parse --short HEAD)"
    test -z "$(git -C "$ROOT" status --porcelain)" || VERSION="$VERSION plus uncommitted changes"
    echo "hai-attack-reader $VERSION, copied $(date -u +%Y-%m-%dT%H:%MZ)" > "$ROOT/training/VERSION"
    ( cd "$ROOT" && COPYFILE_DISABLE=1 tar czf - --no-xattrs --no-mac-metadata --exclude '__pycache__' training \
        $(printf 'data/processed/opentslm/%s.jsonl ' "${FILES[@]}") ) | remote "tar xzf -"
    rm -f "$ROOT/training/VERSION"
    # setup.sh copied the dataset class into the OpenTSLM checkout: keep that copy current
    remote "d=external/OpenTSLM/src/opentslm/time_series_datasets/hai; if [ -d \$d ]; then cp training/HAIAttackQADataset.py \$d/; fi; cat training/VERSION"
    ;;
  setup)
    remote "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
    remote "bash training/setup.sh $LLM_ID && cd external/OpenTSLM && uv run python ../../training/kaggle/gpu_check.py"
    remote "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | sed -n 2p; df -h ~ | tail -n 1"
    ;;
  start)
    NAME="${2:?start needs a run name}"; shift 2
    # with NVIDIA's multi-process service running (see `mps`), the run joins it: several runs then share the card
    remote "mkdir -p work && if pgrep -f nvidia-cuda-mps-control >/dev/null && [ -d \$HOME/mps/pipe ]; then export CUDA_MPS_PIPE_DIRECTORY=\$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=\$HOME/mps/log; fi; (setsid nohup python3 training/cloud/run_variant.py $NAME $* > work/$NAME.out 2>&1 < /dev/null &) && sleep 1 && echo started $NAME"
    ;;
  mps)
    remote 'export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY;
      pgrep -f nvidia-cuda-mps-control >/dev/null || { nvidia-cuda-mps-control -d; sleep 2; };
      if pgrep -f nvidia-cuda-mps-control >/dev/null; then echo "MPS on"; else echo "MPS OFF"; exit 1; fi'
    ;;
  status)
    remote 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; uptime | sed "s/.*load/load/";
      for d in work/*/; do n=$(basename "$d"); log="$d/kept/log.txt";
        if [ -f "$d/kept/run_summary.json" ]; then state=done; elif pgrep -f "run_variant.py $n( |\$)" >/dev/null; then state=running; else state=STOPPED; fi;
        echo "== $n: $state";
        grep -E "encoder response|Epoch [0-9]+ . (train|val) +loss|Traceback|Error|command failed" "$log" 2>/dev/null | tail -n 12 | cut -c1-150;
        tail -n 1 "$log" 2>/dev/null | cut -c1-150; done'
    ;;
  log)
    remote "tail -n ${3:-40} work/${2:?log needs a run name}/kept/log.txt"
    ;;
  fetch)
    DIR="${2:?fetch needs a folder}"; mkdir -p "$DIR"
    remote "cd work && tar czf - --exclude best_model.pt */kept" | tar xzf - -C "$DIR"
    ls "$DIR"
    ;;
  fetch-model)
    NAME="${2:?fetch-model needs a run name}"; DIR="${3:?fetch-model needs a folder}"; mkdir -p "$DIR"
    scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$HAI_REMOTE:hai/work/$NAME/kept/best_model.pt" "$DIR/best_model.pt"
    ;;
  fetch-models)
    DIR="${2:?fetch-models needs a folder}"; shift 2
    NAMES="$*"
    test -n "$NAMES" || NAMES="$(remote 'cd work && for d in */; do [ -f "$d/kept/best_model.pt" ] && echo "${d%/}"; done')"
    for NAME in $NAMES; do
      mkdir -p "$DIR/$NAME/kept"
      test -s "$DIR/$NAME/kept/best_model.pt" && { echo "$NAME: already here"; continue; }
      scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new "$HAI_REMOTE:hai/work/$NAME/kept/best_model.pt" "$DIR/$NAME/kept/best_model.pt" && echo "$NAME: copied"
    done
    ;;
  run)
    remote "${2:?run needs a command}"
    ;;
  *)
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
