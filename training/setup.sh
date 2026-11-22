#!/usr/bin/env bash
# Prepare OpenTSLM for stage6_hai. Run from anywhere inside hai-attack-reader:
#
#   bash training/setup.sh [LLM_ID]        # LLM_ID default: meta-llama/Llama-3.2-1B
#
# 1. clone OpenTSLM into external/OpenTSLM at the pinned commit
# 2. install it with uv (its own .venv, editable)
# 3. copy HAIAttackQADataset.py into src/opentslm/time_series_datasets/hai/
# 4. add stage6_hai to curriculum_learning.py (add_stage.py; safe to rerun)
# 5. seed the starting checkpoint (default OpenTSLM/llama-3.2-1b-sleep-sp) where the stage before
#    stage6_hai is expected: results/<llm>/OpenTSLMSP/stage5_ecg_cot/{checkpoints/best_model.pt,results/metrics.json}
#
# Environment: OPENTSLM_DIR (default external/OpenTSLM), SKIP_INSTALL=1 skips step 2,
# HAI_START_CHECKPOINT (Hugging Face repo of the starting checkpoint, file model_checkpoint.pt).
set -euo pipefail

LLM_ID="${1:-meta-llama/Llama-3.2-1B}"
COMMIT="2968f4b891baab4307f7e9d0043e87677b593a30"
REPO_URL="https://github.com/OpenTSLM/OpenTSLM.git"
# stage6_hai loads its starting weights from the slot of the stage before it, stage5_ecg_cot. The default is
# the checkpoint of OpenTSLM's stage 4 (sleep): encoder, projector and adapter (LoRA) weights, 164 MB, and the
# last published stage whose encoder still responds to its input. Do not use `llama-3.2-1b-ecg-sp`, the end of
# the curriculum: its encoder gives the same output for every curve (measured 2026-11-21; training/launch.py
# refuses it). `llama-3.2-1b-tsqa-sp` is stage 1 (29 MB, no adapters) and works too.
CHECKPOINT_REPO="${HAI_START_CHECKPOINT:-OpenTSLM/llama-3.2-1b-sleep-sp}"
CHECKPOINT_FILE="model_checkpoint.pt"
PREVIOUS_STAGE="stage5_ecg_cot"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OPENTSLM_DIR="${OPENTSLM_DIR:-$ROOT/external/OpenTSLM}"

echo "== 1. OpenTSLM at $COMMIT in $OPENTSLM_DIR"
if [ ! -d "$OPENTSLM_DIR/.git" ]; then
  mkdir -p "$(dirname "$OPENTSLM_DIR")"
  git clone "$REPO_URL" "$OPENTSLM_DIR"
fi
git -C "$OPENTSLM_DIR" checkout -q "$COMMIT" -- . 2>/dev/null || true
git -C "$OPENTSLM_DIR" -c advice.detachedHead=false checkout -q "$COMMIT"
test "$(git -C "$OPENTSLM_DIR" rev-parse HEAD)" = "$COMMIT" || { echo "OpenTSLM is not at $COMMIT" >&2; exit 1; }

cd "$OPENTSLM_DIR"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "== 2. install (uv sync)"
  uv sync
else
  echo "== 2. install skipped (SKIP_INSTALL=1)"
fi

echo "== 3. dataset class"
mkdir -p src/opentslm/time_series_datasets/hai
cp "$HERE/HAIAttackQADataset.py" src/opentslm/time_series_datasets/hai/HAIAttackQADataset.py
touch src/opentslm/time_series_datasets/hai/__init__.py

echo "== 4. stage6_hai in curriculum_learning.py"
python3 "$HERE/add_stage.py" curriculum_learning.py

echo "== 5. starting checkpoint $CHECKPOINT_REPO"
uv run python - "$LLM_ID" "$CHECKPOINT_REPO" "$CHECKPOINT_FILE" "$PREVIOUS_STAGE" <<'EOF'
import json, os, shutil, sys
from huggingface_hub import hf_hub_download

llm_id, repo, filename, previous = sys.argv[1:5]
# same rule as CurriculumTrainer._sanitize_llm_id: meta-llama/Llama-3.2-1B -> Llama_3_2_1B
name = llm_id.split("/")[-1].replace(".", "_").replace("-", "_")
while "__" in name:
    name = name.replace("__", "_")
stage = os.path.join("results", name, "OpenTSLMSP", previous)
os.makedirs(os.path.join(stage, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(stage, "results"), exist_ok=True)
src = hf_hub_download(repo, filename)
dst = os.path.join(stage, "checkpoints", "best_model.pt")
shutil.copyfile(src, dst)
metrics = os.path.join(stage, "results", "metrics.json")
with open(metrics, "w") as f:
    json.dump({"seeded_from": f"{repo}/{filename}"}, f)
print(f"seeded {dst} ({os.path.getsize(dst) / 1e6:.1f} MB) and {metrics}")
EOF

echo "done. Run training from $OPENTSLM_DIR with HAI_JSONL_DIR=$ROOT/data/processed/opentslm (see training/README.md)"
