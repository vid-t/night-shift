"""Put a starting checkpoint where stage6_hai looks for the stage before it, under the current folder.

    uv run python training/cloud/seed.py LLM_ID CHECKPOINT_REPO

The same step as number 5 of training/setup.sh, for a run that has its own working folder.
"""
import json
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

PREVIOUS_STAGE = "stage5_ecg_cot"
CHECKPOINT_FILE = "model_checkpoint.pt"


def main() -> None:
    llm_id, repo = sys.argv[1:3]
    # same rule as CurriculumTrainer._sanitize_llm_id: meta-llama/Llama-3.2-1B -> Llama_3_2_1B
    name = llm_id.split("/")[-1].replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    stage = os.path.join("results", name, "OpenTSLMSP", PREVIOUS_STAGE)
    os.makedirs(os.path.join(stage, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(stage, "results"), exist_ok=True)
    dst = os.path.join(stage, "checkpoints", "best_model.pt")
    shutil.copyfile(hf_hub_download(repo, CHECKPOINT_FILE), dst)
    metrics = os.path.join(stage, "results", "metrics.json")
    with open(metrics, "w") as f:
        json.dump({"seeded_from": f"{repo}/{CHECKPOINT_FILE}"}, f)
    print(f"seeded {dst} ({os.path.getsize(dst) / 1e6:.1f} MB) from {repo}")


if __name__ == "__main__":
    main()
