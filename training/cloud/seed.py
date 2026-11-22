"""Put a starting checkpoint where stage6_hai looks for the stage before it, under the current folder.

    uv run python training/cloud/seed.py LLM_ID CHECKPOINT_REPO
    uv run python training/cloud/seed.py LLM_ID /path/to/best_model.pt

The same step as number 5 of training/setup.sh, for a run that has its own working folder.
The second argument is a Hugging Face repo, or an existing local checkpoint file (the saved
model of another run), which is copied as it is.
"""
import json
import os
import shutil
import sys

PREVIOUS_STAGE = "stage5_ecg_cot"
CHECKPOINT_FILE = "model_checkpoint.pt"


def seed(llm_id: str, source: str) -> str:
    """Copy the checkpoint of `source` (a local file or a Hugging Face repo) into place; return where it went."""
    # same rule as CurriculumTrainer._sanitize_llm_id: meta-llama/Llama-3.2-1B -> Llama_3_2_1B
    name = llm_id.split("/")[-1].replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    stage = os.path.join("results", name, "OpenTSLMSP", PREVIOUS_STAGE)
    os.makedirs(os.path.join(stage, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(stage, "results"), exist_ok=True)
    dst = os.path.join(stage, "checkpoints", "best_model.pt")
    if os.path.isfile(source):
        src, seeded_from = source, os.path.abspath(source)
    else:
        from huggingface_hub import hf_hub_download  # only here: a local file needs no download
        src, seeded_from = hf_hub_download(source, CHECKPOINT_FILE), f"{source}/{CHECKPOINT_FILE}"
    shutil.copyfile(src, dst)
    with open(os.path.join(stage, "results", "metrics.json"), "w") as f:
        json.dump({"seeded_from": seeded_from}, f)
    print(f"seeded {dst} ({os.path.getsize(dst) / 1e6:.1f} MB) from {seeded_from}")
    return dst


def main() -> None:
    llm_id, source = sys.argv[1:3]
    seed(llm_id, source)


if __name__ == "__main__":
    main()
