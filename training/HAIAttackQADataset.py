"""HAI control-loop attack QA dataset for OpenTSLM (stage `stage6_hai`).

setup.sh copies this file to src/opentslm/time_series_datasets/hai/ in the OpenTSLM checkout.

Reads the JSONL files that `uv run hai-export` writes. Environment variables:

  HAI_JSONL_DIR      folder with train.jsonl, validation.jsonl and the test file (required)
  HAI_TEST_FILE      test file name inside that folder (default: test.jsonl;
                     test_shuffled.jsonl or test_full.jsonl for the other runs)
  HAI_OVERFIT_ROWS   N > 0: train, validation and test are the same N training rows,
                     drawn with a fixed seed (the overfit check)
"""
import json
import os
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

ROLE_WORDS = {
    "target": "the value the controller aims for",
    "measurement": "the measured value",
    "command": "what the controller sends to the actuator",
}
OVERFIT_SEED = 0
MIN_SPREAD = 1e-6  # below this (relative to the level) a series counts as flat


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def scale(values) -> tuple[float, float, list[float]]:
    """(mean, spread, scaled values). A flat series has zero spread: its scaled values are all zero."""
    x = np.asarray(values, dtype=np.float64)
    mean, std = float(x.mean()), float(x.std())
    if std <= MIN_SPREAD * max(1.0, abs(mean)):
        return mean, 0.0, [0.0] * len(x)
    return mean, std, ((x - mean) / std).tolist()


class HAIAttackQADataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        folder = os.environ.get("HAI_JSONL_DIR")
        if not folder:
            raise RuntimeError("Set HAI_JSONL_DIR to the folder with the files from `uv run hai-export`")
        folder = Path(folder)
        train = _read_jsonl(folder / "train.jsonl")
        overfit = int(os.environ.get("HAI_OVERFIT_ROWS", "0") or 0)
        if overfit > 0:
            rng = np.random.default_rng(OVERFIT_SEED)
            pick = sorted(rng.choice(len(train), size=min(overfit, len(train)), replace=False).tolist())
            rows = [train[i] for i in pick]
            print(f"HAI overfit check: train, validation and test are the same {len(rows)} training rows")
            return rows, list(rows), list(rows)
        val = _read_jsonl(folder / "validation.jsonl")
        test = _read_jsonl(folder / os.environ.get("HAI_TEST_FILE", "test.jsonl"))
        print(f"HAI rows: train {len(train)}, validation {len(val)}, test {len(test)}")
        return train, val, test

    def _get_answer(self, row) -> str:
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        names = [f"the {s['name']}" for s in row["series"]]
        n = len(row["series"][0]["values"])
        return (
            f"You are given {n} seconds of the {row['loop_name']} control loop of a power plant testbed, "
            f"one reading per second. The series are {_join(names)}. "
            "Say whether the loop behaves normally or is under attack. If it is under attack, "
            "say the second where the change starts. "
            "End with 'Answer: attack' or 'Answer: normal'."
        )

    def _get_post_prompt(self, row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        for s in row["series"]:
            mean, std, scaled = scale(s["values"])
            text = (f"This is the {s['name']}, {ROLE_WORDS[s['role']]}; "
                    f"it has mean {mean:.4f} and std {std:.4f}:")
            prompts.append(TextTimeSeriesPrompt(text, scaled))
        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["example_id"] = row["example_id"]
        sample["label"] = row["label"]
        return sample


if __name__ == "__main__":
    ds = HAIAttackQADataset("train", EOS_TOKEN="")
    print(len(ds), "training samples")
    s = ds[0]
    print(s["pre_prompt"])
    for t in s["time_series_text"]:
        print(t)
    print(s["post_prompt"], s["answer"])
