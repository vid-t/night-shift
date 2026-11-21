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
import re
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


VIEWS = ("zscore", "range")
RANGE_UNIT = 0.02   # one unit in the range view is this share of the series' normal range
RANGE_HEAD_S = 30   # the range view shows the change from the mean of these first seconds


def series_view() -> str:
    """How a series is shown to the model: HAI_SERIES_VIEW, `zscore` (default) or `range`.

    zscore  each window is scaled to mean 0 and spread 1, and the text states its mean and spread.
            What OpenTSLM's own datasets do. Here it hides how large a change is: noise and a real
            step both fill the same range, and the size sits only in two numbers of the text.
    range   the change from the first RANGE_HEAD_S seconds, in units of RANGE_UNIT of the series'
            normal range, squashed with a signed log so a huge step does not drown a small one.
            The text states no numbers. A quiet series stays near zero, a change that the answer
            rule calls visible is about one unit or more.
    """
    view = (os.environ.get("HAI_SERIES_VIEW") or "zscore").strip()
    if view not in VIEWS:
        raise RuntimeError(f"HAI_SERIES_VIEW must be one of {VIEWS}, got {view!r}")
    return view


def range_view(values, normal_range: float) -> list[float]:
    """The series as its change from its first seconds, in units of 2% of its normal range, signed-log squashed."""
    x = np.asarray(values, dtype=np.float64)
    unit = RANGE_UNIT * float(normal_range)
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError("the range view needs a positive `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
    d = (x - x[:RANGE_HEAD_S].mean()) / unit
    return (np.sign(d) * np.log1p(np.abs(d))).tolist()


ORDERS = ("label_last", "label_first", "label_only")
START_SENTENCE = re.compile(r"\s*Change starts near second \d+\.")


def answer_order() -> str:
    """Where the label sits in the answer the model learns: HAI_ANSWER_ORDER, `label_last` (default) or `label_first`.

    label_last   the answer as exported: the sentences, then `Answer: attack` or `Answer: normal`. The
                 model has to commit to a label while it writes the first sentence.
    label_first  `attack. Change starts near second S. <sentences>` or `normal. <sentences>`, right after
                 the prompt's closing `Answer:`. The label is the first word the model writes, so the
                 decision is one clear step.
    label_only   `attack` or `normal` and nothing else. Every bit of the training signal is about the
                 decision, none about wording. No start second and no sentence.
    """
    order = (os.environ.get("HAI_ANSWER_ORDER") or "label_last").strip()
    if order not in ORDERS:
        raise RuntimeError(f"HAI_ANSWER_ORDER must be one of {ORDERS}, got {order!r}")
    return order


def label_first(answer: str) -> str:
    """`<sentences> Change starts near second S. Answer: attack` -> `attack. Change starts near second S. <sentences>`."""
    body, sep, label = answer.rpartition("Answer: ")
    label = label.strip()
    if not sep or label not in ("attack", "normal"):
        raise ValueError(f"not an exported answer: {answer!r}")
    body = body.strip()
    start = ""
    m = START_SENTENCE.search(body)
    if m:
        start, body = m.group().strip(), (body[:m.start()] + body[m.end():]).strip()
    return " ".join(part for part in (label + ".", start, body) if part)


def label_only(answer: str) -> str:
    """`<sentences> Answer: attack` -> `attack`."""
    label = answer.rpartition("Answer: ")[2].strip()
    if label not in ("attack", "normal"):
        raise ValueError(f"not an exported answer: {answer!r}")
    return label


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
        order = answer_order()
        if order == "label_first":
            return label_first(row["answer"])
        if order == "label_only":
            return label_only(row["answer"])
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        names = [f"the {s['name']}" for s in row["series"]]
        n = len(row["series"][0]["values"])
        shown = ("Each series is shown as its change from its first 30 seconds, measured against its normal range. "
                 if series_view() == "range" else "")
        return (
            f"You are given {n} seconds of the {row['loop_name']} control loop of a power plant testbed, "
            f"one reading per second. The series are {_join(names)}. {shown}"
            + {"label_first": "Say first whether the loop is under attack or normal: answer 'attack' or 'normal'. "
                              "If it is under attack, say next the second where the change starts. "
                              "Then say in a sentence or two how the series move.",
               "label_only": "Say whether the loop is under attack or normal: "
                             "answer with the one word 'attack' or 'normal'.",
               "label_last": "Say whether the loop behaves normally or is under attack. If it is under attack, "
                             "say the second where the change starts. "
                             "End with 'Answer: attack' or 'Answer: normal'."}[answer_order()]
        )

    def _get_post_prompt(self, row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        view = series_view()
        for s in row["series"]:
            if view == "range":
                if "scale" not in s:
                    raise RuntimeError("the range view needs `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
                prompts.append(TextTimeSeriesPrompt(f"This is the {s['name']}, {ROLE_WORDS[s['role']]}:",
                                                    range_view(s["values"], s["scale"])))
                continue
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
