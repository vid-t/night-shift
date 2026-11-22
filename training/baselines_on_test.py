"""Final evaluation: the baselines on the test rows, scored like the model.

Run it once, when the design is frozen. `hai.baselines.run` refuses the test split on purpose;
this script is the one place that reads it.

    uv run python training/baselines_on_test.py --out runs/baselines

It trains the gradient boosting baselines on the training sample (`train_use`), answers every
row of the test file in the model's own answer format, and scores the answers with
`training/score.py`. So a baseline and the model get the same counts from the same code.

  range_alarm          attack when the measurement leaves the range of the version's normal file;
                       it states the first second outside the range as the start
  gradient_boosting    statistics of the loop's own series; no start second
  gradient_boosting_with_neighbours
                       the same plus the other loops' measurements; no start second
  always_normal        the floor

`--examples` and `--test` point at another build, `--truth` at the test file whose flags the
scorer should use (default: `--test`). Answers are joined by example id.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from hai.baselines import SEED, features, neighbour_features
from hai.build import norm_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_scorer():
    spec = importlib.util.spec_from_file_location("hai_score", HERE / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alarm_answer(r, ranges: dict) -> str:
    """The range alarm as a sentence: the first second a measurement is outside its normal range."""
    first = None
    for tag, role, x in zip(r.tags, r.roles, r.values):
        if role == "measurement":
            lo, hi = ranges[(r.version, tag)]
            x = np.asarray(x, "float64")
            out = np.flatnonzero((x < lo) | (x > hi))
            if len(out):
                first = int(out[0]) if first is None else min(first, int(out[0]))
    return "Answer: normal" if first is None else f"Change starts near second {first}. Answer: attack"


def answers(ex: pd.DataFrame, test_ids: list[str], split: str = "test") -> dict[str, dict[str, str]]:
    """{baseline: {example id: answer text}} for the rows of `split` named by `test_ids`."""
    ranges, scales = {}, {}
    for version in sorted(ex.version.unique()):
        r, s = norm_stats(version)
        ranges.update(r), scales.update(s)
    ex = ex.reset_index(drop=True)
    wanted = ex.example_id.isin(set(test_ids)).to_numpy()
    if wanted.sum() != len(set(test_ids)):
        raise ValueError(f"{len(set(test_ids)) - int(wanted.sum())} test rows are not in the examples table")
    if (ex[wanted].split != split).any() or ex[ex.train_use].split.ne("train").any():
        raise ValueError(f"the answered rows must be {split} rows and the training sample train rows")
    test = ex[wanted]
    out = {"always_normal": {i: "Answer: normal" for i in test.example_id},
           "range_alarm": {r.example_id: alarm_answer(r, ranges) for r in test.itertuples()}}
    train, y = ex.train_use.to_numpy(), (ex.label == "attack").to_numpy()
    loop_only = features(ex, scales)
    variants = {"gradient_boosting": loop_only,
                "gradient_boosting_with_neighbours": np.c_[loop_only, neighbour_features(ex, scales)]}
    for name, x in variants.items():
        model = HistGradientBoostingClassifier(random_state=SEED).fit(x[train], y[train])
        said = model.predict(x[wanted]).astype(bool)
        out[name] = {i: "Answer: attack" if s else "Answer: normal" for i, s in zip(test.example_id, said)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--examples", default=str(ROOT / "data" / "processed" / "examples.parquet"))
    ap.add_argument("--test", default=str(ROOT / "data" / "processed" / "opentslm" / "test.jsonl"))
    ap.add_argument("--truth", help="test file whose labels and flags the scorer uses (default: --test)")
    ap.add_argument("--split", default="test", choices=("test", "validation"),
                    help="split of the rows in --test; validation scores the baselines on the validation file")
    ap.add_argument("--out", default=str(ROOT / "runs" / "baselines"))
    args = ap.parse_args(argv)

    score = load_scorer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test_rows = score.read_jsonl(args.test)
    truth_rows = score.read_jsonl(args.truth or args.test)
    said = answers(pd.read_parquet(args.examples), [r["example_id"] for r in test_rows], args.split)
    results = {}
    for name, by_id in said.items():
        preds = [{"example_id": i, "generated": text} for i, text in by_id.items()]
        with open(out / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        results[name] = score.score(score.join(preds, truth_rows))
        print(score.report(name, results[name]))
    (out / "scores.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out / 'scores.json'}")


if __name__ == "__main__":
    main()
