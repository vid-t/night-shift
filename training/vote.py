r"""Combine several prediction files on the same rows into one, by voting on the verdict (attack or normal).

    uv run python training/vote.py --out runs/vote.jsonl [--min K] \
        runs/<run_a>/test_predictions.jsonl runs/<run_b>/test_predictions.jsonl [...]

Every file must hold the same example ids, each once. A file votes attack on a row when its answer
parses to `attack` (training/score.py `parse`); a normal or unparseable answer is no attack vote.
The row is attack when at least K files vote attack, else normal. K defaults to a strict majority
(2 of 3, 2 of 2, 3 of 4); `--min K` sets it. The answer text is that of the first file, in the order
given, whose own label equals the voted one, so the sentence and the start second come from a model
that agrees; when no file parses to normal on a normal vote, the text is `Answer: normal`. Each output
row also carries `votes` (attack votes), `voters` (files) and `gold` from the first file when it has one.
The output scores with training/score.py or as a `--run` in training/results_table.py like any single model.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAIN_NORMAL = "Answer: normal"


def load_scorer():
    spec = importlib.util.spec_from_file_location("hai_score", HERE / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load_scorer()


def check_ids(rows_per_file: list[list[dict]], names: list[str] | None = None) -> None:
    """Raise ValueError unless every file holds the ids of the first file, each once."""
    names = names or [f"file {i + 1}" for i in range(len(rows_per_file))]
    first = None
    for name, rows in zip(names, rows_per_file):
        ids = [r["example_id"] for r in rows]
        seen = set(ids)
        if len(seen) != len(ids):
            dup = next(i for i in ids if ids.count(i) > 1)
            raise ValueError(f"{name}: {len(ids) - len(seen)} duplicated example ids, e.g. {dup}")
        if first is None:
            first = seen
            continue
        missing, extra = first - seen, seen - first
        if missing or extra:
            example = sorted(missing or extra)[0]
            raise ValueError(f"{name} differs from {names[0]}: {len(missing)} ids missing, {len(extra)} extra, e.g. {example}")


def vote(rows_per_file: list[list[dict]], k: int | None = None, names: list[str] | None = None) -> list[dict]:
    """One row per example of the first file, labelled attack when at least k files say attack."""
    n = len(rows_per_file)
    if n < 2:
        raise ValueError(f"need at least two prediction files, got {n}")
    k = n // 2 + 1 if k is None else k
    if not 1 <= k <= n:
        raise ValueError(f"--min must be between 1 and {n}, got {k}")
    check_ids(rows_per_file, names)
    by_id = [{r["example_id"]: r for r in rows} for rows in rows_per_file]
    out = []
    for first in rows_per_file[0]:
        eid = first["example_id"]
        texts = [rows[eid].get("generated") for rows in by_id]
        labels = [score.parse(t)[0] for t in texts]
        votes = sum(label == "attack" for label in labels)
        label = "attack" if votes >= k else "normal"
        text = next((t for t, lab in zip(texts, labels) if lab == label), PLAIN_NORMAL)
        row = {"example_id": eid, "generated": text, "votes": votes, "voters": n}
        if "gold" in first:
            row["gold"] = first["gold"]
        out.append(row)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help="two or more prediction files on the same rows")
    ap.add_argument("--out", required=True, help="the voted prediction file to write")
    ap.add_argument("--min", type=int, dest="k", metavar="K",
                    help="attack votes needed for attack (default: a strict majority)")
    args = ap.parse_args(argv)
    if len(args.predictions) < 2:
        ap.error("give at least two prediction files")
    try:
        rows = vote([score.read_jsonl(p) for p in args.predictions], args.k, args.predictions)
    except ValueError as e:
        raise SystemExit(f"vote.py: {e}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(args.predictions)
    k = n // 2 + 1 if args.k is None else args.k
    attack = sum(score.parse(r["generated"])[0] == "attack" for r in rows)
    print(f"{len(rows)} rows, {attack} voted attack, K={k} of {n} files -> {args.out}")


if __name__ == "__main__":
    main()
