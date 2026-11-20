"""Export the loop examples for OpenTSLM and the TimeNet connector: `uv run hai-export`.

Reads data/processed/examples.parquet and writes

  data/processed/opentslm/train.jsonl          rows with `train_use`
  data/processed/opentslm/validation.jsonl     every validation row
  data/processed/opentslm/test.jsonl           every loop row of each scored test attack window,
                                               plus every loop row of NORMAL_TEST_WINDOWS normal
                                               test windows per version
  data/processed/opentslm/test_full.jsonl      every test row
  data/processed/opentslm/test_shuffled.jsonl  test.jsonl with the seconds permuted inside the window
  data/processed/timenet/examples_table.csv.gz one row per example, no series values

Every draw uses a fixed seed, so two runs write the same bytes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hai.answers import label_of
from hai.config import PROCESSED
from hai.dataset import LOOP_NAMES, SEED
from hai.loaders import tag_map

NORMAL_TEST_WINDOWS = 50  # normal test windows per version in test.jsonl
SPLIT_FILES = ("train", "validation", "test", "test_full", "test_shuffled")

TABLE_COLUMNS = ["example_id", "window_id", "version", "file", "split", "window_kind", "anchor", "scored", "start",
                 "loop", "label", "attack_ids", "loop_uncertain", "attack_start_s", "attack_end_s",
                 "no_visible_change", "reacting", "train_use", "tags", "roles", "source_tags", "answer"]


def read_examples(path: Path = PROCESSED / "examples.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def test_rows(ex: pd.DataFrame) -> np.ndarray:
    """Mask of the rows in test.jsonl: scored attack windows plus a seeded draw of normal windows per version."""
    test = (ex.split == "test").to_numpy()
    keep = test & ex.scored.to_numpy() & (ex.window_kind == "attack").to_numpy()
    rng = np.random.default_rng([SEED, 11])
    for version in sorted(ex.version.unique()):
        ids = np.array(sorted(ex.loc[test & (ex.window_kind == "normal").to_numpy() & (ex.version == version).to_numpy(),
                                     "window_id"].unique()))
        pick = rng.choice(ids, size=min(NORMAL_TEST_WINDOWS, len(ids)), replace=False)
        keep |= ex.window_id.isin(set(pick)).to_numpy()
    return keep


def _number(v: np.float32) -> float:
    """Shortest decimal that reads back as the same float32."""
    return float(str(v))


def record(r) -> dict:
    """One JSONL line. Series are named by role, never by tag."""
    roles = list(r.roles)
    counts = {ro: roles.count(ro) for ro in roles}
    series = [{"role": ro, "name": label_of(t, ro, counts[ro]), "values": [_number(v) for v in np.asarray(x, "float32")]}
              for t, ro, x in zip(r.tags, roles, r.values)]
    return {"example_id": r.example_id, "version": r.version, "loop": r.loop, "loop_name": LOOP_NAMES[r.loop],
            "split": r.split, "label": r.label, "attack_start_s": int(r.attack_start_s), "scored": bool(r.scored),
            "no_visible_change": bool(r.no_visible_change), "reacting": bool(r.reacting), "window_kind": r.window_kind, "answer": r.answer,
            "series": series}


def shuffle_seconds(recs: list[dict]) -> list[dict]:
    """One seeded permutation of the 300 seconds per example, applied to every series of that example."""
    rng = np.random.default_rng([SEED, 13])
    out = []
    for rec in recs:
        n = len(rec["series"][0]["values"])
        perm = rng.permutation(n)
        out.append({**rec, "series": [{**s, "values": [s["values"][i] for i in perm]} for s in rec["series"]]})
    return out


def write_jsonl(recs: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def table(ex: pd.DataFrame) -> pd.DataFrame:
    """The TimeNet table: one row per example, tags and source tags as space-separated text, no series."""
    src = {(r.version, r.standard): r.source for r in tag_map().itertuples()}
    t = ex.copy()
    t["source_tags"] = [" ".join(src[(v, tag)] for tag in tags) for v, tags in zip(t.version, t.tags)]
    t["tags"] = t.tags.map(" ".join)
    t["roles"] = t.roles.map(" ".join)
    t["start"] = t.start.dt.strftime("%Y-%m-%d %H:%M:%S")
    return t[TABLE_COLUMNS]


def write_table(ex: pd.DataFrame, path: Path) -> None:
    table(ex).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def export(ex: pd.DataFrame, out: Path = PROCESSED) -> dict[str, int]:
    """Write every file under `out` (opentslm/ and timenet/). Returns rows per file."""
    (out / "opentslm").mkdir(parents=True, exist_ok=True)
    (out / "timenet").mkdir(parents=True, exist_ok=True)
    ex = ex.reset_index(drop=True)
    parts = {
        "train": ex.train_use.to_numpy(),
        "validation": (ex.split == "validation").to_numpy(),
        "test": test_rows(ex),
        "test_full": (ex.split == "test").to_numpy(),
    }
    counts = {}
    test_recs = None
    for name, mask in parts.items():
        recs = [record(r) for r in ex[mask].itertuples()]
        write_jsonl(recs, out / "opentslm" / f"{name}.jsonl")
        counts[f"opentslm/{name}.jsonl"] = len(recs)
        if name == "test":
            test_recs = recs
    shuffled = shuffle_seconds(test_recs)
    write_jsonl(shuffled, out / "opentslm" / "test_shuffled.jsonl")
    counts["opentslm/test_shuffled.jsonl"] = len(shuffled)
    write_table(ex, out / "timenet" / "examples_table.csv.gz")
    counts["timenet/examples_table.csv.gz"] = len(ex)
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, default=PROCESSED / "examples.parquet")
    ap.add_argument("--out", type=Path, default=PROCESSED, help="folder that gets opentslm/ and timenet/ (default: %(default)s)")
    args = ap.parse_args(argv)
    if not args.examples.exists():
        raise SystemExit(f"{args.examples} missing: run `uv run hai-build` first")
    counts = export(read_examples(args.examples), args.out)
    for name, n in counts.items():
        print(f"{n:>6} rows  {args.out / name}")


if __name__ == "__main__":
    main()
