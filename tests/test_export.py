import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hai.dataset import LOOP_NAMES, SERIES_LABEL, WIN
from hai.export import NORMAL_TEST_WINDOWS, export

# SHA-256 of the decompressed TimeNet table (hai_examples_table.csv.gz). The label repair (swing test, `reacting`
# column) changed it from 4c99f4e2...137, the table the TimeNet connector was first built on; hand the new file over.
HANDOFF_SHA256 = "a63ff125155b0f3fbd3ad80d6adae1d2b8b04a60df53035e3daf7ff4720ec9e5"
HANDOFF = os.environ.get("HAI_HANDOFF_TABLE")  # optional: path to the hand-off file itself

KEYS = {"example_id", "version", "loop", "loop_name", "split", "label", "attack_start_s", "scored",
        "no_visible_change", "reacting", "window_kind", "answer", "series"}


@pytest.fixture(scope="module")
def exported(examples, tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    counts = export(examples, out)
    return out, counts


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def digest(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_row_counts(examples, exported):
    out, counts = exported
    ex = examples
    test = ex[ex.split == "test"]
    scored_rows = test[test.scored & (test.window_kind == "attack")]
    normal_windows = test[test.window_kind == "normal"].groupby("version").window_id.nunique()
    expected_test = len(scored_rows) + len(ex.loop.unique()) * int(np.minimum(normal_windows, NORMAL_TEST_WINDOWS).sum())
    assert counts["opentslm/train.jsonl"] == int(ex.train_use.sum())
    assert counts["opentslm/validation.jsonl"] == int((ex.split == "validation").sum())
    assert counts["opentslm/test.jsonl"] == expected_test
    assert counts["opentslm/test_full.jsonl"] == len(test)
    assert counts["opentslm/test_shuffled.jsonl"] == expected_test
    for name, n in counts.items():
        if name.endswith(".jsonl"):
            assert sum(1 for _ in open(out / name)) == n


def test_schema_matches_the_parquet(examples, exported):
    out, _ = exported
    by_id = examples.set_index("example_id")
    for name in ("train", "validation", "test", "test_full"):
        recs = read(out / "opentslm" / f"{name}.jsonl")
        assert len({r["example_id"] for r in recs}) == len(recs)
        for rec in recs[:: max(1, len(recs) // 200)]:
            assert set(rec) == KEYS
            row = by_id.loc[rec["example_id"]]
            for k in ("version", "loop", "split", "label", "attack_start_s", "scored", "no_visible_change",
                      "reacting", "window_kind", "answer"):
                assert rec[k] == row[k], (name, k)
            assert rec["loop_name"] == LOOP_NAMES[rec["loop"]]
            assert [s["role"] for s in rec["series"]] == list(row.roles)
            for s, tag, values in zip(rec["series"], row.tags, row["values"]):
                assert s["name"] != tag and s["name"] in set(SERIES_LABEL.values()) | {"target", "measurement", "command"}
                assert len(s["values"]) == WIN
                assert np.array_equal(np.asarray(s["values"], "float32"), values)


def test_split_membership(examples, exported):
    out, _ = exported
    train = read(out / "opentslm" / "train.jsonl")
    assert {r["example_id"] for r in train} == set(examples[examples.train_use].example_id)
    test = read(out / "opentslm" / "test.jsonl")
    assert {r["split"] for r in test} == {"test"}
    kinds = {r["window_kind"] for r in test}
    assert kinds == {"attack", "normal"}
    assert all(r["scored"] for r in test if r["window_kind"] == "attack")
    scored_ids = set(examples[(examples.split == "test") & examples.scored].example_id)
    assert scored_ids <= {r["example_id"] for r in test}


def test_shuffled_permutes_seconds_inside_each_example(exported):
    out, _ = exported
    plain = read(out / "opentslm" / "test.jsonl")
    shuffled = read(out / "opentslm" / "test_shuffled.jsonl")
    assert [r["example_id"] for r in plain] == [r["example_id"] for r in shuffled]
    moved = 0
    for a, b in zip(plain, shuffled):
        assert {k: v for k, v in a.items() if k != "series"} == {k: v for k, v in b.items() if k != "series"}
        x = np.array([s["values"] for s in a["series"]])
        y = np.array([s["values"] for s in b["series"]])
        # one permutation shared by all series: each column (second) of y is a column of x
        cols = {tuple(c) for c in x.T}
        assert all(tuple(c) in cols for c in y.T)
        assert np.array_equal(np.sort(x, axis=1), np.sort(y, axis=1))
        moved += not np.array_equal(x, y)
    assert moved > 0.9 * len(plain)


def test_reacting_is_exported(examples, exported):
    out, _ = exported
    want = set(examples[examples.reacting & (examples.split == "test")].example_id)
    got = {r["example_id"] for r in read(out / "opentslm" / "test_full.jsonl") if r["reacting"]}
    assert want and got == want
    table = pd.read_csv(out / "timenet" / "examples_table.csv.gz")
    assert table.reacting.dtype == bool and int(table.reacting.sum()) == int(examples.reacting.sum())


def test_export_is_deterministic(examples, exported, tmp_path):
    out, _ = exported
    export(examples, tmp_path)
    assert digest(out) == digest(tmp_path)


def test_timenet_table_equals_the_handoff_file(exported):
    out, _ = exported
    content = gzip.decompress((out / "timenet" / "examples_table.csv.gz").read_bytes())
    assert hashlib.sha256(content).hexdigest() == HANDOFF_SHA256
    if HANDOFF and Path(HANDOFF).exists():
        assert content == gzip.decompress(Path(HANDOFF).read_bytes())
