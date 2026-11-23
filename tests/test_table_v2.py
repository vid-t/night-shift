"""Table v2 (`hai-build --table v2`): process-flag runs, exact copies, the audit, and that v1 stays the default."""
import gzip
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hai import audit
from hai.config import FROZEN, PROCESSED, out_dir, table_version, tag_map_file, tidy_dir, use_table
from hai.dataset import WIN, exact_copies, loop_seconds, runs_of
from hai.export import TABLE_COLUMNS, TABLE_COLUMNS_V2

V2_TABLE = PROCESSED / "v2" / "timenet" / "examples_table.csv.gz"


@pytest.fixture
def v2():
    use_table("v2")
    yield
    use_table("v1")


def test_v1_is_the_default():
    assert table_version() == "v1"
    assert out_dir() == PROCESSED
    assert tidy_dir() == PROCESSED / "tidy"
    assert tag_map_file() == FROZEN / "tag_map.csv"


def test_v2_writes_to_its_own_folder(v2):
    assert out_dir() == PROCESSED / "v2"
    assert tidy_dir() == PROCESSED / "v2" / "tidy"
    assert tag_map_file() == FROZEN / "tag_map_v2.csv"


def test_unknown_table_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        use_table("v3")


def test_runs_of_finds_every_run_with_both_ends():
    assert runs_of(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert runs_of(np.array([1, 1, 1])) == [(0, 2)]
    assert runs_of(np.zeros(4)) == []


def test_loop_seconds_unites_the_runs_that_touch_the_window():
    t0 = pd.Timestamp("2020-07-14 03:49:00")
    w = SimpleNamespace(start=t0, end=t0 + pd.Timedelta(seconds=WIN - 1))
    s = pd.Timedelta(seconds=1)
    on = pd.DataFrame({"runs": [
        [(t0 + 10 * s, t0 + 12 * s), (t0 + 20 * s, t0 + 25 * s), (t0 + 400 * s, t0 + 410 * s)],
        [(t0 + 24 * s, t0 + 30 * s), (t0 + 290 * s, t0 + 320 * s)],  # a second step of the same loop
    ]})
    assert loop_seconds(on, w) == [(10, 12), (20, 30), (290, 320)]
    assert loop_seconds(on.iloc[:0], w) == []


def test_exact_copies_keep_the_normal_file():
    times = pd.date_range("2020-07-13 00:00:00", periods=900, freq="s")
    same = pd.DataFrame({"time": times, "P1_PIT01": np.arange(900.0), "attack": 0})
    other = same.assign(P1_PIT01=same.P1_PIT01 + 1)
    wins = pd.DataFrame({"file": ["train1", "test3", "test3"],
                         "start": [times[100], times[100], times[400]]})
    copies = exact_copies("21.03", wins, {"train1": same, "test3": same})
    assert list(copies) == ["", "train1", ""]
    assert list(exact_copies("21.03", wins, {"train1": same, "test3": other})) == ["", "", ""]


def test_v2_table_columns_add_runs_and_matches():
    assert TABLE_COLUMNS_V2 == [*TABLE_COLUMNS[:15], "attack_runs", *TABLE_COLUMNS[15:21], "matched_from", "answer"]


def test_parse_runs():
    assert audit.parse_runs("3-5 9-300") == [(3, 5), (9, 300)]
    assert audit.parse_runs("") == []


def test_tag_map_v2_marks_how_each_column_was_matched():
    tm = pd.read_csv(FROZEN / "tag_map_v2.csv", dtype=str, keep_default_na=False)
    by = {(r.version, r.standard): r.matched_from for r in tm.itertuples()}
    assert by[("23.05", "P1_B2016")] == "manual"
    for tag in ("P1_B3004", "P1_B3005", "P1_B4002", "P1_B4022"):
        assert by[("23.05", tag)] == "data"
    assert by[("20.07", "P3_LIT01")] == "manual"
    assert by[("21.03", "P3_LH01")] == "name"
    assert set(tm.matched_from) == {"same_name", "manual", "data", "name", ""}
    assert (tm.matched_from == "").equals(tm.source == "")
    v1 = pd.read_csv(FROZEN / "tag_map.csv", dtype=str, keep_default_na=False)
    changed = tm.source.ne(v1.source)
    assert set(tm[changed].standard) == {"P1_B4002", "P1_B4022"}
    assert (tm[["standard", "loop", "role", "version"]] == v1[["standard", "loop", "role", "version"]]).all().all()


@pytest.mark.skipif(not V2_TABLE.exists(), reason="run `uv run hai-build --table v2 && uv run hai-export --table v2`")
def test_v2_table_passes_the_audit(tmp_path, v2):
    report = audit.run(V2_TABLE, tmp_path / "AUDIT.md")
    assert report.failed == 0, (tmp_path / "AUDIT.md").read_text()
    by = {c.name: c for c in report.checks}
    assert by["hashes"].checked == 17 and by["labels"].checked == 24132


@pytest.mark.skipif(not V2_TABLE.exists(), reason="run `uv run hai-build --table v2 && uv run hai-export --table v2`")
def test_a_shifted_run_fails_the_audit(tmp_path, v2):
    t = pd.read_csv(V2_TABLE, dtype=str, keep_default_na=False)
    i = t.index[t.attack_runs != ""][0]
    a, b = (int(x) for x in t.at[i, "attack_runs"].split()[0].split("-"))
    t.at[i, "attack_runs"] = " ".join([f"{a + 1}-{b}", *t.at[i, "attack_runs"].split()[1:]])
    path = tmp_path / "shifted.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        t.to_csv(f, index=False)
    report = audit.run(path)
    by = {c.name: c for c in report.checks}
    assert by["labels"].failures and by["edges"].failures


@pytest.mark.skipif(not V2_TABLE.exists(), reason="run `uv run hai-build --table v2 && uv run hai-export --table v2`")
def test_v2_no_visible_change_rows_have_no_answer():
    t = pd.read_csv(V2_TABLE, dtype={"version": str}, keep_default_na=False)
    assert ((t.answer == "") == t.no_visible_change).all()
    assert t[t.answer != ""].apply(lambda r: r.answer.endswith(f"Answer: {r.label}"), axis=1).all()
