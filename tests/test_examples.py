import re

import pytest

from hai.baselines import run
from hai.config import SPLITS
from hai.dataset import LOOPS, OFFSETS, WIN


def test_no_attack_crosses_splits(examples):
    att = examples[examples.label == "attack"]
    ids = att.assign(aid=att.attack_ids.str.split()).explode("aid")
    assert (ids.groupby(["version", "aid"]).split.nunique() == 1).all()
    anchors = examples[examples.window_kind == "attack"].assign(aid=lambda d: d.version + "/" + d.anchor)
    assert (anchors.groupby("aid").split.nunique() == 1).all()


def test_splits_follow_the_file_table(examples):
    for r in examples.drop_duplicates(["version", "file"]).itertuples():
        expected = [s for s, by in SPLITS.items() if r.file in by.get(r.version, [])]
        assert [r.split] == (expected or ["train"])


def test_window_shape(examples):
    assert set(examples.loop) == set(LOOPS)
    assert examples.groupby("window_id").loop.nunique().eq(len(LOOPS)).all()
    for r in examples.sample(500, random_state=0).itertuples():
        assert all(len(v) == WIN for v in r.values)


def test_attack_windows_start_where_stated(examples):
    a = examples[(examples.window_kind == "attack") & (examples.label == "attack")]
    anchored = a[a.apply(lambda r: r.anchor in r.attack_ids.split(), axis=1)]
    assert set(anchored.offset) <= set(OFFSETS)
    first = anchored[anchored.attack_ids == anchored.anchor]
    assert (first.attack_start_s == first.offset).all()


def test_normal_windows_are_far_from_attacks(examples):
    assert (examples[examples.window_kind == "normal"].label == "normal").all()


def test_answers_end_with_the_label(examples):
    end = examples.answer.str.extract(r"Answer: (attack|normal)$")[0]
    assert (end == examples.label).all()


def test_answers_hold_no_measured_values(examples):
    for text in examples.answer:
        stripped = re.sub(r"second \d+", "second N", text)
        assert not re.search(r"\d", stripped), text


def test_baselines_refuse_the_test_split(examples):
    with pytest.raises(ValueError):
        run(examples[examples.split == "test"].head(10), {}, {})
