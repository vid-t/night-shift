import re

import pandas as pd
import pytest

from hai.baselines import run
from hai.build import TRAIN_NORMAL_PER_ATTACK
from hai.config import SPLITS
from hai.dataset import GAP, LOOPS, N_PER_ATTACK, START_HI, START_LO, WIN


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
    assert anchored.offset.between(START_LO, START_HI).all()
    first = anchored[anchored.attack_ids == anchored.anchor]
    assert (first.attack_start_s == first.offset).all()


def test_start_seconds_cannot_be_guessed(examples):
    """Fixed start seconds would let a model score on timing without reading the curves."""
    a = examples[examples.label == "attack"]
    for part in (a, a[a.scored]):
        share = part.attack_start_s.value_counts(normalize=True)
        assert len(share) > 60, f"only {len(share)} distinct start seconds"
        assert share.iloc[0] < 0.05, f"second {share.index[0]} holds {share.iloc[0]:.0%} of the examples"
        assert part.attack_start_s.min() < START_LO + 40 and part.attack_start_s.max() > START_HI - 40


def test_one_scored_window_per_attack(windows):
    a = windows[windows.kind == "attack"]
    per = a.groupby(["version", "anchor"]).agg(n=("scored", "size"), scored=("scored", "sum"))
    assert (per.scored == 1).all() and (per.n <= N_PER_ATTACK).all()
    assert not windows[windows.kind == "normal"].scored.any()


def test_normal_windows_are_far_from_attacks(examples, windows, checks):
    assert (examples[examples.window_kind == "normal"].label == "normal").all()
    c = checks.assign(start=pd.to_datetime(checks.flag_start))
    c["end"] = c.start + pd.to_timedelta(c.flag_duration - 1, unit="s")
    gap = pd.Timedelta(seconds=GAP)
    for (version, stem), w in windows[windows.kind == "normal"].groupby(["version", "file"]):
        for a in c[(c.version == version) & (c.file == stem)].itertuples():
            near = (w.end > a.start - gap) & (w.start < a.end + gap)
            assert not near.any(), f"{version} {stem}: a normal window is within {GAP} s of {a.attack_id}"


def test_no_visible_change_is_an_attack_flag(examples):
    hidden = examples[examples.no_visible_change]
    assert (hidden.label == "attack").all()
    assert hidden.answer.str.startswith("No series shows a clear step").all()
    shown = examples[(examples.label == "attack") & ~examples.no_visible_change]
    assert not shown.answer.str.startswith("No series").any()


def test_training_sample(examples):
    used = examples[examples.train_use]
    assert (used.split == "train").all()
    assert not used.no_visible_change.any()
    train_attacks = examples[(examples.split == "train") & (examples.label == "attack") & ~examples.no_visible_change]
    assert set(train_attacks.example_id) <= set(used.example_id)
    share = (used.label == "attack").mean()
    assert 1 / (2 * TRAIN_NORMAL_PER_ATTACK + 1) - 0.02 <= share <= 0.5
    normal = used[used.label == "normal"]
    assert set(normal.window_kind) == {"attack", "normal"}


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
