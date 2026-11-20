"""The label repair: the swing test, swing eligibility, `reacting` rows and the training-sample exclusion."""
import numpy as np
import pandas as pd
import pytest

from hai.answers import (SWING_ELIGIBLE_MAX_RATE, attack_text, normal_text, swing, swing_onset, turns)
from hai.build import eligible_loops, process, swing_eligibility

SCALE = 10.0
rng = np.random.default_rng(0)


def quiet(n, level=5.0, noise=0.001):
    return level + rng.normal(0, noise, n)


def swinging(n, level=5.0, amp=0.5, period=12):
    return level + amp * np.sin(2 * np.pi * np.arange(n) / period)


# --- the swing test on made-up series -----------------------------------------

def test_swing_sees_a_series_that_starts_and_stops_oscillating():
    assert swing(quiet(60), swinging(120), SCALE) == "starts swinging"
    assert swing(swinging(60), quiet(120), SCALE) == "stops swinging"


def test_swing_ignores_steady_noise_and_tiny_wiggles():
    assert swing(quiet(60), quiet(120), SCALE) is None
    assert swing(swinging(60), swinging(120), SCALE) is None
    # ten times the noise, but far below the floor of the tag's scale
    assert swing(quiet(60, noise=1e-5), quiet(120, noise=1e-4), SCALE) is None


def test_swing_ignores_ramps_humps_and_steps():
    """These raise the step noise too; the turns condition keeps them out."""
    before = quiet(60)
    ramp = 5.0 + np.linspace(0, 3, 120)
    hump = 5.0 + 2 * np.sin(np.linspace(0, np.pi, 120))
    step = np.r_[np.full(30, 5.0), np.full(90, 8.0)]
    for after in (ramp, hump, step):
        assert turns(after) < 3
        assert swing(before, after, SCALE) is None


def test_swing_needs_enough_seconds():
    assert swing(quiet(4), swinging(120), SCALE) is None
    assert swing(quiet(60), swinging(4), SCALE) is None


def test_swing_onset_dates_the_change():
    x = np.r_[quiet(150), swinging(150)]
    t = swing_onset(x, 120, x[60:120], 300, SCALE, "starts swinging")
    assert 140 <= t <= 152
    y = np.r_[swinging(150), quiet(150)]
    t = swing_onset(y, 120, y[60:120], 300, SCALE, "stops swinging")
    assert 140 <= t <= 152


def test_attack_text_uses_the_swing_words_only_when_eligible():
    target, meas = np.full(300, 5.0), quiet(300)
    cmd = np.r_[quiet(150), swinging(150)]
    args = (["T", "M", "C"], ["target", "measurement", "command"], [target, meas, cmd], 140, 299, {"T": SCALE, "M": SCALE, "C": SCALE})
    text, hidden = attack_text(*args, swing_ok=True)
    assert text.startswith("The command starts swinging from second 1")
    assert "while the target and measurement stay flat" in text
    assert text.endswith("Change starts near second 140. Answer: attack") and not hidden
    text, hidden = attack_text(*args, swing_ok=False)
    assert text.startswith("No series shows a clear step") and hidden


def test_normal_text_mentions_swinging_only_when_eligible():
    args = (["T", "M", "C"], ["target", "measurement", "command"],
            [np.full(300, 5.0), quiet(300), np.r_[swinging(150), quiet(150)]], {"T": SCALE, "M": SCALE, "C": SCALE})
    assert normal_text(*args, swing_ok=True) == \
        "The target and the measurement stay flat. The command stops swinging. Answer: normal"
    assert normal_text(*args, swing_ok=False) == \
        "The target and the measurement stay flat. The command stays flat. Answer: normal"


def test_process_is_the_prefix():
    assert [process(lp) for lp in ("P1-PC", "P1-TC", "P2-SC", "P3-LC")] == ["P1", "P1", "P2", "P3"]


# --- eligibility --------------------------------------------------------------

def _table(rates: dict) -> pd.DataFrame:
    return pd.DataFrame([{"version": "all", "loop": lp, "rate": r, "eligible": r < SWING_ELIGIBLE_MAX_RATE}
                         for lp, r in rates.items()])


def test_eligible_loops_reads_the_pooled_rows():
    assert eligible_loops(_table({"P1-PC": 0.02, "P1-TC": 0.15})) == {"P1-PC"}


@pytest.fixture(scope="module")
def eligibility(examples):
    from hai.build import norm_stats
    scales = {}
    for v in sorted(examples.version.unique()):
        scales.update(norm_stats(v)[1])
    return swing_eligibility(examples, scales), swing_eligibility(examples, scales)


def test_eligibility_is_reproducible_and_never_reads_test(eligibility, examples):
    a, b = eligibility
    pd.testing.assert_frame_equal(a, b)
    pooled = a[a.version == "all"].set_index("loop")
    quiet_seen = examples[(examples.window_kind == "normal") & (examples.split != "test")]
    assert (pooled.quiet_examples == quiet_seen.groupby("loop").size()).all()
    assert (pooled.eligible == (pooled.rate < SWING_ELIGIBLE_MAX_RATE)).all()
    assert not pooled.loc["P1-TC", "eligible"]  # swings on its own
    assert pooled.loc[["P1-PC", "P1-LC", "P1-FC", "P2-SC"], "eligible"].all()


# --- reacting rows and the training sample -----------------------------------

def test_reacting_rows_are_normal_neighbours_of_a_same_process_attack(examples, eligibility):
    eligible = eligible_loops(eligibility[0])
    r = examples[examples.reacting]
    assert len(r) > 0
    assert (r.label == "normal").all() and (r.window_kind == "attack").all()
    assert r.loop.isin(eligible).all()
    attacked = examples[(examples.label == "attack") & (examples.attack_start_s >= 10)]
    pairs = set(zip(attacked.window_id, attacked.loop.map(process)))
    assert all((w, process(lp)) in pairs for w, lp in zip(r.window_id, r.loop))
    assert not examples[examples.window_kind == "normal"].reacting.any()


def test_reacting_rows_are_never_trained_on(examples):
    assert not examples[examples.train_use].reacting.any()


def test_hidden_attacks_show_no_swing_words(examples):
    hidden = examples[examples.no_visible_change]
    assert not hidden.answer.str.contains("swinging").any()
