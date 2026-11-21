"""Tests for training/: the stage patcher, the scorer and the dataset class. No GPU, no network."""
import json
import re

import pytest

from tests.training_helpers import OPENTSLM, load_script, make_examples, opentslm_without_init, pinned_file

add_stage = load_script("add_stage")
score = load_script("score")


# --- patcher ---------------------------------------------------------------

def test_patcher_applies_to_the_pinned_commit():
    src = pinned_file("curriculum_learning.py")
    new = add_stage.patch(src)
    assert new != src
    assert add_stage.IMPORT_LINE in new
    assert '    "stage5_ecg_cot",\n    "stage6_hai",\n]' in new
    assert "    def stage6_hai(\n" in new and "dataset_class=HAIAttackQADataset" in new
    assert 'elif stage == "stage6_hai":' in new
    assert new.count(add_stage.LORA_NEW) == 2 and add_stage.LORA_OLD not in new
    assert 'result["example_id"] = sample["example_id"]' in new
    compile(new, "curriculum_learning.py", "exec")


def test_patcher_is_idempotent(tmp_path):
    path = tmp_path / "curriculum_learning.py"
    path.write_text(pinned_file("curriculum_learning.py"))
    add_stage.main([str(path)])
    once = path.read_text()
    add_stage.main([str(path)])
    assert path.read_text() == once
    assert add_stage.patch(once) == once


def test_patcher_fails_loudly_on_a_missing_anchor(tmp_path):
    src = pinned_file("curriculum_learning.py").replace('    "stage5_ecg_cot",\n]', '    "stage5_ecg",\n]')
    path = tmp_path / "curriculum_learning.py"
    path.write_text(src)
    with pytest.raises(SystemExit, match="CURRICULUM_STAGES"):
        add_stage.main([str(path)])
    assert path.read_text() == src  # untouched


# --- scorer ----------------------------------------------------------------

def predictions(rows, answer_of, with_id=True):
    out = []
    for r in rows:
        p = {"pre_prompt": "", "time_series_text": [], "post_prompt": "Answer:", "generated": answer_of(r),
             "gold": r["answer"] + "<|end_of_text|>"}
        if with_id:
            p["example_id"] = r["example_id"]
        out.append(p)
    return out


def overall(result):
    return next(v for v in result["versions"] if v["version"] == "all")


def run(rows, answer_of, with_id=True):
    return score.score(score.join(predictions(rows, answer_of, with_id), rows))


def test_parse():
    assert score.parse("The measurement rises. Change starts near second 87. Answer: attack") == ("attack", 87)
    assert score.parse("The change is already under way when the window opens. Answer: attack") == ("attack", 0)
    assert score.parse("Stays flat. Answer: normal and then Answer: attack") == ("normal", None)
    for bad in ("", None, "maybe attack", "Answer:", "Answer: maybe", "The loop is under attack."):
        assert score.parse(bad) == (None, None)
    assert score.parse("attack") == ("attack", None)   # a label-first answer opens with its label


def test_perfect_answers_find_every_attack():
    rows = make_examples()
    res = overall(run(rows, lambda r: r["answer"]))
    assert res["attacks"] == 8
    assert res["found"] == res["right_loop_only"] == res["timed"] == 8
    assert res["false_alarms_per_100"] == 0.0
    assert res["valid_share"] == 1.0 and res["label_right_share"] == 1.0
    loops = {(l["version"], l["kind"]): l for l in run(rows, lambda r: r["answer"])["loops"]}
    assert loops[("all", "no_visible_change")]["examples"] == 1
    assert loops[("all", "visible")]["examples"] == 7 and loops[("all", "visible")]["answered_attack"] == 7


def test_perfect_answers_join_by_order_without_ids():
    rows = make_examples()
    assert overall(run(rows, lambda r: r["answer"], with_id=False))["found"] == 8


def test_order_join_refuses_the_wrong_test_file():
    rows = make_examples()
    preds = predictions(rows, lambda r: r["answer"], with_id=False)
    with pytest.raises(ValueError):
        score.join(preds, list(reversed(rows)))


def test_always_normal_finds_nothing():
    res = overall(run(make_examples(), lambda r: "Everything stays flat. Answer: normal"))
    assert res["found"] == res["right_loop_only"] == res["timed"] == 0
    assert res["false_alarms_per_100"] == 0.0


def test_always_second_120_scores_low_on_timing():
    rows = make_examples(n_attack_windows=20)
    res = overall(run(rows, lambda r: "Change starts near second 120. Answer: attack"))
    assert res["found"] == 20
    assert res["right_loop_only"] == 0
    assert res["timed"] <= 6  # true starts spread over 30..270; only those within 30 s of 120 count
    guess = run(rows, lambda r: r["answer"])["constant_guess"]
    assert guess["attacks"] == 20 and guess["timed_at_120"] == res["timed"]
    assert guess["timed_at_best"] >= guess["timed_at_120"]
    assert res["false_alarms_per_100"] == 100.0


def test_empty_answers_count_as_wrong():
    rows = make_examples()
    res = overall(run(rows, lambda r: ""))
    assert res["found"] == res["right_loop_only"] == res["timed"] == 0
    assert res["valid_share"] == 0.0 and res["label_right_share"] == 0.0
    assert res["false_alarms_per_100"] == 100.0  # an empty answer on a normal loop is not a `normal`


def test_missing_predictions_count_as_wrong():
    rows = make_examples()
    preds = [p for p in predictions(rows, lambda r: r["answer"]) if p["example_id"].split("/")[-1] != "P1-PC"]
    res = overall(score.score(score.join(preds, rows)))
    assert res["right_loop_only"] < res["attacks"]
    assert res["valid_share"] < 1.0


def _with_reacting(rows):
    """Mark one normal loop of each scored attack window as reacting."""
    seen = set()
    for r in rows:
        w = r["example_id"].rsplit("/", 1)[0]
        if r["window_kind"] == "attack" and r["label"] == "normal" and w not in seen:
            r["reacting"] = True
            seen.add(w)
    return rows, seen


def test_reacting_loops_do_not_count_for_right_loop_only():
    rows, windows = _with_reacting(make_examples())
    # a perfect reader that also says attack on the reacting loop
    res = overall(run(rows, lambda r: r["answer"] if not r.get("reacting") else "Change starts near second 50. Answer: attack"))
    assert res["right_loop_only"] == res["attacks"] == 8
    assert res["reacting"] == len(windows) == 8 and res["reacting_said_attack"] == 8
    assert res["false_alarms_per_100"] == 0.0
    # without the mark the same answers break right_loop_only in every window
    plain = [{k: v for k, v in r.items() if k != "reacting"} for r in rows]
    res = overall(run(plain, lambda r: r["answer"] if r["example_id"] not in {x["example_id"] for x in rows if x.get("reacting")}
                      else "Answer: attack"))
    assert res["right_loop_only"] == 0 and res["reacting"] == 0


def test_strict_count_keeps_the_old_rule():
    rows, windows = _with_reacting(make_examples())
    # says attack on the reacting loop: fine for right_loop_only, wrong for the strict count
    res = overall(run(rows, lambda r: r["answer"] if not r.get("reacting") else "Change starts near second 50. Answer: attack"))
    assert res["right_loop_only"] == len(windows) and res["right_loop_strict"] == 0
    # a perfect reader is right under both rules
    res = overall(run(rows, lambda r: r["answer"]))
    assert res["right_loop_only"] == res["right_loop_strict"] == len(windows)
    # without the field the two counts are the same number
    plain = [{k: v for k, v in r.items() if k != "reacting"} for r in rows]
    res = overall(run(plain, lambda r: r["answer"]))
    assert res["right_loop_only"] == res["right_loop_strict"] == len(windows)


def test_reacting_loops_still_need_the_attacked_loop():
    rows, _ = _with_reacting(make_examples())
    res = overall(run(rows, lambda r: "Answer: normal"))
    assert res["right_loop_only"] == res["found"] == 0 and res["reacting_said_attack"] == 0


def test_score_report_prints_the_reacting_line():
    rows, _ = _with_reacting(make_examples())
    text = score.report("plain", run(rows, lambda r: r["answer"]))
    assert "Reacting loops (normal, not counted in right_loop_only): 8 in the scored windows, 0 answered attack." in text


def test_score_cli_scores_plain_and_shuffled(tmp_path, capsys):
    rows = make_examples()
    for name in ("plain", "shuffled"):
        (tmp_path / f"{name}_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    perfect = predictions(rows, lambda r: r["answer"])
    blind = predictions(rows, lambda r: "Answer: normal")
    (tmp_path / "plain_pred.jsonl").write_text("".join(json.dumps(p) + "\n" for p in perfect))
    (tmp_path / "shuffled_pred.jsonl").write_text("".join(json.dumps(p) + "\n" for p in blind))
    score.main(["--run", "plain", str(tmp_path / "plain_pred.jsonl"), str(tmp_path / "plain_test.jsonl"),
                "--run", "shuffled", str(tmp_path / "shuffled_pred.jsonl"), str(tmp_path / "shuffled_test.jsonl"),
                "--json", str(tmp_path / "scores.json")])
    out = capsys.readouterr().out
    assert "== plain ==" in out and "== shuffled ==" in out
    res = json.loads((tmp_path / "scores.json").read_text())
    assert overall(res["plain"])["found"] == 8 and overall(res["shuffled"])["found"] == 0


# --- dataset class ---------------------------------------------------------

@pytest.fixture
def hai_dataset(tmp_path, monkeypatch):
    opentslm_without_init()
    rows = make_examples(n_attack_windows=2, n_normal_windows=2)[:20]
    rows[0]["series"][1]["values"] = [3.0] * 300  # flat measurement: zero spread
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("HAI_JSONL_DIR", str(tmp_path))
    mod = load_script("HAIAttackQADataset")

    def build(split, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        cls = type("HAIAttackQADatasetUnderTest", (mod.HAIAttackQADataset,), {})  # fresh class: QADataset caches per class
        return cls(split, EOS_TOKEN="<eos>")

    return rows, build, mod


def test_dataset_builds_prompts_for_20_rows(hai_dataset):
    import numpy as np

    rows, build, _ = hai_dataset
    ds = build("train")
    assert len(ds) == 20 and len(build("test")) == 20
    for i, s in enumerate(ds):
        r = rows[i]
        assert "boiler level control loop" in s["pre_prompt"]
        assert "the target, the measurement and the command" in s["pre_prompt"]
        assert "under attack" in s["pre_prompt"] and "P1_" not in s["pre_prompt"]
        assert s["post_prompt"] == "Answer:"
        assert s["answer"] == r["answer"] + "<eos>"
        assert s["example_id"] == r["example_id"]
        assert len(s["time_series"]) == len(s["time_series_text"]) == 3
        for text, x in zip(s["time_series_text"], s["time_series"]):
            assert "mean" in text and "std" in text
            x = np.asarray(x)
            assert x.shape == (300,) and np.isfinite(x).all()
    flat_text, flat = ds[0]["time_series_text"][1], np.asarray(ds[0]["time_series"][1])
    assert "mean 3.0000 and std 0.0000" in flat_text and (flat == 0).all()
    moving = np.asarray(ds[0]["time_series"][2])
    assert abs(moving.mean()) < 1e-6 and abs(moving.std() - 1) < 1e-6


def test_dataset_overfit_option_limits_rows(hai_dataset):
    _, build, _ = hai_dataset
    train = build("train", HAI_OVERFIT_ROWS="5")
    assert len(train) == 5
    ids = [s["example_id"] for s in train]
    cls = type(train)
    assert [s["example_id"] for s in cls("test", EOS_TOKEN="<eos>")] == ids
    assert [s["example_id"] for s in cls("validation", EOS_TOKEN="<eos>")] == ids


def test_dataset_needs_the_folder(hai_dataset, monkeypatch):
    _, build, _ = hai_dataset
    monkeypatch.delenv("HAI_JSONL_DIR")
    with pytest.raises(RuntimeError, match="HAI_JSONL_DIR"):
        build("train")


# --- the share of attack rows in training ----------------------------------------

def _share_rows():
    """10 attack, 20 neighbour-normal and 20 quiet-normal rows."""
    rows = [{"example_id": f"a{i}", "label": "attack", "window_kind": "attack"} for i in range(10)]
    rows += [{"example_id": f"n{i}", "label": "normal", "window_kind": "attack"} for i in range(20)]
    rows += [{"example_id": f"q{i}", "label": "normal", "window_kind": "normal"} for i in range(20)]
    return [rows[i] for i in __import__("numpy").random.default_rng(1).permutation(len(rows))]  # mixed order


def _ids(rows):
    return [r["example_id"] for r in rows]


def test_dataset_train_rows_are_the_file_rows_by_default(hai_dataset, monkeypatch):
    rows, build, _ = hai_dataset
    monkeypatch.delenv("HAI_ATTACK_SHARE", raising=False)
    assert [s["example_id"] for s in build("train")] == _ids(rows)
    monkeypatch.setenv("HAI_ATTACK_SHARE", "")
    assert [s["example_id"] for s in build("train")] == _ids(rows)


def test_with_attack_share_redraws_the_rows():
    from collections import Counter

    mod = load_script("HAIAttackQADataset")
    rows = _share_rows()
    for share in (0.35, 0.5):
        out = mod.with_attack_share(rows, share)
        assert len(out) == len(rows)
        attack = [r for r in out if r["label"] == "attack"]
        assert len(attack) == round(share * len(rows))
        assert {r["example_id"] for r in attack} == {r["example_id"] for r in rows if r["label"] == "attack"}
        near = sum(r["label"] == "normal" and r["window_kind"] == "attack" for r in out)
        quiet = sum(r["label"] == "normal" and r["window_kind"] == "normal" for r in out)
        assert abs(near - quiet) <= 1                                  # 20 : 20 in the input
        assert set(_ids(out)) <= set(_ids(rows))
        assert _ids(mod.with_attack_share(rows, share)) == _ids(out)
        order = {r["example_id"]: i for i, r in enumerate(rows)}
        assert [order[i] for i in _ids(out)] == sorted(order[i] for i in _ids(out))
        assert all(any(o is r for r in rows) for o in out)             # the same dicts, no copies
        counts = Counter(r["example_id"] for r in attack)
        assert max(counts.values()) - min(counts.values()) <= 1        # every attack row about equally often


def test_with_attack_share_below_the_current_share_draws_without_replacement():
    from collections import Counter

    mod = load_script("HAIAttackQADataset")
    rows = _share_rows()
    out = mod.with_attack_share(rows, 0.1)
    attack = Counter(r["example_id"] for r in out if r["label"] == "attack")
    assert sum(attack.values()) == 5 and max(attack.values()) == 1
    assert len(out) == len(rows)


def test_with_attack_share_needs_both_labels():
    mod = load_script("HAIAttackQADataset")
    rows = _share_rows()
    with pytest.raises(RuntimeError):
        mod.with_attack_share([r for r in rows if r["label"] == "attack"], 0.5)
    with pytest.raises(RuntimeError):
        mod.with_attack_share([r for r in rows if r["label"] == "normal"], 0.5)
    only_near = mod.with_attack_share([r for r in rows if r["window_kind"] == "attack"], 0.5)
    assert len(only_near) == 30 and sum(r["label"] == "attack" for r in only_near) == 15


def test_dataset_attack_share_changes_only_the_training_rows(hai_dataset):
    rows, build, _ = hai_dataset
    train = build("train", HAI_ATTACK_SHARE="0.5")
    cls = type(train)
    assert len(train) == len(rows) == 20
    assert sum(s["label"] == "attack" for s in train) == 10
    for split in ("validation", "test"):
        assert [s["example_id"] for s in cls(split, EOS_TOKEN="<eos>")] == _ids(rows)


def test_dataset_attack_share_in_the_overfit_check(hai_dataset):
    _, build, _ = hai_dataset
    plain = build("train", HAI_OVERFIT_ROWS="10")
    picked = [s["example_id"] for s in type(plain)("test", EOS_TOKEN="<eos>")]
    assert [s["example_id"] for s in plain] == picked
    train = build("train", HAI_OVERFIT_ROWS="10", HAI_ATTACK_SHARE="0.5")
    cls = type(train)
    assert [s["example_id"] for s in cls("test", EOS_TOKEN="<eos>")] == picked
    assert [s["example_id"] for s in cls("validation", EOS_TOKEN="<eos>")] == picked
    assert len(train) == 10 and sum(s["label"] == "attack" for s in train) == 5
    assert {s["example_id"] for s in train} <= set(picked)


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "abc", "-0.2", "nan"])
def test_attack_share_setting_is_checked(hai_dataset, monkeypatch, bad):
    _, build, mod = hai_dataset
    monkeypatch.setenv("HAI_ATTACK_SHARE", bad)
    with pytest.raises(RuntimeError, match="HAI_ATTACK_SHARE"):
        mod.attack_share()
    with pytest.raises(RuntimeError, match="HAI_ATTACK_SHARE"):
        build("train")


def test_run_variant_reads_the_attack_share():
    import importlib.util  # main() needs a rented machine: check the parser only

    from tests.training_helpers import ROOT
    spec = importlib.util.spec_from_file_location("run_variant", ROOT / "training" / "cloud" / "run_variant.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    assert rv.arguments(["x"]).attack_share is None
    assert rv.arguments(["x", "--attack-share", "0.35"]).attack_share == 0.35


# --- mirrored copies of training rows -------------------------------------------

MIRRORS = [
    ("The command falls from second 127 while the target stays flat. The measurement falls after it. "
     "Change starts near second 116. Answer: attack",
     "The command rises from second 127 while the target stays flat. The measurement rises after it. "
     "Change starts near second 116. Answer: attack"),
    ("The measurement rises from second 40 while the target and the command stay flat. The command falls with it. "
     "Change starts near second 38. Answer: attack",
     "The measurement falls from second 40 while the target and the command stay flat. The command rises with it. "
     "Change starts near second 38. Answer: attack"),
    ("The command starts swinging from second 90. The measurement rises after it. Change starts near second 88. "
     "Answer: attack",
     "The command starts swinging from second 90. The measurement falls after it. Change starts near second 88. "
     "Answer: attack"),
    ("The measurement follows the target as it rises. The command falls. Answer: normal",
     "The measurement follows the target as it falls. The command rises. Answer: normal"),
    ("The target falls and the measurement rises. The commands move. Answer: normal",
     "The target rises and the measurement falls. The commands move. Answer: normal"),
    ("The target stays flat while the measurement falls. The command stops swinging. Answer: normal",
     "The target stays flat while the measurement rises. The command stops swinging. Answer: normal"),
    ("The measurement moves. The command rises. Answer: normal",
     "The measurement moves. The command falls. Answer: normal"),
]


def test_mirror_answer_swaps_rises_and_falls():
    mod = load_script("HAIAttackQADataset")
    for answer, want in MIRRORS:
        assert mod.mirror_answer(answer) == want
        assert mod.mirror_answer(want) == answer
        assert mod.mirror_answer(mod.mirror_answer(answer)) == answer
        assert score.parse(mod.mirror_answer(answer)) == score.parse(answer)   # same label and start second
    for plain in ("The target and the measurement stay flat. The command stays flat. Answer: normal",
                  "No series shows a clear step. Change starts near second 104. Answer: attack",
                  "The change is already under way when the window opens. Answer: attack",
                  "Change starts near second 7. Answer: attack"):
        assert mod.mirror_answer(plain) == plain
    assert mod.mirror_answer("risesx falls1 rises.") == "risesx falls1 falls."      # whole words only


def _mirror_source():
    head = [1.0, 3.0] * 15                                         # first 30 values: mean 2
    return {"example_id": "w/P1-LC", "label": "attack", "attack_start_s": 40, "window_kind": "attack",
            "answer": "The measurement rises from second 41. Change starts near second 40. Answer: attack",
            "series": [{"role": "target", "name": "target", "values": [5.0] * 60, "scale": 2.0},
                       {"role": "measurement", "name": "measurement", "values": head + [2.0] * 10 + [10.0] * 20,
                        "scale": 4.0}]}


def test_mirror_row_flips_every_series_around_its_first_seconds():
    import copy

    mod = load_script("HAIAttackQADataset")
    row = _mirror_source()
    before = copy.deepcopy(row)
    out = mod.mirror_row(row)
    assert row == before                                           # the input row and its series are untouched
    assert out is not row and all(a is not b for a, b in zip(out["series"], row["series"]))
    assert out["series"][0]["values"] == [5.0] * 60
    assert out["series"][1]["values"] == [3.0, 1.0] * 15 + [2.0] * 10 + [-6.0] * 20   # 2 * 2 - v
    assert out["answer"] == "The measurement falls from second 41. Change starts near second 40. Answer: attack"
    assert out["mirrored"] is True and "mirrored" not in row
    for key in ("example_id", "label", "attack_start_s", "window_kind"):
        assert out[key] == row[key]
    assert [s["scale"] for s in out["series"]] == [2.0, 4.0]
    assert [(s["role"], s["name"]) for s in out["series"]] == [(s["role"], s["name"]) for s in row["series"]]
    twice = mod.mirror_row(out)
    assert [s["values"] for s in twice["series"]] == [s["values"] for s in row["series"]]
    assert twice["answer"] == row["answer"]


def test_mirrored_row_has_the_negative_range_view():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    row = _mirror_source()
    out = mod.mirror_row(row)
    for s, m in zip(row["series"], out["series"]):
        assert np.allclose(mod.range_view(m["values"], m["scale"]), -np.asarray(mod.range_view(s["values"], s["scale"])))


def _mirror_rows(n=10):
    return [{"example_id": f"r{i}", "label": "attack" if i % 3 == 0 else "normal", "window_kind": "attack",
             "answer": f"The command rises from second {40 + i}. Answer: attack",
             "series": [{"role": "command", "name": "command", "values": [float(i)] * 30 + [i + 1.0] * 30}]}
            for i in range(n)]


def test_with_mirrors_adds_mirrored_copies_after_the_rows():
    mod = load_script("HAIAttackQADataset")
    rows = _mirror_rows(10)
    for share, added in ((0.5, 5), (0.25, 2), (1.0, 10)):
        out = mod.with_mirrors(rows, share)
        assert len(out) == len(rows) + added == len(rows) + round(share * len(rows))
        assert all(a is b for a, b in zip(out, rows))              # the input rows themselves, in order
        copies = out[len(rows):]
        order = {r["example_id"]: i for i, r in enumerate(rows)}
        positions = [order[c["example_id"]] for c in copies]
        assert positions == sorted(set(positions))                 # distinct originals, in their order
        for c, p in zip(copies, positions):
            assert c == mod.mirror_row(rows[p])
        assert mod.with_mirrors(rows, share) == out
    assert [c["example_id"] for c in mod.with_mirrors(rows, 1.0)[10:]] == _ids(rows)
    assert all("mirrored" not in r for r in rows)


def test_dataset_train_rows_have_no_mirrors_by_default(hai_dataset, monkeypatch):
    rows, build, _ = hai_dataset
    monkeypatch.delenv("HAI_MIRROR_SHARE", raising=False)
    assert [s["example_id"] for s in build("train")] == _ids(rows)
    monkeypatch.setenv("HAI_MIRROR_SHARE", "")
    ds = build("train")
    assert [s["example_id"] for s in ds] == _ids(rows)
    assert [s["answer"] for s in ds] == [r["answer"] + "<eos>" for r in rows]


def test_dataset_mirror_share_adds_copies_to_the_training_rows_only(hai_dataset):
    rows, build, mod = hai_dataset
    train = build("train", HAI_MIRROR_SHARE="0.5")
    cls = type(train)
    assert len(train) == len(rows) + round(0.5 * len(rows)) == 30
    assert [train[i]["example_id"] for i in range(20)] == _ids(rows)
    want = mod.with_mirrors(rows, 0.5)[20:]
    for i, w in enumerate(want):
        assert train[20 + i]["answer"] == w["answer"] + "<eos>" and train[20 + i]["example_id"] == w["example_id"]
    for split in ("validation", "test"):
        ds = cls(split, EOS_TOKEN="<eos>")
        assert [s["example_id"] for s in ds] == _ids(rows)
        assert [s["answer"] for s in ds] == [r["answer"] + "<eos>" for r in rows]


def test_dataset_attack_share_then_mirrors(hai_dataset):
    rows, build, mod = hai_dataset
    train = build("train", HAI_ATTACK_SHARE="0.5", HAI_MIRROR_SHARE="0.5")
    shared = mod.with_attack_share(rows, 0.5)
    want = mod.with_mirrors(shared, 0.5)
    assert len(train) == len(want) == 30
    assert [s["example_id"] for s in train] == _ids(want)
    assert [s["answer"] for s in train] == [r["answer"] + "<eos>" for r in want]
    assert sum(s["label"] == "attack" for s in [train[i] for i in range(20)]) == 10
    for split in ("validation", "test"):
        assert [s["example_id"] for s in type(train)(split, EOS_TOKEN="<eos>")] == _ids(rows)


def test_dataset_mirror_share_in_the_overfit_check(hai_dataset):
    _, build, _ = hai_dataset
    plain = build("train", HAI_OVERFIT_ROWS="10")
    picked = [s["example_id"] for s in type(plain)("test", EOS_TOKEN="<eos>")]
    train = build("train", HAI_OVERFIT_ROWS="10", HAI_MIRROR_SHARE="0.5")
    cls = type(train)
    assert len(train) == 15 and [train[i]["example_id"] for i in range(10)] == picked
    for split in ("validation", "test"):
        assert [s["example_id"] for s in cls(split, EOS_TOKEN="<eos>")] == picked
    both = build("train", HAI_OVERFIT_ROWS="10", HAI_ATTACK_SHARE="0.5", HAI_MIRROR_SHARE="0.5")
    assert len(both) == 15 and sum(both[i]["label"] == "attack" for i in range(10)) == 5
    assert [s["example_id"] for s in type(both)("test", EOS_TOKEN="<eos>")] == picked


@pytest.mark.parametrize("bad", ["0", "1.5", "abc", "-0.5", "nan"])
def test_mirror_share_setting_is_checked(hai_dataset, monkeypatch, bad):
    _, build, mod = hai_dataset
    monkeypatch.setenv("HAI_MIRROR_SHARE", bad)
    with pytest.raises(RuntimeError, match="HAI_MIRROR_SHARE"):
        mod.mirror_share()
    with pytest.raises(RuntimeError, match="HAI_MIRROR_SHARE"):
        build("train")


def test_mirror_share_setting_accepts_one(hai_dataset, monkeypatch):
    _, _, mod = hai_dataset
    monkeypatch.setenv("HAI_MIRROR_SHARE", "1")
    assert mod.mirror_share() == 1.0
    monkeypatch.delenv("HAI_MIRROR_SHARE")
    assert mod.mirror_share() is None


def test_run_variant_reads_the_mirror_share():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x"]).mirror_share is None
    assert rv.arguments(["x", "--mirror-share", "0.5"]).mirror_share == 0.5


# --- the range view of a series ------------------------------------------------

def test_range_view_keeps_the_size_of_a_change():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    quiet = [10.0 + 0.001 * (i % 2) for i in range(300)]          # ripple of 0.1% of the normal range
    step = [10.0] * 150 + [10.5] * 150                             # a step of half the normal range
    q, s = np.array(mod.range_view(quiet, 1.0)), np.array(mod.range_view(step, 1.0))
    assert np.abs(q).max() < 0.05                                  # quiet stays near zero
    assert abs(s[:150]).max() == 0 and s[-1] == pytest.approx(np.log1p(0.5 / 0.02))   # 25 units, squashed
    visible = np.array(mod.range_view([0.0] * 150 + [0.02] * 150, 1.0))
    assert visible[-1] == pytest.approx(np.log1p(1.0))            # 2% of the range is one unit
    assert mod.range_view([5.0] * 300, 3.0) == [0.0] * 300
    down = np.array(mod.range_view([0.0] * 150 + [-0.5] * 150, 1.0))
    assert down[-1] == pytest.approx(-s[-1])                       # the sign survives
    with pytest.raises(RuntimeError):
        mod.range_view(step, 0.0)


def test_range_view_prompts_state_no_numbers(hai_dataset):
    import re

    import numpy as np

    rows, build, mod = hai_dataset
    with pytest.raises(RuntimeError):                              # rows without `scale` cannot be shown this way
        build("train", HAI_SERIES_VIEW="range")[0]
    folder = __import__("pathlib").Path(__import__("os").environ["HAI_JSONL_DIR"])
    scaled = [{**r, "series": [{**s, "scale": 2.0} for s in r["series"]]} for r in rows]
    for name in ("train", "validation", "test"):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scaled))
    ds = build("train", HAI_SERIES_VIEW="range")
    sample = ds[1]
    assert "measured against its normal range" in sample["pre_prompt"]
    for text in sample["time_series_text"]:
        assert not re.search(r"\d", text) and "mean" not in text and text.endswith(":")
    got = np.asarray(sample["time_series"][1], dtype=float)
    want = np.asarray(mod.range_view(scaled[1]["series"][1]["values"], 2.0))
    assert np.allclose(got, want, atol=1e-5)
    assert sample["answer"].startswith(scaled[1]["answer"])


def test_series_view_setting_is_checked(hai_dataset, monkeypatch):
    _, _, mod = hai_dataset
    monkeypatch.delenv("HAI_SERIES_VIEW", raising=False)
    assert mod.series_view() == "zscore"
    monkeypatch.setenv("HAI_SERIES_VIEW", "wavelets")
    with pytest.raises(RuntimeError):
        mod.series_view()
    monkeypatch.setenv("HAI_SERIES_VIEW", "looprange")
    assert mod.series_view() == "looprange"


# --- the looprange view: change against the series' usual quiet movement ---------

def _quiet_row(version, loop, label, kind, moves, scale=1.0):
    """A row with one series `cmd`: flat 30 seconds at 0, then a step of `moves` range units (2% of scale each)."""
    values = [0.0] * 30 + [moves * 0.02 * scale] * 30
    return {"example_id": f"{version}/{loop}/{label}/{kind}/{moves}", "version": version, "loop": loop,
            "label": label, "window_kind": kind, "answer": "Answer: normal",
            "series": [{"role": "command", "name": "cmd", "values": values, "scale": scale}]}


def test_quiet_units_measure_how_much_each_series_moves_in_quiet_rows():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    calm = [0.05, 0.1, 0.1, 0.2, 0.1, 0.15]                        # barely moves: below the floor
    pump = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]                    # swings in every quiet window
    rows = [_quiet_row("21.03", "P1-LC", "normal", "normal", m) for m in calm]
    rows += [_quiet_row("21.03", "P2-SC", "normal", "normal", m, scale=3.0) for m in pump]
    rows += [_quiet_row("21.03", "P2-SC", "attack", "attack", 500.0),      # attack rows are ignored
             _quiet_row("21.03", "P2-SC", "normal", "attack", 400.0),      # and so are neighbour rows
             _quiet_row("21.03", "P1-LC", "normal", "attack", 400.0)]
    rows += [_quiet_row("22.04", "P2-SC", "normal", "normal", m) for m in (1.0, 2.0)]   # too few rows of its own
    units = mod.quiet_units(rows)
    assert isinstance(units, dict)
    assert units[("21.03", "P1-LC", "cmd")] == (mod.LOOPRANGE_FLOOR, 6)
    assert units[("21.03", "P2-SC", "cmd")] == (pytest.approx(np.percentile(pump, 95)), 6)
    assert units[("22.04", "P2-SC", "cmd")][1] == 2
    assert units[(None, "P2-SC", "cmd")] == (pytest.approx(np.percentile(pump + [1.0, 2.0], 95)), 8)
    assert mod.unit_for(units, "21.03", "P1-LC", "cmd") == mod.LOOPRANGE_FLOOR
    assert mod.unit_for(units, "21.03", "P2-SC", "cmd") == pytest.approx(np.percentile(pump, 95))
    assert mod.unit_for(units, "22.04", "P2-SC", "cmd") == pytest.approx(np.percentile(pump + [1.0, 2.0], 95))
    assert mod.unit_for(units, "22.04", "P1-LC", "cmd") == mod.LOOPRANGE_FLOOR        # pooled, no 22.04 rows
    assert mod.unit_for(units, "21.03", "P3-LC", "cmd") == 1.0                         # unknown loop
    assert mod.unit_for(units, "21.03", "P1-LC", "other") == 1.0
    assert mod.quiet_units([r for r in rows if r["window_kind"] == "attack"]) == {}


def test_quiet_units_take_the_largest_change_from_the_first_seconds():
    mod = load_script("HAIAttackQADataset")
    row = _quiet_row("21.03", "P1-LC", "normal", "normal", 0.0, scale=2.0)
    row["series"][0]["values"] = [1.0, 3.0] * 15 + [2.0] * 20 + [-1.0] * 10   # head mean 2, largest change 3
    units = mod.quiet_units([row])
    assert units[("21.03", "P1-LC", "cmd")] == (pytest.approx(3.0 / (0.02 * 2.0)), 1)
    del row["series"][0]["scale"]
    with pytest.raises(RuntimeError):
        mod.quiet_units([row])


def test_looprange_view_divides_the_range_view_change_by_the_unit():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    step = [10.0] * 150 + [10.5] * 150
    noisy = (np.random.default_rng(0).normal(size=300) * 0.3 + 4.0).tolist()
    for values, scale in ((step, 1.0), (noisy, 2.5), ([5.0] * 300, 3.0)):
        assert mod.looprange_view(values, scale, 1.0) == mod.range_view(values, scale)
    one = np.array(mod.looprange_view([0.0] * 150 + [0.02 * 7] * 150, 1.0, 7.0))
    assert one[-1] == pytest.approx(np.log1p(1.0)) and one[0] == 0             # one unit of quiet movement
    down = np.array(mod.looprange_view([0.0] * 150 + [-0.02 * 7] * 150, 1.0, 7.0))
    assert down[-1] == pytest.approx(-np.log1p(1.0))                            # the sign survives
    wide = np.array(mod.looprange_view(step, 1.0, 40.0))
    assert 0 < wide[-1] < mod.range_view(step, 1.0)[-1]                         # a swinging series shrinks
    for scale, unit in ((0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0), (1.0, 0.0), (1.0, -2.0)):
        with pytest.raises(RuntimeError):
            mod.looprange_view(step, scale, unit)


def _write_scaled_rows(rows):
    """The fixture's rows with `scale` on every series, written to all three files of HAI_JSONL_DIR."""
    folder = __import__("pathlib").Path(__import__("os").environ["HAI_JSONL_DIR"])
    scaled = [{**r, "series": [{**s, "scale": 2.0} for s in r["series"]]} for r in rows]
    for name in ("train", "validation", "test"):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scaled))
    return scaled


def test_dataset_looprange_view(hai_dataset):
    import re

    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    assert sum(r["label"] == "normal" and r["window_kind"] == "normal" for r in scaled) >= 1
    train = build("train", HAI_SERIES_VIEW="looprange")
    cls = type(train)
    units = cls.looprange_units
    assert units == mod.quiet_units(scaled) and units
    for split in ("train", "validation", "test"):
        ds = cls(split, EOS_TOKEN="<eos>")
        assert len(ds) == len(scaled)
        for i, sample in enumerate(ds):
            r = scaled[i]
            assert "measured against how much it normally moves in this loop" in sample["pre_prompt"]
            assert "measured against its normal range" not in sample["pre_prompt"]
            for text, s, got in zip(sample["time_series_text"], r["series"], sample["time_series"]):
                assert not re.search(r"\d", text) and "mean" not in text and text.endswith(":")
                want = mod.looprange_view(s["values"], 2.0, mod.unit_for(units, r["version"], r["loop"], s["name"]))
                assert np.allclose(np.asarray(got, dtype=float), want, atol=1e-5)


def test_dataset_looprange_units_come_from_the_file_rows(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    want = mod.quiet_units(scaled)
    for env in ({"HAI_OVERFIT_ROWS": "5"}, {"HAI_ATTACK_SHARE": "0.5"}, {"HAI_MIRROR_SHARE": "0.5"},
                {"HAI_OVERFIT_ROWS": "10", "HAI_ATTACK_SHARE": "0.5", "HAI_MIRROR_SHARE": "1"}):
        for k in ("HAI_OVERFIT_ROWS", "HAI_ATTACK_SHARE", "HAI_MIRROR_SHARE"):
            monkeypatch.delenv(k, raising=False)
        assert type(build("train", HAI_SERIES_VIEW="looprange", **env)).looprange_units == want
    assert type(build("train", HAI_SERIES_VIEW="range")).looprange_units is None


def test_range_and_zscore_views_are_unchanged_by_looprange(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    ds = build("train", HAI_SERIES_VIEW="range")
    for i in (0, 1, 7):
        sample = ds[i]
        assert sample["pre_prompt"].count("Each series is shown") == 1
        assert "measured against its normal range" in sample["pre_prompt"]
        for text, s, got in zip(sample["time_series_text"], scaled[i]["series"], sample["time_series"]):
            assert text == f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}:"
            assert np.allclose(np.asarray(got, dtype=float), mod.range_view(s["values"], 2.0), atol=1e-5)
    zs = build("train", HAI_SERIES_VIEW="zscore")
    for i in (0, 1, 7):
        sample = zs[i]
        assert "Each series is shown" not in sample["pre_prompt"]
        for text, s, got in zip(sample["time_series_text"], scaled[i]["series"], sample["time_series"]):
            mean, std, values = mod.scale(s["values"])
            assert text == f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}; it has mean {mean:.4f} and std {std:.4f}:"
            assert np.allclose(np.asarray(got, dtype=float), values, atol=1e-5)


def test_run_variant_takes_the_looprange_view():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x", "--view", "looprange"]).view == "looprange"
    assert rv.arguments(["x"]).view == "range"


# --- the zsize view: the zscore curve, its size against the usual quiet change in the text ---------

ZSIZE_TEXT = re.compile(r"This is the (.+), (.+); its largest change is (\d+\.\d) times its usual quiet change:")


def _zsize_want(mod, units, row, s):
    unit = mod.unit_for(units, row["version"], row["loop"], s["name"])
    return f"{min(mod.relative_size(s['values'], s['scale'], unit), mod.SIZE_CAP):.1f}"


def test_relative_size_is_the_largest_change_in_quiet_units():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    noisy = (np.random.default_rng(0).normal(size=300) * 0.3 + 4.0).tolist()
    for values, scale, unit in ((noisy, 2.5, 3.0), ([10.0] * 150 + [10.5] * 150, 1.0, 0.25), ([5.0] * 300, 3.0, 7.0)):
        assert mod.relative_size(values, scale, unit) == pytest.approx(mod.largest_change(values, scale) / unit)
    assert mod.relative_size([0.0] * 150 + [0.02 * 7] * 150, 1.0, 7.0) == pytest.approx(1.0)    # one usual unit
    assert mod.relative_size([0.0] * 150 + [-0.02 * 7] * 150, 1.0, 7.0) == pytest.approx(1.0)   # sign does not count
    assert mod.relative_size([5.0] * 300, 3.0, 7.0) == 0.0
    for unit in (0.0, -2.0, float("nan")):
        with pytest.raises(RuntimeError):
            mod.relative_size(noisy, 2.5, unit)
    with pytest.raises(RuntimeError):
        mod.relative_size(noisy, 0.0, 1.0)


def test_dataset_zsize_view(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    train = build("train", HAI_SERIES_VIEW="zsize")
    cls = type(train)
    units = cls.looprange_units
    assert units == mod.quiet_units(scaled) and units
    sentence = ("Each series is shown scaled to its own mean and spread. Its text says how large its change is "
                "against what is usual for it in this loop. ")
    for split in ("train", "validation", "test"):
        ds = cls(split, EOS_TOKEN="<eos>")
        assert len(ds) == len(scaled)
        for i, sample in enumerate(ds):
            r = scaled[i]
            assert sentence in sample["pre_prompt"] and sample["pre_prompt"].count("Each series is shown") == 1
            assert sample["post_prompt"] == "Answer:" and sample["answer"] == r["answer"] + "<eos>"
            for text, s, got in zip(sample["time_series_text"], r["series"], sample["time_series"]):
                m = ZSIZE_TEXT.fullmatch(text)
                assert m and m.group(1) == s["name"] and m.group(2) == mod.ROLE_WORDS[s["role"]]
                assert len(re.findall(r"\d+(?:\.\d+)?", text)) == 1 and "mean" not in text
                assert m.group(3) == _zsize_want(mod, units, r, s)
                assert np.allclose(np.asarray(got, dtype=float), mod.scale(s["values"])[2], atol=1e-5)
    flat_text, flat = train[0]["time_series_text"][1], np.asarray(train[0]["time_series"][1])
    assert ZSIZE_TEXT.fullmatch(flat_text).group(3) == "0.0" and (flat == 0).all()


def test_dataset_zsize_values_are_those_of_the_zscore_view(hai_dataset):
    import numpy as np

    rows, build, _ = hai_dataset
    _write_scaled_rows(rows)
    zs, zsize = build("train", HAI_SERIES_VIEW="zscore"), build("train", HAI_SERIES_VIEW="zsize")
    for i in (0, 1, 7):
        for a, b in zip(zs[i]["time_series"], zsize[i]["time_series"]):
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_dataset_zsize_caps_a_huge_change(hai_dataset):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    folder = __import__("pathlib").Path(__import__("os").environ["HAI_JSONL_DIR"])
    huge = [{**r, "series": [{**s, "values": s["values"][:150] + [v + 1e6 for v in s["values"][150:]]}
                             for s in r["series"]]} for r in scaled]
    (folder / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in huge))
    ds = type(build("train", HAI_SERIES_VIEW="zsize"))("validation", EOS_TOKEN="<eos>")
    unit = mod.unit_for(ds.looprange_units, huge[0]["version"], huge[0]["loop"], huge[0]["series"][0]["name"])
    assert mod.relative_size(huge[0]["series"][0]["values"], 2.0, unit) > mod.SIZE_CAP
    for text in ds[0]["time_series_text"]:
        assert ZSIZE_TEXT.fullmatch(text).group(3) == "99.9"


def test_dataset_zsize_units_are_the_looprange_units(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    want = type(build("train", HAI_SERIES_VIEW="looprange")).looprange_units
    assert want == mod.quiet_units(scaled)
    for env in ({}, {"HAI_OVERFIT_ROWS": "5"}, {"HAI_ATTACK_SHARE": "0.5"}, {"HAI_MIRROR_SHARE": "0.5"},
                {"HAI_OVERFIT_ROWS": "10", "HAI_ATTACK_SHARE": "0.5", "HAI_MIRROR_SHARE": "1"}):
        for k in ("HAI_OVERFIT_ROWS", "HAI_ATTACK_SHARE", "HAI_MIRROR_SHARE"):
            monkeypatch.delenv(k, raising=False)
        assert type(build("train", HAI_SERIES_VIEW="zsize", **env)).looprange_units == want


def test_dataset_zsize_needs_scale_and_units(hai_dataset):
    rows, build, mod = hai_dataset
    with pytest.raises(RuntimeError, match="needs `scale`"):    # the fixture rows have no `scale`
        build("train", HAI_SERIES_VIEW="zsize")
    scaled = _write_scaled_rows(rows)
    ds = build("train", HAI_SERIES_VIEW="zsize")
    with pytest.raises(RuntimeError, match="zsize view needs `scale`"):
        ds._get_text_time_series_prompt_list(rows[1])
    type(ds).looprange_units = None
    with pytest.raises(RuntimeError, match="zsize view needs the units"):
        ds._get_text_time_series_prompt_list(scaled[1])


def test_mirrored_row_has_the_same_sizes_and_the_negative_zsize_curve(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    ds = build("train", HAI_SERIES_VIEW="zsize")
    for r in (scaled[1], scaled[7]):
        plain, mirrored = ds._get_text_time_series_prompt_list(r), ds._get_text_time_series_prompt_list(mod.mirror_row(r))
        for a, b in zip(plain, mirrored):
            assert a.get_text() == b.get_text()
            assert np.allclose(np.asarray(b.get_time_series()), -np.asarray(a.get_time_series()), atol=1e-9)


def test_zscore_range_and_looprange_views_are_unchanged_by_zsize(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    for view in ("zscore", "range", "looprange"):
        ds = build("train", HAI_SERIES_VIEW=view)
        units = type(ds).looprange_units
        sample, r = ds[1], scaled[1]
        assert "its usual quiet change" not in " ".join(sample["time_series_text"])
        assert "scaled to its own mean and spread" not in sample["pre_prompt"]
        for text, s, got in zip(sample["time_series_text"], r["series"], sample["time_series"]):
            head = f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}"
            if view == "zscore":
                mean, std, want = mod.scale(s["values"])
                assert text == f"{head}; it has mean {mean:.4f} and std {std:.4f}:"
            elif view == "range":
                want = mod.range_view(s["values"], 2.0)
                assert text == f"{head}:"
            else:
                want = mod.looprange_view(s["values"], 2.0, mod.unit_for(units, r["version"], r["loop"], s["name"]))
                assert text == f"{head}:"
            assert np.allclose(np.asarray(got, dtype=float), want, atol=1e-5)


def test_run_variant_takes_the_zsize_view():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x", "--view", "zsize"]).view == "zsize"
    with pytest.raises(SystemExit):
        rv.arguments(["x", "--view", "wavelets"])


# --- text clauses of the zsize view: the largest one-second step ------------------------------

STEP_TEXT = re.compile(r"This is the (.+), (.+); its largest change is (\d+\.\d) times its usual quiet change; "
                       r"(?:its largest one-second step is (\d+\.\d) times its usual one, at second (\d+)"
                       r"|it never changes from one second to the next):")


def _step_want(mod, units, row, s):
    unit = mod.unit_for(units, row["version"], row["loop"], s["name"])
    return mod.step_clause(s["values"], s["scale"], unit)


def test_largest_step_is_the_largest_one_second_step_in_range_units():
    mod = load_script("HAIAttackQADataset")
    jump = [1.0] * 100 + [1.0 + 0.02 * 2.0 * 7] * 200            # one step of 7 range units at scale 2
    size, second = mod.largest_step(jump, 2.0)
    assert size == pytest.approx(7.0) and second == 100
    assert mod.largest_step([-v for v in jump], 2.0) == (pytest.approx(7.0), 100)   # the sign does not count
    small = [0.0] * 300
    small[50], small[200] = 0.02, 0.06                             # up 1 at 50, down 1 at 51, up 3 at 200
    assert mod.largest_step(small, 1.0) == (pytest.approx(3.0), 200)
    tie = [0.0] * 10 + [0.04] * 10 + [0.0] * 280                   # up 2 at 10, down 2 at 20: the first one
    assert mod.largest_step(tie, 1.0) == (pytest.approx(2.0), 10)
    assert mod.largest_step([5.0] * 300, 3.0) == (0.0, 1)
    assert mod.largest_step([0.0] * 299 + [0.02], 1.0)[1] == 299
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(RuntimeError, match="positive `scale`"):
            mod.largest_step(jump, scale)


def test_quiet_step_units_take_the_largest_step_of_the_quiet_rows():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    calm = [0.05, 0.1, 0.1, 0.2, 0.1, 0.15]                        # tiny steps: below the floor
    pump = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    rows = [_quiet_row("21.03", "P1-LC", "normal", "normal", m) for m in calm]
    rows += [_quiet_row("21.03", "P2-SC", "normal", "normal", m, scale=3.0) for m in pump]
    rows += [_quiet_row("21.03", "P2-SC", "attack", "attack", 500.0),      # attack rows are ignored
             _quiet_row("21.03", "P2-SC", "normal", "attack", 400.0)]      # and so are neighbour rows
    rows += [_quiet_row("22.04", "P2-SC", "normal", "normal", m) for m in (1.0, 2.0)]   # too few rows of its own
    units = mod.quiet_step_units(rows)
    assert units[("21.03", "P1-LC", "cmd")] == (mod.LOOPRANGE_FLOOR, 6)
    assert units[("21.03", "P2-SC", "cmd")] == (pytest.approx(np.percentile(pump, 95)), 6)
    assert units[(None, "P2-SC", "cmd")] == (pytest.approx(np.percentile(pump + [1.0, 2.0], 95)), 8)
    assert mod.unit_for(units, "22.04", "P2-SC", "cmd") == pytest.approx(np.percentile(pump + [1.0, 2.0], 95))
    assert mod.unit_for(units, "22.04", "P1-LC", "cmd") == mod.LOOPRANGE_FLOOR
    assert mod.unit_for(units, "21.03", "P3-LC", "cmd") == 1.0
    # a slow ramp: a large change, but small steps
    ramp = _quiet_row("21.03", "P1-LC", "normal", "normal", 0.0)
    ramp["series"][0]["values"] = [0.02 * 0.5 * i for i in range(300)]    # 0.5 range units a second
    assert mod.quiet_units([ramp])[("21.03", "P1-LC", "cmd")][0] == pytest.approx(149.5 - 7.25)   # from the head mean
    assert mod.quiet_step_units([ramp])[("21.03", "P1-LC", "cmd")] == (pytest.approx(0.5), 1)
    assert mod.quiet_step_units([r for r in rows if r["window_kind"] == "attack"]) == {}
    del ramp["series"][0]["scale"]
    with pytest.raises(RuntimeError):
        mod.quiet_step_units([ramp])


def test_step_clause_states_the_step_in_usual_steps_and_its_second():
    mod = load_script("HAIAttackQADataset")
    jump = [0.0] * 120 + [0.02 * 6] * 180                          # a step of 6 range units at second 120
    assert mod.step_clause(jump, 1.0, 2.0) == "its largest one-second step is 3.0 times its usual one, at second 120"
    assert mod.step_clause(jump, 1.0, 0.7) == "its largest one-second step is 8.6 times its usual one, at second 120"
    assert mod.step_clause([0.0] * 120 + [1e4] * 180, 1.0, 0.25) == (
        "its largest one-second step is 99.9 times its usual one, at second 120")
    assert mod.step_clause([4.0] * 300, 1.0, 0.25) == "it never changes from one second to the next"
    for unit in (0.0, -1.0, float("nan")):
        with pytest.raises(RuntimeError):
            mod.step_clause(jump, 1.0, unit)


def test_dataset_zsize_step_clause(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    train = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step")
    cls = type(train)
    units, steps = cls.looprange_units, cls.step_units
    assert units == mod.quiet_units(scaled) and steps == mod.quiet_step_units(scaled) and steps
    for split in ("train", "validation", "test"):
        ds = cls(split, EOS_TOKEN="<eos>")
        assert len(ds) == len(scaled)
        for i, sample in enumerate(ds):
            r = scaled[i]
            for text, s, got in zip(sample["time_series_text"], r["series"], sample["time_series"]):
                m = STEP_TEXT.fullmatch(text)
                assert m and m.group(1) == s["name"] and m.group(2) == mod.ROLE_WORDS[s["role"]]
                assert m.group(3) == _zsize_want(mod, units, r, s)
                assert text == (f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}; its largest change is "
                                f"{m.group(3)} times its usual quiet change; {_step_want(mod, steps, r, s)}:")
                assert np.allclose(np.asarray(got, dtype=float), mod.scale(s["values"])[2], atol=1e-5)
    moving = STEP_TEXT.fullmatch(train[0]["time_series_text"][2])
    assert moving.group(4) is not None and 1 <= int(moving.group(5)) <= 299
    assert train[0]["time_series_text"][1].endswith("; it never changes from one second to the next:")   # flat


def test_dataset_zsize_step_clause_keeps_the_curve(hai_dataset):
    import numpy as np

    rows, build, _ = hai_dataset
    _write_scaled_rows(rows)
    plain = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="")
    step = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step")
    for i in range(len(plain)):
        assert plain[i]["pre_prompt"] == step[i]["pre_prompt"] and plain[i]["answer"] == step[i]["answer"]
        for a, b in zip(plain[i]["time_series"], step[i]["time_series"]):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(plain[i]["time_series_text"], step[i]["time_series_text"]):
            assert b.startswith(a[:-1] + "; ") and a.endswith(":")


def test_dataset_step_units_come_from_the_file_rows(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    want = mod.quiet_step_units(scaled)
    for env in ({}, {"HAI_OVERFIT_ROWS": "5"}, {"HAI_ATTACK_SHARE": "0.5"}, {"HAI_MIRROR_SHARE": "0.5"},
                {"HAI_OVERFIT_ROWS": "10", "HAI_ATTACK_SHARE": "0.5", "HAI_MIRROR_SHARE": "1"}):
        for k in ("HAI_OVERFIT_ROWS", "HAI_ATTACK_SHARE", "HAI_MIRROR_SHARE"):
            monkeypatch.delenv(k, raising=False)
        ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step", **env)
        assert type(ds).step_units == want
        assert all(STEP_TEXT.fullmatch(t) for sample in ds for t in sample["time_series_text"])
    monkeypatch.delenv("HAI_TEXT_CLAUSES")
    assert type(build("train", HAI_SERIES_VIEW="zsize")).step_units is None


def test_mirrored_row_has_the_same_step_clause(hai_dataset):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step")
    for r in (scaled[1], scaled[7]):
        plain, mirrored = ds._get_text_time_series_prompt_list(r), ds._get_text_time_series_prompt_list(mod.mirror_row(r))
        assert [a.get_text() for a in plain] == [b.get_text() for b in mirrored]
        assert any("one-second step is" in a.get_text() for a in plain)


def test_dataset_zsize_without_clauses_is_unchanged(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    for value in (None, "", " , "):
        if value is None:
            monkeypatch.delenv("HAI_TEXT_CLAUSES", raising=False)
            ds = build("train", HAI_SERIES_VIEW="zsize")
        else:
            ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES=value)
        assert mod.text_clauses() == () and type(ds).step_units is None
        units = type(ds).looprange_units
        for i in (0, 1, 7):
            for text, s in zip(ds[i]["time_series_text"], scaled[i]["series"]):
                assert text == (f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}; "
                                f"its largest change is {_zsize_want(mod, units, scaled[i], s)} times its usual quiet change:")


def test_text_clauses_setting_is_checked(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    _write_scaled_rows(rows)
    monkeypatch.setenv("HAI_SERIES_VIEW", "zsize")
    monkeypatch.setenv("HAI_TEXT_CLAUSES", " step ,step")
    assert mod.text_clauses() == ("step", "step")
    monkeypatch.setenv("HAI_TEXT_CLAUSES", "step,jump")
    with pytest.raises(RuntimeError, match="'jump'"):
        mod.text_clauses()
    with pytest.raises(RuntimeError, match="HAI_TEXT_CLAUSES"):
        build("train")
    for view in ("zscore", "range"):
        monkeypatch.setenv("HAI_SERIES_VIEW", view)
        monkeypatch.setenv("HAI_TEXT_CLAUSES", "step")
        with pytest.raises(RuntimeError, match="needs HAI_SERIES_VIEW=zsize"):
            mod.text_clauses()
        with pytest.raises(RuntimeError, match="needs HAI_SERIES_VIEW=zsize"):
            build("train")
        monkeypatch.setenv("HAI_TEXT_CLAUSES", "")
        assert len(build("train")) == len(rows)


def test_run_variant_takes_the_clauses():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x"]).clauses is None
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "step"]).clauses == "step"
    with pytest.raises(SystemExit):
        rv.arguments(["x", "--view", "range", "--clauses", "step"])
    assert rv.arguments(["x", "--answer-only", "test", "--clauses", "step"]).clauses == "step"


# --- text clauses of the zsize view: the command against the controller's law --------------------

LAW = (0.3, -1.2, 0.05)


def _law_row(version, loop, label, kind, law=LAW, seed=0, scale=10.0):
    """A row whose first command follows `law` exactly: du[k] = a*g[k] + b*(g[k] - g[k-1]) + c, g = target - measurement."""
    import numpy as np

    rng = np.random.default_rng(seed)
    t = 5.0 + np.cumsum(rng.normal(size=300)) * 0.1
    m = t - rng.normal(size=300) - 0.5
    g = t - m
    a, b, c = law
    u = np.cumsum(np.concatenate([[2.0], a * g[1:] + b * np.diff(g) + c]))
    return {"example_id": f"{version}/{loop}/{label}/{kind}/{seed}", "version": version, "loop": loop,
            "label": label, "window_kind": kind, "answer": "Answer: normal",
            "series": [{"role": "target", "name": "target", "values": t.tolist(), "scale": scale},
                       {"role": "measurement", "name": "measurement", "values": m.tolist(), "scale": scale},
                       {"role": "command", "name": "command", "values": u.tolist(), "scale": scale}]}


def test_fit_laws_recovers_the_controller_law_from_quiet_rows():
    mod = load_script("HAIAttackQADataset")
    other = (-5.0, 3.0, 1.0)
    rows = [_law_row("21.03", "P1-PC", "normal", "normal", seed=i) for i in range(6)]
    rows += [_law_row("22.04", "P1-PC", "normal", "normal", seed=10 + i) for i in range(2)]   # too few rows of its own
    rows += [_law_row("21.03", "P1-PC", "attack", "attack", other, seed=20),      # attack rows are ignored
             _law_row("21.03", "P1-PC", "normal", "attack", other, seed=21)]      # and so are neighbour rows
    two_targets = _law_row("21.03", "P1-PC", "normal", "normal", other, seed=22)
    two_targets["series"].insert(1, {**two_targets["series"][0], "name": "target 2"})
    no_target = _law_row("21.03", "P1-PC", "normal", "normal", other, seed=23)
    no_target["series"] = no_target["series"][1:]
    no_command = _law_row("21.03", "P1-PC", "normal", "normal", other, seed=24)
    no_command["series"] = no_command["series"][:2]
    rows += [two_targets, no_target, no_command]                                 # not eligible: ignored
    laws = mod.fit_laws(rows)
    assert set(laws) == {("21.03", "P1-PC"), ("22.04", "P1-PC"), (None, "P1-PC")}
    for key, n in ((("21.03", "P1-PC"), 6), (("22.04", "P1-PC"), 2), ((None, "P1-PC"), 8)):
        assert laws[key][1] == n and laws[key][0] == pytest.approx(LAW, abs=1e-8)
        assert all(isinstance(v, float) for v in laws[key][0])
        assert isinstance(laws[key][2], float) and laws[key][2] == pytest.approx(1.0, abs=1e-9)
    pooled = {("22.04", "P1-PC"): ((9.0, 9.0, 9.0), 4, 0.9), (None, "P1-PC"): ((1.0, 2.0, 3.0), 10, 0.5),
              ("21.03", "P1-PC"): ((4.0, 5.0, 6.0), 5, 0.4)}
    assert mod.law_for(pooled, "21.03", "P1-PC") == (4.0, 5.0, 6.0)                 # 5 rows: its own
    assert mod.law_for(pooled, "22.04", "P1-PC") == (1.0, 2.0, 3.0)                 # under 5 rows: all versions
    assert mod.law_for(pooled, "23.01", "P1-PC") == (1.0, 2.0, 3.0)                 # no rows of its own
    assert mod.law_for(pooled, "21.03", "P3-LC") is None                            # unknown loop
    assert mod.fit_laws([r for r in rows if r["window_kind"] == "attack"]) == {}


def _noise_row(version, loop, seed):
    """A quiet row whose first command is a random walk, unrelated to the gap between target and measurement."""
    import numpy as np

    row = _law_row(version, loop, "normal", "normal", seed=seed)
    row["series"][2]["values"] = np.cumsum(np.random.default_rng(1000 + seed).normal(size=300)).tolist()
    return row


def test_fit_laws_r2_tells_a_law_from_noise():
    mod = load_script("HAIAttackQADataset")
    rows = [_law_row("21.03", "P1-PC", "normal", "normal", seed=i) for i in range(6)]
    rows += [_noise_row("21.03", "P1-LC", seed=i) for i in range(6)]
    laws = mod.fit_laws(rows)
    assert laws[("21.03", "P1-PC")][2] == pytest.approx(1.0, abs=1e-9)
    assert laws[("21.03", "P1-LC")][2] < mod.LAW_MIN_R2 and laws[(None, "P1-LC")][2] < mod.LAW_MIN_R2
    assert mod.law_for(laws, "21.03", "P1-PC") == pytest.approx(LAW, abs=1e-8)
    assert mod.law_for(laws, "21.03", "P1-LC") is None
    assert mod.law_clause(_law_row("21.03", "P1-PC", "attack", "attack", seed=50), laws) is not None
    assert mod.law_clause(_noise_row("21.03", "P1-LC", seed=50), laws) is None
    flat = _law_row("21.03", "P2-SC", "normal", "normal", seed=60)
    flat["series"][2]["values"] = [4.0] * 300                                       # du never varies
    assert mod.fit_laws([flat])[("21.03", "P2-SC")][2] == 0.0
    assert mod.LAW_MIN_R2 == 0.4


def test_law_for_falls_back_to_the_pooled_law_only_under_5_rows_and_only_if_it_explains_the_command():
    mod = load_script("HAIAttackQADataset")
    own, pooled = (4.0, 5.0, 6.0), (1.0, 2.0, 3.0)

    def laws(n, r2, pooled_r2):
        return {("21.03", "P1-PC"): (own, n, r2), (None, "P1-PC"): (pooled, 20, pooled_r2)}

    assert mod.law_for(laws(5, 0.39, 0.9), "21.03", "P1-PC") is None                # enough rows, poor fit: no fallback
    assert mod.law_for(laws(5, 0.4, 0.1), "21.03", "P1-PC") == own                  # enough rows, good fit
    assert mod.law_for(laws(4, 0.9, 0.9), "21.03", "P1-PC") == pooled               # under 5 rows: the pooled law
    assert mod.law_for(laws(4, 0.9, 0.39), "21.03", "P1-PC") is None                # whose fit must pass too
    assert mod.law_for(laws(4, 0.9, 0.39), "23.01", "P1-PC") is None                # no rows of its own, poor pool
    assert mod.law_for(laws(4, 0.9, 0.4), "23.01", "P1-PC") == pooled
    # from fitted rows: 2 quiet rows of a noisy version, pooled with 6 lawful ones of another
    rows = [_law_row("21.03", "P1-PC", "normal", "normal", seed=i) for i in range(6)]
    rows += [_noise_row("22.04", "P1-PC", seed=i) for i in range(2)]
    fitted = mod.fit_laws(rows)
    assert fitted[("22.04", "P1-PC")][2] < mod.LAW_MIN_R2 <= fitted[(None, "P1-PC")][2]
    assert mod.law_for(fitted, "22.04", "P1-PC") == fitted[(None, "P1-PC")][0]


def test_wrong_way_seconds_counts_the_seconds_against_the_law():
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    t, m, u = (np.asarray(s["values"]) for s in _law_row("21.03", "P1-PC", "normal", "normal", seed=3)["series"])
    assert mod.wrong_way_seconds(t, m, u, 10.0, LAW) == 0                          # follows the law exactly
    mirrored = 2 * u[0] - u                                                         # mirrored around its start
    duh = u - u[:mod.RANGE_HEAD_S].mean()
    want = int(np.sum(np.abs(duh) > mod.RANGE_UNIT * 10.0))
    assert want > 100
    got = mod.wrong_way_seconds(t, m, mirrored, 10.0, LAW)
    assert isinstance(got, int) and got == want
    assert mod.wrong_way_seconds(t, m, [3.0] * 300, 10.0, LAW) == 0                # a flat command
    for bad in (0.0, -1.0, float("nan"), float("inf"), None):
        with pytest.raises(RuntimeError, match="positive `scale`"):
            mod.wrong_way_seconds(t, m, u, bad, LAW)


def test_law_clause_phrasings():
    mod = load_script("HAIAttackQADataset")
    row = _law_row("21.03", "P1-PC", "normal", "normal", scale=1.0)
    laws = {("21.03", "P1-PC"): ((0.0, 0.0, 1.0), 10, 1.0)}       # the law's command rises one a second
    cmd = row["series"][2]
    cmd["values"] = [0.0] * 300
    assert mod.law_clause(row, laws) == "it moves the way the controller would"
    cmd["values"][200] = -1.0
    assert mod.law_clause(row, laws) == "for 1 second it moves the opposite way to what the controller would do"
    cmd["values"] = [0.0] * 150 + [-1.0] * 150
    assert mod.law_clause(row, laws) == "for 150 seconds it moves the opposite way to what the controller would do"
    second = {**cmd, "name": "command 2", "values": [0.0] * 300}       # only the first command counts
    row["series"].append(second)
    assert mod.law_clause(row, laws).startswith("for 150 seconds")
    row["series"].insert(2, second)
    assert mod.law_clause(row, laws) == "it moves the way the controller would"
    assert mod.law_clause(row, {("22.04", "P1-PC"): ((0.0, 0.0, 1.0), 10, 1.0)}) is None      # no law for this loop
    assert mod.law_clause(row, {(None, "P1-PC"): ((0.0, 0.0, 1.0), 2, 1.0)}) is not None       # the pooled law
    assert mod.law_clause(row, {("21.03", "P1-PC"): ((0.0, 0.0, 1.0), 10, 0.39)}) is None     # a law that explains too little
    row["series"].insert(0, {**row["series"][0], "name": "target 2"})
    assert mod.law_clause(row, laws) is None                                             # not eligible


def _texts(ds, i):
    return list(ds[i]["time_series_text"])


def _lawful(rows, loop="P1-PC"):
    """The rows with the first command of every row of `loop` following LAW exactly; the other loops as they are."""
    import numpy as np

    out = []
    for r in rows:
        if r["loop"] == loop:
            series = [dict(s) for s in r["series"]]
            t, m = (np.asarray(s["values"], dtype=np.float64) for s in series[:2])
            g = t - m
            a, b, c = LAW
            series[2]["values"] = np.cumsum(np.concatenate([[2.0], a * g[1:] + b * np.diff(g) + c])).tolist()
            r = {**r, "series": series}
        out.append(r)
    return out


def test_dataset_zsize_law_clause(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(rows))      # P1-PC follows a law, the other loops do not
    plain = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="")
    train = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="law")
    cls = type(train)
    laws = cls.control_laws
    assert laws == mod.fit_laws(scaled) and laws and cls.step_units is None
    assert laws[(None, "P1-PC")][2] >= mod.LAW_MIN_R2
    assert all(r2 < mod.LAW_MIN_R2 for (_, loop), (_, _, r2) in laws.items() if loop != "P1-PC")
    assert {r["loop"] for r in scaled} > {"P1-PC"}
    plain_cls = type(plain)
    for split in ("train", "validation", "test"):
        ds, base = cls(split, EOS_TOKEN="<eos>"), plain_cls(split, EOS_TOKEN="<eos>")
        assert len(ds) == len(scaled)
        for i, sample in enumerate(ds):
            r = scaled[i]
            want = mod.law_clause(r, laws)
            assert (want is not None) == (r["loop"] == "P1-PC")
            for j, (text, before) in enumerate(zip(sample["time_series_text"], base[i]["time_series_text"])):
                if want is not None and r["series"][j]["role"] == "command":
                    assert text == before[:-1] + "; " + want + ":"
                else:
                    assert text == before
            for a, b in zip(sample["time_series"], base[i]["time_series"]):
                assert np.array_equal(np.asarray(a), np.asarray(b))
            assert sample["pre_prompt"] == base[i]["pre_prompt"] and sample["answer"] == base[i]["answer"]
    # a row with two commands: only the first one's line carries the clause
    assert scaled[6]["loop"] == "P1-PC"
    two = {**scaled[6], "series": scaled[6]["series"] + [{**scaled[6]["series"][2], "name": "second command"}]}
    got = [p.get_text() for p in train._get_text_time_series_prompt_list(two)]
    assert got[2].endswith("; " + mod.law_clause(two, laws) + ":")
    assert "controller would" not in got[3] and "controller would" not in " ".join(got[:2])
    # a loop without a law: no clause
    type(train).control_laws = {}
    assert all("controller would" not in p.get_text() for p in train._get_text_time_series_prompt_list(scaled[6]))
    type(train).control_laws = None
    with pytest.raises(RuntimeError, match="control laws"):
        train._get_text_time_series_prompt_list(scaled[1])


def test_dataset_zsize_step_and_law_clauses_in_a_fixed_order(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(rows))
    step = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step")
    both = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step,law")
    swapped = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="law,step")
    laws = type(both).control_laws
    assert type(both).step_units == type(step).step_units == mod.quiet_step_units(scaled)
    assert laws == type(swapped).control_laws == mod.fit_laws(scaled)
    for i in range(len(both)):
        r = scaled[i]
        assert _texts(both, i) == _texts(swapped, i)
        assert (mod.law_clause(r, laws) is not None) == (r["loop"] == "P1-PC")
        for j, (text, before) in enumerate(zip(_texts(both, i), _texts(step, i))):
            if j == 2 and r["loop"] == "P1-PC":
                assert STEP_TEXT.fullmatch(before)
                assert text == before[:-1] + "; " + mod.law_clause(r, laws) + ":"
                assert text.index("its largest change") < text.index("one-second step") < text.index("controller would")
            else:
                assert text == before
        for a, b in zip(both[i]["time_series"], step[i]["time_series"]):
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_dataset_control_laws_come_from_the_file_rows(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    want = mod.fit_laws(scaled)
    for env in ({}, {"HAI_OVERFIT_ROWS": "5"}, {"HAI_ATTACK_SHARE": "0.5"}, {"HAI_MIRROR_SHARE": "0.5"},
                {"HAI_OVERFIT_ROWS": "10", "HAI_ATTACK_SHARE": "0.5", "HAI_MIRROR_SHARE": "1"}):
        for k in ("HAI_OVERFIT_ROWS", "HAI_ATTACK_SHARE", "HAI_MIRROR_SHARE"):
            monkeypatch.delenv(k, raising=False)
        ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="law", **env)
        assert type(ds).control_laws == want
        assert len(ds) > 0
    monkeypatch.setenv("HAI_TEXT_CLAUSES", "step")
    assert type(build("train", HAI_SERIES_VIEW="zsize")).control_laws is None
    monkeypatch.delenv("HAI_TEXT_CLAUSES")
    assert type(build("train", HAI_SERIES_VIEW="zsize")).control_laws is None


def test_dataset_zsize_and_step_texts_are_unchanged_by_the_law_clause(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(rows)
    monkeypatch.setenv("HAI_SERIES_VIEW", "zsize")
    monkeypatch.setenv("HAI_TEXT_CLAUSES", "step,law")
    assert mod.text_clauses() == ("step", "law") and "law" in mod.CLAUSES
    for clauses in ("", "step"):
        ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES=clauses)
        units, steps = type(ds).looprange_units, type(ds).step_units
        assert type(ds).control_laws is None
        for i in range(len(ds)):
            for text, s in zip(_texts(ds, i), scaled[i]["series"]):
                want = (f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}; "
                        f"its largest change is {_zsize_want(mod, units, scaled[i], s)} times its usual quiet change")
                if clauses == "step":
                    want += "; " + _step_want(mod, steps, scaled[i], s)
                assert text == want + ":" and "controller would" not in text


def test_run_variant_takes_the_law_clause():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "step,law"]).clauses == "step,law"
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "law"]).clauses == "law"


# --- text clauses of the looprange view, and the size clause ----------------------------------

def test_size_clause_states_the_size_in_usual_quiet_changes():
    mod = load_script("HAIAttackQADataset")
    assert mod.CLAUSES == ("size", "step", "law", "plant")
    step = [0.0] * 150 + [0.02 * 7] * 150                          # 7 range units at scale 1
    assert mod.size_clause(step, 1.0, 7.0) == "its largest change is 1.0 times its usual quiet change"
    assert mod.size_clause(step, 1.0, 3.0) == "its largest change is 2.3 times its usual quiet change"
    assert mod.size_clause([-v for v in step], 1.0, 0.25) == "its largest change is 28.0 times its usual quiet change"
    assert mod.size_clause([5.0] * 300, 3.0, 7.0) == "its largest change is 0.0 times its usual quiet change"
    assert mod.size_clause([0.0] * 150 + [1e6] * 150, 1.0, 0.25) == (
        "its largest change is 99.9 times its usual quiet change")
    noisy = (__import__("numpy").random.default_rng(0).normal(size=300) * 0.3 + 4.0).tolist()
    want = f"{min(mod.relative_size(noisy, 2.5, 3.0), mod.SIZE_CAP):.1f}"
    assert mod.size_clause(noisy, 2.5, 3.0) == f"its largest change is {want} times its usual quiet change"
    for unit in (0.0, -2.0, float("nan")):
        with pytest.raises(RuntimeError):
            mod.size_clause(step, 1.0, unit)


def _looprange_want(mod, cls, row, s, clauses):
    """The looprange text line of series `s` of `row`, built from the clause functions."""
    unit = mod.unit_for(cls.looprange_units, row["version"], row["loop"], s["name"])
    text = f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}"
    if "size" in clauses:
        text += "; " + mod.size_clause(s["values"], s["scale"], unit)
    if "step" in clauses:
        text += "; " + _step_want(mod, cls.step_units, row, s)
    first = next(x for x in row["series"] if x["role"] == "command")
    if "law" in clauses and s is first:
        law = mod.law_clause(row, cls.control_laws)
        if law is not None:
            text += "; " + law
    return text + ":"


@pytest.mark.parametrize("clauses", ["size", "step", "law", "size,step,law"])
def test_dataset_looprange_clauses(hai_dataset, clauses):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(rows))      # P1-PC follows a law, the other loops do not
    plain = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES="")
    train = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES=clauses)
    swapped = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES=",".join(reversed(clauses.split(","))))
    cls, plain_cls, swapped_cls = type(train), type(plain), type(swapped)
    names = clauses.split(",")
    assert cls.looprange_units == mod.quiet_units(scaled)
    assert cls.step_units == (mod.quiet_step_units(scaled) if "step" in names else None)
    assert cls.control_laws == (mod.fit_laws(scaled) if "law" in names else None)
    with_law = 0
    for split in ("train", "validation", "test"):
        ds, base, other = cls(split, EOS_TOKEN="<eos>"), plain_cls(split, EOS_TOKEN="<eos>"), swapped_cls(split, EOS_TOKEN="<eos>")
        assert len(ds) == len(scaled)
        for i, sample in enumerate(ds):
            r = scaled[i]
            assert sample["pre_prompt"] == base[i]["pre_prompt"] and sample["answer"] == base[i]["answer"]
            assert sample["time_series_text"] == other[i]["time_series_text"]      # the order of the names does not matter
            first = next(j for j, s in enumerate(r["series"]) if s["role"] == "command")
            for j, (text, before, s) in enumerate(zip(sample["time_series_text"], base[i]["time_series_text"], r["series"])):
                assert text == _looprange_want(mod, cls, r, s, names)
                assert before == f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}:"
                assert text.startswith(before[:-1]) and text.endswith(":")
                has_law = "controller would" in text
                assert has_law == ("law" in names and j == first and r["loop"] == "P1-PC")
                with_law += has_law
                if "size" in names:
                    assert "; its largest change is " in text
                    parts = [text.index(p) for p in ("its largest change", "one-second", "controller would") if p in text]
                    assert parts == sorted(parts)
                if clauses == "size,step,law":
                    assert text.count("; ") == 2 + has_law
            for a, b in zip(sample["time_series"], base[i]["time_series"]):
                assert np.array_equal(np.asarray(a), np.asarray(b))     # the curve is that of plain looprange
    assert ("law" in names) == (with_law > 0)


def test_dataset_looprange_with_all_clauses_has_the_texts_of_zsize_with_step_and_law(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(rows))
    loop = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES="size,step,law")
    zsize = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step,law")
    plain = build("train", HAI_SERIES_VIEW="looprange")
    assert any("controller would" in t for i in range(len(loop)) for t in _texts(loop, i))
    for i in range(len(loop)):
        assert _texts(loop, i) == _texts(zsize, i)
        for a, b in zip(loop[i]["time_series"], plain[i]["time_series"]):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        assert loop[i]["pre_prompt"] == plain[i]["pre_prompt"] != zsize[i]["pre_prompt"]
    s = scaled[6]["series"][2]
    assert scaled[6]["loop"] == "P1-PC"
    assert _texts(loop, 6)[2].startswith(f"This is the {s['name']}, {mod.ROLE_WORDS['command']}; its largest change is ")


def test_dataset_texts_without_the_new_clauses_are_unchanged(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(rows))
    for value in (None, "", " , "):
        if value is None:
            monkeypatch.delenv("HAI_TEXT_CLAUSES", raising=False)
            ds = build("train", HAI_SERIES_VIEW="looprange")
        else:
            ds = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES=value)
        assert type(ds).step_units is None and type(ds).control_laws is None
        for i in range(len(ds)):
            assert _texts(ds, i) == [f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}:" for s in scaled[i]["series"]]
    alone = build("train", HAI_SERIES_VIEW="zsize")
    for clauses in ("step", "law", "step,law"):
        ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES=clauses)
        with_size = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="size," + clauses)
        cls = type(ds)
        for i in range(len(ds)):
            r = scaled[i]
            want = []
            for s in r["series"]:
                text = (f"This is the {s['name']}, {mod.ROLE_WORDS[s['role']]}; "
                        f"its largest change is {_zsize_want(mod, cls.looprange_units, r, s)} times its usual quiet change")
                if "step" in clauses:
                    text += "; " + _step_want(mod, cls.step_units, r, s)
                if "law" in clauses and s is r["series"][2] and mod.law_clause(r, cls.control_laws) is not None:
                    text += "; " + mod.law_clause(r, cls.control_laws)
                want.append(text + ":")
            assert _texts(ds, i) == want and _texts(with_size, i) == want
    size = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="size")
    for i in range(len(alone)):
        assert _texts(size, i) == _texts(alone, i)
        assert all(t.count("its largest change is") == 1 for t in _texts(size, i))
    for view in ("zscore", "range"):
        monkeypatch.setenv("HAI_SERIES_VIEW", view)
        for clauses in ("step", "size"):
            monkeypatch.setenv("HAI_TEXT_CLAUSES", clauses)
            with pytest.raises(RuntimeError, match="zsize or looprange"):
                mod.text_clauses()


def test_run_variant_takes_clauses_with_the_looprange_view():
    rv = _cloud_script("run_variant")
    args = rv.arguments(["x", "--view", "looprange", "--clauses", "size,step,law"])
    assert args.view == "looprange" and args.clauses == "size,step,law"
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "size"]).clauses == "size"
    for view in ("range", "zscore"):
        with pytest.raises(SystemExit):
            rv.arguments(["x", "--view", view, "--clauses", "step"])


# --- text clauses of the zsize and looprange views: the measurement against the plant model -------

def _plant_row(loop="P1-FC", label="normal", kind="normal", version="21.03", scale=1.0):
    """A row with a flat target, a flat command and a flat measurement, each 300 seconds at 0."""
    return {"example_id": f"{version}/{loop}/{label}/{kind}", "version": version, "loop": loop,
            "label": label, "window_kind": kind, "answer": "Answer: normal",
            "series": [{"role": "target", "name": "target", "values": [0.0] * 300, "scale": scale},
                       {"role": "measurement", "name": "measurement", "values": [0.0] * 300, "scale": scale},
                       {"role": "command", "name": "command", "values": [0.0] * 300, "scale": scale}]}


def _units_step(at, size, scale=1.0):
    """300 seconds: 0, then `size` range units (RANGE_UNIT of `scale` each) from second `at` on."""
    return [0.0] * at + [size * 0.02 * scale] * (300 - at)


def test_plant_ratio_is_the_largest_residual_in_res95_units():
    opentslm_without_init()
    mod = load_script("HAIAttackQADataset")
    assert mod.CLAUSES == ("size", "step", "law", "plant") and mod.PLANT_MIN_R2 == 0.5
    assert mod.PLANT_LAGS == (0, 5, 10, 15, 20, 25, 30)
    lag5 = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)          # the measurement is the command 5 seconds back
    plants = {"P1-FC": (lag5, 2.0, 10, 0.9)}
    row = _plant_row(scale=3.0)
    row["series"][2]["values"] = _units_step(50, 3.0, scale=3.0)      # the command steps 3 units at second 50
    row["series"][1]["values"] = _units_step(55, 3.0, scale=3.0)      # the measurement follows 5 seconds later
    assert mod.plant_ratio(row, plants) == 0.0
    assert mod.plant_clause(row, plants) == "it strays from what its command predicts by 0.0 times its usual amount"
    row["series"][1]["values"][200] += 4.0 * 0.02 * 3.0               # 4 units off at second 200
    assert mod.plant_ratio(row, plants) == 2.0
    assert mod.plant_clause(row, plants) == "it strays from what its command predicts by 2.0 times its usual amount"
    row["series"][1]["values"] = _units_step(50, 3.0, scale=3.0)      # 5 seconds too early: 3 units off
    assert mod.plant_ratio(row, plants) == 1.5
    row["series"][1]["values"] = _units_step(55, 3.0, scale=3.0)
    row["series"][1]["values"][10] = 30 * 0.02 * 3.0                  # the head moves the baseline by 1 unit
    assert mod.plant_ratio(row, plants) == pytest.approx(0.5)
    row["series"][1]["values"] = _units_step(55, 1e6, scale=3.0)
    assert mod.plant_ratio(row, plants) == mod.SIZE_CAP
    assert mod.plant_clause(row, plants).endswith("by 99.9 times its usual amount")
    row["series"][1]["values"] = _units_step(55, 3.0 + 0.04 * 2.0, scale=3.0)   # 0.08 units: shown 0.04 -> 0.0
    assert mod.plant_clause(row, plants).endswith("by 0.0 times its usual amount")
    row = _plant_row()                                                 # the measurement is twice the target, plus 7
    row["series"][0]["values"] = _units_step(100, 5.0)
    row["series"][1]["values"] = [7.0 + 2.0 * v for v in _units_step(100, 5.0)]
    twice = (0.0,) * 7 + (2.0, 0.0)
    assert mod.plant_ratio(row, {"P1-FC": (twice, 1.0, 10, 0.9)}) == 0.0             # the 7 goes with the head mean
    assert mod.plant_ratio(row, {"P1-FC": (twice[:8] + (1.0,), 1.0, 10, 0.9)}) == 1.0   # a constant of 1 unit
    row["series"][2]["scale"] = 0.0
    with pytest.raises(RuntimeError, match="positive `scale`"):
        mod.plant_ratio(row, plants)


def _plant_quiet_rows(n, loop="P1-FC", seed=0, noise=0.1, follows=True):
    """Quiet rows whose measurement is 0.6 * the command 5 seconds back + 0.3 * the command 15 seconds back,
    in range units, plus noise; with follows=False the measurement is a random walk."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        u = np.cumsum(rng.normal(size=300))
        mm = np.zeros(300)
        mm[15:] = 0.6 * u[10:295] + 0.3 * u[:285]
        mm += rng.normal(size=300) * noise
        if not follows:
            mm = np.cumsum(rng.normal(size=300))
        row = _plant_row(loop, version=["21.03", "22.04"][k % 2])
        row["example_id"] += f"/{seed}/{k}"
        row["series"][2]["values"] = (u * 0.02).tolist()
        row["series"][1]["values"] = (mm * 0.02).tolist()
        row["series"][0]["values"] = (rng.normal(size=300) * 0.02).tolist()
        rows.append(row)
    return rows


def test_fit_plants_recovers_the_plant_from_quiet_rows_of_all_versions():
    opentslm_without_init()
    import numpy as np

    mod = load_script("HAIAttackQADataset")
    rows = _plant_quiet_rows(8)
    other = _plant_quiet_rows(2, seed=5, follows=False)
    rows += [{**r, "label": "attack", "window_kind": "attack"} for r in other]        # attack rows are ignored
    rows += [{**r, "label": "normal", "window_kind": "attack"} for r in other]        # and so are neighbour rows
    two_targets = _plant_quiet_rows(1, seed=7, follows=False)[0]
    two_targets["series"].insert(1, {**two_targets["series"][0], "name": "target 2"})
    rows.append(two_targets)                                                           # not eligible: ignored
    plants = mod.fit_plants(rows)
    assert set(plants) == {"P1-FC"}                                                    # one fit per loop, not per version
    coef, res95, n, r2 = plants["P1-FC"]
    assert n == 8 and isinstance(r2, float) and r2 > 0.99 and all(isinstance(v, float) for v in coef)
    assert coef[1] == pytest.approx(0.6, abs=0.01) and coef[3] == pytest.approx(0.3, abs=0.01)
    assert max(abs(c) for i, c in enumerate(coef[:7]) if i not in (1, 3)) < 0.02   # the other lags
    # res95: the 95th percentile of the largest residual of each row, under the fit
    want = []
    for r in rows[:8]:
        x, y = mod._plant_design(r)
        want.append(np.abs(y - x @ np.asarray(coef)).max())
    assert res95 == pytest.approx(np.percentile(want, 95)) and res95 > 0
    assert mod.plant_for(plants, "P1-FC") == plants["P1-FC"]
    assert mod.plant_ratio(rows[0], plants) == float(f"{min(want[0] / res95, mod.SIZE_CAP):.1f}")
    attacked = _plant_quiet_rows(1, seed=9)[0]
    attacked["series"][1]["values"][150:] = [v + 20 * 0.02 for v in attacked["series"][1]["values"][150:]]
    assert mod.plant_ratio(attacked, plants) > 5
    assert mod.fit_plants([r for r in rows if r["window_kind"] == "attack"]) == {}


def test_plant_r2_gate_switches_the_clause_off():
    opentslm_without_init()
    mod = load_script("HAIAttackQADataset")
    rows = _plant_quiet_rows(6) + _plant_quiet_rows(6, loop="P2-SC", seed=1, follows=False)
    plants = mod.fit_plants(rows)
    assert plants["P1-FC"][3] >= mod.PLANT_MIN_R2 > plants["P2-SC"][3]
    assert mod.plant_for(plants, "P2-SC") is None and mod.plant_for(plants, "P3-LC") is None
    assert mod.plant_ratio(rows[-1], plants) is None and mod.plant_clause(rows[-1], plants) is None
    assert mod.plant_clause(rows[0], plants) is not None
    fit = ((0.0,) * 9, 2.0, 10)
    assert mod.plant_for({"P1-FC": fit + (0.49,)}, "P1-FC") is None
    assert mod.plant_for({"P1-FC": fit + (0.5,)}, "P1-FC") == fit + (0.5,)
    for res95 in (0.0, -1.0, float("nan"), float("inf")):                              # res95 must be positive, finite
        assert mod.plant_for({"P1-FC": ((0.0,) * 9, res95, 10, 0.9)}, "P1-FC") is None
    exact = _plant_quiet_rows(6)                                                        # the measurement is the command
    for r in exact:
        r["series"][1]["values"] = list(r["series"][2]["values"])
    perfect = mod.fit_plants(exact)["P1-FC"]
    assert perfect[3] == pytest.approx(1.0) and perfect[1] == pytest.approx(0.0, abs=1e-9)
    assert mod.plant_for({"P1-FC": perfect[:1] + (0.0,) + perfect[2:]}, "P1-FC") is None   # a perfect fit: res95 0


def test_plant_clause_needs_one_target_one_measurement_and_a_command():
    opentslm_without_init()
    mod = load_script("HAIAttackQADataset")
    plants = {"P1-FC": ((0.0,) * 9, 1.0, 10, 0.9)}
    row = _plant_row()
    assert mod.plant_clause(row, plants) is not None
    two = {**row, "series": [row["series"][0], {**row["series"][0], "name": "target 2"}] + row["series"][1:]}
    assert mod._plant_design(two) is None and mod.plant_ratio(two, plants) is None and mod.plant_clause(two, plants) is None
    assert mod.plant_clause({**row, "series": row["series"][1:]}, plants) is None                 # no target
    assert mod.plant_clause({**row, "series": row["series"][:2]}, plants) is None                 # no command
    assert mod.fit_plants([two]) == {}


def _plantful(rows, loop="P1-PC"):
    """The rows with the measurement of every row of `loop` following its command 5 and 15 seconds back, plus a
    little noise; the other loops as they are."""
    import numpy as np

    out = []
    for k, r in enumerate(rows):
        if r["loop"] == loop:
            series = [dict(s) for s in r["series"]]
            u = np.asarray(series[2]["values"], dtype=np.float64)
            m = np.zeros(300)
            m[15:] = 0.6 * u[10:295] + 0.3 * u[:285]
            m += np.random.default_rng(k).normal(size=300) * 0.05
            series[1]["values"] = m.tolist()
            r = {**r, "series": series}
        out.append(r)
    return out


def test_dataset_plant_clause_on_the_measurement_line_only(hai_dataset):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_plantful(rows))      # P1-PC follows a plant, the other loops do not
    plain = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step,law")
    train = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="step,law,plant")
    swapped = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="plant,law,step")
    cls = type(train)
    plants = cls.plant_models
    assert plants == mod.fit_plants(scaled) == type(swapped).plant_models and type(plain).plant_models is None
    assert mod.plant_for(plants, "P1-PC") is not None
    assert all(mod.plant_for(plants, loop) is None for loop in plants if loop != "P1-PC")
    with_plant = 0
    for split in ("train", "validation", "test"):
        ds, base = cls(split, EOS_TOKEN="<eos>"), type(plain)(split, EOS_TOKEN="<eos>")
        for i, sample in enumerate(ds):
            r = scaled[i]
            want = mod.plant_clause(r, plants)
            assert (want is not None) == (r["loop"] == "P1-PC")
            assert sample["time_series_text"] == swapped[i]["time_series_text"]   # the three files hold the same rows
            for text, before, s in zip(sample["time_series_text"], base[i]["time_series_text"], r["series"]):
                if want is not None and s["role"] == "measurement":
                    assert text == before[:-1] + "; " + want + ":"
                    with_plant += 1
                else:
                    assert text == before and "strays" not in text
            for a, b in zip(sample["time_series"], base[i]["time_series"]):
                assert np.array_equal(np.asarray(a), np.asarray(b))
            assert sample["pre_prompt"] == base[i]["pre_prompt"] and sample["answer"] == base[i]["answer"]
    assert with_plant > 0
    # the looprange view: last on the line, after size, step and law
    loop = build("train", HAI_SERIES_VIEW="looprange", HAI_TEXT_CLAUSES="size,step,law,plant")
    for i in range(len(loop)):
        assert _texts(loop, i) == _texts(train, i)
    row = next(r for r in scaled if r["loop"] == "P1-PC")
    got = [p.get_text() for p in loop._get_text_time_series_prompt_list(row)]
    assert got[1].index("its largest change") < got[1].index("one-second step") < got[1].index("strays from")
    assert "strays" not in got[0] + got[2]
    type(loop).plant_models = None
    with pytest.raises(RuntimeError, match="plant models"):
        loop._get_text_time_series_prompt_list(row)


def test_dataset_plant_models_come_from_the_file_rows(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_plantful(rows))
    want = mod.fit_plants(scaled)
    for env in ({}, {"HAI_OVERFIT_ROWS": "5"}, {"HAI_ATTACK_SHARE": "0.5"},
                {"HAI_OVERFIT_ROWS": "10", "HAI_ATTACK_SHARE": "0.5"}):
        for k in ("HAI_OVERFIT_ROWS", "HAI_ATTACK_SHARE"):
            monkeypatch.delenv(k, raising=False)
        ds = build("train", HAI_SERIES_VIEW="zsize", HAI_TEXT_CLAUSES="plant", **env)
        assert type(ds).plant_models == want and len(ds) > 0
        assert type(ds).control_laws is None and type(ds).step_units is None
    for clauses in ("", "step", "law", "size,step,law"):
        monkeypatch.setenv("HAI_TEXT_CLAUSES", clauses)
        assert type(build("train", HAI_SERIES_VIEW="zsize")).plant_models is None
    monkeypatch.delenv("HAI_TEXT_CLAUSES")
    assert type(build("train", HAI_SERIES_VIEW="looprange")).plant_models is None


def test_dataset_texts_without_the_plant_clause_are_unchanged(hai_dataset, monkeypatch):
    import numpy as np

    rows, build, mod = hai_dataset
    scaled = _write_scaled_rows(_lawful(_plantful(rows)))
    for view, clauses in (("zsize", ""), ("zsize", "step,law"), ("looprange", ""), ("looprange", "size,step,law")):
        without = build("train", HAI_SERIES_VIEW=view, HAI_TEXT_CLAUSES=clauses)
        assert type(without).plant_models is None
        with_plant = build("train", HAI_SERIES_VIEW=view, HAI_TEXT_CLAUSES=",".join(filter(None, [clauses, "plant"])))
        for i in range(len(without)):
            plain = _texts(without, i)
            for text, before, s in zip(_texts(with_plant, i), plain, scaled[i]["series"]):
                assert "strays" not in before
                if text != before:
                    assert s["role"] == "measurement" and scaled[i]["loop"] == "P1-PC"
                    assert text == before[:-1] + "; " + mod.plant_clause(scaled[i], type(with_plant).plant_models) + ":"
            for a, b in zip(without[i]["time_series"], with_plant[i]["time_series"]):
                assert np.array_equal(np.asarray(a), np.asarray(b))
    monkeypatch.delenv("HAI_TEXT_CLAUSES")
    zscore = build("train", HAI_SERIES_VIEW="zscore")
    assert type(zscore).plant_models is None
    assert all("strays" not in t for i in range(len(zscore)) for t in _texts(zscore, i))


def test_plant_clause_names_are_checked(hai_dataset, monkeypatch):
    rows, build, mod = hai_dataset
    _write_scaled_rows(rows)
    monkeypatch.setenv("HAI_SERIES_VIEW", "zsize")
    for raw, want in (("plant", ("plant",)), ("step,law,plant", ("step", "law", "plant")), (" plant , size", ("plant", "size"))):
        monkeypatch.setenv("HAI_TEXT_CLAUSES", raw)
        assert mod.text_clauses() == want
    for bad in ("plants", "plant,jump", "Plant"):
        monkeypatch.setenv("HAI_TEXT_CLAUSES", bad)
        with pytest.raises(RuntimeError, match="HAI_TEXT_CLAUSES takes names"):
            mod.text_clauses()
    monkeypatch.setenv("HAI_SERIES_VIEW", "range")
    monkeypatch.setenv("HAI_TEXT_CLAUSES", "plant")
    with pytest.raises(RuntimeError, match="needs HAI_SERIES_VIEW=zsize"):
        mod.text_clauses()


def test_run_variant_takes_the_plant_clause():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "plant"]).clauses == "plant"
    assert rv.arguments(["x", "--view", "zsize", "--clauses", "step,law,plant"]).clauses == "step,law,plant"
    assert rv.arguments(["x", "--view", "looprange", "--clauses", "size,step,law,plant"]).clauses == "size,step,law,plant"
    with pytest.raises(SystemExit):
        rv.arguments(["x", "--view", "zscore", "--clauses", "plant"])


# --- label-first answers --------------------------------------------------------

TEMPLATES = [
    ("The command falls from second 127 while the target stays flat. The measurement falls after it. "
     "Change starts near second 116. Answer: attack",
     "attack. Change starts near second 116. The command falls from second 127 while the target stays flat. "
     "The measurement falls after it."),
    ("No series shows a clear step. Change starts near second 104. Answer: attack",
     "attack. Change starts near second 104. No series shows a clear step."),
    ("Change starts near second 7. Answer: attack", "attack. Change starts near second 7."),
    ("The change is already under way when the window opens. Answer: attack",
     "attack. The change is already under way when the window opens."),
    ("The target and the measurement stay flat. The command stays flat. Answer: normal",
     "normal. The target and the measurement stay flat. The command stays flat."),
]


def test_label_first_moves_the_label_and_the_start_to_the_front():
    mod = load_script("HAIAttackQADataset")
    for exported, want in TEMPLATES:
        assert mod.label_first(exported) == want
        assert score.parse(want) == score.parse(exported)          # same label and start second either way
    with pytest.raises(ValueError):
        mod.label_first("The command rises.")


def test_label_first_keeps_label_and_start_of_every_exported_answer():
    from tests.training_helpers import ROOT

    path = ROOT / "data" / "processed" / "opentslm" / "validation.jsonl"
    if not path.exists():
        pytest.skip("no exported files: run `uv run hai-build` and `uv run hai-export`")
    mod = load_script("HAIAttackQADataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(rows) > 1000
    for r in rows:
        first = mod.label_first(r["answer"])
        assert first.startswith(r["label"] + ".")
        assert score.parse(first) == score.parse(r["answer"]) and score.parse(first)[0] == r["label"]


def test_scorer_reads_what_a_label_first_model_writes():
    assert score.parse(" attack. Change starts near second 116. The command falls from second 127.") == ("attack", 116)
    assert score.parse("normal. The target and the measurement stay flat.") == ("normal", None)
    assert score.parse(" attack. The change is already under way when the window opens.") == ("attack", 0)
    assert score.parse("The command rises.") == (None, None)
    assert score.parse("Answer: normal") == ("normal", None)
    assert score.parse("Change starts near second 12. Answer: attack") == ("attack", 12)


def test_dataset_can_put_the_label_first(hai_dataset):
    rows, build, mod = hai_dataset
    ds = build("train", HAI_ANSWER_ORDER="label_first")
    for i in range(len(ds)):
        s = ds[i]
        assert s["answer"] == mod.label_first(rows[i]["answer"]) + "<eos>"
        assert s["answer"].startswith(rows[i]["label"] + ".")
        assert "Say first whether" in s["pre_prompt"] and "End with" not in s["pre_prompt"]
        assert s["post_prompt"] == "Answer:"


def test_label_only_keeps_just_the_label():
    mod = load_script("HAIAttackQADataset")
    assert mod.label_only("The command rises from second 37. Change starts near second 33. Answer: attack") == "attack"
    assert mod.label_only("The target and the measurement stay flat. Answer: normal") == "normal"
    with pytest.raises(ValueError):
        mod.label_only("The command rises.")


def test_dataset_can_ask_for_the_label_only(hai_dataset):
    rows, build, mod = hai_dataset
    ds = build("train", HAI_ANSWER_ORDER="label_only")
    for i in range(len(ds)):
        s = ds[i]
        assert s["answer"] == rows[i]["label"] + "<eos>"
        assert "one word" in s["pre_prompt"] and "Answer: attack" not in s["pre_prompt"]


def test_scorer_reads_what_a_label_only_model_writes():
    score = load_script("score")
    assert score.parse("attack") == ("attack", None)
    assert score.parse(" normal") == ("normal", None)


# --- starting from another run -------------------------------------------------

def _cloud_script(name: str):
    import importlib.util

    from tests.training_helpers import ROOT
    spec = importlib.util.spec_from_file_location(f"cloud_{name}", ROOT / "training" / "cloud" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_start_source_names_a_repo_or_the_model_of_a_finished_run(tmp_path):
    rv = _cloud_script("run_variant")
    assert rv.start_source("sleep", tmp_path) == "OpenTSLM/llama-3.2-1b-sleep-sp"
    assert rv.start_source("some-org/some-repo", tmp_path) == "some-org/some-repo"
    kept = tmp_path / "work" / "abc" / "kept"
    kept.mkdir(parents=True)
    (kept / "best_model.pt").write_bytes(b"model")
    (kept / "run_summary.json").write_text("{}")
    assert rv.start_source("run:abc", tmp_path) == str((kept / "best_model.pt").resolve())
    with pytest.raises(SystemExit):
        rv.start_source("run:missing", tmp_path)
    unfinished = tmp_path / "work" / "half" / "kept"
    unfinished.mkdir(parents=True)
    (unfinished / "best_model.pt").write_bytes(b"model")
    with pytest.raises(SystemExit):
        rv.start_source("run:half", tmp_path)


def test_seed_copies_a_local_checkpoint(tmp_path, monkeypatch):
    seed = _cloud_script("seed")
    local = tmp_path / "other" / "best_model.pt"
    local.parent.mkdir()
    local.write_bytes(b"\x00\x01dummy checkpoint")
    monkeypatch.chdir(tmp_path)
    dst = seed.seed("unsloth/Llama-3.2-1B", str(local))
    stage = tmp_path / "results" / "Llama_3_2_1B" / "OpenTSLMSP" / "stage5_ecg_cot"
    assert (tmp_path / dst).resolve() == (stage / "checkpoints" / "best_model.pt").resolve()
    assert (stage / "checkpoints" / "best_model.pt").read_bytes() == local.read_bytes()
    assert json.loads((stage / "results" / "metrics.json").read_text())["seeded_from"] == str(local.resolve())


def test_run_variant_parser_takes_a_run_as_start():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x"]).start == "sleep"
    assert rv.arguments(["x", "--start", "sleep"]).start == "sleep"
    assert rv.arguments(["x", "--start", "run:abc"]).start == "run:abc"


# --- choosing the pass whose model is kept --------------------------------------

def test_pass_files_are_found_in_pass_order(tmp_path):
    rv = _cloud_script("run_variant")
    for name in ("pass_1.pt", "pass_10.pt", "pass_2.pt", "best_model.pt", "pass_x.pt"):
        (tmp_path / name).write_bytes(b"model")
    got = rv.pass_files(tmp_path)
    assert list(got) == [1, 2, 10] and got[10] == tmp_path / "pass_10.pt"
    assert rv.pass_files(tmp_path / "missing") == {}


def _lines(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_pass_score_counts_attacks_found_minus_normal_rows_alarmed(tmp_path):
    rv = _cloud_script("run_variant")
    truth = _lines(tmp_path / "truth.jsonl", [
        {"example_id": "a1", "label": "attack"}, {"example_id": "a2", "label": "attack"},
        {"example_id": "a3", "label": "attack"}, {"example_id": "n1", "label": "normal"},
        {"example_id": "n2", "label": "normal"}, {"example_id": "n3", "label": "attack"}])   # n3: no prediction
    predictions = _lines(tmp_path / "predictions.jsonl", [
        {"example_id": "a1", "generated": "Change starts near second 12. Answer: attack"},
        {"example_id": "a2", "generated": " attack. Change starts near second 40."},        # label first
        {"example_id": "a3", "generated": "The command rises."},                            # does not parse
        {"example_id": "n1", "generated": "attack"},
        {"example_id": "n2", "generated": "The target stays flat. Answer: normal"}])
    assert rv.pass_score(predictions, truth) == {"attack_rows_found": 2, "normal_rows_alarmed": 1, "score": 1}


def test_best_pass_takes_the_highest_score_and_the_later_pass_on_a_tie():
    rv = _cloud_script("run_variant")
    assert rv.best_pass({1: {"score": 3}, 2: {"score": 7}, 3: {"score": -1}}) == 2
    assert rv.best_pass({1: {"score": 5}, 2: {"score": 2}, 3: {"score": 5}}) == 3


def test_run_variant_parser_takes_the_pass_choice():
    rv = _cloud_script("run_variant")
    assert rv.arguments(["x"]).choose_pass == "loss"
    for mode in ("loss", "last", "validation"):
        assert rv.arguments(["x", "--choose-pass", mode]).choose_pass == mode
    with pytest.raises(SystemExit):
        rv.arguments(["x", "--choose-pass", "best"])


def _pass_stage(tmp_path, passes):
    """A stage after training with HAI_KEEP_PASSES: pass files, the best-loss model and its answers in kept/."""
    stage, kept = tmp_path / "stage", tmp_path / "kept"
    (stage / "checkpoints").mkdir(parents=True)
    (stage / "results").mkdir()
    kept.mkdir()
    for n in passes:
        (stage / "checkpoints" / f"pass_{n}.pt").write_text(f"pass {n}")
    (stage / "checkpoints" / "best_model.pt").write_text("pass 2 with optimizer")
    truth = _lines(tmp_path / "validation.jsonl", [{"example_id": "a", "label": "attack"},
                                                   {"example_id": "b", "label": "attack"},
                                                   {"example_id": "n", "label": "normal"}])
    return stage, kept, truth


def _answers(stage, says):
    """A stand-in for answer_again: the model in best_model.pt answers with the labels in `says`."""
    answered = []

    def answer():
        model = (stage / "checkpoints" / "best_model.pt").read_text()
        answered.append(model)
        rows = [{"example_id": i, "generated": f"Answer: {label}"} for i, label in zip("abn", says[model])]
        _lines(stage / "results" / "test_predictions.jsonl", rows)
        return {"model": model}

    return answer, answered


def test_choose_validation_keeps_the_pass_best_on_the_validation_rows(tmp_path):
    rv = _cloud_script("run_variant")
    stage, kept, truth = _pass_stage(tmp_path, (1, 2, 3))
    first = kept / "test_predictions_validation.jsonl"
    _lines(first, [{"example_id": i, "generated": "Answer: normal"} for i in "abn"])   # pass 2: finds nothing
    says = {"pass 1": ("attack", "normal", "attack"), "pass 3": ("attack", "attack", "normal")}
    answer, answered = _answers(stage, says)
    got = rv.choose("validation", stage, kept, "validation", 2, truth, answer, say=lambda text: None)
    assert answered == ["pass 1", "pass 3"]                               # the best-loss pass is not answered again
    assert got["kept_pass"] == 3 and [a["model"] for a in got["pass_answers"]] == answered
    assert got["pass_scores"] == {1: {"attack_rows_found": 1, "normal_rows_alarmed": 1, "score": 0},
                                  2: {"attack_rows_found": 0, "normal_rows_alarmed": 0, "score": 0},
                                  3: {"attack_rows_found": 2, "normal_rows_alarmed": 0, "score": 2}}
    assert (stage / "checkpoints" / "best_model.pt").read_text() == "pass 3"
    assert first.read_text() == (kept / "passes" / "test_predictions_validation_pass_3.jsonl").read_text()
    assert sorted(f.name for f in (kept / "passes").iterdir()) == [
        f"test_predictions_validation_pass_{n}.jsonl" for n in (1, 2, 3)]


def test_choose_validation_restores_the_best_loss_model_when_it_wins(tmp_path):
    rv = _cloud_script("run_variant")
    stage, kept, truth = _pass_stage(tmp_path, (1, 2))
    _lines(kept / "test_predictions_validation.jsonl",
           [{"example_id": i, "generated": f"Answer: {label}"} for i, label in zip("abn", ("attack", "attack", "normal"))])
    answer, _ = _answers(stage, {"pass 1": ("normal", "normal", "attack")})
    got = rv.choose("validation", stage, kept, "validation", 2, truth, answer, say=lambda text: None)
    assert got["kept_pass"] == 2
    assert (stage / "checkpoints" / "best_model.pt").read_text() == "pass 2 with optimizer"   # OpenTSLM's own file


def test_choose_last_answers_again_only_when_the_last_pass_is_not_the_best_loss_one(tmp_path):
    rv = _cloud_script("run_variant")
    stage, kept, truth = _pass_stage(tmp_path, (1, 2, 3))
    first = kept / "test_predictions_validation.jsonl"
    first.write_text("best-loss answers\n")
    answer, answered = _answers(stage, {"pass 3": ("attack", "attack", "attack")})
    got = rv.choose("last", stage, kept, "validation", 2, truth, answer, say=lambda text: None)
    assert got["kept_pass"] == 3 and answered == ["pass 3"] and got["pass_scores"] is None
    assert (stage / "checkpoints" / "best_model.pt").read_text() == "pass 3"
    assert first.read_text() == (stage / "results" / "test_predictions.jsonl").read_text()

    stage, kept, truth = _pass_stage(tmp_path / "again", (1, 2))
    answer, answered = _answers(stage, {})
    got = rv.choose("last", stage, kept, "validation", 2, truth, answer, say=lambda text: None)
    assert got["kept_pass"] == 2 and answered == []
    assert (stage / "checkpoints" / "best_model.pt").read_text() == "pass 2 with optimizer"


def test_choose_loss_changes_nothing(tmp_path):
    rv = _cloud_script("run_variant")
    stage, kept, truth = _pass_stage(tmp_path, ())
    answer, answered = _answers(stage, {})
    assert rv.choose("loss", stage, kept, "validation", 4, truth, answer) == {
        "kept_pass": 4, "pass_scores": None, "pass_answers": []}
    assert answered == [] and (stage / "checkpoints" / "best_model.pt").read_text() == "pass 2 with optimizer"


def test_loss_pass_is_read_from_the_metrics(tmp_path):
    rv = _cloud_script("run_variant")
    assert rv.loss_pass(tmp_path) is None
    (tmp_path / "metrics.json").write_text(json.dumps({"epoch": 3, "test_loss": 0.5}))
    assert rv.loss_pass(tmp_path) == 3
