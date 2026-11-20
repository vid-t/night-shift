"""Tests for training/: the stage patcher, the scorer and the dataset class. No GPU, no network."""
import json

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
    for bad in ("", None, "attack", "Answer:", "Answer: maybe"):
        assert score.parse(bad) == (None, None)


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
