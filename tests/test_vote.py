"""Tests for training/vote.py: the verdict vote over several prediction files."""
import json

import pytest

from tests.training_helpers import load_script

vote_mod = load_script("vote")
score = load_script("score")

ATTACK = "The measurement jumps. Change starts near second {}. Answer: attack"
NORMAL = "The loop stays flat. Answer: normal"


def files(*answers_per_file, ids=None):
    """One file per argument; each argument lists the generated answers row by row."""
    ids = ids or [f"w/{i}" for i in range(len(answers_per_file[0]))]
    return [[{"example_id": i, "generated": a, "gold": f"gold {i}"} for i, a in zip(ids, answers)]
            for answers in answers_per_file]


def labels(rows):
    return [score.parse(r["generated"])[0] for r in rows]


def test_majority_of_three():
    a, n = ATTACK.format(50), NORMAL
    out = vote_mod.vote(files([a, a, a, n], [a, a, n, n], [a, n, n, a]))
    assert [r["votes"] for r in out] == [3, 2, 1, 1]
    assert labels(out) == ["attack", "attack", "normal", "normal"]
    out = vote_mod.vote(files([n], [n], [n]))
    assert labels(out) == ["normal"] and out[0]["votes"] == 0


def test_two_files_need_both():
    a, n = ATTACK.format(50), NORMAL
    assert labels(vote_mod.vote(files([a, a, n], [a, n, n]))) == ["attack", "normal", "normal"]


def test_four_files_need_three():
    a, n = ATTACK.format(50), NORMAL
    out = vote_mod.vote(files([a, a], [a, a], [a, n], [n, n]))
    assert labels(out) == ["attack", "normal"]


def test_min_k():
    a, n = ATTACK.format(50), NORMAL
    rows = files([a, a, n], [n, a, n], [n, a, n])
    assert labels(vote_mod.vote(rows, k=1)) == ["attack", "attack", "normal"]
    assert labels(vote_mod.vote(rows, k=3)) == ["normal", "attack", "normal"]
    for bad in (0, 4):
        with pytest.raises(ValueError, match="between 1 and 3"):
            vote_mod.vote(rows, k=bad)


def test_text_comes_from_the_first_agreeing_file():
    out = vote_mod.vote(files([NORMAL], [ATTACK.format(40)], [ATTACK.format(90)]))
    assert out[0]["generated"] == ATTACK.format(40)
    assert score.parse(out[0]["generated"]) == ("attack", 40)
    out = vote_mod.vote(files([ATTACK.format(10)], ["The loop is calm. Answer: normal"], [NORMAL]))
    assert out[0]["generated"] == "The loop is calm. Answer: normal"


def test_label_first_and_verdict_only_answers_vote_attack():
    lead = "attack. Change starts near second 40. The measurement drifts."
    out = vote_mod.vote(files([lead], ["attack"], [NORMAL]))
    assert out[0]["votes"] == 2 and out[0]["generated"] == lead
    assert score.parse(out[0]["generated"]) == ("attack", 40)


def test_unparsed_answer_is_no_attack_vote():
    out = vote_mod.vote(files(["I am not sure."], [ATTACK.format(50)], [""]))
    assert out[0]["votes"] == 1
    assert out[0]["generated"] == "Answer: normal"  # no file parses to normal
    assert labels(out) == ["normal"]


def test_missing_extra_and_duplicated_ids():
    a = ATTACK.format(50)
    base = files([a, a])[0]
    with pytest.raises(ValueError, match="file 2 differs from file 1: 1 ids missing, 0 extra"):
        vote_mod.vote([base, base[:1]])
    with pytest.raises(ValueError, match="file 2 differs from file 1: 0 ids missing, 1 extra"):
        vote_mod.vote([base, base + [{"example_id": "w/9", "generated": a}]])
    with pytest.raises(ValueError, match="file 2: 1 duplicated example ids, e.g. w/0"):
        vote_mod.vote([base, base + [base[0]]])
    with pytest.raises(ValueError, match="at least two"):
        vote_mod.vote([base])


def test_order_of_first_file_and_fields():
    a, n = ATTACK.format(50), NORMAL
    first = files([a, n, a], ids=["c", "a", "b"])[0]
    second = list(reversed(files([a, n, n], ids=["c", "a", "b"])[0]))
    out = vote_mod.vote([first, second, first])
    assert [r["example_id"] for r in out] == ["c", "a", "b"]
    assert all(r["voters"] == 3 for r in out)
    assert [r["votes"] for r in out] == [3, 0, 2]
    assert [r["gold"] for r in out] == ["gold c", "gold a", "gold b"]
    no_gold = [{"example_id": "x", "generated": a}]
    assert "gold" not in vote_mod.vote([no_gold, no_gold])[0]


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_main_end_to_end(tmp_path, capsys):
    a, n = ATTACK.format(50), NORMAL
    rows = files([a, a, n, "junk"], [a, n, n, a], [n, a, n, a])
    paths = [write(tmp_path / f"p{i}.jsonl", r) for i, r in enumerate(rows)]
    out = tmp_path / "vote.jsonl"
    vote_mod.main(["--out", str(out), *paths])
    lines = out.read_text().splitlines()
    got = [json.loads(line) for line in lines]
    assert len(got) == 4
    want = ["attack", "attack", "normal", "attack"]
    assert [score.parse(r["generated"])[0] for r in got] == want
    assert "4 rows, 3 voted attack, K=2 of 3" in capsys.readouterr().out


def test_main_errors(tmp_path):
    a = ATTACK.format(50)
    p1 = write(tmp_path / "p1.jsonl", files([a, a])[0])
    p2 = write(tmp_path / "p2.jsonl", files([a])[0])
    with pytest.raises(SystemExit, match="p2.jsonl differs"):
        vote_mod.main(["--out", str(tmp_path / "o.jsonl"), p1, p2])
    with pytest.raises(SystemExit, match="between 1 and 2"):
        vote_mod.main(["--out", str(tmp_path / "o.jsonl"), "--min", "3", p1, p1])
    with pytest.raises(SystemExit):
        vote_mod.main(["--out", str(tmp_path / "o.jsonl"), p1])
