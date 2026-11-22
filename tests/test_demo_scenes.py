"""Checks on demo/data/scenes.json, the control room's data. Reads only the committed file."""
import json
import math
from pathlib import Path

import pytest

SCENES = Path(__file__).resolve().parent.parent / "demo" / "data" / "scenes.json"
LOOPS = ["P1-PC", "P1-LC", "P1-FC", "P1-TC", "P2-SC", "P3-LC"]
KEYS = ["caught", "two_loops", "one_missed", "missed", "false_alarm", "quiet"]


@pytest.fixture(scope="module")
def data():
    return json.loads(SCENES.read_text(encoding="utf-8"))


def scene(data, key):
    return next(s for s in data["scenes"] if s["key"] == key)


def loop(sc, lp):
    return next(x for x in sc["loops"] if x["loop"] == lp)


def test_scene_keys_and_loop_order(data):
    assert [s["key"] for s in data["scenes"]] == KEYS
    assert [lo["loop"] for lo in data["loops"]] == LOOPS
    for s in data["scenes"]:
        assert [x["loop"] for x in s["loops"]] == LOOPS


def test_series_are_300_finite_numbers(data):
    assert data["seconds"] == 300
    for s in data["scenes"]:
        for x in s["loops"]:
            assert x["series"]
            for ser in x["series"]:
                vals = ser["values"]
                assert len(vals) == 300, (x["id"], ser["name"])
                assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals), (x["id"], ser["name"])


def test_series_match_the_loop_scales(data):
    scales = {lo["loop"]: [(s["role"], s["name"]) for s in lo["series"]] for lo in data["loops"]}
    for s in data["scenes"]:
        for x in s["loops"]:
            keys = [(ser["role"], ser["name"]) for ser in x["series"]]
            it = iter(scales[x["loop"]])
            assert all(k in it for k in keys), (x["id"], keys)


def test_votes_and_vote_rule(data):
    assert data["voters"] == 7 and data["votes_needed"] == 4
    for s in data["scenes"]:
        for x in s["loops"]:
            assert len(x["voters"]) == 7
            assert x["votes_attack"] == sum(1 for v in x["voters"] if v["label"] == "attack")
            assert (x["vote"] == "attack") == (x["votes_attack"] >= 4)
            assert (x["vote_start"] is None) == (x["vote"] == "normal")


def test_attack_start_only_on_attacked_loops(data):
    for s in data["scenes"]:
        for x in s["loops"]:
            if x["kind"] == "attack":
                assert isinstance(x["attack_start"], int) and 0 <= x["attack_start"] <= 299, x["id"]
            else:
                assert x["attack_start"] is None, x["id"]


def test_flagged_and_attacked_lists(data):
    for s in data["scenes"]:
        assert s["flagged"] == [x["loop"] for x in s["loops"] if x["vote"] == "attack"]
        assert s["attacked"] == [x["loop"] for x in s["loops"] if x["kind"] == "attack"]


def test_check_set_counts(data):
    c = data["check_set"]
    assert c["rows"] == 1092 and c["window_times"] == 182
    assert (c["vote"]["caught"], c["vote"]["false_alarms"]) == (101, 11)
    assert (c["gradient_boosting"]["caught"], c["gradient_boosting"]["false_alarms"]) == (108, 26)


def test_scene_facts_of_the_brief(data):
    sc = scene(data, "caught")
    assert sc["time_id"] == "22.04/test3/20210715T020720"
    assert sc["attacked"] == ["P1-LC"] and loop(sc, "P1-LC")["attack_start"] == 110
    x = loop(sc, "P1-LC")
    assert (x["votes_attack"], x["vote_start"], x["gb"]) == (7, 127, "attack")
    assert all(loop(sc, lp)["votes_attack"] == 0 for lp in LOOPS if lp != "P1-LC")

    sc = scene(data, "two_loops")
    assert sc["time_id"] == "21.03/test4/20200728T124026"
    assert set(sc["attacked"]) == {"P1-LC", "P1-PC"}
    assert loop(sc, "P1-LC")["attack_start"] == 157 and loop(sc, "P1-PC")["attack_start"] == 157
    assert loop(sc, "P1-LC")["votes_attack"] == 7 and loop(sc, "P1-PC")["votes_attack"] == 7
    assert loop(sc, "P1-LC")["vote_start"] == 155 and loop(sc, "P1-PC")["vote_start"] == 183

    sc = scene(data, "one_missed")
    assert sc["time_id"] == "22.04/test3/20210714T232009"
    assert set(sc["attacked"]) == {"P1-LC", "P3-LC"}
    lc, wl = loop(sc, "P1-LC"), loop(sc, "P3-LC")
    assert (lc["attack_start"], lc["votes_attack"], lc["vote_start"], lc["gb"]) == (112, 7, 127, "attack")
    assert (wl["attack_start"], wl["votes_attack"], wl["vote"], wl["gb"]) == (112, 0, "normal", "attack")

    sc = scene(data, "missed")
    assert sc["time_id"] == "22.04/test3/20210715T013858"
    assert sc["attacked"] == ["P3-LC"] and sc["flagged"] == []
    wl = loop(sc, "P3-LC")
    assert (wl["attack_start"], wl["votes_attack"], wl["gb"]) == (129, 0, "normal")

    sc = scene(data, "false_alarm")
    assert sc["time_id"] == "21.03/test4/20200728T141317"
    assert sc["attacked"] == [] and sc["flagged"] == ["P1-LC"]
    lc = loop(sc, "P1-LC")
    assert (lc["votes_attack"], lc["vote_start"], lc["gb"]) == (5, 53, "normal")

    sc = scene(data, "quiet")
    assert sc["time_id"] == "21.03/test4/20200728T120817"
    assert sc["attacked"] == [] and sc["flagged"] == []
    assert all(loop(sc, lp)["votes_attack"] == 0 and loop(sc, lp)["gb"] == "normal" for lp in LOOPS)


def test_window_start_matches_time_id(data):
    for s in data["scenes"]:
        stamp = s["time_id"].rsplit("/", 1)[1]
        assert s["window_start"].replace("-", "").replace(":", "") == stamp
