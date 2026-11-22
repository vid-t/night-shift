"""Build demo/data/scenes.json for the control room of the demo page.

Run from the repo root:  uv run python demo/build_scenes.py

A scene is one 300-second moment of the plant seen through all six loops. The check set holds all six
loops for each of its 182 window times. This script reads only check-set (validation) files and the
answers on them; it never opens a test file. It reuses demo/build_data.py for parsing and counting rules.
"""
import importlib.util
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_build_data():
    spec = importlib.util.spec_from_file_location("demo_build_data", HERE / "build_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = load_build_data()
ROOT = bd.ROOT
OUT = ROOT / "demo" / "data" / "scenes.json"
LOOPS = ["P1-PC", "P1-LC", "P1-FC", "P1-TC", "P2-SC", "P3-LC"]
SECONDS = 300
NEED = 4  # votes of seven needed for an attack call

# The scene table of the brief. Every fact here is checked against the data; the script stops on a mismatch.
# attacked: loop -> true start second. votes: loop -> attack votes (loops not named: checked only if
# others_votes is set). starts: loop -> vote_start. gb: loop -> gradient boosting's call.
SCENES = [
    {"key": "caught", "time_id": "22.04/test3/20210715T020720",
     "attacked": {"P1-LC": 110}, "votes": {"P1-LC": 7}, "others_votes": 0, "starts": {"P1-LC": 127},
     "gb": {"P1-LC": "attack"}},
    {"key": "two_loops", "time_id": "21.03/test4/20200728T124026",
     "attacked": {"P1-LC": 157, "P1-PC": 157}, "votes": {"P1-LC": 7, "P1-PC": 7}, "others_votes": None,
     "starts": {"P1-LC": 155, "P1-PC": 183}, "gb": {}},
    {"key": "one_missed", "time_id": "22.04/test3/20210714T232009",
     "attacked": {"P1-LC": 112, "P3-LC": 112}, "votes": {"P1-LC": 7, "P3-LC": 0}, "others_votes": None,
     "starts": {"P1-LC": 127}, "gb": {"P1-LC": "attack", "P3-LC": "attack"}},
    {"key": "missed", "time_id": "22.04/test3/20210715T013858",
     "attacked": {"P3-LC": 129}, "votes": {"P3-LC": 0}, "others_votes": None, "starts": {},
     "gb": {"P3-LC": "normal"}},
    {"key": "false_alarm", "time_id": "21.03/test4/20200728T141317",
     "attacked": {}, "votes": {"P1-LC": 5}, "others_votes": None, "starts": {"P1-LC": 53},
     "gb": {"P1-LC": "normal"}},
    {"key": "quiet", "time_id": "21.03/test4/20200728T120817",
     "attacked": {}, "votes": {}, "others_votes": 0, "starts": {}, "gb": {lp: "normal" for lp in LOOPS}},
]


class DataDisagrees(SystemExit):
    pass


def check(ok, msg):
    if not ok:
        raise DataDisagrees(f"build_scenes: the data disagrees with the brief: {msg}")


def window_start(time_id):
    """The time part of example_id is the window's first second (src/hai/dataset.py: w.start, asserted
    equal to the first row's time of the 300-second segment), written as %Y%m%dT%H%M%S."""
    stamp = time_id.rsplit("/", 1)[1]
    return datetime.strptime(stamp, "%Y%m%dT%H%M%S").isoformat()


def p90(xs):
    return float(np.percentile(np.asarray(xs, dtype=float), 90))  # linear interpolation (numpy default)


def main():
    rows = bd.read_jsonl(bd.VALIDATION)
    assert len(rows) == 1092, len(rows)
    by_id = {r["example_id"]: r for r in rows}
    preds = {rn: {p["example_id"]: p for p in
                  bd.read_jsonl(bd.NIGHT / rn / "kept" / "test_predictions_validation.jsonl")}
             for rn in bd.VOTERS}
    gb = {p["example_id"]: p["generated"] for p in
          bd.read_jsonl(bd.BASELINES / "predictions_gradient_boosting.jsonl")}
    gbn = {p["example_id"]: p["generated"] for p in
           bd.read_jsonl(bd.BASELINES / "predictions_gradient_boosting_with_neighbours.jsonl")}
    for rn in bd.VOTERS:
        assert set(preds[rn]) == set(by_id), rn
    assert set(gb) == set(by_id) and set(gbn) == set(by_id)

    # ---- per-row facts, same rules as build_data.py -----------------------------------------
    facts = {}
    for r in rows:
        e = r["example_id"]
        answers = [bd.parse(preds[rn][e]["generated"]) for rn in bd.VOTERS]
        votes = sum(1 for lab, _ in answers if lab == "attack")
        facts[e] = {
            "kind": bd.kind_of(r),
            "votes": votes,
            "vote": "attack" if votes >= NEED else "normal",
            "gb": bd.parse(gb[e])[0],
            "gbn": bd.parse(gbn[e])[0],
            "start": bd.median([st for lab, st in answers if lab == "attack" and st is not None]),
        }

    def count(pick):
        att = sum(1 for e, f in facts.items() if f["kind"] == "attack" and pick(e) == "attack")
        nor = sum(1 for e, f in facts.items() if f["kind"] != "attack" and pick(e) == "attack")
        return {"caught": att, "false_alarms": nor, "score": att - nor}

    vote = count(lambda e: facts[e]["vote"])
    gbc = count(lambda e: facts[e]["gb"])
    assert (vote["caught"], vote["false_alarms"]) == (101, 11), vote
    assert (gbc["caught"], gbc["false_alarms"]) == (108, 26), gbc
    times = sorted({e.rsplit("/", 1)[0] for e in by_id})
    check_set = {
        "rows": len(rows),
        "attack_rows": sum(1 for f in facts.values() if f["kind"] == "attack"),
        "normal_rows": sum(1 for f in facts.values() if f["kind"] != "attack"),
        "window_times": len(times),
        "vote": vote,
        "gradient_boosting": gbc,
    }
    for t in times:  # every window time holds all six loops
        assert all(f"{t}/{lp}" in by_id for lp in LOOPS), t

    # ---- per-loop scales ----------------------------------------------------------------------
    # A loop's series must have the same roles and names in the same order within each version.
    # Across versions one exception exists: version 21.03 records no target for P2-SC. So every
    # window's series must be an ordered subset of the loop's full list; each series is matched by
    # (role, name), and "windows" says how many check-set windows hold it.
    loops_out = []
    for lp in LOOPS:
        lrows = [r for r in rows if r["loop"] == lp]
        shapes = {}
        for r in lrows:
            sh = tuple((s["role"], s["name"]) for s in r["series"])
            prev = shapes.setdefault(r["version"], sh)
            if prev != sh:
                raise SystemExit(f"build_scenes: {lp} has two series layouts in version {r['version']}: "
                                 f"{prev} vs {sh} ({r['example_id']})")
            if len(set(sh)) != len(sh):
                raise SystemExit(f"build_scenes: {lp} repeats a (role, name) pair: {sh}")
        full = max(shapes.values(), key=len)
        for v, sh in shapes.items():
            it = iter(full)
            if not all(x in it for x in sh):
                raise SystemExit(f"build_scenes: {lp} version {v} layout {sh} is not an ordered subset of {full}")
        series = []
        for key in full:
            spans, lo, hi, n = [], math.inf, -math.inf, 0
            for r in lrows:
                s = next((s for s in r["series"] if (s["role"], s["name"]) == key), None)
                if s is None:
                    continue
                vals = s["values"]
                assert len(vals) == SECONDS and all(math.isfinite(x) for x in vals), r["example_id"]
                n += 1
                lo, hi = min(lo, min(vals)), max(hi, max(vals))
                if facts[r["example_id"]]["kind"] == "normal":
                    spans.append(max(vals) - min(vals))
            assert spans, (lp, key)
            series.append({"role": key[0], "name": key[1],
                           "usual_span": bd.round_sig(statistics.median(spans)),
                           "usual_span_p90": bd.round_sig(p90(spans)),
                           "check_min": bd.round_sig(lo), "check_max": bd.round_sig(hi),
                           "windows": n})
        loops_out.append({"loop": lp, "loop_name": lrows[0]["loop_name"], "plant": bd.PLANTS[lp[:2]],
                          "series": series})
        assert len(lrows) == len(times), (lp, len(lrows))

    # ---- prompts and hints: do the seven voters' files agree? --------------------------------
    disagree = []

    # ---- scenes -------------------------------------------------------------------------------
    scenes = []
    for sc in SCENES:
        t = sc["time_id"]
        check(t in times, f"{sc['key']}: window time {t} is not in the check set")
        entries = []
        for lp in LOOPS:
            e = f"{t}/{lp}"
            r, f = by_id[e], facts[e]
            p0 = preds[bd.VOTERS[0]][e]
            for rn in bd.VOTERS[1:]:
                p = preds[rn][e]
                for k in ("pre_prompt", "time_series_text"):
                    if p[k] != p0[k]:
                        disagree.append((e, rn, k))
            voters = []
            for rn in bd.VOTERS:
                lab, start = bd.parse(preds[rn][e]["generated"])
                voters.append({"run": rn, "label": lab, "start": start, "text": preds[rn][e]["generated"]})
            # the sentence shown for the vote: same rule as build_data.py
            side = [v for v in voters if v["label"] == f["vote"]]
            vt = (min(side, key=lambda v: abs((v["start"] or 0) - (f["start"] or 0))) if f["vote"] == "attack"
                  else side[0])
            entries.append({
                "id": e,
                "loop": lp,
                "loop_name": r["loop_name"],
                "plant": bd.PLANTS[lp[:2]],
                "kind": f["kind"],
                "label": r["label"],
                "attack_start": r["attack_start_s"] if f["kind"] == "attack" and r["attack_start_s"] >= 0 else None,
                "reacting": r["reacting"],
                "no_visible_change": r["no_visible_change"],
                "reference_answer": r["answer"],
                "series": [{"role": s["role"], "name": s["name"], "values": [bd.round_sig(x) for x in s["values"]]}
                           for s in r["series"]],
                "prompt": p0["pre_prompt"],
                "hints": p0["time_series_text"],
                "voters": voters,
                "votes_attack": f["votes"],
                "vote": f["vote"],
                "vote_start": f["start"] if f["vote"] == "attack" else None,
                "vote_example": {"run": vt["run"], "start": vt["start"], "text": vt["text"]},
                "gb": f["gb"],
                "gb_neighbours": f["gbn"],
            })
        by_loop = {x["loop"]: x for x in entries}
        attacked = [x["loop"] for x in entries if x["kind"] == "attack"]
        flagged = [x["loop"] for x in entries if x["vote"] == "attack"]

        # ---- assert the brief's facts ----
        k = sc["key"]
        check(set(attacked) == set(sc["attacked"]), f"{k}: attacked loops {attacked}, brief says {list(sc['attacked'])}")
        for lp, st in sc["attacked"].items():
            check(by_loop[lp]["attack_start"] == st, f"{k}: {lp} attacked from {by_loop[lp]['attack_start']}, brief says {st}")
        for lp, n in sc["votes"].items():
            check(by_loop[lp]["votes_attack"] == n, f"{k}: {lp} has {by_loop[lp]['votes_attack']} attack votes, brief says {n}")
        if sc["others_votes"] is not None:
            for lp in LOOPS:
                if lp not in sc["votes"]:
                    check(by_loop[lp]["votes_attack"] == sc["others_votes"],
                          f"{k}: {lp} has {by_loop[lp]['votes_attack']} attack votes, brief says {sc['others_votes']}")
        for lp, st in sc["starts"].items():
            check(by_loop[lp]["vote_start"] == st, f"{k}: {lp} vote start {by_loop[lp]['vote_start']}, brief says {st}")
        for lp, call in sc["gb"].items():
            check(by_loop[lp]["gb"] == call, f"{k}: gradient boosting says {by_loop[lp]['gb']} on {lp}, brief says {call}")

        scenes.append({"key": k, "time_id": t, "version": by_loop[LOOPS[0]]["id"].split("/")[0],
                       "window_start": window_start(t), "flagged": flagged, "attacked": attacked,
                       "loops": entries})

    data = {
        "source": "HAI dataset (github.com/icsdataset/hai), CC BY-SA 4.0; check-set windows only",
        "seconds": SECONDS, "voters": len(bd.VOTERS), "votes_needed": NEED,
        "loops": loops_out,
        "check_set": check_set,
        "scenes": scenes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    size = OUT.stat().st_size

    print(f"wrote {OUT.relative_to(ROOT)}: {size / 1024:.0f} KB ({size} bytes)")
    print(f"check set: {check_set['rows']} rows, {check_set['window_times']} window times; "
          f"vote {vote}  gradient boosting {gbc}")
    print("prompt and hints: " + ("the seven voters agree on all 36 rows" if not disagree else
                                  f"{len(disagree)} disagreements, first: {disagree[:3]}"))
    for lo in loops_out:
        print(f"  {lo['loop']:6} " + "; ".join(f"{s['role']}/{s['name']} span {s['usual_span']} p90 "
                                              f"{s['usual_span_p90']} n={s['windows']}" for s in lo["series"]))
    for s in scenes:
        detail = ", ".join(f"{x['loop']} {x['kind']} {x['votes_attack']}/7"
                           + (f" from {x['attack_start']}" if x["attack_start"] is not None else "")
                           + (f" start {x['vote_start']}" if x["vote_start"] is not None else "")
                           + f" gb {x['gb']}" for x in s["loops"])
        print(f"  {s['key']:12} {s['window_start']}  attacked {s['attacked']} flagged {s['flagged']} | {detail}")
    assert size < 600 * 1024, size


if __name__ == "__main__":
    main()
