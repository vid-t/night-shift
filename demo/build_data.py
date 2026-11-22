"""Build demo/data/demo.json for the static demo page.

Run from the repo root:  uv run python demo/build_data.py

Reads only check-set (validation) files, the night's answers on them, the baseline answers,
the curve check and the night's run table. It never opens a test file.
"""
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "data" / "demo.json"
NIGHT = ROOT / "runs" / "nebius" / "2026-11-21" / "night"
VALIDATION = ROOT / "data" / "processed" / "opentslm" / "validation.jsonl"
BASELINES = ROOT / "runs" / "baselines" / "repaired_validation"
CURVECHECK = NIGHT / "analysis" / "curvecheck" / "out"
NUMBERS = ROOT / "runs" / "nebius" / "summary_source" / "night_numbers.json"

VOTERS = [
    "r5-looprange-steplaw", "r5-looprange-steplaw-s1", "r6-lr-steplaw-s2", "r6-lr-steplaw-s3",
    "r7-lr-steplaw-s4", "r7-lr-steplaw-s5", "r7-lr-steplaw-s6",
]
CURVE_RUN = "r5-looprange-steplaw"
VARIANTS = ["plain", "flat", "curveshuffle"]
# Number of attacks in the check set, quoted from runs/nebius/NIGHT-SUMMARY.md line 5.
ATTACKS_QUOTED = 15
PLANTS = {"P1": "boiler", "P2": "turbine", "P3": "water treatment"}


def load_parse():
    spec = importlib.util.spec_from_file_location("hai_score", ROOT / "training" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


parse = load_parse()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def permutation(example_id, n=300):
    """The seconds order the curve check used for its shuffled variant (same rule as curve_check.py)."""
    import numpy as np
    seed = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:16], 16)
    return [int(i) for i in np.random.default_rng(seed).permutation(n)]


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2)


def kind_of(row):
    if row["label"] == "attack":
        return "attack"
    return "neighbour" if row["window_kind"] == "attack" else "normal"


def round_sig(x, sig=5):
    if x == 0:
        return 0
    return float(f"{x:.{sig}g}")


def main():
    rows = read_jsonl(VALIDATION)
    by_id = {r["example_id"]: r for r in rows}
    assert len(rows) == 1092, len(rows)

    preds = {rn: {p["example_id"]: p for p in read_jsonl(NIGHT / rn / "kept" / "test_predictions_validation.jsonl")}
             for rn in VOTERS}
    gb = {p["example_id"]: p["generated"] for p in read_jsonl(BASELINES / "predictions_gradient_boosting.jsonl")}
    gbn = {p["example_id"]: p["generated"] for p in
           read_jsonl(BASELINES / "predictions_gradient_boosting_with_neighbours.jsonl")}
    cc = defaultdict(dict)
    for v in VARIANTS:
        for d in read_jsonl(CURVECHECK / f"{v}.jsonl"):
            assert d["variant"] == v
            cc[d["example_id"]][v] = d

    # ---- per-row facts ------------------------------------------------------------------
    facts = {}
    for r in rows:
        e = r["example_id"]
        answers = [parse(preds[rn][e]["generated"]) for rn in VOTERS]
        votes = sum(1 for lab, _ in answers if lab == "attack")
        facts[e] = {
            "kind": kind_of(r),
            "votes": votes,
            "vote": "attack" if votes >= 4 else "normal",
            "gb": parse(gb[e])[0],
            "gbn": parse(gbn[e])[0],
            "cc": {v: parse(cc[e][v]["generated"])[0] for v in VARIANTS} if e in cc else None,
            "start": median([st for lab, st in answers if lab == "attack" and st is not None]),
        }

    # ---- headline counts, recounted from the raw answers ---------------------------------
    def count(pick):
        att = sum(1 for e, f in facts.items() if f["kind"] == "attack" and pick(e) == "attack")
        nor = sum(1 for e, f in facts.items() if f["kind"] != "attack" and pick(e) == "attack")
        return {"caught": att, "false_alarms": nor, "score": att - nor}

    totals = {
        "rows": len(rows),
        "attack_rows": sum(1 for f in facts.values() if f["kind"] == "attack"),
        "normal_rows": sum(1 for f in facts.values() if f["kind"] != "attack"),
        "neighbour_rows": sum(1 for f in facts.values() if f["kind"] == "neighbour"),
        "quiet_rows": sum(1 for f in facts.values() if f["kind"] == "normal"),
        # Quoted, not counted: the rows carry no attack id, and distinct window times give 90, not 15.
        # Source: runs/nebius/NIGHT-SUMMARY.md, "144 attack rows from 15 attacks".
        "attacks": ATTACKS_QUOTED,
        "attack_window_times": len({e.rsplit("/", 1)[0] for e, f in facts.items() if f["kind"] == "attack"}),
        "no_visible_change_rows": sum(1 for r in rows if r["label"] == "attack" and r["no_visible_change"]),
    }
    vote = count(lambda e: facts[e]["vote"])
    gb_counts = count(lambda e: facts[e]["gb"])
    gbn_counts = count(lambda e: facts[e]["gbn"])
    singles = {rn: count(lambda e, rn=rn: parse(preds[rn][e]["generated"])[0]) for rn in VOTERS}

    # ---- curve check totals ---------------------------------------------------------------
    machine = {e: parse(preds[CURVE_RUN][e]["generated"])[0] for e in cc}
    curve_totals = {"machine": {
        "attack_called": sum(1 for e in cc if facts[e]["kind"] == "attack" and machine[e] == "attack"),
        "neighbour_called": sum(1 for e in cc if facts[e]["kind"] != "attack" and machine[e] == "attack"),
    }}
    rerun_differs = sum(1 for e in cc if machine[e] != facts[e]["cc"]["plain"])
    for v in VARIANTS:
        att = sum(1 for e, f in facts.items() if f["cc"] and f["kind"] == "attack" and f["cc"][v] == "attack")
        nei = sum(1 for e, f in facts.items() if f["cc"] and f["kind"] != "attack" and f["cc"][v] == "attack")
        curve_totals[v] = {"attack_called": att, "neighbour_called": nei}
    curve_rows = {
        "rerun_differs": rerun_differs,
        "attack": sum(1 for f in facts.values() if f["cc"] and f["kind"] == "attack"),
        "neighbour": sum(1 for f in facts.values() if f["cc"] and f["kind"] != "attack"),
    }

    # ---- choose the example windows -------------------------------------------------------
    chosen, reasons = [], {}

    def take(cands, key, n, why, used_loops=None):
        cands = sorted(cands, key=lambda e: (key(e), e))
        got = 0
        for e in cands:
            if got == n:
                break
            if e in reasons:
                continue
            if used_loops is not None and by_id[e]["loop"] in used_loops:
                continue
            chosen.append(e)
            reasons[e] = why
            got += 1
            if used_loops is not None:
                used_loops.add(by_id[e]["loop"])

    def mid(e):  # prefer a start that sits well inside the window
        return abs(by_id[e]["attack_start_s"] - 120)

    def flips(e):
        c = facts[e]["cc"]
        return c is not None and len(set(c.values())) > 1

    ids = list(facts)
    # caught attacks: one per loop, preferring windows where the curve check changes the answer
    loops = set()
    caught = [e for e in ids if facts[e]["kind"] == "attack" and facts[e]["vote"] == "attack"]
    def timed(e):
        st = facts[e]["start"]
        return st is not None and abs(st - by_id[e]["attack_start_s"]) <= 30

    take(caught, lambda e: (not flips(e), not timed(e), facts[e]["votes"] < 7, mid(e)), 5, "caught", loops)
    # missed attacks
    missed = [e for e in ids if facts[e]["kind"] == "attack" and facts[e]["vote"] == "normal"]
    take([e for e in missed if by_id[e]["loop"] == "P3-LC" and facts[e]["gb"] == "attack"],
         lambda e: (facts[e]["votes"] > 0, mid(e)), 1, "missed")
    take([e for e in missed if by_id[e]["loop"] == "P1-TC"], lambda e: (not by_id[e]["no_visible_change"], mid(e)),
         1, "missed")
    take([e for e in missed if by_id[e]["loop"] not in ("P3-LC", "P1-TC")],
         lambda e: (-facts[e]["votes"], mid(e)), 1, "missed")
    # false alarms of the vote
    fa = [e for e in ids if facts[e]["kind"] != "attack" and facts[e]["vote"] == "attack"]
    take([e for e in fa if facts[e]["kind"] == "neighbour"], lambda e: (not flips(e), -facts[e]["votes"]), 1,
         "false_alarm")
    take([e for e in fa if facts[e]["kind"] == "normal"], lambda e: (-facts[e]["votes"],), 1, "false_alarm")
    if sum(1 for e in chosen if reasons[e] == "false_alarm") < 2:
        take(fa, lambda e: (-facts[e]["votes"],), 2 - sum(1 for e in chosen if reasons[e] == "false_alarm"),
             "false_alarm")
    # quiet normal windows, three loops, one where gradient boosting raised a false alarm
    quiet = [e for e in ids if facts[e]["kind"] == "normal" and facts[e]["votes"] == 0]
    qloops = set()
    take([e for e in quiet if facts[e]["gb"] == "attack"], lambda e: (0,), 1, "quiet", qloops)
    take(quiet, lambda e: (by_id[e]["loop"] not in ("P1-LC", "P2-SC", "P3-LC"),), 2, "quiet", qloops)
    # neighbour windows that stay normal: one marked reacting, one whose answer the curve check changes
    nb = [e for e in ids if facts[e]["kind"] == "neighbour" and facts[e]["vote"] == "normal"]
    take([e for e in nb if by_id[e]["reacting"]], lambda e: (facts[e]["votes"] != 0, mid(e)), 1, "neighbour")
    take([e for e in nb if flips(e)], lambda e: (mid(e),), 2, "neighbour")

    hero = next(e for e in chosen if reasons[e] == "caught" and by_id[e]["loop"] == "P1-LC")

    windows = []
    for e in chosen:
        r, f = by_id[e], facts[e]
        p0 = preds[VOTERS[0]][e]
        series = []
        for s in r["series"]:
            vals = [round_sig(x) for x in s["values"]]
            series.append({"role": s["role"], "name": s["name"], "values": vals})
        voters = []
        for rn in VOTERS:
            lab, start = parse(preds[rn][e]["generated"])
            voters.append({"run": rn, "label": lab, "start": start, "text": preds[rn][e]["generated"]})
        # the sentence shown for the vote: a voter on the winning side, nearest the median start
        side = [v for v in voters if v["label"] == f["vote"]]
        vote_text = min(side, key=lambda v: abs((v["start"] or 0) - (f["start"] or 0))) if f["vote"] == "attack" else side[0]
        curve = None
        if f["cc"]:
            lab, start = parse(preds[CURVE_RUN][e]["generated"])
            curve = {"machine": {"label": lab, "start": start, "text": preds[CURVE_RUN][e]["generated"]}}
            curve["order"] = permutation(e, len(r["series"][0]["values"]))
            for v in VARIANTS:
                lab, start = parse(cc[e][v]["generated"])
                curve[v] = {"label": lab, "start": start, "text": cc[e][v]["generated"]}
        windows.append({
            "id": e,
            "group": reasons[e],
            "loop": r["loop"],
            "loop_name": r["loop_name"],
            "plant": PLANTS[r["loop"][:2]],
            "version": r["version"],
            "kind": f["kind"],
            "label": r["label"],
            "attack_start": r["attack_start_s"] if f["kind"] == "attack" and r["attack_start_s"] >= 0 else None,
            "reacting": r["reacting"],
            "no_visible_change": r["no_visible_change"],
            "reference_answer": r["answer"],
            "series": series,
            "prompt": p0["pre_prompt"],
            "hints": p0["time_series_text"],
            "voters": voters,
            "votes_attack": f["votes"],
            "vote": f["vote"],
            "vote_example": {"run": vote_text["run"], "start": vote_text["start"], "text": vote_text["text"]},
            "vote_start": f["start"] if f["vote"] == "attack" else None,
            "gb": f["gb"],
            "gb_neighbours": f["gbn"],
            "curve_check": curve,
        })

    # ---- the night ------------------------------------------------------------------------
    nn = json.loads(NUMBERS.read_text())

    def family(design):
        if design.startswith("Range view"):
            return "range"
        if design.startswith("Z-score view"):
            return "zscore"
        if "no clause" in design:
            return "nohint"
        return "hints"

    runs = []
    for name, r in nn["runs"].items():
        runs.append({
            "run": name, "round": r["round"], "seed": r["seed"], "design": r["design"],
            "family": family(r["design"]), "caught": r["attack"], "false_alarms": r["normal"], "score": r["score"],
            "voter": name in VOTERS,
        })
    runs.sort(key=lambda x: (x["round"], x["run"]))
    assert len(runs) == 64, len(runs)
    for rn in VOTERS:  # the run table and the raw answers must agree
        rec = next(x for x in runs if x["run"] == rn)
        assert rec["score"] == singles[rn]["score"], (rn, rec["score"], singles[rn])

    designs = []
    for d in nn["designs"]:
        designs.append({k: d[k] for k in ("design", "runs", "scores", "mean", "low", "high")} |
                       {"family": family(d["design"]),
                        "vote": ({k: d["vote"][k] for k in ("need", "attack", "normal", "score")} | {
                            "n": len(d["vote"]["runs"])}) if d.get("vote") else None,
                        "vote_sizes": d.get("vote_sizes")})
    voting_design = next(d for d in designs if d["design"] == "Loop-relative + step and law")
    assert voting_design["vote"]["score"] == vote["score"], (voting_design["vote"], vote)

    start_runs = [x["score"] for x in runs if x["design"] in ("Range view", "Range view, pass chosen by loss")]
    leaders = sorted((d for d in designs if len(d["scores"]) >= 3), key=lambda d: -d["mean"])[:6]
    hint_runs = [x["score"] for x in runs if x["family"] == "hints" and x["design"].split(",")[0].split(" +")[0]
                 in ("Hybrid", "Loop-relative")]

    score = {
        "perfect": totals["attack_rows"],
        "start_of_night": {"low": min(start_runs), "high": max(start_runs), "n": len(start_runs)},
        "leading_designs": {"low_mean": min(d["mean"] for d in leaders), "high_mean": max(d["mean"] for d in leaders),
                            "names": [d["design"] for d in leaders]},
        "voting_design_single": {"low": voting_design["low"], "high": voting_design["high"],
                                 "mean": voting_design["mean"], "scores": voting_design["scores"],
                                 "n": len(voting_design["scores"])},
        "vote": vote | {"n": len(VOTERS)},
        "gradient_boosting": gb_counts,
        "gradient_boosting_neighbours": gbn_counts,
        "vote_sizes": voting_design["vote_sizes"],
        "hint_runs": {"n": len(hint_runs), "mean": round(sum(hint_runs) / len(hint_runs), 1),
                      "low": min(hint_runs), "high": max(hint_runs)},
    }

    data = {
        "source": "HAI dataset (github.com/icsdataset/hai), CC BY-SA 4.0; check-set windows only",
        "totals": totals,
        "score": score,
        "curve_check": {"run": CURVE_RUN, "rows": curve_rows, "totals": curve_totals},
        "hero": hero,
        "windows": windows,
        "night": {"runs": runs, "designs": designs, "rounds": max(x["round"] for x in runs)},
        "counts_by_group": dict(Counter(reasons.values())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    extra = OUT.parent / "results_extra.json"
    if not extra.exists():
        print("note: data/results_extra.json is missing; run: uv run python demo/build_results.py")
    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(ROOT)}: {size / 1024:.0f} KB, {len(windows)} windows {data['counts_by_group']}")
    print(f"vote {vote}  gb {gb_counts}  gb+neighbours {gbn_counts}")
    print(f"curve check {curve_rows} {curve_totals}")
    print(f"start of night {score['start_of_night']}  leaders {score['leading_designs']['low_mean']}-"
          f"{score['leading_designs']['high_mean']}  voting design single {voting_design['low']}-{voting_design['high']}")
    assert size < 2 * 1024 * 1024


if __name__ == "__main__":
    main()
