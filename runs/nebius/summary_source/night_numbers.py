"""Every number of the night summary, counted from the answer files under runs/nebius/2026-11-21/night (plain validation only).

    uv run python night_numbers.py OUT.json

Per run: attack rows called attack, normal rows called attack, score, per loop. Per design: seeds, mean, lowest,
highest, and the majority vote over its seeds (an even number of seeds leaves out the highest-numbered seed).
No test file is opened.
"""
import collections
import glob
import importlib.util
import itertools
import json
import os
import statistics
import sys

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[3])
NIGHT = f"{REPO}/runs/nebius/2026-11-21/night"
spec = importlib.util.spec_from_file_location("s", f"{REPO}/training/score.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
truth = {json.loads(l)["example_id"]: json.loads(l) for l in open(f"{REPO}/data/processed/opentslm/validation.jsonl")}
LOOPS = ["P1-FC", "P1-LC", "P1-PC", "P1-TC", "P2-SC", "P3-LC"]

VIEW = {"range": "Range view", "zscore": "Z-score view", "looprange": "Loop-relative", "zsize": "Hybrid"}
CLAUSE = {"": "no clause", "step": "step", "law": "law", "step,law": "step and law", "step,law,plant": "step, law and plant",
          "size,step,law": "size, step and law", "law,plant": "law and plant"}


def design_name(s: dict) -> str:
    view, clauses = s.get("series_view"), s.get("clauses") or ""
    name = VIEW[view] + (" + " + CLAUSE[clauses] if clauses else ", no clause" if view in ("looprange", "zsize") else "")
    extra = []
    start = "run" if s.get("start_file") else (s.get("start_checkpoint") or "").split("-")[-2]
    if start != "sleep":
        extra.append("starts from another run" if start == "run" else f"start `{start}`")
    if s.get("answer_order") != "label_last":
        extra.append({"label_first": "verdict first", "label_only": "verdict only"}[s["answer_order"]])
    if s.get("epochs_planned") != 4:
        extra.append(f"{s['epochs_planned']} passes")
    if s.get("attack_share"):
        extra.append(f"attack share {s['attack_share']}")
    if s.get("mirror_share"):
        extra.append(f"mirrored rows {s['mirror_share']}")
    if s.get("lr_encoder"):
        extra.append(f"encoder step {s['lr_encoder']}")
    keep = s.get("choose_pass") or "loss"
    if keep == "loss":
        extra.append("pass chosen by loss")
    return name + (", " + ", ".join(extra) if extra else "")


def labels(run: str) -> dict:
    return {json.loads(l)["example_id"]: S.parse(json.loads(l).get("generated"))[0]
            for l in open(f"{NIGHT}/{run}/kept/test_predictions_validation.jsonl")}


def count(said_attack) -> dict:
    """said_attack: example_id -> bool. Returns attack rows and normal rows called attack, overall and per loop."""
    per = {k: [0, 0] for k in LOOPS}
    for e, r in truth.items():
        if said_attack[e]:
            per[r["loop"]][r["label"] != "attack"] += 1
    a, n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"attack": a, "normal": n, "score": a - n, "loops": per}


def vote(label_sets: list, k: int) -> dict:
    return count({e: sum(l[e] == "attack" for l in label_sets) >= k for e in truth})


def main() -> None:
    runs = {}
    for d in sorted(glob.glob(f"{NIGHT}/r[1-9]-*/kept")):
        run = d.split("/")[-2]
        if not (os.path.exists(f"{d}/test_predictions_validation.jsonl") and os.path.exists(f"{d}/run_summary.json")):
            continue
        s = json.load(open(f"{d}/run_summary.json"))
        lab = labels(run)
        c = count({e: lab[e] == "attack" for e in truth})
        c.update(run=run, round=int(run[1]), seed=s.get("seed"), design=design_name(s), kept_pass=s.get("kept_pass"),
                 rows=len(lab), invalid=sum(v is None for v in lab.values()),
                 labels=lab)
        runs[run] = c
    groups = collections.defaultdict(list)
    for r in runs.values():
        groups[r["design"]].append(r)
    designs = []
    for name, rs in groups.items():
        rs.sort(key=lambda r: (r["seed"], r["run"]))
        sc = [r["score"] for r in rs]
        d = {"design": name, "runs": [r["run"] for r in rs], "seeds": [r["seed"] for r in rs], "scores": sc,
             "mean": statistics.mean(sc), "low": min(sc), "high": max(sc),
             "mean_attack": statistics.mean(r["attack"] for r in rs), "mean_normal": statistics.mean(r["normal"] for r in rs),
             "loops_mean": {k: [statistics.mean(r["loops"][k][0] for r in rs), statistics.mean(r["loops"][k][1] for r in rs)] for k in LOOPS}}
        distinct = len(set(d["seeds"])) == len(rs)
        if len(rs) >= 3 and distinct:
            members = rs if len(rs) % 2 else rs[:-1]   # an even number: leave out the highest-numbered seed
            v = vote([r["labels"] for r in members], len(members) // 2 + 1)
            d["vote"] = {"runs": [r["run"] for r in members], "need": len(members) // 2 + 1, **v}
            if len(rs) >= 5:
                d["vote_sizes"] = {}
                for m in (3, 5, 7):
                    if m <= len(rs):
                        s2 = [vote([r["labels"] for r in c], m // 2 + 1)["score"] for c in itertools.combinations(rs, m)]
                        d["vote_sizes"][m] = {"mean": statistics.mean(s2), "low": min(s2), "high": max(s2), "choices": len(s2)}
        designs.append(d)
    designs.sort(key=lambda d: -d["mean"])
    for r in runs.values():
        del r["labels"]
    out = {"runs": runs, "designs": designs, "n_runs": len(runs)}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(out["n_runs"], "runs")
    for d in designs:
        v = d.get("vote")
        print(f"{d['mean']:6.2f} {d['low']:>3}..{d['high']:<3} n={len(d['runs'])} {d['design']} | {d['scores']}"
              + (f" | vote of {len(v['runs'])}: {v['attack']}/{v['normal']} = {v['score']}" if v else ""))


if __name__ == "__main__":
    main()
