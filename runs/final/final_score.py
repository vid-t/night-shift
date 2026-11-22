"""The final scoring, once: our two designs, the baselines and the big model on the test rows.

    uv run python runs/final/final_score.py

Everything is counted from the raw answer files with training/score.py. Writes runs/final/final_results.json and
runs/final/FINAL-RESULTS.md. Designs, votes, the sample and the wording were fixed in runs/final/DECLARED.md before
any test result was read.
"""
import collections
import importlib.util
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NIGHT = REPO / "runs/nebius/2026-11-21/night"
OUT = REPO / "runs/final"
LC = REPO / "runs/llm_compare"
spec = importlib.util.spec_from_file_location("s", REPO / "training/score.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
spec2 = importlib.util.spec_from_file_location("rt", REPO / "training/results_table.py")
RT = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(RT)

LOOPS = ["P1-FC", "P1-LC", "P1-PC", "P1-TC", "P2-SC", "P3-LC"]
A_RUNS = ["r4-zsize-law", "r5-zsize-law-s1", "r7-zsize-law-s2", "r8-zsize-law-s3"]
B_RUNS = ["r5-looprange-steplaw", "r5-looprange-steplaw-s1", "r6-lr-steplaw-s2", "r6-lr-steplaw-s3",
          "r7-lr-steplaw-s4", "r7-lr-steplaw-s5", "r7-lr-steplaw-s6", "r7-lr-steplaw-s7"]


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


TRUTH_ROWS = read(REPO / "data/processed/opentslm/test.jsonl")
TRUTH = {r["example_id"]: r for r in TRUTH_ROWS}
VAL_ROWS = read(REPO / "data/processed/opentslm/validation.jsonl")
VAL = {r["example_id"]: r for r in VAL_ROWS}


def kind(r):
    return "attack" if r["label"] == "attack" else ("neighbour" if r["window_kind"] == "attack" else "quiet")


def labels(path):
    """{example id: label} from a prediction file; a later line replaces an earlier one."""
    return {r["example_id"]: S.parse(r.get("generated"))[0] for r in read(path)}


def tally(lab, ids, truth=TRUTH):
    c = collections.Counter()
    per = {k: [0, 0] for k in LOOPS}
    for e in ids:
        r, l = truth[e], lab.get(e)
        k = kind(r)
        c["n_" + k] += 1
        c["valid"] += l is not None
        if l == "attack":
            c["said_" + k] += 1
            per[r["loop"]][k != "attack"] += 1
    fa = c["said_neighbour"] + c["said_quiet"]
    return {"rows": len(ids), "attack_rows": c["n_attack"], "caught": c["said_attack"], "neighbour_rows": c["n_neighbour"],
            "fa_neighbour": c["said_neighbour"], "quiet_rows": c["n_quiet"], "fa_quiet": c["said_quiet"],
            "false_alarms": fa, "score": c["said_attack"] - fa, "valid": c["valid"], "loops": per}


def vote_labels(label_sets, need):
    ids = set.intersection(*[set(l) for l in label_sets])
    return {e: ("attack" if sum(l[e] == "attack" for l in label_sets) >= need else "normal") for e in ids}


def found(pred_path, truth_rows=TRUTH_ROWS):
    """'Found X of N' attacks by the project's own table tool."""
    try:
        row = RT.row(S, "x", read(pred_path), truth_rows)
        return row["Found"]
    except Exception as e:   # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_ids = [r["example_id"] for r in TRUTH_ROWS]
    sample = [l.strip() for l in open(LC / "ids_final_sample.txt") if l.strip()]
    think = [l.strip() for l in open(LC / "ids_final_thinking.txt") if l.strip()]
    totals = collections.Counter(kind(r) for r in TRUTH_ROWS)
    s_tot = collections.Counter(kind(TRUTH[e]) for e in sample)
    scale = {"neighbour": totals["neighbour"] / s_tot["neighbour"], "quiet": totals["quiet"] / s_tot["quiet"]}

    systems = {}   # name -> {id: label} on test
    files = {}     # name -> prediction file (for 'found')
    missing = []
    for name, runs in (("A", A_RUNS), ("B", B_RUNS)):
        for run in runs:
            f = NIGHT / run / "kept" / "test_predictions_plain.jsonl"
            if f.exists():
                systems[run] = labels(f)
                files[run] = f
            else:
                missing.append(run)
    base = REPO / "runs/baselines/repaired"
    for name, f in (("Gradient boosting", "predictions_gradient_boosting.jsonl"),
                    ("Gradient boosting with neighbours", "predictions_gradient_boosting_with_neighbours.jsonl"),
                    ("Range alarm", "predictions_range_alarm.jsonl")):
        systems[name] = labels(base / f)
        files[name] = base / f
    # the big model: a cut-off answer is replaced by its "more room" answer
    def kimi(mode):
        strict = labels(LC / f"final_k31_{mode}.jsonl")
        merged = dict(strict)
        more = LC / f"final_k31_{mode}_more_room.jsonl"
        if more.exists():
            for e, l in labels(more).items():
                if merged.get(e) is None:
                    merged[e] = l
        return strict, merged
    k_cold_strict, k_cold = kimi("cold")
    k_he_strict, k_he = kimi("hints_examples")
    k_think = labels(LC / "final_k31_cold_thinking.jsonl")

    have_a = [r for r in A_RUNS if r in systems]
    have_b = [r for r in B_RUNS if r in systems]
    votes = {}
    if len(have_a) == 4:
        votes["A vote (seeds 0 to 2, 2 of 3)"] = vote_labels([systems[r] for r in A_RUNS[:3]], 2)
    if len(have_b) == 8:
        votes["B vote (seeds 0 to 6, 4 of 7)"] = vote_labels([systems[r] for r in B_RUNS[:7]], 4)

    res = {"test_rows": dict(totals), "sample_rows": dict(s_tot), "scale": scale, "missing_runs": missing, "full": {}, "sample": {},
           "thinking": {}, "shuffled": {}, "validation": {}}

    def add(table, name, lab, ids, extra=None):
        t = tally(lab, ids)
        if extra:
            t.update(extra)
        res[table][name] = t
        return t

    # 1. the whole test set: ours and the baselines
    for run in have_a + have_b:
        add("full", run, systems[run], all_ids, {"found": found(files[run])})
    for name, lab in votes.items():
        add("full", name, lab, all_ids)
    for name in ("Gradient boosting", "Gradient boosting with neighbours", "Range alarm"):
        add("full", name, systems[name], all_ids, {"found": found(files[name])})

    # 2. the same 811 windows: every system
    def add_sample(name, lab, extra=None):
        t = add("sample", name, lab, sample, extra)
        t["estimated_full_false_alarms"] = round(t["fa_neighbour"] * scale["neighbour"] + t["fa_quiet"] * scale["quiet"], 1)
        t["estimated_full_score"] = round(t["caught"] - t["estimated_full_false_alarms"], 1)
    for run in have_a + have_b:
        add_sample(run, systems[run])
    for name, lab in votes.items():
        add_sample(name, lab)
    for name in ("Gradient boosting", "Gradient boosting with neighbours", "Range alarm"):
        add_sample(name, systems[name])
    add_sample("Kimi K3.1 Pro, cold", k_cold, {"valid_strict": tally(k_cold_strict, sample)["valid"]})
    add_sample("Kimi K3.1 Pro, hints and solved examples", k_he, {"valid_strict": tally(k_he_strict, sample)["valid"]})

    # 3. the 58 windows of the thinking side run
    for name, lab in (("Kimi K3.1 Pro, cold, thinking off", k_cold), ("Kimi K3.1 Pro, cold, thinking on (low)", k_think),
                      ("Kimi K3.1 Pro, hints and solved examples, thinking off", k_he),
                      ("Gradient boosting", systems["Gradient boosting"])):
        add("thinking", name, lab, think)
    for name, lab in votes.items():
        add("thinking", name, lab, think)

    # 4. our runs with the seconds shuffled (test)
    for run in have_a + have_b:
        f = NIGHT / run / "kept" / "test_predictions_shuffled.jsonl"
        if f.exists():
            add("shuffled", run, labels(f), all_ids)

    # 5. the same runs on the check set, for the drop from check to test
    val_ids = [r["example_id"] for r in VAL_ROWS]
    val_lab = {}
    for run in A_RUNS + B_RUNS:
        f = NIGHT / run / "kept" / "test_predictions_validation.jsonl"
        val_lab[run] = labels(f)
        res["validation"][run] = tally(val_lab[run], val_ids, VAL)
    res["validation"]["A vote (seeds 0 to 2, 2 of 3)"] = tally(vote_labels([val_lab[r] for r in A_RUNS[:3]], 2), val_ids, VAL)
    res["validation"]["B vote (seeds 0 to 6, 4 of 7)"] = tally(vote_labels([val_lab[r] for r in B_RUNS[:7]], 4), val_ids, VAL)
    vb = REPO / "runs/baselines/repaired_validation"
    for name, f in (("Gradient boosting", "predictions_gradient_boosting.jsonl"),
                    ("Gradient boosting with neighbours", "predictions_gradient_boosting_with_neighbours.jsonl"),
                    ("Range alarm", "predictions_range_alarm.jsonl")):
        res["validation"][name] = tally(labels(vb / f), val_ids, VAL)

    # 6. what the big model cost
    cost = {}
    for f in sorted(LC.glob("*.jsonl")):
        rows = read(f)
        ok = [r for r in rows if not r.get("error")]
        cost[f.name] = {"questions": len(rows), "usd": round(sum(r.get("cost_usd", 0) for r in rows), 4),
                        "usd_per_1000": round(1000 * sum(r.get("cost_usd", 0) for r in rows) / max(1, len(rows)), 2),
                        "seconds": round(statistics.mean(r["seconds"] for r in ok), 1) if ok else None,
                        "tokens_in": round(statistics.mean(r["prompt_tokens"] for r in ok)) if ok else None,
                        "tokens_out": round(statistics.mean(r["completion_tokens"] for r in ok)) if ok else None}
    res["big_model_cost"] = cost
    res["big_model_total_usd"] = round(sum(c["usd"] for c in cost.values()), 4)

    (OUT / "final_results.json").write_text(json.dumps(res, indent=1))

    # ---- markdown
    L = []
    def table(title, rows, cols):
        L.append(f"\n## {title}\n")
        L.append("| " + " | ".join(c for c, _ in cols) + " |")
        L.append("|" + "---|" * len(cols))
        for name, t in rows:
            L.append("| " + " | ".join(str(fn(name, t)) for _, fn in cols) + " |")
    def mean_row(names, tab):
        ts = [res[tab][n] for n in names if n in res[tab]]
        if not ts:
            return None
        return {"caught": round(statistics.mean(t["caught"] for t in ts), 2), "false_alarms": round(statistics.mean(t["false_alarms"] for t in ts), 2),
                "score": round(statistics.mean(t["score"] for t in ts), 2), "low": min(t["score"] for t in ts), "high": max(t["score"] for t in ts),
                "fa_neighbour": round(statistics.mean(t["fa_neighbour"] for t in ts), 2), "fa_quiet": round(statistics.mean(t["fa_quiet"] for t in ts), 2),
                "valid": "", "found": "", "rows": ts[0]["rows"], "attack_rows": ts[0]["attack_rows"]}

    L.append("# Final results on the test set\n")
    L.append(f"Test set: {len(all_ids)} windows: {totals['attack']} attack, {totals['neighbour']} neighbour, {totals['quiet']} quiet. "
             f"Score = attack windows called attack minus normal windows called attack. Perfect is {totals['attack']}.")
    if missing:
        L.append(f"\n**Not in yet:** {', '.join(missing)}")
    full_rows = [(n, res["full"][n]) for n in res["full"]]
    for label, runs in (("A mean of the four single runs (hybrid + law)", A_RUNS), ("B mean of the eight single runs (loop-relative + step and law)", B_RUNS)):
        m = mean_row(runs, "full")
        if m:
            full_rows.append((label + f" [{m['low']} to {m['high']}]", m))
    table("The whole test set", full_rows,
          [("System", lambda n, t: n), (f"Attack windows caught (of {totals['attack']})", lambda n, t: t["caught"]),
           (f"False alarms (of {totals['neighbour'] + totals['quiet']})", lambda n, t: t["false_alarms"]), ("Score", lambda n, t: t["score"]),
           ("Attacks found", lambda n, t: t.get("found", ""))])
    samp_rows = [(n, res["sample"][n]) for n in res["sample"]]
    table(f"The same {len(sample)} windows for every system ({s_tot['attack']} attack, {s_tot['neighbour']} neighbour, {s_tot['quiet']} quiet)", samp_rows,
          [("System", lambda n, t: n), (f"Caught (of {s_tot['attack']})", lambda n, t: t["caught"]),
           (f"False alarms, neighbour (of {s_tot['neighbour']})", lambda n, t: t["fa_neighbour"]), (f"False alarms, quiet (of {s_tot['quiet']})", lambda n, t: t["fa_quiet"]),
           ("Score on these windows", lambda n, t: t["score"]), ("Estimated score on the whole set", lambda n, t: t.get("estimated_full_score", "")),
           ("Answers with a verdict", lambda n, t: f"{t['valid']} of {t['rows']}" + (f" ({t['valid_strict']} before the cut-off answers were asked again)" if "valid_strict" in t else ""))])
    table(f"Per loop on those {len(sample)} windows (caught / false alarms)", [(n, res["sample"][n]) for n in res["sample"] if not n.startswith("r")],
          [("System", lambda n, t: n)] + [(k, (lambda k: lambda n, t: f"{t['loops'][k][0]} / {t['loops'][k][1]}")(k)) for k in LOOPS])
    tk = collections.Counter(kind(TRUTH[e]) for e in think)
    table(f"The {len(think)} windows of the side run with thinking on ({tk['attack']} attack, {tk['neighbour'] + tk['quiet']} normal)", [(n, res["thinking"][n]) for n in res["thinking"]],
          [("System", lambda n, t: n), (f"Caught (of {tk['attack']})", lambda n, t: t["caught"]), (f"False alarms (of {tk['neighbour'] + tk['quiet']})", lambda n, t: t["false_alarms"]),
           ("Score", lambda n, t: t["score"]), ("Answers with a verdict", lambda n, t: f"{t['valid']} of {t['rows']}")])
    if res["shuffled"]:
        table("Our runs with the seconds shuffled (test set)", [(n, res["shuffled"][n]) for n in res["shuffled"]],
              [("Run", lambda n, t: n), ("Attack windows called attack", lambda n, t: t["caught"]), ("Normal windows called attack", lambda n, t: t["false_alarms"])])
    vrows = [(n, res["validation"][n]) for n in res["validation"]]
    table("The same systems on the check set (1,092 windows, 144 attack), for the drop from check to test", vrows,
          [("System", lambda n, t: n), ("Caught (of 144)", lambda n, t: t["caught"]), ("False alarms (of 948)", lambda n, t: t["false_alarms"]), ("Score", lambda n, t: t["score"])])
    table("What the big model cost", [(n, c) for n, c in cost.items()],
          [("File", lambda n, t: n), ("Questions", lambda n, t: t["questions"]), ("Spent (USD)", lambda n, t: t["usd"]), ("USD per 1,000 questions", lambda n, t: t["usd_per_1000"]),
           ("Seconds per answer", lambda n, t: t["seconds"]), ("Tokens read", lambda n, t: t["tokens_in"]), ("Tokens written", lambda n, t: t["tokens_out"])])
    L.append(f"\nTotal spent on the big model: {res['big_model_total_usd']} USD.\n")
    (OUT / "FINAL-RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
