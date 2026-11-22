"""Build demo/data/results_extra.json: the test-set result and the big-model comparison.

Run from the repo root:

    uv run python demo/build_results.py

It reads only runs/final/final_results.json (every count) and runs/final/results_table_test.md
(attacks found by the votes, start times, the constant-guess comparison). It opens no answer file.
A few cost and speed figures for our own model are not in either file; they are quoted below from
runs/final/FACTS.md and marked as such. The script stops if any count disagrees with FACTS.md.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "runs" / "final" / "final_results.json"
TABLE = ROOT / "runs" / "final" / "results_table_test.md"
OUT = Path(__file__).resolve().parent / "data" / "results_extra.json"

# Run names inside final_results.json.
GB = "Gradient boosting"
GBN = "Gradient boosting with neighbours"
VOTE3 = "A vote (seeds 0 to 2, 2 of 3)"
VOTE7 = "B vote (seeds 0 to 6, 4 of 7)"
FLOOR = "Range alarm"
KIMI_COLD = "Kimi K3.1 Pro, cold"
KIMI_HINTS = "Kimi K3.1 Pro, hints and solved examples"
SIMPLER = ["r4-zsize-law", "r5-zsize-law-s1", "r7-zsize-law-s2", "r8-zsize-law-s3"]
HINT_HEAVY = ["r5-looprange-steplaw", "r5-looprange-steplaw-s1", "r6-lr-steplaw-s2", "r6-lr-steplaw-s3",
              "r7-lr-steplaw-s4", "r7-lr-steplaw-s5", "r7-lr-steplaw-s6", "r7-lr-steplaw-s7"]

# Rows of results_table_test.md.
T_VOTE3 = "Hybrid + law, vote of three"
T_VOTE7 = "Loop-relative + step and law, vote of seven"
T_ONE = "Hybrid + law, one model (seed 1)"
T_ONE_HEAVY = "Loop-relative + step and law, one model (seed 0)"
T_GB = "Gradient boosting"

# Quoted from runs/final/FACTS.md, "Cost and speed" (not in the two source files).
FACTS_OURS = {
    "card_usd_per_hour": 5.40,
    "card_run": "12 models answered 1,536 windows each in 34 minutes",
    "usd_per_1000": {"one": 0.17, "three": 0.50, "seven": 1.16},
    "cpu_seconds_per_window": "about 3",
    "train_one": "about 67 minutes on one eighth of the card, under 1 dollar",
    "night_search": "about 57 dollars of card time for the 64 runs",
    "gb_train": "trains in 5 seconds on a laptop",
}
# Quoted from FACTS.md, to check the counts against.
FACTS_CHECK = {
    GB: (71, 17, 54), GBN: (68, 14, 54), VOTE3: (66, 15, 51), VOTE7: (65, 15, 50), FLOOR: (38, 173, -135),
}
FACTS_SAMPLE = {GB: (71, 8), VOTE3: (66, 7), VOTE7: (65, 8), KIMI_HINTS: (78, 259), KIMI_COLD: (78, 362)}


def counts(run):
    return {k: run[k] for k in ("caught", "false_alarms", "score")}


def parse_table(text):
    lines = [l for l in text.splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = dict(zip(head, cells))
    m = re.search(r"second (\d+) would time (\d+) of (\d+)", text)
    return rows, {"second": int(m.group(1)), "within_30": int(m.group(2)), "of": int(m.group(3))}


def found(text):
    return int(text.split(" of ")[0])


def main():
    F = json.loads(FINAL.read_text())
    table, constant = parse_table(TABLE.read_text())
    full, sample, thinking, cost = F["full"], F["sample"], F["thinking"], F["big_model_cost"]

    for name, want in FACTS_CHECK.items():
        got = (full[name]["caught"], full[name]["false_alarms"], full[name]["score"])
        assert got == want, (name, got, want)
    for name, want in FACTS_SAMPLE.items():
        got = (sample[name]["caught"], sample[name]["false_alarms"])
        assert got == want, (name, got, want)

    tr = F["test_rows"]
    normal = tr["neighbour"] + tr["quiet"]
    attacks = constant["of"]

    def single(names, label, short):
        runs = [full[n] for n in names]
        scores = [r["score"] for r in runs]
        return {
            "name": label, "short": short, "repeats": len(runs), "scores": scores,
            "mean": round(sum(scores) / len(scores), 2),
            "caught": [min(r["caught"] for r in runs), max(r["caught"] for r in runs)],
            "false_alarms": [min(r["false_alarms"] for r in runs), max(r["false_alarms"] for r in runs)],
            "found": [min(found(r["found"]) for r in runs), max(found(r["found"]) for r in runs)],
            "scores_low": min(scores), "scores_high": max(scores),
            "scores_list": ", ".join(str(x) for x in scores),
        }

    simpler = single(SIMPLER, "Our simpler design, one model", "simpler design, one model")
    heavy = single(HINT_HEAVY, "Our feature-heavy design, one model", "feature-heavy design, one model")
    assert simpler["mean"] == 47.25 and heavy["mean"] == 45, (simpler["mean"], heavy["mean"])

    systems = [
        {"key": "gb", "kind": "baseline", "name": "Gradient boosting", "sub": "classic statistical method",
         **counts(full[GB]), "found": found(full[GB]["found"])},
        {"key": "gbn", "kind": "baseline", "name": "Gradient boosting reading the other loops", "sub": "also sees the neighbouring loops' signals",
         **counts(full[GBN]), "found": found(full[GBN]["found"])},
        {"key": "vote3", "kind": "ours", "name": "Our simpler design, three models voting", "sub": "hybrid view + law hint",
         **counts(full[VOTE3]), "found": found(table[T_VOTE3]["Found"])},
        {"key": "vote7", "kind": "ours", "name": "Our feature-heavy design, seven models voting", "sub": "loop view + step and law hints",
         **counts(full[VOTE7]), "found": found(table[T_VOTE7]["Found"])},
    ]
    floor = {"key": "floor", "kind": "floor", "name": "Range alarm", "sub": "the floor: alarm when a signal leaves its usual range",
             **counts(full[FLOOR]), "found": found(full[FLOOR]["found"])}

    def start_row(row, name, key):
        r = table[row]
        return {"key": key, "name": name, "within_30": int(r["Start within 30 s"]),
                "median_error": int(r["Median start error (s)"]) if r["Median start error (s)"] else None}

    start = [
        start_row(T_ONE, "Our simpler design, one model (seed 1)", "one"),
        start_row(T_VOTE3, "Our simpler design, three models voting", "vote3"),
        start_row(T_VOTE7, "Our feature-heavy design, seven models voting", "vote7"),
        start_row(T_ONE_HEAVY, "Our feature-heavy design, one model (seed 0)", "one_heavy"),
    ]
    assert [r["within_30"] for r in start] == [39, 30, 30, 28], start
    gb_start = start_row(T_GB, "Gradient boosting", "gb")
    assert gb_start["within_30"] == 0

    # Test rows answered again with the seconds of every series shuffled (FACTS.md, "The test rows
    # with their seconds shuffled"). The hints are recomputed from the shuffled seconds.
    sh = F["shuffled"]

    def shuffled(names, true_run):
        runs = [sh[n] for n in names]
        return {
            "repeats": len(runs),
            "caught": [min(r["caught"] for r in runs), max(r["caught"] for r in runs)],
            "false_alarms": [min(r["false_alarms"] for r in runs), max(r["false_alarms"] for r in runs)],
            "caught_list": ", ".join(str(r["caught"]) for r in runs),
            "false_alarms_list": ", ".join(str(r["false_alarms"]) for r in runs),
            "true_caught": true_run["caught"], "true_false_alarms": true_run["false_alarms"],
        }

    shuf = {"normal": normal, "simpler": shuffled(SIMPLER, simpler), "heavy": shuffled(HINT_HEAVY, heavy)}
    for key, want in {"simpler": ((25, 33), (4, 12), (61, 68), (13, 23)),
                      "heavy": ((29, 44), (395, 677), (59, 64), (12, 21))}.items():
        x = shuf[key]
        got = (tuple(x["caught"]), tuple(x["false_alarms"]), tuple(x["true_caught"]), tuple(x["true_false_alarms"]))
        assert got == want, (key, got, want)
    assert shuf["normal"] == 1454, shuf["normal"]

    sr = F["sample_rows"]
    sample_normal = sr["neighbour"] + sr["quiet"]
    big_systems = [
        {"key": "gb", "kind": "baseline", "name": "Gradient boosting", **counts(sample[GB])},
        {"key": "vote3", "kind": "ours", "name": "Our simpler design, three models voting", **counts(sample[VOTE3])},
        {"key": "vote7", "kind": "ours", "name": "Our feature-heavy design, seven models voting", **counts(sample[VOTE7])},
        {"key": "kimi_hints", "kind": "big", "name": "Kimi K3.1 Pro with our computed features and six solved examples", **counts(sample[KIMI_HINTS])},
        {"key": "kimi_cold", "kind": "big", "name": "Kimi K3.1 Pro, cold", **counts(sample[KIMI_COLD])},
    ]

    th = thinking
    t_off, t_on = th["Kimi K3.1 Pro, cold, thinking off"], th["Kimi K3.1 Pro, cold, thinking on (low)"]
    think = {
        "rows": t_off["rows"], "attack": t_off["attack_rows"], "normal": t_off["rows"] - t_off["attack_rows"],
        "off": counts(t_off), "on": counts(t_on),
        "gb": counts(th[GB]), "vote3": counts(th[VOTE3]), "vote7": counts(th[VOTE7]),
    }

    c_cold, c_hints, c_think = cost["final_k31_cold.jsonl"], cost["final_k31_hints_examples.jsonl"], cost["final_k31_cold_thinking.jsonl"]
    o = FACTS_OURS["usd_per_1000"]
    cost_rows = [
        {"key": "gb", "kind": "baseline", "name": "Gradient boosting", "usd_per_1000": None,
         "speed": FACTS_OURS["gb_train"], "data": "stays on site"},
        {"key": "one", "kind": "ours", "name": "Our model, one model", "usd_per_1000": o["one"],
         "speed": f"{FACTS_OURS['cpu_seconds_per_window']} s a window on a plain desktop processor, no graphics card", "data": "stays on site"},
        {"key": "vote3", "kind": "ours", "name": "Our model, three voting", "usd_per_1000": o["three"], "speed": None, "data": "stays on site"},
        {"key": "vote7", "kind": "ours", "name": "Our model, seven voting", "usd_per_1000": o["seven"], "speed": None, "data": "stays on site"},
        {"key": "kimi_cold", "kind": "big", "name": "Kimi K3.1 Pro, cold", "usd_per_1000": c_cold["usd_per_1000"],
         "speed": f"{c_cold['seconds']} s an answer", "data": "leaves the site"},
        {"key": "kimi_hints", "kind": "big", "name": "Kimi K3.1 Pro, features and examples", "usd_per_1000": c_hints["usd_per_1000"],
         "speed": f"{c_hints['seconds']} s an answer", "data": "leaves the site"},
        {"key": "kimi_think", "kind": "big", "name": "Kimi K3.1 Pro, thinking on", "usd_per_1000": c_think["usd_per_1000"],
         "speed": f"{c_think['seconds']} s an answer", "data": "leaves the site"},
    ]

    data = {
        "source": "runs/final/final_results.json and runs/final/results_table_test.md, built by demo/build_results.py",
        "test": {
            "rows": {"total": tr["attack"] + normal, "attack": tr["attack"], "normal": normal,
                     "neighbour": tr["neighbour"], "quiet": tr["quiet"], "attacks": attacks},
            "perfect": tr["attack"],
            "systems": systems,
            "single": [simpler, heavy],
            "floor": floor,
            "check": {
                "perfect": F["validation"][VOTE7]["attack_rows"],
                "vote7": F["validation"][VOTE7]["score"], "vote3": F["validation"][VOTE3]["score"],
                "gb": F["validation"][GB]["score"], "gbn": F["validation"][GBN]["score"],
            },
            "start": start, "start_gb": gb_start, "constant": constant,
            "shuffled": shuf,
        },
        "big_model": {
            "rows": {"total": sr["attack"] + sample_normal, "attack": sr["attack"], "normal": sample_normal,
                     "neighbour": sr["neighbour"], "quiet": sr["quiet"]},
            "systems": big_systems,
            "thinking": think,
            "cost": cost_rows,
            "ours": FACTS_OURS,
            "total_usd": round(F["big_model_total_usd"], 2),
        },
    }
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for s in systems + [floor]:
        print(f"  test  {s['name']}: {s['caught']} - {s['false_alarms']} = {s['score']}, found {s['found']} of {attacks}")
    for s in big_systems:
        print(f"  811   {s['name']}: {s['caught']} of {sr['attack']}, {s['false_alarms']} of {sample_normal}")


if __name__ == "__main__":
    main()
