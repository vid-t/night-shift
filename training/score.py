"""Score OpenTSLM test predictions on the HAI attack examples, per attack. Standard library only.

    python score.py --run plain    results/.../stage6_hai/results/test_predictions.jsonl data/processed/opentslm/test.jsonl \
                    --run shuffled path/to/shuffled/test_predictions.jsonl data/processed/opentslm/test_shuffled.jsonl

Predictions file: one JSON object per line, as `_evaluate_stage` in OpenTSLM's curriculum_learning.py
writes it: `pre_prompt`, `time_series_text`, `post_prompt`, `generated`, `gold`, plus `example_id`
once add_stage.py has patched the file. Without `example_id`, predictions are joined to the test
file by order, and every `gold` must match that row's answer.

An answer is valid when it holds `Answer: attack` or `Answer: normal`. The first one counts.
An empty or unparseable answer is wrong: it never counts as normal nor as attack.

Per version and overall, on the scored window of each attack (one per attack):
  attacks           attacks with at least one attacked loop among the six
  found             an attacked loop answers attack
  right_loop_only   every attacked loop answers attack and every other loop answers normal; a
                    `reacting` loop (normal, but its series react to an attack on another loop of the
                    same process) counts neither for nor against it
  right_loop_strict the same without that allowance: a reacting loop must answer normal too. This
                    is the rule as it was before reacting loops were marked
  timed             the earliest start second stated by the attacked loops that answer attack is
                    within 30 s of the attack's start (the earliest true start among its attacked
                    loops). "Already under way" reads as second 0; a start before the window counts as 0.
                    In the per-loop lines, `timed` compares each loop with its own start.
On loop examples from normal windows:
  false_alarms_per_100   answers that are not a valid `normal`, per 100 examples
On all examples:
  valid_share       share of valid answers
Attacked loop examples in the scored windows, on separate lines for those with a visible change
and those marked no_visible_change: how many answer attack, how many are timed.
Reacting loop examples in the scored windows: how many there are, how many answer attack
(`reacting`, `reacting_said_attack`). A test file without the `reacting` field has none.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

ANSWER = re.compile(r"Answer:\s*(attack|normal)\b", re.IGNORECASE)
START = re.compile(r"Change starts near second\s+(\d+)", re.IGNORECASE)
UNDER_WAY = re.compile(r"already under way", re.IGNORECASE)
TIME_TOLERANCE_S = 30


def parse(text) -> tuple[str | None, int | None]:
    """(label or None, start second or None) from a generated answer."""
    if not isinstance(text, str):
        return None, None
    m = ANSWER.search(text)
    if not m:
        return None, None
    label = m.group(1).lower()
    head = text[: m.end()]
    s = START.search(head)
    if s:
        return label, int(s.group(1))
    if UNDER_WAY.search(head):
        return label, 0
    return label, None


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def join(preds: list[dict], examples: list[dict], only_predicted: bool = False) -> list[tuple[dict, str | None]]:
    """Pair each example with its generated text (None when it has no prediction)."""
    if preds and all("example_id" in p for p in preds):
        by_id = {p["example_id"]: p.get("generated") for p in preds}
        known = {e["example_id"] for e in examples}
        stray = [i for i in by_id if i not in known]
        if stray:
            raise ValueError(f"{len(stray)} predictions are for examples not in the test file, e.g. {stray[0]}")
        if only_predicted:
            examples = [e for e in examples if e["example_id"] in by_id]
        return [(e, by_id.get(e["example_id"])) for e in examples]
    if len(preds) != len(examples):
        raise ValueError(f"no example_id in the predictions and {len(preds)} predictions for {len(examples)} examples")
    for i, (p, e) in enumerate(zip(preds, examples)):
        if not str(p.get("gold", "")).strip().startswith(e["answer"].strip()):
            raise ValueError(f"prediction {i}: gold does not match the answer of {e['example_id']}; wrong test file?")
    return [(e, p.get("generated")) for p, e in zip(preds, examples)]


def _timed(e: dict, start: int | None) -> bool:
    return start is not None and abs(start - max(0, int(e["attack_start_s"]))) <= TIME_TOLERANCE_S


def score(pairs: list[tuple[dict, str | None]]) -> dict:
    """Returns {"versions": [row per version + "all"], "loops": [row per version x visibility]}."""
    groups = defaultdict(list)
    for e, text in pairs:
        label, start = parse(text)
        r = {**e, "said": label, "start": start}
        groups[e["version"]].append(r)
        groups["all"].append(r)
    versions, loops = [], []
    for version in sorted(k for k in groups if k != "all") + ["all"]:
        rows = groups[version]
        windows = defaultdict(list)
        for r in rows:
            if r["scored"] and r["window_kind"] == "attack":
                windows[r["example_id"].rsplit("/", 1)[0]].append(r)
        n = found = right = strict = timed = reacting = reacting_attack = 0
        truths = []
        vis = defaultdict(lambda: {"examples": 0, "answered_attack": 0, "timed": 0})
        for w in windows.values():
            hit = [r for r in w if r["label"] == "attack"]
            if not hit:
                continue
            n += 1
            found += any(r["said"] == "attack" for r in hit)
            others = [r for r in w if r["label"] == "normal" and not r.get("reacting", False)]
            right += all(r["said"] == "attack" for r in hit) and all(r["said"] == "normal" for r in others)
            strict += all(r["said"] == "attack" for r in hit) and all(r["said"] == "normal" for r in w if r["label"] == "normal")
            react = [r for r in w if r["label"] == "normal" and r.get("reacting", False)]
            reacting += len(react)
            reacting_attack += sum(r["said"] == "attack" for r in react)
            truth = min(max(0, int(r["attack_start_s"])) for r in hit)
            truths.append(truth)
            said = [r["start"] for r in hit if r["said"] == "attack" and r["start"] is not None]
            timed += bool(said) and abs(min(said) - truth) <= TIME_TOLERANCE_S
            for r in hit:
                v = vis["no_visible_change" if r["no_visible_change"] else "visible"]
                v["examples"] += 1
                v["answered_attack"] += r["said"] == "attack"
                v["timed"] += r["said"] == "attack" and _timed(r, r["start"])
        normal = [r for r in rows if r["window_kind"] == "normal"]
        alarms = sum(r["said"] != "normal" for r in normal)
        versions.append({
            "version": version, "attacks": n, "found": found, "right_loop_only": right,
            "right_loop_strict": strict, "timed": timed,
            "reacting": reacting, "reacting_said_attack": reacting_attack,
            "normal_examples": len(normal),
            "false_alarms_per_100": round(100 * alarms / len(normal), 1) if normal else None,
            "examples": len(rows), "valid_share": round(sum(r["said"] is not None for r in rows) / len(rows), 3) if rows else None,
            "label_right_share": round(sum(r["said"] == r["label"] for r in rows) / len(rows), 3) if rows else None,
        })
        for kind in ("visible", "no_visible_change"):
            loops.append({"version": version, "kind": kind, **vis[kind]})
    return {"versions": versions, "loops": loops, "constant_guess": constant_guess(truths)}


def constant_guess(truths: list[int]) -> dict:
    """How many attacks a model would time right by always stating the same second (over all versions).

    The `timed` count of a real model means little unless it beats the best constant."""
    def hits(s):
        return sum(abs(s - t) <= TIME_TOLERANCE_S for t in truths)
    best = max(range(0, 301), key=hits) if truths else None
    return {"attacks": len(truths), "timed_at_120": hits(120), "best_second": best,
            "timed_at_best": hits(best) if truths else 0}


def _table(rows: list[dict], cols: list[str]) -> str:
    cells = [[str(c) for c in cols]] + [["" if r[c] is None else str(r[c]) for c in cols] for r in rows]
    width = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    return "\n".join("  ".join(v.rjust(width[i]) for i, v in enumerate(row)) for row in cells)


def report(name: str, result: dict) -> str:
    v_cols = ["version", "attacks", "found", "right_loop_only", "right_loop_strict", "timed", "reacting", "reacting_said_attack",
              "normal_examples", "false_alarms_per_100", "examples", "valid_share", "label_right_share"]
    l_cols = ["version", "kind", "examples", "answered_attack", "timed"]
    c = result["constant_guess"]
    a = next(v for v in result["versions"] if v["version"] == "all")
    return (f"== {name} ==\nPer attack (scored windows) and false alarms (normal windows):\n"
            f"{_table(result['versions'], v_cols)}\n\nAttacked loop examples in the scored windows:\n"
            f"{_table(result['loops'], l_cols)}\n\n"
            f"Reacting loops (normal, not counted in right_loop_only): {a['reacting']} in the scored windows, "
            f"{a['reacting_said_attack']} answered attack.\n"
            f"Constant guess on timing: always second 120 times {c['timed_at_120']} of {c['attacks']} attacks; "
            f"the best constant (second {c['best_second']}) times {c['timed_at_best']}.\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "PREDICTIONS", "TEST_JSONL"),
                    help="a name, the predictions file and the test JSONL it was made from; repeat for more runs")
    ap.add_argument("--only-predicted", action="store_true",
                    help="score only the examples that have a prediction (the overfit check); otherwise a missing prediction is wrong")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args(argv)
    out = {}
    for name, pred_path, test_path in args.run:
        pairs = join(read_jsonl(pred_path), read_jsonl(test_path), args.only_predicted)
        out[name] = score(pairs)
        sys.stdout.write(report(name, out[name]) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
