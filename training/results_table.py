"""One table for several systems on the same test rows: `training/score.py` counts, side by side.

    uv run python training/results_table.py \
        --truth data/processed/opentslm/test.jsonl \
        --run "Range alarm" runs/baselines/repaired/predictions_range_alarm.jsonl \
        --run "Model" runs/kaggle/<day>/<run>/test_predictions_plain.jsonl \
        --shuffled "Model" runs/kaggle/<day>/<run>/test_predictions_shuffled.jsonl \
        --out runs/results.md

Every prediction file is scored against the same truth file, so the flags (`reacting`,
`no_visible_change`) are the same for all. A `--shuffled` file is scored against the shuffled
truth file next to `--truth` (`<name>_shuffled.jsonl`). Prints Markdown and writes it to `--out`.
"""
import argparse
import importlib.util
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_scorer():
    spec = importlib.util.spec_from_file_location("hai_score", HERE / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_seconds(score, preds: list[dict], truth: list[dict]) -> dict:
    """How the stated start seconds of the attacked loops relate to the true ones (scored windows only)."""
    by_id = {r["example_id"]: r for r in truth}
    pairs = []
    for p in preds:
        r = by_id.get(p["example_id"])
        if r and r["label"] == "attack" and r["scored"]:
            label, start = score.parse(p.get("generated"))
            if label == "attack" and start is not None:
                pairs.append((max(0, int(r["attack_start_s"])), start))
    if len(pairs) < 3:
        return {"stated": len(pairs), "median_error_s": None, "correlation": None}
    true, said = [a for a, _ in pairs], [b for _, b in pairs]
    spread = statistics.pstdev(true) * statistics.pstdev(said)
    corr = (statistics.fmean((a - statistics.fmean(true)) * (b - statistics.fmean(said)) for a, b in pairs) / spread
            if spread else None)
    return {"stated": len(pairs), "median_error_s": statistics.median(abs(a - b) for a, b in pairs),
            "correlation": None if corr is None else round(corr, 2)}


def row_counts(score, preds: list[dict], truth: list[dict]) -> dict:
    """Per row, over every row of the file: attack rows called attack, normal rows called attack."""
    said = {p["example_id"]: score.parse(p.get("generated"))[0] for p in preds}
    attack = [r for r in truth if r["label"] == "attack" and r["example_id"] in said]
    normal = [r for r in truth if r["label"] == "normal" and r["example_id"] in said]
    return {"attack_rows": len(attack), "attack_rows_found": sum(said[r["example_id"]] == "attack" for r in attack),
            "normal_rows": len(normal), "normal_rows_alarmed": sum(said[r["example_id"]] == "attack" for r in normal)}


def row(score, name: str, preds: list[dict], truth: list[dict]) -> dict:
    result = score.score(score.join(preds, truth))
    rows = row_counts(score, preds, truth)
    a = next(v for v in result["versions"] if v["version"] == "all")
    per_version = " ".join(f"{v['found']}/{v['attacks']}" for v in result["versions"] if v["version"] != "all")
    timing = start_seconds(score, preds, truth)
    return {"System": name, "Found": f"{a['found']} of {a['attacks']}", "Right loop only": a["right_loop_only"],
            "Strict": a["right_loop_strict"],
            "Attack rows called attack": f"{rows['attack_rows_found']} of {rows['attack_rows']}",
            "Normal rows called attack": f"{rows['normal_rows_alarmed']} of {rows['normal_rows']}",
            "Start within 30 s": a["timed"],
            "Median start error (s)": "" if timing["median_error_s"] is None else f"{timing['median_error_s']:.0f}",
            "Start correlation": "" if timing["correlation"] is None else timing["correlation"],
            "False alarms per 100": a["false_alarms_per_100"],
            "Reacting loops called attack": f"{a['reacting_said_attack']} of {a['reacting']}",
            "Valid answers": f"{a['valid_share']:.0%}", "Found per version": per_version,
            "_constant": result["constant_guess"]}


def markdown(rows: list[dict]) -> str:
    cols = [c for c in rows[0] if not c.startswith("_")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--run", nargs=2, action="append", default=[], metavar=("NAME", "PREDICTIONS"))
    ap.add_argument("--shuffled", nargs=2, action="append", default=[], metavar=("NAME", "PREDICTIONS"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    score = load_scorer()
    truth = score.read_jsonl(args.truth)
    rows = [row(score, name, score.read_jsonl(path), truth) for name, path in args.run]
    if args.shuffled:
        shuffled_truth = score.read_jsonl(str(Path(args.truth).with_name(Path(args.truth).stem + "_shuffled.jsonl")))
        rows += [row(score, f"{name}, seconds shuffled", score.read_jsonl(path), shuffled_truth)
                 for name, path in args.shuffled]
    c = rows[0]["_constant"]
    text = (markdown(rows) + f"\n\nFound per version: 20.07, 21.03, 22.04, 23.05. Always answering second "
            f"{c['best_second']} would time {c['timed_at_best']} of {c['attacks']} attacks; second 120 would time "
            f"{c['timed_at_120']}. `Strict` counts a reacting loop that is called attack as an error.\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
