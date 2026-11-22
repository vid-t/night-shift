"""How much could chance move the gap between two systems on the test set?

    uv run python runs/final/gap_check.py

Made AFTER the test result was read (21 November 2026, 23:58). It changes no result: it takes the answers that
runs/final/final_score.py scored and asks how far the score difference moves when the test set is drawn again.

Method: a paired bootstrap over time windows. The test set holds 256 time windows, each with one row per loop:
56 of them carry an attack (its attack rows and its neighbour rows) and 200 are quiet. One draw takes 56 attack
time windows and 200 quiet ones with replacement, the same ones for both systems, and counts each system's score
(attack rows called attack minus normal rows called attack). 20,000 draws, fixed seed.

The test rows come from four recording files, one per HAI release: too few to draw whole files instead. Windows of
one file are alike, so the true spread may be somewhat wider than this check shows.
Two plainer counts go with it, on the same answers:
- the time windows on which two systems score differently, and a two-sided sign test on them;
- the score of every vote of three that the four models of the simpler design could have formed (the declared vote
  was seeds 0 to 2), to show how much the choice of training repeats moves a vote.
Writes runs/final/gap_check.json and prints it.
"""
import collections
import importlib.util
import itertools
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fs", HERE / "final_score.py")
FS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FS)

DRAWS = 20000
SEED = 20261122


def window_of(example_id: str) -> str:
    return example_id.rsplit("/", 1)[0]            # version/file/timestamp


def per_cluster(lab: dict, cluster) -> dict:
    """{cluster: this system's score on the rows of that cluster}."""
    out = collections.Counter()
    for e, r in FS.TRUTH.items():
        if lab.get(e) == "attack":
            out[cluster(e)] += 1 if r["label"] == "attack" else -1
    return out


def boot(a: dict, b: dict, strata: list, rng) -> dict:
    """Difference a - b over draws; `strata` is a list of cluster-name lists, each drawn on its own."""
    diffs = []
    for _ in range(DRAWS):
        d = 0
        for names in strata:
            for _ in range(len(names)):
                c = names[rng.randrange(len(names))]
                d += a.get(c, 0) - b.get(c, 0)
        diffs.append(d)
    diffs.sort()
    q = lambda p: diffs[min(len(diffs) - 1, int(p * len(diffs)))]
    return {"low_90": q(0.05), "high_90": q(0.95), "low_95": q(0.025), "high_95": q(0.975), "median": q(0.5),
            "share_b_at_least_a": round(sum(x <= 0 for x in diffs) / len(diffs), 3),
            "share_b_above_a": round(sum(x < 0 for x in diffs) / len(diffs), 3)}


def main():
    night = FS.NIGHT
    lab = {r: FS.labels(night / r / "kept" / "test_predictions_plain.jsonl") for r in FS.A_RUNS + FS.B_RUNS}
    base = FS.REPO / "runs/baselines/repaired"
    systems = {
        "gradient boosting": FS.labels(base / "predictions_gradient_boosting.jsonl"),
        "gradient boosting with neighbours": FS.labels(base / "predictions_gradient_boosting_with_neighbours.jsonl"),
        "vote of three (simpler design)": FS.vote_labels([lab[r] for r in FS.A_RUNS[:3]], 2),
        "vote of seven (hint-heavy design)": FS.vote_labels([lab[r] for r in FS.B_RUNS[:7]], 4),
    }
    ids = list(FS.TRUTH)
    kind_of_window = {}
    for e in ids:
        w = window_of(e)
        kind_of_window[w] = "attack" if FS.TRUTH[e]["window_kind"] == "attack" else kind_of_window.get(w, "quiet")
    attack_w = sorted(w for w, k in kind_of_window.items() if k == "attack")
    quiet_w = sorted(w for w, k in kind_of_window.items() if k == "quiet")
    files = sorted({"/".join(e.split("/")[:2]) for e in ids})
    out = {"made": "after the test result was read; changes no result", "draws": DRAWS, "seed": SEED,
           "time_windows": {"attack": len(attack_w), "quiet": len(quiet_w)}, "recording_files": len(files), "scores": {}, "pairs": {}}
    for name, l in systems.items():
        out["scores"][name] = sum(per_cluster(l, window_of).values())
    pairs = [("gradient boosting", "vote of three (simpler design)"), ("gradient boosting", "vote of seven (hint-heavy design)"),
             ("gradient boosting with neighbours", "vote of three (simpler design)"),
             ("vote of three (simpler design)", "vote of seven (hint-heavy design)")]
    for a, b in pairs:
        rng = random.Random(SEED)
        by_window = boot(per_cluster(systems[a], window_of), per_cluster(systems[b], window_of), [attack_w, quiet_w], rng)
        out["pairs"][f"{a} minus {b}"] = {"observed": out["scores"][a] - out["scores"][b], **by_window}
    out["disagree"] = {}
    for a, b in pairs:
        pa, pb = per_cluster(systems[a], window_of), per_cluster(systems[b], window_of)
        d = [pa.get(w, 0) - pb.get(w, 0) for w in attack_w + quiet_w]
        up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
        n, k = up + down, min(up, down)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        out["disagree"][f"{a} against {b}"] = {"time_windows": len(d), "first_better": up, "second_better": down,
                                               "same": len(d) - n, "sign_test_p": round(p, 2)}
    out["votes_of_three_simpler_design"] = {
        "+".join(r for r in tri): sum(per_cluster(FS.vote_labels([lab[r] for r in tri], 2), window_of).values())
        for tri in itertools.combinations(FS.A_RUNS, 3)}
    (HERE / "gap_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
