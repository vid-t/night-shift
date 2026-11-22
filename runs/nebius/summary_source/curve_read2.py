"""Read a curve check: answers with unchanged text and changed curves, against the machine's plain answers of the same run."""
import json, collections, importlib.util, sys
run, folder = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("s", "training/score.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
gpu = {json.loads(l)["example_id"]: S.parse(json.loads(l).get("generated"))[0] for l in open(f"runs/nebius/2026-11-21/night/{run}/kept/test_predictions_validation.jsonl")}
LOOPS = ["P1-FC", "P1-LC", "P1-PC", "P1-TC", "P2-SC", "P3-LC"]
out = {"run": run}
for v in ("flat", "curveshuffle"):
    try: R = {json.loads(l)["example_id"]: json.loads(l) for l in open(f"{folder}/{v}.jsonl") if l.strip()}
    except FileNotFoundError: continue
    res = {"rows": len(R)}
    for kind, sel in (("attack", lambda r: r["label"] == "attack"), ("neighbour", lambda r: r["label"] == "normal")):
        rows = [r for r in R.values() if sel(r)]
        per = {k: [0, 0, 0] for k in LOOPS}
        for r in rows:
            per[r["loop"]][0] += 1; per[r["loop"]][1] += gpu[r["example_id"]] == "attack"; per[r["loop"]][2] += r["said"] == "attack"
        res[kind] = {"rows": len(rows), "plain": sum(x[1] for x in per.values()), "changed": sum(x[2] for x in per.values()),
                     "same": sum(gpu[r["example_id"]] == r["said"] for r in rows), "loops": per}
    out[v] = res
    print(f"== {run} {v}: {res['rows']} rows | attack rows {res['attack']['rows']}: plain {res['attack']['plain']} -> {res['attack']['changed']} (same verdict {res['attack']['same']}) | neighbour rows {res['neighbour']['rows']}: plain {res['neighbour']['plain']} -> {res['neighbour']['changed']}")
    print("   attack per loop (rows, plain, changed): " + "; ".join(f"{k} {x[0]}, {x[1]}, {x[2]}" for k, x in res["attack"]["loops"].items()))
    print("   neighbour per loop: " + "; ".join(f"{k} {x[0]}, {x[1]}, {x[2]}" for k, x in res["neighbour"]["loops"].items()))
json.dump(out, open(sys.argv[3], "w"), indent=1) if len(sys.argv) > 3 else None
