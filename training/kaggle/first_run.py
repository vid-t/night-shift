"""Kaggle run 1: the overfit check (step 3 of training/README.md) on one Kaggle GPU.

It answers three questions before the full run:
  1. Does the whole chain work on a GPU: setup, training, answers, scoring?
  2. Does the model learn ROWS training rows it sees again and again: the format, the label, the second?
  3. How long does one row take, so how long will the full run take?

Saves predictions, losses, scores and a summary to /kaggle/working/results.
"""
import glob
import json
import shutil
import sys
import time

sys.path.insert(0, glob.glob("/kaggle/input/**/kaggle_common.py", recursive=True)[0].rsplit("/", 1)[0])
import kaggle_common as k  # noqa: E402

ROWS = 50            # training rows used for training, validation and test alike
EPOCHS = 40          # 1,000 steps at batch 2. The stage's 10 epochs are only 250 steps: too few to learn 50 rows by heart
BATCH_SIZE = 2
# From gpu_probe.py on the T4: float16 is 2.5 times faster than float32 and its losses follow
# float32. sdpa attention gives the same losses with a quarter less memory.
DTYPE = "float16"
ATTENTION = "sdpa"


def main() -> None:
    started = time.time()
    paths = k.prepare()
    result = k.train(paths, rows=ROWS, batch_size=BATCH_SIZE, dtype=DTYPE, attention=ATTENTION, epochs=EPOCHS)

    stage = paths.checkout / k.STAGE
    for name in ("results/test_predictions.jsonl", "results/metrics.json", "checkpoints/loss_history.txt"):
        if (stage / name).exists():
            shutil.copy(stage / name, k.RESULTS / name.split("/")[-1])
    k.run(f"{sys.executable} training/score.py --only-predicted --run overfit "
          f"{k.RESULTS / 'test_predictions.jsonl'} {paths.jsonl / 'train.jsonl'} --json {k.RESULTS / 'scores.json'}",
          cwd=paths.root)

    k.say("\nFirst five answers (model, then truth):")
    with open(k.RESULTS / "test_predictions.jsonl", encoding="utf-8") as f:
        for line in list(f)[:5]:
            row = json.loads(line)
            k.say(f"  model: {row['generated'].strip()!r}")
            k.say(f"  truth: {row['gold'].strip()!r}")

    summary = {"version": paths.version, "llm_id": k.LLM_ID, "gpu": paths.gpu, "setup_seconds": paths.setup_seconds,
               "total_seconds": round(time.time() - started), "run": result,
               "full_run_estimate": k.full_run_estimate(result["seconds_per_row"], k.count_rows(paths.jsonl))}
    (k.RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    k.say("\nSummary:")
    k.say(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
