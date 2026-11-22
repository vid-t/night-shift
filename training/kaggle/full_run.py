"""Kaggle run 2: the full fine-tune (steps 5 to 7 of training/README.md) on one Kaggle GPU.

Trains stage6_hai on the 3,880 training rows, answers the 1,536 test rows, answers them again
with the seconds shuffled inside each window, answers the validation rows the same two ways, and
scores all four. While it trains it copies the
best checkpoint and the loss history to /kaggle/working/results every five minutes, so a run
that is cut short still leaves a model behind.

Kept in /kaggle/working/results: best_model.pt, loss_history.txt, the prediction files (plain,
shuffled, validation, validation_shuffled), scores.json, run_summary.json and log.txt.
"""
import glob
import json
import shutil
import sys
import threading
import time

sys.path.insert(0, glob.glob("/kaggle/input/**/kaggle_common.py", recursive=True)[0].rsplit("/", 1)[0])
import kaggle_common as k  # noqa: E402

REHEARSAL_ROWS = 0   # 50 walks through every step on 50 training rows in a few minutes
REHEARSAL_EPOCHS = 3
# The learning rate falls in a straight line to zero over the planned epochs. Planned over the stage's
# 10, the best epoch of the first runs was the third, at 72% of the peak rate, and the five epochs after
# it were thrown away (about 2.4 of 4.4 hours). Four epochs let the rate reach zero.
EPOCHS = 4
# From the test runs on the T4 (2026-11-20). float16 is 2.5 times faster than float32 with the
# same losses, sdpa attention needs a quarter less memory, and at batch 1 a row looks the same
# in training as in testing: no padding between its prompt and its answer.
DTYPE = "float16"
ATTENTION = "sdpa"
BATCH_SIZE = 1
SERIES_VIEW = "zscore"  # or "range": the change from the first seconds in units of the normal range, no numbers in the text
ANSWER_ORDER = "label_last"  # or "label_first": the model writes attack or normal as its first word
# Answered after training, each scored against its own file. The validation answers are for choosing
# between designs, so that the test answers are not used for that.
EXTRA_ANSWERS = ("test_shuffled", "validation", "validation_shuffled")
KEEP_EVERY = 300     # seconds between copies of the checkpoint
# Hugging Face repo of the starting checkpoint; None takes the default of training/setup.sh (stage 4, sleep).
# The end-of-curriculum checkpoint (ecg) has a dead encoder and is refused by training/launch.py.
START_CHECKPOINT = None


def keep_checkpoint(stage, stop: threading.Event) -> None:
    """Copy the best checkpoint and the loss history to the kept folder whenever they change."""
    seen: dict[str, float] = {}
    while True:
        last_round = stop.wait(KEEP_EVERY)
        for name in ("best_model.pt", "loss_history.txt"):
            path = stage / "checkpoints" / name
            try:
                changed = path.stat().st_mtime
                if seen.get(name) != changed:
                    shutil.copy(path, k.RESULTS / (name + ".part"))
                    (k.RESULTS / (name + ".part")).replace(k.RESULTS / name)
                    seen[name] = changed
            except OSError:
                pass  # not written yet, or being written: the next round gets it
        if last_round:
            return


def main() -> None:
    started = time.time()
    paths = k.prepare(START_CHECKPOINT)
    stage = paths.checkout / k.STAGE
    rehearsal = REHEARSAL_ROWS > 0
    if rehearsal:
        k.say(f"REHEARSAL on {REHEARSAL_ROWS} training rows: the scores mean nothing")

    stop = threading.Event()
    keeper = threading.Thread(target=keep_checkpoint, args=(stage, stop), daemon=True)
    keeper.start()
    try:
        result = k.train(paths, rows=REHEARSAL_ROWS, batch_size=BATCH_SIZE, dtype=DTYPE, attention=ATTENTION,
                         epochs=REHEARSAL_EPOCHS if rehearsal else EPOCHS, series_view=SERIES_VIEW, answer_order=ANSWER_ORDER)
    finally:
        stop.set()
        keeper.join()
    shutil.copy(stage / "results" / "test_predictions.jsonl", k.RESULTS / "test_predictions_plain.jsonl")
    shutil.copy(stage / "results" / "metrics.json", k.RESULTS / "metrics.json")
    truth = {"plain": "train" if rehearsal else "test"}
    extra = []
    for name in EXTRA_ANSWERS:
        extra.append(k.answer_again(paths, f"{name}.jsonl", dtype=DTYPE, attention=ATTENTION, rows=REHEARSAL_ROWS,
                                    series_view=SERIES_VIEW, answer_order=ANSWER_ORDER))
        short = name.replace("test_", "")
        shutil.copy(stage / "results" / "test_predictions.jsonl", k.RESULTS / f"test_predictions_{short}.jsonl")
        truth[short] = "train" if rehearsal else name

    runs = " ".join(f"--run {short} {k.RESULTS / f'test_predictions_{short}.jsonl'} {paths.jsonl / (file + '.jsonl')}"
                    for short, file in truth.items())
    k.run(f"{sys.executable} training/score.py {'--only-predicted ' if rehearsal else ''}{runs} "
          f"--json {k.RESULTS / 'scores.json'}", cwd=paths.root)

    summary = {"version": paths.version, "llm_id": k.LLM_ID, "gpu": paths.gpu, "rehearsal_rows": REHEARSAL_ROWS,
               "setup_seconds": paths.setup_seconds, "total_seconds": round(time.time() - started),
               "start_checkpoint": START_CHECKPOINT or "default of setup.sh",
               "series_view": SERIES_VIEW, "answer_order": ANSWER_ORDER, "epochs_planned": REHEARSAL_EPOCHS if rehearsal else EPOCHS,
               "run": result, "extra_answers": extra}
    (k.RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2))
    k.say("\nSummary:")
    k.say(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
