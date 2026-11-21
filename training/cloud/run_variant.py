"""One design of stage6_hai on a rented GPU machine, in its own folder, so several can run at once.

    python3 training/cloud/run_variant.py NAME [--start sleep] [--view range] [--order label_last]
        [--epochs 4] [--seed 0] [--rows 0] [--dtype bfloat16] [--batch-size 1] [--answers validation]
    python3 training/cloud/run_variant.py NAME --answer-only test test_shuffled

Run it from the repo root on the machine, after `bash training/setup.sh unsloth/Llama-3.2-1B`.

A run trains, then answers the validation rows, plain and with the seconds shuffled, and scores
both. Designs are chosen on these. The test rows are for the design that is picked:
`--answer-only test test_shuffled` answers them later with the run's saved model, without
training again. `--answers all` answers all four files in one go.

Everything of the run is under work/NAME:

  work/NAME/results/...   OpenTSLM's own tree (checkpoints, loss history, predictions)
  work/NAME/kept/         log.txt, the prediction files, scores.json, run_summary.json,
                          loss_history.txt and best_model.pt

`--rows 50` is the rehearsal: train, validation and test are the same 50 training rows.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
STARTS = {name: f"OpenTSLM/llama-3.2-1b-{name}-sp" for name in ("tsqa", "m4", "har", "sleep")}
ANSWER_SETS = {"validation": ("validation", "validation_shuffled"),
               "all": ("validation", "validation_shuffled", "test", "test_shuffled")}
FILES = ("validation", "validation_shuffled", "test", "test_shuffled")


def short(file: str) -> str:
    """Name of a file's predictions in kept/: test -> plain, test_shuffled -> shuffled, the others as they are."""
    return {"test": "plain", "test_shuffled": "shuffled"}.get(file, file)


def arguments(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="folder name of the run under work/")
    ap.add_argument("--start", default="sleep",
                    help="starting checkpoint: tsqa, m4, har, sleep, or a Hugging Face repo (default: %(default)s)")
    ap.add_argument("--view", default="range", choices=("range", "zscore"))
    ap.add_argument("--order", default="label_last", choices=("label_last", "label_first", "label_only"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=0, help="N > 0: rehearsal on N training rows")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "float16"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads of this run")
    ap.add_argument("--answers", default="validation", choices=sorted(ANSWER_SETS),
                    help="files answered after training (default: %(default)s)")
    ap.add_argument("--answer-only", nargs="+", choices=FILES, metavar="FILE",
                    help="no training: answer these files with the saved model of the finished run NAME")
    return ap.parse_args(argv)


def score(k, kept: Path, jsonl: Path, rehearsal: bool, env: dict) -> None:
    """Score every prediction file in kept/ against its own truth file."""
    done = [f for f in FILES if (kept / f"test_predictions_{short(f)}.jsonl").exists()]
    runs = " ".join(f"--run {short(f)} {kept / f'test_predictions_{short(f)}.jsonl'} "
                    f"{jsonl / (('train' if rehearsal else f) + '.jsonl')}" for f in done)
    k.run(f"uv run python {ROOT / 'training' / 'score.py'} {'--only-predicted ' if rehearsal else ''}{runs} "
          f"--json {kept / 'scores.json'}", cwd=ROOT, **env)


def main(argv=None) -> None:
    args = arguments(argv)
    work = ROOT / "work" / args.name
    kept = work / "kept"
    summary_file = kept / "run_summary.json"
    if args.answer_only and not summary_file.exists():
        sys.exit(f"no {summary_file}: --answer-only needs a finished run")
    if not args.answer_only and summary_file.exists():
        sys.exit(f"{summary_file} exists: this run is done. Pick another name, or use --answer-only.")
    kept.mkdir(parents=True, exist_ok=True)
    os.environ["HAI_KEEP_DIR"] = str(kept)  # before the import: kaggle_common reads it once
    sys.path.insert(0, str(ROOT / "training" / "kaggle"))
    import kaggle_common as k

    checkout = ROOT / "external" / "OpenTSLM"
    if not (checkout / "curriculum_learning.py").exists():
        sys.exit(f"no OpenTSLM in {checkout}: run `bash training/setup.sh {k.LLM_ID}` first")
    # every command of the run uses OpenTSLM's Python, from this run's own folder
    env = {"UV_PROJECT": str(checkout), "UV_NO_SYNC": "1", "OPENTSLM_DIR": str(checkout),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
    paths = SimpleNamespace(root=ROOT, checkout=work, jsonl=ROOT / "data" / "processed" / "opentslm", cache=env)
    stage = work / k.STAGE
    started = time.time()

    if args.answer_only:
        summary = json.loads(summary_file.read_text())
        rows = summary["rehearsal_rows"]
        k.say(f"\nanswer only: {' '.join(args.answer_only)} with the saved model of {args.name}")
        for name in args.answer_only:
            summary["extra_answers"].append(
                k.answer_again(paths, f"{name}.jsonl", dtype=summary["dtype"], attention="sdpa", rows=rows,
                               series_view=summary["series_view"], answer_order=summary["answer_order"]))
            shutil.copy(stage / "results" / "test_predictions.jsonl", kept / f"test_predictions_{short(name)}.jsonl")
        score(k, kept, paths.jsonl, rows > 0, env)
        summary_file.write_text(json.dumps(summary, indent=2))
        return

    k.LOG.write_text("")
    repo = STARTS.get(args.start, args.start)
    files = ANSWER_SETS[args.answers]
    design = {"name": args.name, "start_checkpoint": repo, "series_view": args.view, "answer_order": args.order,
              "epochs_planned": args.epochs, "seed": args.seed, "rehearsal_rows": args.rows, "dtype": args.dtype,
              "batch_size": args.batch_size, "answers": list(files)}
    k.say(f"design: {json.dumps(design)}")
    k.run(f"uv run python {ROOT / 'training' / 'cloud' / 'seed.py'} {k.LLM_ID} {repo}", cwd=work, **env)

    result = k.train(paths, rows=args.rows, batch_size=args.batch_size, dtype=args.dtype, attention="sdpa",
                     epochs=args.epochs, seed=args.seed, series_view=args.view, answer_order=args.order,
                     test_file=f"{files[0]}.jsonl")
    shutil.copy(stage / "results" / "test_predictions.jsonl", kept / f"test_predictions_{short(files[0])}.jsonl")
    shutil.copy(stage / "results" / "metrics.json", kept / "metrics.json")
    for name in ("best_model.pt", "loss_history.txt"):
        shutil.copy(stage / "checkpoints" / name, kept / name)
    extra = []
    for name in files[1:]:
        extra.append(k.answer_again(paths, f"{name}.jsonl", dtype=args.dtype, attention="sdpa", rows=args.rows,
                                    series_view=args.view, answer_order=args.order))
        shutil.copy(stage / "results" / "test_predictions.jsonl", kept / f"test_predictions_{short(name)}.jsonl")

    score(k, kept, paths.jsonl, args.rows > 0, env)
    summary = {**design, "llm_id": k.LLM_ID, "total_seconds": round(time.time() - started), "run": result,
               "extra_answers": extra}
    summary_file.write_text(json.dumps(summary, indent=2))
    k.say("\nSummary:")
    k.say(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
