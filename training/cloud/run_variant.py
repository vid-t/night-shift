"""One design of stage6_hai on a rented GPU machine, in its own folder, so several can run at once.

    python3 training/cloud/run_variant.py NAME [--start sleep|run:OTHER] [--view range|zscore|looprange] [--order label_last]
        [--epochs 4] [--seed 0] [--rows 0] [--dtype bfloat16] [--batch-size 1] [--answers validation]
        [--lr-encoder 5e-5] [--attack-share 0.35] [--mirror-share 0.5] [--choose-pass loss|last|validation]
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
`--start run:OTHER` starts from work/OTHER/kept/best_model.pt, the saved model of the finished run OTHER.

`--choose-pass` picks the pass (epoch) whose model is kept. `loss` (default) keeps OpenTSLM's choice,
the pass with the lowest validation loss. `last` and `validation` keep the model of every pass, then
keep the last pass, or the pass that does best on the validation rows: attack rows called attack
minus normal rows called attack (the later pass on a tie). The answers of every pass are in
kept/passes/. The other files are answered with the kept model.
"""
import argparse
import importlib.util
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
CHOICES = ("loss", "last", "validation")


def short(file: str) -> str:
    """Name of a file's predictions in kept/: test -> plain, test_shuffled -> shuffled, the others as they are."""
    return {"test": "plain", "test_shuffled": "shuffled"}.get(file, file)


def start_source(start: str, root: Path = ROOT) -> str:
    """What seed.py gets for --start: a Hugging Face repo, or the path of the saved model of a finished run."""
    if not start.startswith("run:"):
        return STARTS.get(start, start)
    kept = root / "work" / start[len("run:"):] / "kept"
    for need in ("run_summary.json", "best_model.pt"):
        if not (kept / need).is_file():
            sys.exit(f"no {kept / need}: --start {start} needs a finished run with a saved model")
    return str((kept / "best_model.pt").resolve())


def arguments(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="folder name of the run under work/")
    ap.add_argument("--start", default="sleep",
                    help="starting checkpoint: tsqa, m4, har, sleep, a Hugging Face repo, or run:OTHER for the "
                         "saved model of the finished run OTHER on this machine (default: %(default)s)")
    ap.add_argument("--view", default="range", choices=("range", "zscore", "looprange"))
    ap.add_argument("--order", default="label_last", choices=("label_last", "label_first", "label_only"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=0, help="N > 0: rehearsal on N training rows")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "float16"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads of this run")
    ap.add_argument("--lr-encoder", type=float, default=None,
                    help="learning rate of the encoder in stage6_hai (default: the 2e-4 of the stage)")
    ap.add_argument("--attack-share", type=float, default=None,
                    help="share of attack rows in the training rows, in (0, 1) (default: the rows as exported)")
    ap.add_argument("--mirror-share", type=float, default=None,
                    help="share of the training rows that get an added mirrored copy (series flipped, rises and "
                         "falls swapped), in (0, 1] (default: no copies)")
    ap.add_argument("--answers", default="validation", choices=sorted(ANSWER_SETS),
                    help="files answered after training (default: %(default)s)")
    ap.add_argument("--choose-pass", default="loss", choices=CHOICES,
                    help="pass whose model is kept: loss (lowest validation loss, OpenTSLM's choice), last, or "
                         "validation (best on the validation rows: attacks found minus normal rows alarmed) "
                         "(default: %(default)s)")
    ap.add_argument("--answer-only", nargs="+", choices=FILES, metavar="FILE",
                    help="no training: answer these files with the saved model of the finished run NAME")
    return ap.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parser():
    """parse() of training/score.py, which is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("training_score", ROOT / "training" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


def pass_files(checkpoints: Path) -> dict[int, Path]:
    """Pass number -> its model file checkpoints/pass_<N>.pt, in pass order."""
    found = {}
    for path in checkpoints.glob("pass_*.pt"):
        number = path.stem[len("pass_"):]
        if number.isdigit():
            found[int(number)] = path
    return dict(sorted(found.items()))


def loss_pass(results: Path) -> int | None:
    """The pass of OpenTSLM's best-loss model, from results/metrics.json (None if it does not say)."""
    metrics = results / "metrics.json"
    epoch = json.loads(metrics.read_text()).get("epoch") if metrics.exists() else None
    return int(epoch) if epoch is not None else None


def pass_score(predictions: Path, truth: Path) -> dict:
    """Attack rows called attack minus normal rows called attack. Truth rows without a prediction are left out."""
    parse = parser()
    said = {p["example_id"]: parse(p.get("generated"))[0] for p in read_jsonl(predictions)}
    found = alarmed = 0
    for row in read_jsonl(truth):
        if said.get(row["example_id"]) != "attack":
            continue
        found += row["label"] == "attack"
        alarmed += row["label"] == "normal"
    return {"attack_rows_found": found, "normal_rows_alarmed": alarmed, "score": found - alarmed}


def best_pass(scores: dict[int, dict]) -> int:
    """The pass with the highest score; the later pass on a tie."""
    return max(scores, key=lambda n: (scores[n]["score"], n))


def choose(mode: str, stage: Path, kept: Path, first: str, best_loss: int | None, truth: Path, answer,
           say=print) -> dict:
    """Put the chosen pass's model in checkpoints/best_model.pt and its answers of `first` in kept/.

    `answer()` answers `first` again with checkpoints/best_model.pt into results/test_predictions.jsonl
    and returns its record. kept/test_predictions_<first>.jsonl holds the answers of the best-loss
    model when this is called. Returns kept_pass, pass_scores and the records of the answers.
    """
    checkpoints, results = stage / "checkpoints", stage / "results"
    best, predictions = checkpoints / "best_model.pt", kept / f"test_predictions_{short(first)}.jsonl"
    out = {"kept_pass": best_loss, "pass_scores": None, "pass_answers": []}
    if mode == "loss":
        return out
    passes = pass_files(checkpoints)
    if not passes:
        say(f"no pass_<N>.pt in {checkpoints}: the model of the lowest validation loss is kept")
        return out
    if mode == "last":
        n = max(passes)
        if n != best_loss:
            shutil.copy(passes[n], best)
            out["pass_answers"].append(answer())
            shutil.copy(results / "test_predictions.jsonl", predictions)
        say(f"kept the model of the last pass, {n} (lowest validation loss: pass {best_loss})")
        return {**out, "kept_pass": n}
    loss_model = checkpoints / "loss_model.pt"   # with its optimizer state, as OpenTSLM saved it
    shutil.copy(best, loss_model)
    folder = kept / "passes"
    folder.mkdir(parents=True, exist_ok=True)
    scores, files = {}, {}
    for n, model in passes.items():
        files[n] = folder / f"test_predictions_{short(first)}_pass_{n}.jsonl"
        if n == best_loss:
            shutil.copy(predictions, files[n])   # the trainer's own answers with this model
        else:
            shutil.copy(model, best)
            out["pass_answers"].append(answer())
            shutil.copy(results / "test_predictions.jsonl", files[n])
        scores[n] = pass_score(files[n], truth)
        say(f"pass {n}: {json.dumps(scores[n])}")
    n = best_pass(scores)
    shutil.copy(loss_model if n == best_loss else passes[n], best)
    shutil.copy(files[n], predictions)
    say(f"kept the model of pass {n}, the best on the validation rows (lowest validation loss: pass {best_loss})")
    return {**out, "kept_pass": n, "pass_scores": scores}


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
    if not args.answer_only:
        if args.start == f"run:{args.name}":
            sys.exit(f"--start {args.start}: a run cannot start from itself")
        source = start_source(args.start)  # exits before anything is made if the other run is not done
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
    if args.lr_encoder is not None:
        env["HAI_LR_ENCODER"] = str(args.lr_encoder)
    if args.attack_share is not None:
        env["HAI_ATTACK_SHARE"] = str(args.attack_share)
    if args.mirror_share is not None:
        env["HAI_MIRROR_SHARE"] = str(args.mirror_share)
    if args.choose_pass != "loss":
        env["HAI_KEEP_PASSES"] = "1"
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
    from_run = args.start.startswith("run:")
    files = ANSWER_SETS[args.answers]
    design = {"name": args.name, "start_checkpoint": args.start if from_run else source,
              "start_file": source if from_run else None, "series_view": args.view, "answer_order": args.order,
              "epochs_planned": args.epochs, "seed": args.seed, "rehearsal_rows": args.rows, "dtype": args.dtype,
              "batch_size": args.batch_size, "answers": list(files), "lr_encoder": args.lr_encoder,
              "attack_share": args.attack_share, "mirror_share": args.mirror_share,
              "choose_pass": args.choose_pass}
    k.say(f"design: {json.dumps(design)}")
    k.run(f"uv run python {ROOT / 'training' / 'cloud' / 'seed.py'} {k.LLM_ID} {source}", cwd=work, **env)

    result = k.train(paths, rows=args.rows, batch_size=args.batch_size, dtype=args.dtype, attention="sdpa",
                     epochs=args.epochs, seed=args.seed, series_view=args.view, answer_order=args.order,
                     test_file=f"{files[0]}.jsonl")
    best_loss = loss_pass(stage / "results")   # before answer_again moves metrics.json aside
    shutil.copy(stage / "results" / "test_predictions.jsonl", kept / f"test_predictions_{short(files[0])}.jsonl")
    shutil.copy(stage / "results" / "metrics.json", kept / "metrics.json")
    truth = paths.jsonl / ("train.jsonl" if args.rows > 0 else f"{files[0]}.jsonl")

    def answer_first():
        return k.answer_again(paths, f"{files[0]}.jsonl", dtype=args.dtype, attention="sdpa", rows=args.rows,
                              series_view=args.view, answer_order=args.order)

    chosen = choose(args.choose_pass, stage, kept, files[0], best_loss, truth, answer_first, say=k.say)
    for name in ("best_model.pt", "loss_history.txt"):   # after the choice: best_model.pt is the chosen pass
        shutil.copy(stage / "checkpoints" / name, kept / name)
    extra = []
    for name in files[1:]:
        extra.append(k.answer_again(paths, f"{name}.jsonl", dtype=args.dtype, attention="sdpa", rows=args.rows,
                                    series_view=args.view, answer_order=args.order))
        shutil.copy(stage / "results" / "test_predictions.jsonl", kept / f"test_predictions_{short(name)}.jsonl")

    score(k, kept, paths.jsonl, args.rows > 0, env)
    summary = {**design, "llm_id": k.LLM_ID, "total_seconds": round(time.time() - started), "run": result,
               "extra_answers": extra, "loss_pass": best_loss, **chosen}
    summary_file.write_text(json.dumps(summary, indent=2))
    k.say("\nSummary:")
    k.say(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
