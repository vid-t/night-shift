"""Shared by the Kaggle run scripts: find the run files, set up OpenTSLM, run and time training.

The run files come from the attached Kaggle dataset (see push.sh). Everything heavy goes to a
scratch folder that Kaggle throws away. Only /kaggle/working/results is kept.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RESULTS = Path("/kaggle/working/results")
LOG = RESULTS / "log.txt"
PROGRESS_EVERY = 30              # seconds between progress-bar lines in the log
LLM_ID = "unsloth/Llama-3.2-1B"  # open copy; its weights file has the same SHA-256 as meta-llama/Llama-3.2-1B
STAGE = Path("results") / "Llama_3_2_1B" / "OpenTSLMSP" / "stage6_hai"
FULL_ROWS = {"train": 3770, "validation": 1092, "test": 1536}
MAX_EPOCHS = 10


def say(text: str = "") -> None:
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run(command: str, cwd: Path | None = None, check: bool = True, **env: str) -> tuple[int, float, list[str]]:
    """Run a shell command and log its output. Returns exit code, seconds and the lines kept."""
    say(f"\n$ {command}")
    start = time.time()
    process = subprocess.Popen(["bash", "-c", "set -euo pipefail; " + command], cwd=cwd,
                               env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines, pending, last_progress = [], "", 0.0
    fd = process.stdout.fileno()
    while True:
        data = os.read(fd, 65536)
        if not data:
            break
        pending += data.decode("utf-8", "replace")
        while True:
            cut = re.search(r"[\r\n]", pending)
            if not cut:
                break
            line, end, pending = pending[:cut.start()], cut.group(), pending[cut.end():]
            if not line.strip():
                continue
            # a progress bar redraws itself with \r: keep its last state and one line every 30 s
            if end == "\n" or "100%|" in line or time.time() - last_progress > PROGRESS_EVERY:
                if end == "\r":
                    last_progress = time.time()
                lines.append(line)
                say(line)
    if pending.strip():
        lines.append(pending)
        say(pending)
    code = process.wait()
    seconds = time.time() - start
    say(f"[{seconds:.0f} s, exit {code}]")
    if code and check:
        raise SystemExit(f"command failed with exit {code}: {command}")
    return code, seconds, lines


def bar_seconds(lines: list[str]) -> dict[str, list[float]]:
    """Seconds each finished progress bar took, by kind: Epoch, Validating, Evaluating."""
    found: dict[str, list[float]] = {"Epoch": [], "Validating": [], "Evaluating": []}
    last: dict[str, tuple[int, str]] = {}
    for i, line in enumerate(lines):
        m = re.search(r"(Epoch \d+/\d+|Validating[^|:]*|Evaluating[^|:]*): 100%\|.*?\| (\d+)/\2 \[([\d:]+)<", line)
        if not m:
            continue
        seconds = 0.0
        for part in m.group(3).split(":"):
            seconds = seconds * 60 + float(part)
        kind = m.group(1).split()[0]
        if last.get(kind) == (i - 1, m.group(1)):
            found[kind][-1] = seconds  # a bar prints its last state twice: same bar, keep one
        else:
            found[kind].append(seconds)
        last[kind] = (i, m.group(1))
    return found


def losses(lines: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"train": [], "validation": []}
    for line in lines:
        m = re.search(r"Epoch \d+ — (train|val)\s+loss: ([\d.]+|nan|inf)", line)
        if m:
            out["train" if m.group(1) == "train" else "validation"].append(float(m.group(2)))
    return out


def prepare() -> SimpleNamespace:
    """Lay the run files out like the repo, set up OpenTSLM, report the GPU. Returns the paths."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    hits = glob.glob("/kaggle/input/**/add_stage.py", recursive=True)
    if not hits:
        raise SystemExit("run files not found under /kaggle/input: attach the dataset made by push.sh")
    source = Path(hits[0]).parent
    scratch = max(("/kaggle/temp", "/tmp"), key=lambda p: shutil.disk_usage(p).free if os.path.isdir(p) else -1)
    root = Path(scratch) / "hai"
    version = (source / "VERSION").read_text().strip() if (source / "VERSION").exists() else "unknown"
    say(f"run files: {source}")
    say(f"version: {version}")
    say(f"scratch: {root} ({shutil.disk_usage(scratch).free / 1e9:.0f} GB free)")

    (root / "training").mkdir(parents=True, exist_ok=True)
    jsonl = root / "data" / "processed" / "opentslm"
    jsonl.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix == ".jsonl":
            shutil.copy(path, jsonl / path.name)
        elif path.suffix in (".py", ".sh"):
            shutil.copy(path, root / "training" / path.name)
    cache = {"HF_HOME": f"{scratch}/hf", "UV_CACHE_DIR": f"{scratch}/uv-cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

    if not shutil.which("uv"):
        run(f"{sys.executable} -m pip install -q uv")
    _, setup_seconds, _ = run(f"bash training/setup.sh {LLM_ID}", cwd=root, **cache)
    checkout = root / "external" / "OpenTSLM"
    _, _, check = run("uv run python ../../training/gpu_check.py", cwd=checkout, **cache)
    return SimpleNamespace(root=root, checkout=checkout, jsonl=jsonl, cache=cache, version=version,
                           setup_seconds=round(setup_seconds), gpu=json.loads(check[-1]))


def train(paths: SimpleNamespace, rows: int, batch_size: int, dtype: str, epochs: int | None = None,
          max_new_tokens: int = 128, seed: int = 0, check: bool = True, attention: str | None = None,
          loss_scale: float | None = None) -> dict:
    """One run of stage6_hai on the first `rows` training rows (0 = the real split). Starts fresh."""
    stage = paths.checkout / STAGE
    if stage.exists():
        shutil.rmtree(stage)
    env = dict(paths.cache, HAI_JSONL_DIR=str(paths.jsonl), HAI_LLM_DTYPE=dtype, HAI_SEED=str(seed),
               HAI_MAX_NEW_TOKENS=str(max_new_tokens))
    if rows:
        env["HAI_OVERFIT_ROWS"] = str(rows)
    if epochs:
        env["HAI_EPOCHS"] = str(epochs)
    if attention:
        env["HAI_ATTENTION"] = attention
    if loss_scale:
        env["HAI_LOSS_SCALE"] = str(loss_scale)
    code, seconds, lines = run(
        f"uv run python ../../training/launch.py --model OpenTSLMSP --llm_id {LLM_ID} "
        f"--stages stage6_hai --batch_size {batch_size} --device cuda", cwd=paths.checkout, check=check, **env)
    bars, text = bar_seconds(lines), "\n".join(lines)
    n = {"train": rows or FULL_ROWS["train"], "validation": rows or FULL_ROWS["validation"], "test": rows or FULL_ROWS["test"]}
    later = bars["Epoch"][1:] or bars["Epoch"]   # the first epoch also warms the GPU up
    per_row = {
        "train": sum(later) / (len(later) * n["train"]) if later else None,
        "validation": sum(bars["Validating"]) / (len(bars["Validating"]) * n["validation"]) if bars["Validating"] else None,
        "answer": bars["Evaluating"][-1] / n["test"] if bars["Evaluating"] else None,
    }
    peak = re.search(r"peak GPU memory: ([\d.]+) GB", text)
    scale = re.search(r"loss scale ended at ([\d.e+]+); steps with a gradient that was not finite: (\d+) of (\d+)", text)
    return {"dtype": dtype, "attention": attention or "eager", "batch_size": batch_size, "rows": rows, "exit": code,
            "seconds": round(seconds),
            "out_of_memory": "OutOfMemoryError" in text, "epochs_run": len(bars["Epoch"]),
            "epoch_seconds": bars["Epoch"], "seconds_per_row": per_row, "losses": losses(lines),
            "peak_gpu_memory_gb": float(peak.group(1)) if peak else None,
            "loss_scale_at_end": float(scale.group(1)) if scale else None,
            "overflow_steps": int(scale.group(2)) if scale else 0,
            "train_steps": int(scale.group(3)) if scale else None}


def answer_again(paths: SimpleNamespace, test_file: str, dtype: str, attention: str | None = None,
                 max_new_tokens: int = 128, rows: int = 0) -> dict:
    """Answers of the trained stage on another test file, without training (step 6 of the README).

    Keeps the stage and its checkpoint. The first results move aside as metrics_plain.json and
    test_predictions_plain.jsonl, because OpenTSLM skips a stage whose metrics.json says completed.
    """
    results = paths.checkout / STAGE / "results"
    for name, aside in (("metrics.json", "metrics_plain.json"), ("test_predictions.jsonl", "test_predictions_plain.jsonl")):
        if (results / name).exists():
            (results / name).rename(results / aside)
    env = dict(paths.cache, HAI_JSONL_DIR=str(paths.jsonl), HAI_LLM_DTYPE=dtype, HAI_TEST_FILE=test_file,
               HAI_MAX_NEW_TOKENS=str(max_new_tokens))
    if rows:
        env["HAI_OVERFIT_ROWS"] = str(rows)
    if attention:
        env["HAI_ATTENTION"] = attention
    code, seconds, lines = run(
        f"uv run python ../../training/launch.py --model OpenTSLMSP --llm_id {LLM_ID} "
        f"--stages stage6_hai --eval_only --device cuda", cwd=paths.checkout, **env)
    bars = bar_seconds(lines)
    return {"test_file": test_file, "exit": code, "seconds": round(seconds),
            "answer_seconds": bars["Evaluating"][-1] if bars["Evaluating"] else None}


def full_run_estimate(per_row: dict) -> dict | None:
    """Hours for the full run from the seconds per row of a short one."""
    if not all(per_row.values()):
        return None
    epoch = (FULL_ROWS["train"] * per_row["train"] + FULL_ROWS["validation"] * per_row["validation"]) / 3600
    answers = FULL_ROWS["test"] * per_row["answer"] / 3600
    return {"hours_per_epoch": round(epoch, 2), "hours_for_test_answers": round(answers, 2),
            "hours_for_max_epochs": round(MAX_EPOCHS * epoch + answers, 2), "max_epochs": MAX_EPOCHS}
