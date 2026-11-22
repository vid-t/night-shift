"""Shared helpers for the tests of training/: load its scripts, find the OpenTSLM checkout, make examples."""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
OPENTSLM = ROOT / "external" / "OpenTSLM"
PINNED = "2968f4b891baab4307f7e9d0043e87677b593a30"
LOOPS = ["P1-PC", "P1-LC", "P1-FC", "P1-TC", "P2-SC", "P3-LC"]


def load_script(name: str):
    if name == "HAIAttackQADataset":
        opentslm_without_init()   # the class imports opentslm and torch: skip the test without them
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinned_file(rel: str) -> str:
    """A file of OpenTSLM as it is at the pinned commit (ignores local edits). Skips if there is no checkout."""
    if not (OPENTSLM / ".git").exists():
        pytest.skip("external/OpenTSLM missing: run training/setup.sh or clone it at the pinned commit")
    out = subprocess.run(["git", "-C", str(OPENTSLM), "show", f"{PINNED}:{rel}"], capture_output=True, text=True)
    if out.returncode:
        pytest.skip(f"pinned commit not in external/OpenTSLM: {out.stderr.strip()}")
    return out.stdout


def opentslm_without_init():
    """Make `opentslm.*` importable from the checkout without running its package __init__
    (that loads transformers and the model). Needs torch for QADataset."""
    pytest.importorskip("torch", reason="install the training group: uv sync --group training")
    src = OPENTSLM / "src" / "opentslm"
    if not src.exists():
        pytest.skip("external/OpenTSLM missing")
    if "opentslm" not in sys.modules:
        pkg = types.ModuleType("opentslm")
        pkg.__path__ = [str(src)]
        sys.modules["opentslm"] = pkg


def make_examples(n_attack_windows: int = 8, n_normal_windows: int = 6, seed: int = 0) -> list[dict]:
    """Made-up test rows in the export format: scored attack windows (one attacked loop each) and normal windows."""
    rng = np.random.default_rng(seed)
    rows = []

    def row(wid, version, loop, label, start, kind, scored, hidden=False):
        if label == "attack":
            answer = f"The measurement rises from second {start + 2}. Change starts near second {start}. Answer: attack"
        else:
            answer = "The target and the measurement stay flat. The command stays flat. Answer: normal"
        values = rng.normal(size=300).round(3).tolist()
        series = [{"role": "target", "name": "target", "values": [5.0] * 300},
                  {"role": "measurement", "name": "measurement", "values": values},
                  {"role": "command", "name": "command", "values": values[::-1]}]
        return {"example_id": f"{wid}/{loop}", "version": version, "loop": loop, "loop_name": "boiler level",
                "split": "test", "label": label, "attack_start_s": start, "scored": scored,
                "no_visible_change": hidden, "window_kind": kind, "answer": answer, "series": series}

    for i in range(n_attack_windows):
        version = ["21.03", "22.04"][i % 2]
        wid = f"{version}/test/A{i:03d}"
        hit = LOOPS[i % len(LOOPS)]
        start = int(30 + (240 * i) // max(1, n_attack_windows - 1))
        for loop in LOOPS:
            label = "attack" if loop == hit else "normal"
            rows.append(row(wid, version, loop, label, start if label == "attack" else -1, "attack", True,
                            hidden=(label == "attack" and i == 0)))
    for i in range(n_normal_windows):
        version = ["21.03", "22.04"][i % 2]
        wid = f"{version}/test/N{i:03d}"
        for loop in LOOPS:
            rows.append(row(wid, version, loop, "normal", -1, "normal", False))
    return rows
