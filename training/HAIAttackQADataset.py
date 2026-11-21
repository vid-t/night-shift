"""HAI control-loop attack QA dataset for OpenTSLM (stage `stage6_hai`).

setup.sh copies this file to src/opentslm/time_series_datasets/hai/ in the OpenTSLM checkout.

Reads the JSONL files that `uv run hai-export` writes. Environment variables:

  HAI_JSONL_DIR      folder with train.jsonl, validation.jsonl and the test file (required)
  HAI_TEST_FILE      test file name inside that folder (default: test.jsonl;
                     test_shuffled.jsonl or test_full.jsonl for the other runs)
  HAI_OVERFIT_ROWS   N > 0: train, validation and test are the same N training rows,
                     drawn with a fixed seed (the overfit check)
  HAI_ATTACK_SHARE   a share in (0, 1): the training rows are redrawn so this share of them has label
                     attack, same number of rows (see `with_attack_share`). Unset: the rows as in the file.
                     Validation and test are never touched.
  HAI_MIRROR_SHARE   a share in (0, 1]: mirrored copies of this share of the training rows are added after
                     them (see `with_mirrors`), after HAI_ATTACK_SHARE if both are set. A mirrored copy shows
                     every series flipped around the mean of its first RANGE_HEAD_S seconds, with rises and
                     falls swapped in its answer. Unset: no copies. Validation and test are never touched.
  HAI_TEXT_CLAUSES   comma-separated names from CLAUSES (see `text_clauses`): further clauses in the text line
                     of each series, views zsize and looprange only. Unset or empty: no clauses, the text as before.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

ROLE_WORDS = {
    "target": "the value the controller aims for",
    "measurement": "the measured value",
    "command": "what the controller sends to the actuator",
}
OVERFIT_SEED = 0
ATTACK_SHARE_SEED = 0
MIRROR_SEED = 0
MIN_SPREAD = 1e-6  # below this (relative to the level) a series counts as flat


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


VIEWS = ("zscore", "range", "looprange", "zsize")
RANGE_UNIT = 0.02   # one unit in the range view is this share of the series' normal range
RANGE_HEAD_S = 30   # the range view shows the change from the mean of these first seconds
LOOPRANGE_FLOOR = 0.25   # the smallest quiet-window unit of the looprange view, in range units
LOOPRANGE_MIN_ROWS = 5   # a (version, loop, series) key with fewer quiet rows uses the unit of all versions
SIZE_CAP = 99.9   # the zsize view prints at most this size
LAW_MIN_R2 = 0.4   # the law clause is written only for a loop whose fitted law explains this share of du on its quiet rows
PLANT_MIN_R2 = 0.5   # the plant clause is written only for a loop whose fitted plant explains this share of the measurement
PLANT_LAGS = (0, 5, 10, 15, 20, 25, 30)   # the plant model reads the first command this many seconds back


def series_view() -> str:
    """How a series is shown to the model: HAI_SERIES_VIEW, one of VIEWS (default `zscore`).

    zscore  each window is scaled to mean 0 and spread 1, and the text states its mean and spread.
            What OpenTSLM's own datasets do. Here it hides how large a change is: noise and a real
            step both fill the same range, and the size sits only in two numbers of the text.
    range   the change from the first RANGE_HEAD_S seconds, in units of RANGE_UNIT of the series'
            normal range, squashed with a signed log so a huge step does not drown a small one.
            The text states no numbers. A quiet series stays near zero, a change that the answer
            rule calls visible is about one unit or more.
    looprange  the range view, with the change also divided by how much that series normally moves in a
            quiet window of the same loop: the 95th percentile, over the quiet training rows, of its
            largest change in range units (see `quiet_units`). A pump command that switches in every
            quiet window gets a large unit, a series that barely moves a small one (at least
            LOOPRANGE_FLOOR). The text states no numbers, unless clauses are on (see `text_clauses`).
    zsize   the zscore curve, but the text states no mean and spread. It states one number: the series'
            largest change from its first RANGE_HEAD_S seconds, in units of its usual quiet change in this
            loop (the looprange unit, see `relative_size`), at most SIZE_CAP. The curve gives the shape,
            the number the size.
    """
    view = (os.environ.get("HAI_SERIES_VIEW") or "zscore").strip()
    if view not in VIEWS:
        raise RuntimeError(f"HAI_SERIES_VIEW must be one of {VIEWS}, got {view!r}")
    return view


CLAUSES = ("size", "step", "law", "plant")
CLAUSE_VIEWS = ("zsize", "looprange")   # the views that take clauses


def text_clauses() -> tuple[str, ...]:
    """Further clauses in the text line of each series: HAI_TEXT_CLAUSES, names from CLAUSES. Unset or empty: none.

    size    the series' largest change from its first RANGE_HEAD_S seconds, in units of its usual quiet change in
            this loop (see `size_clause`). The zsize view always states it, named or not.
    step    the series' largest one-second step, in units of its usual one in a quiet window of this loop
            (see `quiet_step_units`), and the second where it lands (see `step_clause`).
    law     on the line of the loop's first command only: for how many seconds that command moves the opposite
            way to what the controller would do, given the gap between target and measurement. The controller's
            law is fitted on the quiet training rows of the loop (see `fit_laws`, `wrong_way_seconds`, `law_clause`).
            A loop whose law explains less than LAW_MIN_R2 of the command's changes there gets no law clause.
    plant   on the line of the measurement only: how far the measurement strays from what its first command (and
            target) predict, in units of how far it usually strays in a quiet window of the loop. The plant model is
            fitted on the quiet training rows of the loop, all versions together (see `fit_plants`, `plant_ratio`,
            `plant_clause`). A loop whose model explains less than PLANT_MIN_R2 of the measurement there gets no
            plant clause, and so does a row without exactly one target, one measurement and a command.

    Only the views in CLAUSE_VIEWS take clauses. On a line the order is fixed, whatever the order of the names:
    size, step, law, plant. With size, step and law, the looprange text is the zsize text with step and law.
    Known limit: the law and plant clauses are not valid for mirrored copies (HAI_MIRROR_SHARE): a mirrored command
    and measurement no longer follow the fitted law or plant. Do not combine them.
    """
    raw = (os.environ.get("HAI_TEXT_CLAUSES") or "").strip()
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    for n in names:
        if n not in CLAUSES:
            raise RuntimeError(f"HAI_TEXT_CLAUSES takes names from {CLAUSES}, got {n!r}")
    if names and series_view() not in CLAUSE_VIEWS:
        raise RuntimeError(f"HAI_TEXT_CLAUSES needs HAI_SERIES_VIEW=zsize or looprange, the view is {series_view()!r}")
    return names


def range_view(values, normal_range: float) -> list[float]:
    """The series as its change from its first seconds, in units of 2% of its normal range, signed-log squashed."""
    x = np.asarray(values, dtype=np.float64)
    unit = RANGE_UNIT * float(normal_range)
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError("the range view needs a positive `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
    d = (x - x[:RANGE_HEAD_S].mean()) / unit
    return (np.sign(d) * np.log1p(np.abs(d))).tolist()


def largest_change(values, normal_range: float) -> float:
    """The largest absolute change from the mean of the first RANGE_HEAD_S values, in range units, not squashed."""
    x = np.asarray(values, dtype=np.float64)
    unit = RANGE_UNIT * float(normal_range)
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError("the looprange view needs a positive `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
    return float(np.abs(x - x[:RANGE_HEAD_S].mean()).max() / unit)


def largest_step(values, normal_range: float) -> tuple[float, int]:
    """(size, second) of the largest one-second step, in range units; the second it lands on, the first on a tie."""
    x = np.asarray(values, dtype=np.float64)
    unit = RANGE_UNIT * float(normal_range)
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError("the looprange view needs a positive `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
    d = np.abs(np.diff(x / unit))
    i = int(np.argmax(d))
    return float(d[i]), i + 1


def quiet_units(rows: list[dict], measure=largest_change) -> dict:
    """{(version, loop, series name): (unit, quiet rows)} from the quiet rows (label normal, window_kind normal).

    The unit is the 95th percentile of `measure` (default `largest_change`) over the key's quiet rows, at least
    LOOPRANGE_FLOOR. Keys (None, loop, series name) hold the same over all versions together, for `unit_for`.
    """
    changes: dict = {}
    for r in rows:
        if r["label"] != "normal" or r["window_kind"] != "normal":
            continue
        for s in r["series"]:
            if "scale" not in s:
                raise RuntimeError("the looprange view needs `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
            c = measure(s["values"], s["scale"])
            for key in ((r["version"], r["loop"], s["name"]), (None, r["loop"], s["name"])):
                changes.setdefault(key, []).append(c)
    return {k: (max(LOOPRANGE_FLOOR, float(np.percentile(v, 95))), len(v)) for k, v in changes.items()}


def quiet_step_units(rows: list[dict]) -> dict:
    """quiet_units with the size of `largest_step` in place of `largest_change`. `unit_for` reads it the same way."""
    return quiet_units(rows, measure=lambda values, normal_range: largest_step(values, normal_range)[0])


def step_clause(values, normal_range: float, unit: float) -> str:
    """The step clause of the text: the largest one-second step in `unit`s, at most SIZE_CAP, and its second."""
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError(f"the step clause needs a positive unit, got {unit!r}")
    size, second = largest_step(values, normal_range)
    if size == 0:
        return "it never changes from one second to the next"
    return f"its largest one-second step is {min(size / float(unit), SIZE_CAP):.1f} times its usual one, at second {second}"


def _law_series(row: dict):
    """(target, measurement, first command) series of the row, or None if it has not exactly one target, exactly
    one measurement and at least one command."""
    roles = [s["role"] for s in row["series"]]
    if roles.count("target") != 1 or roles.count("measurement") != 1 or "command" not in roles:
        return None
    pick = {}
    for s in row["series"]:
        pick.setdefault(s["role"], s)
    return pick["target"], pick["measurement"], pick["command"]


def fit_laws(rows: list[dict]) -> dict:
    """{(version, loop): ((a, b, c), quiet rows, r2)}: the controller's law du = a*g + b*dg + c, fitted by least squares.

    g = target - measurement and u is the first command, raw values; du and dg are one-second changes. Fitted over
    all eligible quiet rows of the key (label normal, window_kind normal; see `_law_series`). Keys (None, loop) hold
    the same over all versions together, for `law_for`. r2 is 1 - SS_res / SS_tot of the fit on those rows, 0.0 if du
    never varies.
    """
    xs: dict = {}
    ys: dict = {}
    counts: dict = {}
    for r in rows:
        if r["label"] != "normal" or r["window_kind"] != "normal":
            continue
        found = _law_series(r)
        if found is None:
            continue
        t, m, u = (np.asarray(s["values"], dtype=np.float64) for s in found)
        g = t - m
        x = np.column_stack([g[1:], np.diff(g), np.ones(len(g) - 1)])
        for key in ((r["version"], r["loop"]), (None, r["loop"])):
            xs.setdefault(key, []).append(x)
            ys.setdefault(key, []).append(np.diff(u))
            counts[key] = counts.get(key, 0) + 1
    laws = {}
    for key in xs:
        x, y = np.vstack(xs[key]), np.concatenate(ys[key])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - x @ coef) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        laws[key] = (tuple(float(v) for v in coef), counts[key], r2)
    return laws


def law_for(laws: dict, version, loop):
    """The (a, b, c) of the key if it has at least LOOPRANGE_MIN_ROWS quiet rows, else that of all versions, else None;
    a law with r2 under LAW_MIN_R2 counts as None, and a key with enough rows but a poor fit does not fall back."""
    own = laws.get((version, loop))
    if own is not None and own[1] >= LOOPRANGE_MIN_ROWS:
        return own[0] if own[2] >= LAW_MIN_R2 else None
    pooled = laws.get((None, loop))
    return pooled[0] if pooled is not None and pooled[2] >= LAW_MIN_R2 else None


def wrong_way_seconds(t, m, u, scale_u: float, law) -> int:
    """Seconds where the command and the command the law would give are both beyond RANGE_UNIT of the command's
    range from their first RANGE_HEAD_S seconds, on opposite sides."""
    if scale_u is None or not np.isfinite(scale_u) or scale_u <= 0:
        raise RuntimeError(f"the law clause needs a positive `scale` on the command, got {scale_u!r}")
    a, b, c = law
    t, m, u = (np.asarray(v, dtype=np.float64) for v in (t, m, u))
    g = t - m
    uh = np.cumsum(np.concatenate([[u[0]], a * g[1:] + b * np.diff(g) + c]))
    du, duh = u - u[:RANGE_HEAD_S].mean(), uh - uh[:RANGE_HEAD_S].mean()
    thr = RANGE_UNIT * float(scale_u)
    return int(np.sum((np.abs(du) > thr) & (np.abs(duh) > thr) & (np.sign(du) != np.sign(duh))))


def law_clause(row: dict, laws: dict) -> Optional[str]:
    """The law clause of the first command's text, or None if the row is not eligible or its loop has no law."""
    found = _law_series(row)
    law = law_for(laws, row["version"], row["loop"]) if found is not None else None
    if law is None:
        return None
    t, m, u = found
    n = wrong_way_seconds(t["values"], m["values"], u["values"], u.get("scale"), law)
    if n == 0:
        return "it moves the way the controller would"
    return f"for {n} second{'' if n == 1 else 's'} it moves the opposite way to what the controller would do"


def _plant_design(row: dict):
    """(X, y) of the plant model for the row, or None if `_law_series` finds no target, measurement and command.

    Each series is its change from the mean of its first RANGE_HEAD_S seconds, in units of RANGE_UNIT of its scale.
    For each second i from RANGE_HEAD_S on, y is the measurement and the row of X is the first command at the lags of
    PLANT_LAGS, the target, and 1."""
    found = _law_series(row)
    if found is None:
        return None
    z = []
    for s in found:
        sc = s.get("scale")
        if sc is None or not np.isfinite(sc) or sc <= 0:
            raise RuntimeError(f"the plant clause needs a positive `scale` on the target, measurement and command, got {sc!r}")
        v = np.asarray(s["values"], dtype=np.float64)
        z.append((v - v[:RANGE_HEAD_S].mean()) / (RANGE_UNIT * float(sc)))
    tt, mm, uu = z
    n, h = len(mm), RANGE_HEAD_S
    x = np.column_stack([uu[h - lag:n - lag] for lag in PLANT_LAGS] + [tt[h:], np.ones(n - h)])
    return x, mm[h:]


def fit_plants(rows: list[dict]) -> dict:
    """{loop: (coefficients, res95, quiet rows, r2)}: the measurement explained by the first command and the target.

    Least squares over all eligible quiet rows of the loop (label normal, window_kind normal; see `_plant_design`),
    all versions together. res95 is the 95th percentile, over those rows, of the largest absolute residual of a row.
    r2 is 1 - SS_res / SS_tot of the fit on those rows, 0.0 if the measurement never varies.
    """
    found: dict = {}
    for r in rows:
        if r["label"] != "normal" or r["window_kind"] != "normal":
            continue
        d = _plant_design(r)
        if d is not None:
            found.setdefault(r["loop"], []).append(d)
    plants = {}
    for loop, ds in found.items():
        x, y = np.vstack([a for a, _ in ds]), np.concatenate([b for _, b in ds])
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - x @ coef) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        res95 = float(np.percentile([np.abs(b - a @ coef).max() for a, b in ds], 95))
        plants[loop] = (tuple(float(v) for v in coef), res95, len(ds), r2)
    return plants


def plant_for(plants: dict, loop):
    """The fit of the loop, or None if it has none, its r2 is under PLANT_MIN_R2 or its res95 is not positive and finite."""
    fit = plants.get(loop)
    if fit is None or fit[3] < PLANT_MIN_R2 or not np.isfinite(fit[1]) or fit[1] <= 0:
        return None
    return fit


def plant_ratio(row: dict, plants: dict) -> Optional[float]:
    """The row's largest absolute residual under its loop's plant model, in res95 units, at most SIZE_CAP, one
    decimal; None if the row is not eligible or its loop has no plant model (see `plant_for`)."""
    d = _plant_design(row)
    fit = plant_for(plants, row["loop"]) if d is not None else None
    if fit is None:
        return None
    x, y = d
    ratio = float(np.abs(y - x @ np.asarray(fit[0])).max()) / fit[1]
    return float(f"{min(ratio, SIZE_CAP):.1f}")


def plant_clause(row: dict, plants: dict) -> Optional[str]:
    """The plant clause of the measurement's text, or None if the row has no `plant_ratio`."""
    ratio = plant_ratio(row, plants)
    if ratio is None:
        return None
    return f"it strays from what its command predicts by {ratio:.1f} times its usual amount"


def unit_for(units: dict, version, loop, name) -> float:
    """The unit of the key if it has at least LOOPRANGE_MIN_ROWS quiet rows, else the unit of all versions, else 1.0."""
    own = units.get((version, loop, name))
    if own is not None and own[1] >= LOOPRANGE_MIN_ROWS:
        return own[0]
    pooled = units.get((None, loop, name))
    return pooled[0] if pooled is not None else 1.0


def relative_size(values, normal_range: float, unit: float) -> float:
    """`largest_change` divided by `unit`, how much the series normally moves in a quiet window (see `quiet_units`)."""
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError(f"the size needs a positive unit, got {unit!r}")
    return largest_change(values, normal_range) / float(unit)


def size_clause(values, normal_range: float, unit: float) -> str:
    """The size clause of the text: `relative_size` in `unit`s, at most SIZE_CAP, one decimal."""
    size = min(relative_size(values, normal_range, unit), SIZE_CAP)
    return f"its largest change is {size:.1f} times its usual quiet change"


def looprange_view(values, normal_range: float, unit: float) -> list[float]:
    """The range view with the change divided by `unit`, how much the series normally moves (see `quiet_units`)."""
    x = np.asarray(values, dtype=np.float64)
    scale_unit = RANGE_UNIT * float(normal_range)
    if not np.isfinite(scale_unit) or scale_unit <= 0:
        raise RuntimeError("the range view needs a positive `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
    if not np.isfinite(unit) or unit <= 0:
        raise RuntimeError(f"the looprange view needs a positive unit, got {unit!r}")
    d = (x - x[:RANGE_HEAD_S].mean()) / scale_unit / unit
    return (np.sign(d) * np.log1p(np.abs(d))).tolist()


ORDERS = ("label_last", "label_first", "label_only")
START_SENTENCE = re.compile(r"\s*Change starts near second \d+\.")


def answer_order() -> str:
    """Where the label sits in the answer the model learns: HAI_ANSWER_ORDER, `label_last` (default) or `label_first`.

    label_last   the answer as exported: the sentences, then `Answer: attack` or `Answer: normal`. The
                 model has to commit to a label while it writes the first sentence.
    label_first  `attack. Change starts near second S. <sentences>` or `normal. <sentences>`, right after
                 the prompt's closing `Answer:`. The label is the first word the model writes, so the
                 decision is one clear step.
    label_only   `attack` or `normal` and nothing else. Every bit of the training signal is about the
                 decision, none about wording. No start second and no sentence.
    """
    order = (os.environ.get("HAI_ANSWER_ORDER") or "label_last").strip()
    if order not in ORDERS:
        raise RuntimeError(f"HAI_ANSWER_ORDER must be one of {ORDERS}, got {order!r}")
    return order


def label_first(answer: str) -> str:
    """`<sentences> Change starts near second S. Answer: attack` -> `attack. Change starts near second S. <sentences>`."""
    body, sep, label = answer.rpartition("Answer: ")
    label = label.strip()
    if not sep or label not in ("attack", "normal"):
        raise ValueError(f"not an exported answer: {answer!r}")
    body = body.strip()
    start = ""
    m = START_SENTENCE.search(body)
    if m:
        start, body = m.group().strip(), (body[:m.start()] + body[m.end():]).strip()
    return " ".join(part for part in (label + ".", start, body) if part)


def label_only(answer: str) -> str:
    """`<sentences> Answer: attack` -> `attack`."""
    label = answer.rpartition("Answer: ")[2].strip()
    if label not in ("attack", "normal"):
        raise ValueError(f"not an exported answer: {answer!r}")
    return label


def attack_share() -> Optional[float]:
    """Share of attack rows in the training rows: HAI_ATTACK_SHARE, a float in (0, 1). Unset or empty: None."""
    raw = (os.environ.get("HAI_ATTACK_SHARE") or "").strip()
    if not raw:
        return None
    try:
        share = float(raw)
    except ValueError:
        share = None
    if share is None or not 0 < share < 1:
        raise RuntimeError(f"HAI_ATTACK_SHARE must be a number strictly between 0 and 1, got {raw!r}")
    return share


def _draw(group: list[int], n: int, rng) -> list[int]:
    """n positions from group: each one n // len(group) times, plus n % len(group) more drawn without replacement."""
    if n == 0:
        return []
    times, rest = divmod(n, len(group))
    extra = rng.choice(len(group), size=rest, replace=False).tolist() if rest else []
    counts = [times] * len(group)
    for i in extra:
        counts[i] += 1
    return [p for p, c in zip(group, counts) for _ in range(c)]


def with_attack_share(rows: list[dict], share: float) -> list[dict]:
    """The rows redrawn so that round(share * len(rows)) of them have label attack; same length, fixed seed.

    Attack rows and the two groups of normal rows (by `window_kind`) are each drawn with `_draw`; the
    two normal groups keep their proportion. The output is in input order, repeats of a row side by side.
    Repeated rows are the same dict.
    """
    n = len(rows)
    attack = [i for i, r in enumerate(rows) if r["label"] == "attack"]
    near = [i for i, r in enumerate(rows) if r["label"] != "attack" and r["window_kind"] == "attack"]
    quiet = [i for i, r in enumerate(rows) if r["label"] != "attack" and r["window_kind"] != "attack"]
    if not attack or not (near or quiet):
        raise RuntimeError(f"HAI_ATTACK_SHARE needs both attack and normal rows, got {len(attack)} attack "
                           f"and {len(near) + len(quiet)} normal")
    rng = np.random.default_rng(ATTACK_SHARE_SEED)
    want_attack = round(share * n)
    want_normal = n - want_attack
    if not near:
        want_near = 0
    elif not quiet:
        want_near = want_normal
    else:
        want_near = round(want_normal * len(near) / (len(near) + len(quiet)))
    pick = _draw(attack, want_attack, rng) + _draw(near, want_near, rng) + _draw(quiet, want_normal - want_near, rng)
    return [rows[i] for i in sorted(pick)]


def mirror_share() -> Optional[float]:
    """Share of the training rows that get a mirrored copy: HAI_MIRROR_SHARE, a float in (0, 1]. Unset or empty: None."""
    raw = (os.environ.get("HAI_MIRROR_SHARE") or "").strip()
    if not raw:
        return None
    try:
        share = float(raw)
    except ValueError:
        share = None
    if share is None or not 0 < share <= 1:
        raise RuntimeError(f"HAI_MIRROR_SHARE must be a number greater than 0 and at most 1, got {raw!r}")
    return share


MIRROR_WORDS = {"rises": "falls", "falls": "rises"}   # every direction word src/hai/answers.py writes
MIRROR_PATTERN = re.compile(r"\b(" + "|".join(MIRROR_WORDS) + r")\b")


def mirror_answer(answer: str) -> str:
    """The answer of the series flipped upside down: rises and falls swapped in one pass, the rest as it is."""
    return MIRROR_PATTERN.sub(lambda m: MIRROR_WORDS[m.group()], answer)


def mirror_row(row: dict) -> dict:
    """A new row: every series flipped around the mean of its first RANGE_HEAD_S values, the answer mirrored."""
    series = []
    for s in row["series"]:
        x = np.asarray(s["values"], dtype=np.float64)
        series.append({**s, "values": (2 * x[:RANGE_HEAD_S].mean() - x).tolist()})
    return {**row, "series": series, "answer": mirror_answer(row["answer"]), "mirrored": True}


def with_mirrors(rows: list[dict], share: float) -> list[dict]:
    """The rows, then mirrored copies of round(share * len(rows)) of them (fixed seed, in the order of their originals)."""
    rng = np.random.default_rng(MIRROR_SEED)
    pick = sorted(rng.choice(len(rows), size=round(share * len(rows)), replace=False).tolist())
    return list(rows) + [mirror_row(rows[i]) for i in pick]


def scale(values) -> tuple[float, float, list[float]]:
    """(mean, spread, scaled values). A flat series has zero spread: its scaled values are all zero."""
    x = np.asarray(values, dtype=np.float64)
    mean, std = float(x.mean()), float(x.std())
    if std <= MIN_SPREAD * max(1.0, abs(mean)):
        return mean, 0.0, [0.0] * len(x)
    return mean, std, ((x - mean) / std).tolist()


class HAIAttackQADataset(QADataset):
    looprange_units = None   # set by _load_splits for the looprange and zsize views: quiet_units of the file's training rows
    step_units = None        # set by _load_splits for the clause `step`: quiet_step_units of the file's training rows
    control_laws = None      # set by _load_splits for the clause `law`: fit_laws of the file's training rows
    plant_models = None      # set by _load_splits for the clause `plant`: fit_plants of the file's training rows

    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        folder = os.environ.get("HAI_JSONL_DIR")
        if not folder:
            raise RuntimeError("Set HAI_JSONL_DIR to the folder with the files from `uv run hai-export`")
        folder = Path(folder)
        train = _read_jsonl(folder / "train.jsonl")
        self.__class__.looprange_units = self._units(train) if series_view() in ("looprange", "zsize") else None
        self.__class__.step_units = self._step_units(train) if "step" in text_clauses() else None
        self.__class__.control_laws = self._laws(train) if "law" in text_clauses() else None
        self.__class__.plant_models = self._plants(train) if "plant" in text_clauses() else None
        share = attack_share()
        mirror = mirror_share()
        overfit = int(os.environ.get("HAI_OVERFIT_ROWS", "0") or 0)
        if overfit > 0:
            rng = np.random.default_rng(OVERFIT_SEED)
            pick = sorted(rng.choice(len(train), size=min(overfit, len(train)), replace=False).tolist())
            rows = [train[i] for i in pick]
            print(f"HAI overfit check: train, validation and test are the same {len(rows)} training rows")
            train = rows if share is None else self._with_share(rows, share)
            if mirror is not None:
                train = self._with_mirrors(train, mirror)
            return train, list(rows), list(rows)
        val = _read_jsonl(folder / "validation.jsonl")
        test = _read_jsonl(folder / os.environ.get("HAI_TEST_FILE", "test.jsonl"))
        if share is not None:
            train = self._with_share(train, share)
        if mirror is not None:
            train = self._with_mirrors(train, mirror)
        print(f"HAI rows: train {len(train)}, validation {len(val)}, test {len(test)}")
        return train, val, test

    @staticmethod
    def _units(rows: list[dict]) -> dict:
        """quiet_units, and one line that says what it found."""
        units = quiet_units(rows)
        own = [u for (version, _, _), (u, _) in units.items() if version is not None]
        quiet = sum(r["label"] == "normal" and r["window_kind"] == "normal" for r in rows)
        span = f"unit {min(own):.2f} to {max(own):.1f}" if own else "no units"
        print(f"HAI looprange units: {len(own)} series keys from {quiet} quiet training rows, {span}")
        return units

    @staticmethod
    def _step_units(rows: list[dict]) -> dict:
        """quiet_step_units, and one line that says what it found."""
        units = quiet_step_units(rows)
        own = [u for (version, _, _), (u, _) in units.items() if version is not None]
        quiet = sum(r["label"] == "normal" and r["window_kind"] == "normal" for r in rows)
        span = f"unit {min(own):.2f} to {max(own):.1f}" if own else "no units"
        print(f"HAI step units: {len(own)} series keys from {quiet} quiet training rows, {span}")
        return units

    @staticmethod
    def _laws(rows: list[dict]) -> dict:
        """fit_laws, and one line that says what it found."""
        laws = fit_laws(rows)
        own = sum(version is not None for version, _ in laws)
        good = sum(r2 >= LAW_MIN_R2 for (version, _), (_, _, r2) in laws.items() if version is not None)
        quiet = sum(n for (version, _), (_, n, _) in laws.items() if version is None)
        print(f"HAI control laws: {good} of {own} (version, loop) keys explain the command (R2 at least {LAW_MIN_R2}), "
              f"from {quiet} eligible quiet training rows")
        return laws

    @staticmethod
    def _plants(rows: list[dict]) -> dict:
        """fit_plants, and one line that says what it found."""
        plants = fit_plants(rows)
        good = sum(plant_for(plants, loop) is not None for loop in plants)
        quiet = sum(n for _, _, n, _ in plants.values())
        print(f"HAI plant models: {good} of {len(plants)} loops explain the measurement (R2 at least {PLANT_MIN_R2}), "
              f"from {quiet} eligible quiet training rows")
        return plants

    @staticmethod
    def _with_share(rows: list[dict], share: float) -> list[dict]:
        """with_attack_share, and one line that says what it did."""
        out = with_attack_share(rows, share)
        before = sum(r["label"] == "attack" for r in rows)
        after = sum(r["label"] == "attack" for r in out)
        print(f"HAI attack share {share}: train rows {len(out)} (attack {after}, normal {len(out) - after}; "
              f"before: attack {before}, normal {len(rows) - before})")
        return out

    @staticmethod
    def _with_mirrors(rows: list[dict], share: float) -> list[dict]:
        """with_mirrors, and one line that says what it did."""
        out = with_mirrors(rows, share)
        print(f"HAI mirror share {share}: train rows {len(out)} ({len(out) - len(rows)} mirrored copies added to {len(rows)})")
        return out

    def _get_answer(self, row) -> str:
        order = answer_order()
        if order == "label_first":
            return label_first(row["answer"])
        if order == "label_only":
            return label_only(row["answer"])
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        names = [f"the {s['name']}" for s in row["series"]]
        n = len(row["series"][0]["values"])
        shown = {"range": "Each series is shown as its change from its first 30 seconds, "
                          "measured against its normal range. ",
                 "looprange": "Each series is shown as its change from its first 30 seconds, "
                              "measured against how much it normally moves in this loop. ",
                 "zsize": "Each series is shown scaled to its own mean and spread. Its text says how large its "
                          "change is against what is usual for it in this loop. "}.get(series_view(), "")
        return (
            f"You are given {n} seconds of the {row['loop_name']} control loop of a power plant testbed, "
            f"one reading per second. The series are {_join(names)}. {shown}"
            + {"label_first": "Say first whether the loop is under attack or normal: answer 'attack' or 'normal'. "
                              "If it is under attack, say next the second where the change starts. "
                              "Then say in a sentence or two how the series move.",
               "label_only": "Say whether the loop is under attack or normal: "
                             "answer with the one word 'attack' or 'normal'.",
               "label_last": "Say whether the loop behaves normally or is under attack. If it is under attack, "
                             "say the second where the change starts. "
                             "End with 'Answer: attack' or 'Answer: normal'."}[answer_order()]
        )

    def _get_post_prompt(self, row) -> str:
        return "Answer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prompts = []
        view = series_view()
        clauses = text_clauses()
        first_command = next((s for s in row["series"] if s["role"] == "command"), None)
        for s in row["series"]:
            if view == "range":
                if "scale" not in s:
                    raise RuntimeError("the range view needs `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
                prompts.append(TextTimeSeriesPrompt(f"This is the {s['name']}, {ROLE_WORDS[s['role']]}:",
                                                    range_view(s["values"], s["scale"])))
                continue
            if view == "looprange":
                if "scale" not in s:
                    raise RuntimeError("the looprange view needs `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
                if self.looprange_units is None:
                    raise RuntimeError("the looprange view needs the units of the training rows: _load_splits sets them")
                unit = unit_for(self.looprange_units, row["version"], row["loop"], s["name"])
                text = self._clause_text(row, s, clauses, first_command, size="size" in clauses, unit=unit)
                prompts.append(TextTimeSeriesPrompt(text, looprange_view(s["values"], s["scale"], unit)))
                continue
            if view == "zsize":
                if "scale" not in s:
                    raise RuntimeError("the zsize view needs `scale` on every series: rerun `uv run hai-build` and `uv run hai-export`")
                if self.looprange_units is None:
                    raise RuntimeError("the zsize view needs the units of the training rows: _load_splits sets them")
                unit = unit_for(self.looprange_units, row["version"], row["loop"], s["name"])
                text = self._clause_text(row, s, clauses, first_command, size=True, unit=unit)
                prompts.append(TextTimeSeriesPrompt(text, scale(s["values"])[2]))
                continue
            mean, std, scaled = scale(s["values"])
            text = (f"This is the {s['name']}, {ROLE_WORDS[s['role']]}; "
                    f"it has mean {mean:.4f} and std {std:.4f}:")
            prompts.append(TextTimeSeriesPrompt(text, scaled))
        return prompts

    def _clause_text(self, row, s, clauses, first_command, size: bool, unit: float) -> str:
        """`This is the {name}, {role words}`, the clauses that are on (size, step, law, plant, in this order), and `:`."""
        text = f"This is the {s['name']}, {ROLE_WORDS[s['role']]}"
        if size:
            text += "; " + size_clause(s["values"], s["scale"], unit)
        if "step" in clauses:
            if self.step_units is None:
                raise RuntimeError("the step clause needs the step units of the training rows: _load_splits sets them")
            step_unit = unit_for(self.step_units, row["version"], row["loop"], s["name"])
            text += "; " + step_clause(s["values"], s["scale"], step_unit)
        if "law" in clauses and s is first_command:
            if self.control_laws is None:
                raise RuntimeError("the law clause needs the control laws of the training rows: _load_splits sets them")
            law = law_clause(row, self.control_laws)
            if law is not None:
                text += "; " + law
        if "plant" in clauses and s["role"] == "measurement":
            if self.plant_models is None:
                raise RuntimeError("the plant clause needs the plant models of the training rows: _load_splits sets them")
            plant = plant_clause(row, self.plant_models)
            if plant is not None:
                text += "; " + plant
        return text + ":"

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["example_id"] = row["example_id"]
        sample["label"] = row["label"]
        return sample


if __name__ == "__main__":
    ds = HAIAttackQADataset("train", EOS_TOKEN="")
    print(len(ds), "training samples")
    s = ds[0]
    print(s["pre_prompt"])
    for t in s["time_series_text"]:
        print(t)
    print(s["post_prompt"], s["answer"])
