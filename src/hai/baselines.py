"""Step 10: baselines, scored on the validation split only, per attack.

1. Range alarm: flag a loop when its measurement leaves the range seen in that
   version's normal file.
2. Gradient boosting on per-loop window statistics (shape of each series and how
   the measurement relates to the target and command; no absolute levels),
   trained on the training sample (`train_use`) of all versions together.
3. The same, plus the shape of the other five loops' measurements in the same
   window. This tests whether the neighbouring loops carry evidence the loop
   view hides.

Per-attack scoring uses the one `scored` window of each attack, whose start
second is random. The test split is never read here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from hai.dataset import LOOPS

SLOTS = {"target": 2, "measurement": 1, "command": 2}  # series per role, at most
SEED = 0


def range_alarm(ex: pd.DataFrame, ranges: dict) -> np.ndarray:
    out = np.zeros(len(ex), bool)
    for i, r in enumerate(ex.itertuples()):
        for tag, role, x in zip(r.tags, r.roles, r.values):
            if role == "measurement":
                lo, hi = ranges[(r.version, tag)]
                out[i] |= bool((np.asarray(x) < lo).any() or (np.asarray(x) > hi).any())
    return out


def _stats(x: np.ndarray, scale: float) -> list[float]:
    x = np.asarray(x, "float64") / scale
    d = np.diff(x)
    head = x[:30].mean()
    return [x.std(), x.max() - x.min(), x[-30:].mean() - head, np.abs(d).max(), np.abs(x - head).max(),
            np.abs(d).sum(), float((d != 0).mean())]


N_STATS = 7


def _relations(r, scales: dict) -> list[float]:
    """How the measurement relates to the target and the first command (spoofing breaks these)."""
    by = {}
    for t, ro, x in zip(r.tags, r.roles, r.values):
        by.setdefault(ro, []).append((t, np.asarray(x, "float64") / scales[(r.version, t)]))
    m = by["measurement"][0][1]
    out = []
    if "target" in by:
        e = m - by["target"][0][1]
        out += [np.abs(e - e[:60].mean()).max(), e[-60:].mean() - e[:60].mean()]
    else:
        out += [np.nan, np.nan]
    if "command" in by:
        c = by["command"][0][1]
        dm, dc = np.diff(m), np.diff(c)
        ok = dm.std() > 0 and dc.std() > 0
        out += [float(np.corrcoef(dm, dc)[0, 1]) if ok else np.nan]
        h = len(dm) // 2
        a = float(np.corrcoef(dm[:h], dc[:h])[0, 1]) if dm[:h].std() > 0 and dc[:h].std() > 0 else np.nan
        b = float(np.corrcoef(dm[h:], dc[h:])[0, 1]) if dm[h:].std() > 0 and dc[h:].std() > 0 else np.nan
        out += [b - a]
    else:
        out += [np.nan, np.nan]
    return out


def features(ex: pd.DataFrame, scales: dict) -> np.ndarray:
    """Per role, up to two series; missing series are NaN. Plus one-hot loop."""
    rows = []
    for r in ex.itertuples():
        f = []
        for role, n in SLOTS.items():
            xs = [(t, x) for t, ro, x in zip(r.tags, r.roles, r.values) if ro == role][:n]
            for k in range(n):
                f += _stats(xs[k][1], scales[(r.version, xs[k][0])]) if k < len(xs) else [np.nan] * N_STATS
        f += _relations(r, scales)
        f += [float(r.loop == l) for l in LOOPS]
        rows.append(f)
    return np.asarray(rows, "float64")


def neighbour_features(ex: pd.DataFrame, scales: dict) -> np.ndarray:
    """Shape statistics of the other five loops' measurements in the same window, in LOOPS order.

    The example's own slot is NaN. `ex` must hold every loop example of each window.
    """
    meas = {}
    for r in ex.itertuples():
        for t, ro, x in zip(r.tags, r.roles, r.values):
            if ro == "measurement":
                meas[(r.window_id, r.loop)] = (t, x)
    rows = []
    for r in ex.itertuples():
        f = []
        for loop in LOOPS:
            got = meas.get((r.window_id, loop)) if loop != r.loop else None
            f += _stats(got[1], scales[(r.version, got[0])]) if got else [np.nan] * N_STATS
        rows.append(f)
    return np.asarray(rows, "float64")


def score(ex: pd.DataFrame, flagged: np.ndarray) -> pd.DataFrame:
    """Per version, on the scored window of each attack.

    attacks: attacks with a scored window and an attacked loop among the six.
    found_on_attacked_loop / detected_any_loop / right_loop_only: counts of attacks.
    visible_* and hidden_*: attacked loop examples in those windows, split by whether
    their series show a visible change, and how many of each were flagged.
    False alarms are per 100 loop examples from normal windows.
    """
    ex = ex.assign(flag=flagged)
    rows = []
    for version, g in list(ex.groupby("version")) + [("all", ex)]:
        att = g[(g.window_kind == "attack") & g.scored]
        n = found = right = detected = 0
        vis = vis_f = hid = hid_f = 0
        for _, w in att.groupby("window_id"):
            hit = w[w.label == "attack"]
            if hit.empty:
                continue
            n += 1
            found += bool(hit.flag.any())
            detected += bool(w.flag.any())
            right += bool(hit.flag.all() and not w[w.label == "normal"].flag.any())
            v, h = hit[~hit.no_visible_change], hit[hit.no_visible_change]
            vis, vis_f, hid, hid_f = vis + len(v), vis_f + int(v.flag.sum()), hid + len(h), hid_f + int(h.flag.sum())
        normal = g[g.window_kind == "normal"]
        rows.append({"version": version, "attacks": n, "found_on_attacked_loop": found,
                     "detected_any_loop": detected, "right_loop_only": right,
                     "visible_loops": vis, "visible_flagged": vis_f, "hidden_loops": hid, "hidden_flagged": hid_f,
                     "normal_examples": len(normal),
                     "false_alarms_per_100": round(100 * normal.flag.mean(), 1) if len(normal) else np.nan})
    return pd.DataFrame(rows)


def run(ex: pd.DataFrame, ranges: dict, scales: dict) -> dict[str, pd.DataFrame]:
    if (ex.split == "test").any():
        raise ValueError("baselines must not see the test split")
    ex = ex.reset_index(drop=True)
    train, va = ex.train_use.to_numpy(), (ex.split == "validation").to_numpy()
    y = (ex.label == "attack").to_numpy()
    val = ex[va]
    out = {"range_alarm": score(val, range_alarm(val, ranges))}
    loop_only = features(ex, scales)
    variants = {"gradient_boosting": loop_only,
                "gradient_boosting_with_neighbours": np.c_[loop_only, neighbour_features(ex, scales)]}
    for name, x in variants.items():
        model = HistGradientBoostingClassifier(random_state=SEED).fit(x[train], y[train])
        out[name] = score(val, model.predict(x[va]).astype(bool))
    return out
