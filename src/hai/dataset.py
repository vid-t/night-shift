"""Steps 6, 7 and 9: windows, loop examples and splits.

A window is 300 seconds of one file. Attack windows: six per attack. The attack
starts at a random second between 30 and 270, one draw from each of six equal
bands, so the start second cannot be guessed. One of the six, picked at random,
is the scored window of that attack. Normal windows: at least 600 seconds from
any attack, half from attack files, half from the version's normal file.
A loop example is one window seen through one loop.

Table v2 (hai.config.use_table) changes two things here. Where HAI has one flag per process
(20.07, 21.03), a loop's attack seconds are the seconds of its attack's run in which the flag of the
loop's own process is on; a loop is labelled `attack` when one of them falls in the window, and the
example lists every run (`attack_runs`). And a window that is an exact copy of a window in another
file of the version (same start, same rows) is dropped.
"""
import numpy as np
import pandas as pd

from hai.config import DATA_FILES, NORMAL_FILES, PROCESS_FLAG_VERSIONS, split_of, table_version
from hai.loaders import load, tag_map

WIN = 300
N_PER_ATTACK = 6
START_LO, START_HI = 30, 270  # the attack starts at a random second in this range
GAP = 600
SEED = 20261120
NORMAL_PER_VERSION = 750  # 4 versions x 750 = about 3,000 normal windows
LOOPS = ("P1-PC", "P1-LC", "P1-FC", "P1-TC", "P2-SC", "P3-LC")
LOOP_NAMES = {"P1-PC": "boiler pressure", "P1-LC": "boiler level", "P1-FC": "boiler flow",
              "P1-TC": "boiler temperature", "P2-SC": "turbine speed", "P3-LC": "water level"}
# label of each series in the answer text; roles with two series get a qualifier
SERIES_LABEL = {"P1_B4022": "demand target", "P1_B4002": "outlet target",
                "P1_PCV01D": "first valve command", "P1_PCV02D": "second valve command",
                "P1_FCV01D": "first valve command", "P1_FCV02D": "second valve command",
                "P3_LH01": "high-level target", "P3_LL01": "low-level target",
                "P3_LCP01D": "pump command", "P3_LCV01D": "valve command"}


def loop_tags(version: str) -> dict[str, list[tuple[str, str]]]:
    """loop -> [(standard tag, role)] for the tags that version records."""
    tm = tag_map()[lambda d: (d.version == version) & (d.source != "")]
    return {loop: list(zip(g.standard, g.role)) for loop, g in tm.groupby("loop", sort=False)}


def file_spans(version: str) -> pd.DataFrame:
    rows = []
    for stem in DATA_FILES[version]:
        t = load(version, stem)["time"]
        rows.append({"file": stem, "first": t.iloc[0], "last": t.iloc[-1], "rows": len(t)})
    return pd.DataFrame(rows).set_index("file")


def attack_windows(version: str, attacks: pd.DataFrame, spans: pd.DataFrame, rng) -> pd.DataFrame:
    """Six windows per attack, one start second drawn from each of six equal bands.

    Fixed start seconds would let a model answer the start without reading the curves.
    One window per attack is marked `scored`: scoring uses it, so the true start second
    of the scored window is spread over the whole range.
    """
    edges = np.linspace(START_LO, START_HI, N_PER_ATTACK + 1).astype(int)
    rows = []
    for _, a in attacks.iterrows():
        f = spans.loc[a.file]
        mine = []
        for k in range(N_PER_ATTACK):
            last_band = k == N_PER_ATTACK - 1
            off = int(rng.integers(edges[k], edges[k + 1] + (1 if last_band else 0)))
            start = a.flag_start - pd.Timedelta(seconds=off)
            if start < f["first"] or start + pd.Timedelta(seconds=WIN - 1) > f["last"]:
                continue
            mine.append({"version": version, "file": a.file, "kind": "attack", "anchor": a.attack_id,
                         "offset": off, "start": start, "scored": False})
        if mine:
            mine[int(rng.integers(len(mine)))]["scored"] = True
        rows += mine
    return pd.DataFrame(rows)


def normal_windows(version: str, steps: pd.DataFrame, spans: pd.DataFrame, rng) -> pd.DataFrame:
    """Non-overlapping 300 s windows on a grid with a random phase, at least GAP s from any attack."""
    def candidates(stem):
        f = spans.loc[stem]
        phase = int(rng.integers(0, WIN))
        starts = pd.date_range(f["first"] + pd.Timedelta(seconds=phase),
                               f["last"] - pd.Timedelta(seconds=WIN - 1), freq=f"{WIN}s")
        ends = starts + pd.Timedelta(seconds=WIN - 1)
        ok = np.ones(len(starts), bool)
        for _, s in steps[steps.file == stem].iterrows():
            far = (ends <= s.flag_start - pd.Timedelta(seconds=GAP)) | (starts >= s.flag_end + pd.Timedelta(seconds=GAP))
            ok &= np.asarray(far)
        return list(starts[ok])

    half = NORMAL_PER_VERSION // 2
    rows = []
    normal = NORMAL_FILES[version]
    pool = [(stem, s) for stem in DATA_FILES[version] if stem != normal for s in candidates(stem)]
    for i in rng.choice(len(pool), size=min(half, len(pool)), replace=False):
        rows.append({"file": pool[i][0], "start": pool[i][1]})
    pool = [(normal, s) for s in candidates(normal)]
    for i in rng.choice(len(pool), size=min(half, len(pool)), replace=False):
        rows.append({"file": pool[i][0], "start": pool[i][1]})
    out = pd.DataFrame(rows)
    out.insert(0, "version", version)
    out["kind"], out["anchor"], out["offset"], out["scored"] = "normal", "", -1, False
    return out


def runs_of(flag: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of ones, both inclusive."""
    f = np.concatenate([[0], np.asarray(flag, "int8") != 0, [0]]).astype("int8")
    d = np.diff(f)
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def step_runs(version: str, steps: pd.DataFrame) -> list[list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """For each attack step, the runs of seconds that count as attacked for its loop.

    v1, and versions without process flags: the whole flag run of the attack. v2 in 20.07 and 21.03:
    the runs inside the attack's flag run in which the flag of the loop's process (attack_P1, _P2,
    _P3) is on. HAI's process flags pulse during some attacks, so one attack can give several runs.
    """
    if table_version() == "v1" or version not in PROCESS_FLAG_VERSIONS:
        return [[(s.flag_start, s.flag_end)] for s in steps.itertuples()]
    out = []
    for s in steps.itertuples():
        d = load(version, s.file)
        i0 = int((s.flag_start - d.time.iloc[0]).total_seconds())
        i1 = int((s.flag_end - d.time.iloc[0]).total_seconds())
        assert d.time.iloc[i0] == s.flag_start and d.time.iloc[i1] == s.flag_end
        flag = d[f"attack_{s.loop.split('-')[0]}"].to_numpy()[i0:i1 + 1]
        out.append([(d.time.iloc[i0 + a], d.time.iloc[i0 + b]) for a, b in runs_of(flag)])
    return out


def exact_copies(version: str, wins: pd.DataFrame, tables: dict) -> pd.Series:
    """For each window, the file of the window it copies exactly, or "".

    A copy has the same start as a window of another file of the version and the same rows (every
    tidy column). HAI 21.03 train1 and test3 share 12 hours of identical rows, so their windows could
    coincide. The window in the version's normal file is kept; between two attack files, the first in
    DATA_FILES.
    """
    order = {stem: (stem != NORMAL_FILES[version], DATA_FILES[version].index(stem)) for stem in DATA_FILES[version]}
    out = pd.Series("", index=wins.index)
    for _, group in wins.groupby("start"):
        if group.file.nunique() < 2:
            continue
        rows = {}
        for w in group.itertuples():
            d = tables[w.file]
            i0 = int((w.start - d.time.iloc[0]).total_seconds())
            rows[w.Index] = d.iloc[i0:i0 + WIN].drop(columns="time").reset_index(drop=True)
        ranked = sorted(group.itertuples(), key=lambda w: order[w.file])
        for i, w in enumerate(ranked):
            for keep in ranked[:i]:
                if keep.file != w.file and rows[w.Index].equals(rows[keep.Index]) and not out[keep.Index]:
                    out[w.Index] = keep.file
                    break
    return out


def loop_seconds(on: pd.DataFrame, w) -> list[tuple[int, int]]:
    """The attacked runs of one loop in window `w`, in seconds from the window start, both ends inclusive.

    Union of the runs of every step of the loop that touches the window. The last second can lie
    after the window end.
    """
    runs = [((a - w.start).total_seconds(), (b - w.start).total_seconds()) for rs in on.runs for a, b in rs
            if a <= w.end and b >= w.start]
    if not runs:
        return []
    lo, hi = int(min(a for a, _ in runs)), int(max(b for _, b in runs))
    mask = np.zeros(hi - lo + 1, "int8")
    for a, b in runs:
        mask[int(a) - lo:int(b) - lo + 1] = 1
    return [(lo + a, lo + b) for a, b in runs_of(mask)]


def build_examples(version: str, attacks: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (windows, loop examples) for one version.

    attacks: checked attacks (one row each, with file, flag_start, flag_duration).
    steps: attack steps joined with their flag runs (file, flag_start, flag_end, loop).
    """
    rng = np.random.default_rng([SEED, int(version.replace(".", ""))])
    rng_starts = np.random.default_rng([SEED, int(version.replace(".", "")), 1])  # own stream: normal windows stay put
    spans = file_spans(version)
    wins = pd.concat([attack_windows(version, attacks, spans, rng_starts),
                      normal_windows(version, steps, spans, rng)], ignore_index=True)
    wins["scored"] = wins["scored"].astype(bool)
    wins["end"] = wins["start"] + pd.Timedelta(seconds=WIN - 1)
    wins["split"] = [split_of(version, f) or ("train" if f == NORMAL_FILES[version] else None) for f in wins.file]
    tags = loop_tags(version)
    tables = {stem: load(version, stem) for stem in wins.file.unique()}
    v2 = table_version() != "v1"
    if v2:
        steps = steps.assign(runs=step_runs(version, steps))
        wins["copy_of"] = exact_copies(version, wins, tables)
    examples, keep = [], []
    for w in wins.itertuples():
        if v2 and w.copy_of:
            keep.append(False)  # an exact copy of a window in another file
            continue
        hit = steps[(steps.file == w.file) & (steps.flag_start <= w.end) & (steps.flag_end >= w.start)]
        covered = hit[hit.loop.isin(LOOPS)]
        if len(hit) and covered.empty:
            keep.append(False)  # only P2-TC, P1-CC, P1-HC or other uncovered loops attacked
            continue
        keep.append(True)
        d = tables[w.file]
        i0 = int((w.start - d.time.iloc[0]).total_seconds())
        seg = d.iloc[i0:i0 + WIN]
        assert len(seg) == WIN and seg.time.iloc[0] == w.start
        for loop in LOOPS:
            on = covered[covered.loop == loop]
            if v2:
                runs = loop_seconds(on, w)
                ids = sorted({a for a, rs in zip(on.attack_id, on.runs)
                              if any(x <= w.end and y >= w.start for x, y in rs)})
                examples.append({
                    "window_id": f"{version}/{w.file}/{w.start:%Y%m%dT%H%M%S}",
                    "version": version, "file": w.file, "split": w.split, "window_kind": w.kind,
                    "anchor": w.anchor, "offset": w.offset, "scored": bool(w.scored), "start": w.start, "loop": loop,
                    "label": "attack" if runs else "normal",
                    "attack_ids": " ".join(ids),
                    "loop_uncertain": bool(hit.note.str.contains("loop uncertain").any()),
                    "attack_start_s": runs[0][0] if runs else -1,
                    "attack_end_s": runs[-1][1] if runs else -1,
                    "attack_runs": " ".join(f"{a}-{b}" for a, b in runs),
                    "tags": [t for t, _ in tags.get(loop, [])],
                    "roles": [r for _, r in tags.get(loop, [])],
                    "values": [seg[t].to_numpy("float32") for t, _ in tags.get(loop, [])],
                })
                continue
            label = "attack" if len(on) else "normal"
            first = on.sort_values("flag_start").iloc[0] if len(on) else None
            examples.append({
                "window_id": f"{version}/{w.file}/{w.start:%Y%m%dT%H%M%S}",
                "version": version, "file": w.file, "split": w.split, "window_kind": w.kind,
                "anchor": w.anchor, "offset": w.offset, "scored": bool(w.scored), "start": w.start, "loop": loop,
                "label": label,
                "attack_ids": " ".join(sorted(on.attack_id.unique())),
                "loop_uncertain": bool(hit.note.str.contains("loop uncertain").any()),
                "attack_start_s": int((first.flag_start - w.start).total_seconds()) if first is not None else -1,
                "attack_end_s": int((first.flag_end - w.start).total_seconds()) if first is not None else -1,
                "tags": [t for t, _ in tags.get(loop, [])],
                "roles": [r for _, r in tags.get(loop, [])],
                "values": [seg[t].to_numpy("float32") for t, _ in tags.get(loop, [])],
            })
    wins["kept"] = keep
    ex = pd.DataFrame(examples)
    ex.insert(0, "example_id", ex.window_id + "/" + ex.loop)
    return wins, ex
