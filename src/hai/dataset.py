"""Steps 6, 7 and 9: windows, loop examples and splits.

A window is 300 seconds of one file. Attack windows: six per attack, with the
attack starting at second 30, 75, 120, 165, 210 and 255. Normal windows: at
least 600 seconds from any attack, half from attack files, half from the
version's normal file. A loop example is one window seen through one loop.
"""
import numpy as np
import pandas as pd

from hai.config import DATA_FILES, NORMAL_FILES, split_of
from hai.loaders import load, tag_map

WIN = 300
OFFSETS = (30, 75, 120, 165, 210, 255)
GAP = 600
SEED = 20261120
NORMAL_PER_VERSION = 750  # 4 versions x 750 = about 3,000 normal windows
LOOPS = ("P1-PC", "P1-LC", "P1-FC", "P1-TC", "P2-SC", "P3-LC")
LOOP_NAMES = {"P1-PC": "boiler pressure", "P1-LC": "boiler level", "P1-FC": "boiler flow",
              "P1-TC": "boiler temperature", "P2-SC": "turbine speed", "P3-LC": "water level"}
# label of each series in the answer text; roles with two series get a qualifier
SERIES_LABEL = {"P1_PCV01D": "first valve command", "P1_PCV02D": "second valve command",
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


def attack_windows(version: str, attacks: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in attacks.iterrows():
        f = spans.loc[a.file]
        for off in OFFSETS:
            start = a.flag_start - pd.Timedelta(seconds=off)
            if start < f["first"] or start + pd.Timedelta(seconds=WIN - 1) > f["last"]:
                continue
            rows.append({"version": version, "file": a.file, "kind": "attack", "anchor": a.attack_id,
                         "offset": off, "start": start})
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
    out["kind"], out["anchor"], out["offset"] = "normal", "", -1
    return out


def build_examples(version: str, attacks: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (windows, loop examples) for one version.

    attacks: checked attacks (one row each, with file, flag_start, flag_duration).
    steps: attack steps joined with their flag runs (file, flag_start, flag_end, loop).
    """
    rng = np.random.default_rng([SEED, int(version.replace(".", ""))])
    spans = file_spans(version)
    wins = pd.concat([attack_windows(version, attacks, spans), normal_windows(version, steps, spans, rng)],
                     ignore_index=True)
    wins["end"] = wins["start"] + pd.Timedelta(seconds=WIN - 1)
    wins["split"] = [split_of(version, f) or ("train" if f == NORMAL_FILES[version] else None) for f in wins.file]
    tags = loop_tags(version)
    tables = {stem: load(version, stem) for stem in wins.file.unique()}
    examples, keep = [], []
    for w in wins.itertuples():
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
            label = "attack" if len(on) else "normal"
            first = on.sort_values("flag_start").iloc[0] if len(on) else None
            examples.append({
                "window_id": f"{version}/{w.file}/{w.start:%Y%m%dT%H%M%S}",
                "version": version, "file": w.file, "split": w.split, "window_kind": w.kind,
                "anchor": w.anchor, "offset": w.offset, "start": w.start, "loop": loop,
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
