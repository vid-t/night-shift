"""Step 3: check the attack table against the attack flags in the data files.

Each attack must switch the flag on inside its stated minute and keep it on
for exactly its stated seconds. Every mismatch is reported; fixes go into the
table (hai.attacks_pdf.CORRECTIONS), never into the data.
"""
import pandas as pd

from hai.config import DATA_FILES, FROZEN, NORMAL_FILES
from hai.loaders import load, read_raw, tag_map


def attacks_table() -> pd.DataFrame:
    df = pd.read_csv(FROZEN / "attacks.csv", dtype={"version": str}, keep_default_na=False)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def flag_runs(version: str) -> pd.DataFrame:
    """Every run of attack == 1 in every data file of a version."""
    rows = []
    for stem in DATA_FILES[version]:
        d = load(version, stem)
        a = d["attack"].to_numpy()
        edges = pd.Series(a).diff().fillna(a[0]).to_numpy()
        starts = (edges == 1).nonzero()[0]
        ends = (edges == -1).nonzero()[0]
        if len(ends) < len(starts):
            ends = list(ends) + [len(a)]
        for s, e in zip(starts, ends):
            rows.append({"version": version, "file": stem, "flag_start": d.time.iloc[s],
                         "flag_end": d.time.iloc[e - 1], "flag_duration": int(e - s)})
    return pd.DataFrame(rows, columns=["version", "file", "flag_start", "flag_end", "flag_duration"])


def process_flags(version: str, run: pd.Series) -> set[str] | None:
    """Processes (P1, P2, P3) whose own flag is on during a run, where the version has them."""
    if version not in ("20.07", "21.03"):
        return None
    raw = _raw_cached(version, run["file"])
    t = pd.to_datetime(raw["time"])
    m = (t >= run["flag_start"]) & (t <= run["flag_end"])
    return {p for p in ("P1", "P2", "P3") if raw.loc[m, f"attack_{p}"].max() == 1}


_RAW: dict = {}


def _raw_cached(version, stem):
    if (version, stem) not in _RAW:
        _RAW[(version, stem)] = read_raw(version, stem)
    return _RAW[(version, stem)]


def check(version: str, table: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per attack with status, unmatched flag runs).
    `table` defaults to the frozen attacks.csv; pass the raw PDF extraction to see the original mismatches."""
    table = attacks_table() if table is None else table
    table = table[table.version == version]
    runs = flag_runs(version)
    used = set()
    out = []
    for aid, steps in table.groupby("attack_id", sort=False):
        start, dur = steps.start_time.iloc[0], int(steps.duration_s.iloc[0])
        row = {"version": version, "attack_id": aid, "table_start": start, "table_duration": dur,
               "loops": " ".join(steps.loop), "file": "", "flag_start": pd.NaT,
               "flag_duration": pd.NA, "status": "", "detail": ""}
        inside = runs[(runs.flag_start >= start) & (runs.flag_start < start + pd.Timedelta(minutes=1))]
        if len(inside) == 1:
            r = inside.iloc[0]
            used.add(r.name)
            row.update(file=r.file, flag_start=r.flag_start, flag_duration=r.flag_duration)
            if r.flag_duration == dur:
                row["status"] = "ok"
            else:
                row["status"] = "duration"
                row["detail"] = f"flag on {r.flag_duration} s, table says {dur} s"
        elif len(inside) > 1:
            row["status"] = "ambiguous"
            row["detail"] = f"{len(inside)} flag runs start in the stated minute"
        else:
            near = runs.assign(gap=(runs.flag_start - start).abs()).sort_values("gap")
            near = near[near.gap <= pd.Timedelta(days=2)]
            row["status"] = "start"
            if len(near):
                r = near.iloc[0]
                row["detail"] = (f"no flag run starts in the stated minute; nearest run starts "
                                 f"{r.flag_start} in {r.file}, {r.flag_duration} s")
            else:
                row["detail"] = "no flag run within two days of the stated start"
        if row["status"] in ("ok", "duration"):
            procs = process_flags(version, runs.loc[list(inside.index)[0]])
            want = {l[:2] for l in steps.loop}
            if procs is not None and procs != want:
                row["detail"] = (row["detail"] + "; " if row["detail"] else "") + \
                    f"process flags on: {sorted(procs)}, table loops imply {sorted(want)}"
                if row["status"] == "ok":
                    row["status"] = "process"
        out.append(row)
    unmatched = runs.drop(index=list(used))
    return pd.DataFrame(out), unmatched


def rename_check(version: str, attacks: pd.DataFrame) -> pd.DataFrame:
    """For each renamed tag (not 'same name'), does its source column move during
    the attacks whose table row names the tag? Evidence for the name map."""
    tm = tag_map()[lambda d: (d.version == version) & (d.basis != "same name") & (d.source != "")]
    table = attacks_table()[lambda d: d.version == version]
    rows = []
    for std, src, basis in zip(tm.standard, tm.source, tm.basis):
        steps = table[table.points.str.split().apply(lambda p: std in p)]
        for _, s in steps.iterrows():
            a = attacks[(attacks.attack_id == s.attack_id) & attacks.status.isin(["ok", "process"])]
            if a.empty:
                continue
            a = a.iloc[0]
            raw = _raw_cached(version, a.file)
            tcol = "time" if "time" in raw else "timestamp"
            t = pd.to_datetime(raw[tcol])
            before = raw.loc[(t >= a.flag_start - pd.Timedelta(minutes=5)) & (t < a.flag_start), src]
            during = raw.loc[(t >= a.flag_start) & (t <= a.flag_start + pd.Timedelta(seconds=int(a.flag_duration) - 1)), src]
            moved = bool((during - before.median()).abs().max() > 5 * max(before.std(), 1e-9)) \
                or bool(during.nunique() > 1 and before.nunique() == 1)
            rows.append({"version": version, "standard": std, "source": src, "attack_id": s.attack_id,
                         "scenario": s.scenario, "moves_during_attack": moved})
    return pd.DataFrame(rows)


def normal_file_clean(version: str) -> int:
    """Attack seconds in the version's normal file (must be 0)."""
    return int(load(version, NORMAL_FILES[version])["attack"].sum())
