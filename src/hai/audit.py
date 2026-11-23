"""Audit a written TimeNet table against the raw HAI files: `uv run hai-audit [--table v2] [TABLE]`.

The audit does not reuse the build. It reads the raw files and the frozen attack table itself and
derives every label and interval again from HAI's own flags. `hai-export --table v2` runs it on the
table it has just written and fails when a check fails.

Checks:

1. hashes: every raw file the table names matches data/raw/SHA256SUMS, the pinned digests.
2. one second: every such file steps exactly one second per row.
3. windows: every window lies inside its file, and no window is an exact copy (same start, same rows)
   of a window in another file of the version.
4. manual: every attack of the version starts with one flag run inside the minute the attack table
   gives, and the run lasts as long as the table says. The frozen attack table must equal the printed
   manual plus the fixes coded in hai.attacks_pdf, each with its reason: most are durations, which
   the manual prints differently from the flags (37 of 38 in 20.07; one second short in 22.04 and 23.05).
5. labels: every example's label, attack ids and attacked runs equal what the flags give: the seconds
   of each attack of the loop (frozen table) in which the loop's flag is on. The loop's flag is the
   flag of its process (attack_P1, _P2, _P3) in 20.07 and 21.03 for a v2 table, and the attack flag
   otherwise.
6. edges: every attacked run sits exactly on the flag it was read from: each of its seconds is
   flagged, and the second before and the second after are not.
7. normal windows: every second within 600 s of a normal window is unflagged.

For a v1 table (no `attack_runs` column) the runs are (attack_start_s, attack_end_s) and the flag is
the attack flag, which is the audit the TimeNet review ran.
"""
import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hai import attacks_pdf
from hai.config import FROZEN, PROCESS_FLAG_VERSIONS, RAW, TABLES, out_dir, raw_path, use_table

GAP = 600
WIN = 300
TIME_COLUMN = {"20.07": "time", "21.03": "time", "22.04": "timestamp", "23.05": "timestamp"}
SEPARATOR = {"20.07": ";", "21.03": ",", "22.04": ",", "23.05": ","}


@dataclass
class Check:
    name: str
    checked: int = 0
    failures: list[str] = field(default_factory=list)
    note: str = ""

    def fail(self, message: str) -> None:
        self.failures.append(message)


@dataclass
class Report:
    checks: list[Check]
    table_sha256: str
    rows: int

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.failures)

    def markdown(self, path: Path) -> str:
        out = ["# Audit of the examples table", "",
               f"Table: `{path.name}`, SHA-256 `{self.table_sha256}`, {self.rows} rows.", "",
               "| check | items checked | failures | note |", "|---|---|---|---|"]
        for c in self.checks:
            out.append(f"| {c.name} | {c.checked} | {len(c.failures)} | {c.note} |")
        for c in self.checks:
            if c.failures:
                out += ["", f"## Failures: {c.name}", ""] + [f"- {m}" for m in c.failures[:50]]
                if len(c.failures) > 50:
                    out.append(f"- ... and {len(c.failures) - 50} more")
        return "\n".join(out) + "\n"


def file_name(version: str, stem: str) -> str:
    return f"{stem}.csv.gz" if version in ("20.07", "21.03") else f"{stem}.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums() -> dict[str, str]:
    out = {}
    for line in (RAW / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        out[rel.strip()] = digest
    return out


FLAG_COLUMNS = {"20.07": ["attack", "attack_P1", "attack_P2", "attack_P3"],
                "21.03": ["attack", "attack_P1", "attack_P2", "attack_P3"], "22.04": ["Attack"], "23.05": []}


def read_file(version: str, stem: str, every_column: bool = False) -> pd.DataFrame:
    """The raw file as published (only its time and flag columns unless `every_column`), with `time`
    parsed and the flags HAI gives for it.

    Adds `flag` (the attack flag: `attack` in 20.07 and 21.03, `Attack` in 22.04, the label file's
    `label` for the 23.05 test files, all zero for the 23.05 training files, which HAI publishes
    without labels). 20.07 and 21.03 keep `attack_P1` to `attack_P3`.
    """
    keep = None if every_column else [TIME_COLUMN[version], *FLAG_COLUMNS[version]]
    d = pd.read_csv(raw_path(version, file_name(version, stem)), sep=SEPARATOR[version], usecols=keep)
    d["time"] = pd.to_datetime(d[TIME_COLUMN[version]], format="%Y-%m-%d %H:%M:%S")
    if version in ("20.07", "21.03"):
        d["flag"] = d["attack"].astype("int8")
    elif version == "22.04":
        d["flag"] = d["Attack"].astype("int8")
    elif stem.startswith("hai-test"):
        labels = pd.read_csv(raw_path(version, f"label-{stem.removeprefix('hai-')}.csv"))
        if len(labels) != len(d):
            raise SystemExit(f"23.05 {stem}: the label file has {len(labels)} rows, the data file {len(d)}")
        minute = pd.to_datetime(labels["timestamp"], format="mixed").dt.floor("min").to_numpy()
        if not (minute == d["time"].dt.floor("min").to_numpy()).all():
            raise SystemExit(f"23.05 {stem}: the label file does not line up with the data file")
        d["flag"] = labels["label"].astype("int8").to_numpy()
    else:
        d["flag"] = np.int8(0)
    return d


def runs_of(flag: np.ndarray) -> list[tuple[int, int]]:
    f = np.concatenate([[0], np.asarray(flag) != 0, [0]]).astype("int8")
    d = np.diff(f)
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def parse_runs(text: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in part.split("-", 1)) for part in str(text).split()] if text else []


def run(table_path: Path, report_path: Path | None = None, use_process_flags: bool | None = None) -> Report:
    t = pd.read_csv(table_path, dtype={"version": str, "attack_runs": str, "attack_ids": str, "anchor": str},
                    keep_default_na=False)
    v2 = "attack_runs" in t.columns
    process = v2 if use_process_flags is None else use_process_flags
    t["start"] = pd.to_datetime(t["start"], format="%Y-%m-%d %H:%M:%S")
    checks = {k: Check(k) for k in ("hashes", "one second", "windows", "manual", "labels", "edges",
                                    "normal windows")}
    sums = read_sums()
    attacks = pd.read_csv(FROZEN / "attacks.csv", dtype=str, keep_default_na=False)
    attacks["start_time"] = pd.to_datetime(attacks.start_time)
    printed = attacks_pdf.build(fixes=False)
    printed["start_time"] = pd.to_datetime(printed.start_time)
    files = sorted(set(zip(t.version, t.file)))
    data = {}
    for version, stem in files:
        rel = f"hai-{version}/{file_name(version, stem)}"
        checks["hashes"].checked += 1
        if sums.get(rel) != sha256(RAW / rel):
            checks["hashes"].fail(f"{rel} does not match SHA256SUMS")
        d = read_file(version, stem)
        checks["one second"].checked += 1
        steps = d["time"].diff().dropna()
        if not (steps == pd.Timedelta(seconds=1)).all():
            checks["one second"].fail(f"{rel}: {int((steps != pd.Timedelta(seconds=1)).sum())} steps are not 1 s")
        data[(version, stem)] = d
    checks["hashes"].note = "every raw file the table names"
    checks["one second"].note = "every row one second after the one before"

    # 4. every attack of the table's versions: one flag run starts in the printed minute, duration as frozen
    overall = {}  # (version, attack id) -> (file, first row, last row)
    versions = sorted(t.version.unique())
    for version in versions:
        stems = sorted({s for v, s in files if v == version})
        runs = [(s, a, b) for s in stems for a, b in runs_of(data[(version, s)]["flag"].to_numpy())]
        for aid, g in attacks[attacks.version == version].groupby("attack_id", sort=False):
            start, dur = g.start_time.iloc[0], int(g.duration_s.iloc[0])
            checks["manual"].checked += 1
            inside = [(s, a, b) for s, a, b in runs
                      if start <= data[(version, s)]["time"].iloc[a] < start + pd.Timedelta(minutes=1)]
            if len(inside) != 1:
                checks["manual"].fail(f"{version} {aid}: {len(inside)} flag runs start in {start:%Y-%m-%d %H:%M}")
                continue
            s, a, b = inside[0]
            if b - a + 1 != dur:
                checks["manual"].fail(f"{version} {aid}: the flag is on {b - a + 1} s, the attack table says {dur} s")
            overall[(version, aid)] = (s, a, b)
    # the frozen attack table against the manual as printed: it must equal the printed table plus the
    # fixes coded in hai.attacks_pdf, each of which carries its reason
    fixed = attacks_pdf.build().astype(str)
    frozen = pd.read_csv(FROZEN / "attacks.csv", dtype=str, keep_default_na=False)
    if not frozen.equals(fixed):
        checks["manual"].fail("data/frozen/attacks.csv is not the printed manual plus the documented fixes")
    same = {"start": 0, "duration": 0, "loops": 0}
    fixes = 0
    for (version, aid), g in attacks[attacks.version.isin(versions)].groupby(["version", "attack_id"], sort=False):
        p = printed[(printed.version == version) & (printed.attack_id == aid)]
        if p.empty:
            checks["manual"].fail(f"{version} {aid} is not in the printed manual")
            continue
        diff = {"start": p.start_time.iloc[0] != g.start_time.iloc[0],
                "duration": int(p.duration_s.iloc[0]) != int(g.duration_s.iloc[0]),
                "loops": list(p.loop) != list(g.loop)}
        for k, v in diff.items():
            same[k] += not v
        fixes += any(diff.values())
    a101 = overall.get(("21.03", "A101"))
    sample = ""
    if a101:
        s, a, b = a101
        loops = " ".join(attacks[(attacks.version == "21.03") & (attacks.attack_id == "A101")].loop)
        sample = f"; e.g. 21.03 A101: {loops}, flag on from {data[('21.03', s)]['time'].iloc[a]:%H:%M:%S}, {b - a + 1} s"
    n = checks["manual"].checked
    checks["manual"].note = (f"one flag run per attack, starting in the stated minute and as long as the attack table "
                             f"says; the attack table is the printed manual plus the fixes coded in hai.attacks_pdf: "
                             f"start minute as printed for {same['start']} of {n} attacks, duration for "
                             f"{same['duration']}, loops for {same['loops']}; {fixes} attacks carry a fix{sample}")

    def loop_flag(version: str, stem: str, loop: str) -> np.ndarray:
        d = data[(version, stem)]
        if process and version in PROCESS_FLAG_VERSIONS:
            return d[f"attack_{loop.split('-')[0]}"].to_numpy()
        return d["flag"].to_numpy()

    # attacked seconds of each loop, per file: the loop's flag inside the overall runs of its attacks
    loop_attacks = {}  # (version, stem, loop) -> list of (attack id, first row, last row) of the overall run
    for (version, aid), (s, a, b) in overall.items():
        for loop in attacks[(attacks.version == version) & (attacks.attack_id == aid)].loop.unique():
            loop_attacks.setdefault((version, s, loop), []).append((aid, a, b))

    windows = t.drop_duplicates("window_id")
    for w in windows.itertuples():
        d = data[(w.version, w.file)]
        i0 = int((w.start - d["time"].iloc[0]).total_seconds())
        checks["windows"].checked += 1
        if i0 < 0 or i0 + WIN > len(d) or d["time"].iloc[i0] != w.start:
            checks["windows"].fail(f"{w.window_id} leaves its file")
        if w.window_kind == "normal":
            checks["normal windows"].checked += 1
            lo, hi = max(0, i0 - GAP), min(len(d), i0 + WIN + GAP)
            if d["flag"].iloc[lo:hi].any():
                checks["normal windows"].fail(f"{w.window_id}: a flagged second lies within {GAP} s")
    # exact copies: the same start in two files of a version, with the same rows
    for (version, start), g in windows.groupby(["version", "start"]):
        if g.file.nunique() < 2:
            continue
        rows = []
        for w in g.itertuples():
            d = read_file(version, w.file, every_column=True)
            i0 = int((w.start - d["time"].iloc[0]).total_seconds())
            rows.append((w.window_id, d.drop(columns=["time", TIME_COLUMN[version]], errors="ignore")
                         .iloc[i0:i0 + WIN].reset_index(drop=True)))
        for i in range(len(rows)):
            for j in range(i):
                if rows[i][1].equals(rows[j][1]):
                    checks["windows"].fail(f"{rows[i][0]} is an exact copy of {rows[j][0]}")
    checks["windows"].note = "inside the file, no exact copy across files"
    checks["normal windows"].note = f"no flagged second within {GAP} s"

    for r in t.itertuples():
        d = data[(r.version, r.file)]
        i0 = int((r.start - d["time"].iloc[0]).total_seconds())
        flag = loop_flag(r.version, r.file, r.loop)
        mask = np.zeros(len(d), "int8")
        ids = set()
        for aid, a, b in loop_attacks.get((r.version, r.file, r.loop), []):
            seg = flag[a:b + 1] != 0
            if seg[max(0, i0 - a):max(0, i0 + WIN - a)].any():
                ids.add(aid)
            mask[a:b + 1] |= seg
        runs = [(a - i0, b - i0) for a, b in runs_of(mask) if a < i0 + WIN and b >= i0]
        want_label = "attack" if runs else "normal"
        got_runs = parse_runs(r.attack_runs) if v2 else (
            [(int(r.attack_start_s), int(r.attack_end_s))] if r.label == "attack" else [])
        checks["labels"].checked += 1
        if r.label != want_label:
            checks["labels"].fail(f"{r.example_id}: label {r.label}, the flags say {want_label}")
        elif set(str(r.attack_ids).split()) != ids:
            checks["labels"].fail(f"{r.example_id}: attack ids {r.attack_ids!r}, the flags say {sorted(ids)}")
        elif v2 and got_runs != runs:
            checks["labels"].fail(f"{r.example_id}: runs {got_runs}, the flags say {runs}")
        elif v2 and runs and (int(r.attack_start_s), int(r.attack_end_s)) != (runs[0][0], runs[-1][1]):
            checks["labels"].fail(f"{r.example_id}: start and end do not match the runs")
        elif not v2 and runs and (int(r.attack_start_s) != runs[0][0]):
            checks["labels"].fail(f"{r.example_id}: start {r.attack_start_s}, the flags say {runs[0][0]}")
        if any(a < 0 or a >= WIN for a, _ in got_runs):
            checks["labels"].fail(f"{r.example_id}: a run starts outside the window, which the TimeNet connector refuses")
        for a, b in got_runs:
            checks["edges"].checked += 1
            lo, hi = i0 + a, i0 + b
            if not (flag[lo:hi + 1] != 0).all():
                checks["edges"].fail(f"{r.example_id}: run {a}-{b} has unflagged seconds")
            elif (lo > 0 and flag[lo - 1]) or (hi + 1 < len(flag) and flag[hi + 1]):
                checks["edges"].fail(f"{r.example_id}: run {a}-{b} does not end where its flag ends")
    flag_name = "the process flag in 20.07 and 21.03, the attack flag elsewhere" if process else "the attack flag"
    checks["labels"].note = f"label, attack ids and runs re-derived from {flag_name}"
    checks["edges"].note = "each run flagged throughout, the second before and after unflagged"

    report = Report(checks=list(checks.values()), table_sha256=sha256(table_path), rows=len(t))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.markdown(table_path))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path, nargs="?", default=None,
                    help="default: timenet/examples_table.csv.gz of the table version")
    ap.add_argument("--table", dest="version", choices=TABLES, default="v2")
    ap.add_argument("--report", type=Path, default=None, help="where to write AUDIT.md (default: next to the table)")
    args = ap.parse_args(argv)
    use_table(args.version)
    path = args.table or out_dir() / "timenet" / "examples_table.csv.gz"
    report = run(path, args.report or path.parent / "AUDIT.md")
    print(report.markdown(path))
    if report.failed:
        raise SystemExit(f"audit failed: {report.failed} checks")


if __name__ == "__main__":
    main()
