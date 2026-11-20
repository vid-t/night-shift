"""Rebuild everything from the pinned source: `uv run hai-build`.

download and verify -> attack table vs PDF -> table vs flags -> windows ->
loop examples -> answer text -> splits -> baselines -> REPORT.md
"""
import argparse
import json

import numpy as np
import pandas as pd

from hai import attacks_pdf, check_table, download, report
from hai.answers import answer
from hai.baselines import run as run_baselines
from hai.config import DATA_FILES, ENABLED, FROZEN, NORMAL_FILES, PROCESSED
from hai.dataset import build_examples
from hai.loaders import count_rows, load, standard_tags


def norm_stats(version: str) -> tuple[dict, dict]:
    """Per tag, from the version's normal file: (min, max) range and a scale (p99 - p1)."""
    d = load(version, NORMAL_FILES[version])
    ranges, scales = {}, {}
    for tag in standard_tags():
        x = d[tag].dropna()
        if x.empty:
            continue
        ranges[(version, tag)] = (float(x.min()), float(x.max()))
        s = float(x.quantile(0.99) - x.quantile(0.01))
        scales[(version, tag)] = s if s > 0 else max(abs(float(x.median())), 1.0)
    return ranges, scales


def file_facts(version: str) -> pd.DataFrame:
    rows = []
    for stem in DATA_FILES[version]:
        d = load(version, stem)
        gaps = int((d.time.diff().dropna() != pd.Timedelta(seconds=1)).sum())
        a = d.attack.to_numpy()
        runs = int(((np.diff(a) == 1).sum()) + (a[0] == 1))
        rows.append({"version": version, "file": stem, "rows": len(d), "raw_lines": count_rows(version, stem),
                     "hours": round(len(d) / 3600, 1), "first": d.time.iloc[0], "last": d.time.iloc[-1],
                     "time_gaps": gaps, "attacks": runs, "attack_seconds": int(a.sum())})
    return pd.DataFrame(rows)


def tracking(version: str) -> pd.DataFrame:
    """Does each loop's measurement follow its target in the normal file? Correlation of 10-minute means."""
    from hai.dataset import loop_tags

    d = load(version, NORMAL_FILES[version])
    g = d.drop(columns="time").groupby(np.arange(len(d)) // 600).mean()
    rows = []
    for loop, tags in loop_tags(version).items():
        meas = [t for t, r in tags if r == "measurement"][0]
        for t, r in tags:
            if r == "target":
                c = g[meas].corr(g[t]) if g[t].std() > 0 else np.nan
                rows.append({"version": version, "loop": loop, "target": t, "measurement": meas,
                             "target_constant": bool(g[t].std() == 0),
                             "corr_10min_means": None if pd.isna(c) else round(float(c), 3)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--versions", default=",".join(ENABLED), help="comma-separated, default: %(default)s")
    args = ap.parse_args(argv)
    versions = args.versions.split(",")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("1. download and verify SHA-256", flush=True)
    download.main()

    print("2. attack table: frozen file vs fresh PDF extraction", flush=True)
    fresh = attacks_pdf.build()
    frozen = pd.read_csv(FROZEN / "attacks.csv", dtype=str, keep_default_na=False)
    table_matches_pdf = frozen.equals(fresh.astype(str))
    if not table_matches_pdf:
        raise SystemExit("data/frozen/attacks.csv differs from the PDF extraction plus CORRECTIONS")
    printed = attacks_pdf.build(fixes=False).assign(start_time=lambda d: pd.to_datetime(d.start_time))

    print("3. table vs flags, per version", flush=True)
    facts, checks, before, unmatched, renames, included, excluded = [], [], [], [], [], [], {}
    tracks = []
    for v in versions:
        facts.append(file_facts(v))
        tracks.append(tracking(v))
        c, u = check_table.check(v)
        b, _ = check_table.check(v, printed)
        checks.append(c), before.append(b), unmatched.append(u)
        renames.append(check_table.rename_check(v, c))
        bad = c[c.status != "ok"]
        if len(bad) or len(u) or check_table.normal_file_clean(v):
            excluded[v] = f"{len(bad)} attacks fail the flag check, {len(u)} flag runs not in the table"
        else:
            included.append(v)
    checks, before = pd.concat(checks, ignore_index=True), pd.concat(before, ignore_index=True)
    checks.to_csv(PROCESSED / "table_check.csv", index=False)

    print(f"4. windows and loop examples for {included}", flush=True)
    table = check_table.attacks_table()
    wins, exs, ranges, scales = [], [], {}, {}
    for v in included:
        att = checks[(checks.version == v) & (checks.status == "ok")]
        steps = table[table.version == v].merge(att[["attack_id", "file", "flag_start", "flag_duration"]], on="attack_id")
        steps["flag_end"] = steps.flag_start + pd.to_timedelta(steps.flag_duration.astype(int) - 1, unit="s")
        w, e = build_examples(v, att, steps)
        r, s = norm_stats(v)
        ranges.update(r), scales.update(s)
        wins.append(w), exs.append(e)
    wins, ex = pd.concat(wins, ignore_index=True), pd.concat(exs, ignore_index=True)

    print("5. answer text", flush=True)
    ex["answer"] = [answer(r, {t: scales[(r.version, t)] for t in r.tags}) for r in ex.itertuples()]

    wins.to_parquet(PROCESSED / "windows.parquet", index=False)
    ex.to_parquet(PROCESSED / "examples.parquet", index=False)

    print("6. baselines on validation", flush=True)
    base = run_baselines(ex[ex.split != "test"], ranges, scales)
    with open(PROCESSED / "baselines.json", "w") as f:
        json.dump({k: v.to_dict("records") for k, v in base.items()}, f, indent=1, default=str)

    print("7. REPORT.md", flush=True)
    report.write(versions=versions, included=included, excluded=excluded,
                 facts=pd.concat(facts, ignore_index=True), checks=checks, before=before,
                 unmatched=pd.concat(unmatched, ignore_index=True),
                 renames=pd.concat(renames, ignore_index=True), tracks=pd.concat(tracks, ignore_index=True),
                 wins=wins, ex=ex, base=base)
    print("done")


if __name__ == "__main__":
    main()
