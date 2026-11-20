"""Rebuild everything from the pinned source: `uv run hai-build`.

download and verify -> attack table vs PDF -> table vs flags -> windows ->
loop examples -> answer text -> splits -> baselines -> REPORT.md
"""
import argparse
import json

import numpy as np
import pandas as pd

from hai import attacks_pdf, check_table, download, report
from hai.answers import SWING_ELIGIBLE_MAX_RATE, answer, swing
from hai.baselines import run as run_baselines
from hai.config import DATA_FILES, ENABLED, FROZEN, NORMAL_FILES, PROCESSED
from hai.dataset import SEED, build_examples
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


def swings_at(r, s: int, e: int, scales: dict) -> bool:
    """Does any series of example `r` show a swing change at second `s` (60 s before against s..e)?"""
    for tag, x in zip(r.tags, r.values):
        x = np.asarray(x, "float64")
        if swing(x[max(0, s - 60):s], x[s:min(e + 1, len(x))], scales[(r.version, tag)]):
            return True
    return False


def swing_eligibility(ex: pd.DataFrame, scales: dict) -> pd.DataFrame:
    """How often each loop swings on its own: the share of quiet examples with a swing change at a placebo second.

    Quiet examples: every loop example of a normal window in the train and validation splits (the test split
    is not read). Each gets one (start, end) pair drawn with a fixed seed from the attack examples of the same
    splits whose attack starts at second 10 or later, so the placebo second has the distribution of a real one.
    A loop is swing-eligible when its rate, pooled over versions, is below SWING_ELIGIBLE_MAX_RATE.
    Returns one row per (version, loop) plus one pooled row per loop (version "all").
    """
    seen = ex[ex.split != "test"]
    att = seen[(seen.label == "attack") & (seen.attack_start_s >= 10)]
    pairs = att[["attack_start_s", "attack_end_s"]].to_numpy()
    quiet = seen[seen.window_kind == "normal"]
    rng = np.random.default_rng([SEED, 17])
    draw = pairs[rng.integers(len(pairs), size=len(quiet))]
    hit = np.array([swings_at(r, int(s), int(e), scales) for r, (s, e) in zip(quiet.itertuples(), draw)], bool)
    q = quiet[["version", "loop"]].assign(swing=hit)
    per = q.groupby(["version", "loop"]).swing.agg(quiet_examples="size", swinging="sum").reset_index()
    pooled = q.groupby("loop").swing.agg(quiet_examples="size", swinging="sum").reset_index().assign(version="all")
    out = pd.concat([per, pooled], ignore_index=True)
    out["rate"] = (out.swinging / out.quiet_examples).round(3)
    out["eligible"] = False
    rate = dict(zip(pooled.loop, pooled.swinging / pooled.quiet_examples))
    out.loc[out.version == "all", "eligible"] = [rate[lp] < SWING_ELIGIBLE_MAX_RATE for lp in pooled.loop]
    return out[["version", "loop", "quiet_examples", "swinging", "rate", "eligible"]]


def eligible_loops(table: pd.DataFrame) -> set[str]:
    return set(table[(table.version == "all") & table.eligible].loop)


def process(loop: str) -> str:
    """P1 boiler, P2 turbine, P3 water: the part of the loop name before the dash."""
    return loop.split("-")[0]


def reacting(ex: pd.DataFrame, scales: dict, eligible: set[str]) -> np.ndarray:
    """Normal rows that look like they react to an attack on another loop of the same process.

    True for a row labelled `normal` in an attack window, in a swing-eligible loop, when the window holds an
    attacked loop of the same process (P1 boiler, P2 turbine, P3 water) and the row's series show a swing
    change at that attack's start. The start is the earliest start among the same-process attacked rows of
    the window, the end the latest end; only attacked rows whose attack starts at second 10 or later count
    (an attack already under way leaves no before stretch to compare), so a row whose only same-process
    attack is already under way is never reacting. Two attacks in one window: earliest start, latest end.
    """
    out = np.zeros(len(ex), bool)
    att = ex[(ex.label == "attack") & (ex.attack_start_s >= 10)].assign(process=lambda d: d.loop.map(process))
    span = att.groupby(["window_id", "process"]).agg(s=("attack_start_s", "min"), e=("attack_end_s", "max"))
    span = span.to_dict("index")
    for i, r in enumerate(ex.itertuples()):
        if r.label != "normal" or r.window_kind != "attack" or r.loop not in eligible:
            continue
        got = span.get((r.window_id, process(r.loop)))
        if got is not None:
            out[i] = swings_at(r, int(got["s"]), int(got["e"]), scales)
    return out


TRAIN_NORMAL_PER_ATTACK = 2  # normal examples kept per attack example, from each of the two kinds of window


def training_sample(ex: pd.DataFrame) -> np.ndarray:
    """Which rows to train on. The full train split is 96% normal, which would teach a model to say `normal`.

    Kept, per version: every train attack example with a visible change, twice as many normal
    examples from attack windows (loops that were not attacked) and twice as many from normal
    windows, drawn with a fixed seed. A `reacting` row is never kept: its label says normal while
    its series react to the attack next door. Validation and test keep every row and are never sampled.
    """
    rng = np.random.default_rng([SEED, 7])
    use = np.zeros(len(ex), bool)
    train = (ex.split == "train").to_numpy()
    attack = (ex.label == "attack").to_numpy()
    for version in sorted(ex.version.unique()):
        v = train & (ex.version == version).to_numpy()
        pos = np.flatnonzero(v & attack & ~ex.no_visible_change.to_numpy())
        use[pos] = True
        for kind in ("attack", "normal"):
            pool = np.flatnonzero(v & ~attack & ~ex.reacting.to_numpy() & (ex.window_kind == kind).to_numpy())
            n = min(TRAIN_NORMAL_PER_ATTACK * len(pos), len(pool))
            use[rng.choice(pool, size=n, replace=False)] = True
    return use


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

    print("5. swing eligibility, answer text, reacting neighbours", flush=True)
    eligibility = swing_eligibility(ex, scales)
    eligible = eligible_loops(eligibility)
    texts = [answer(r, {t: scales[(r.version, t)] for t in r.tags}, r.loop in eligible) for r in ex.itertuples()]
    ex["answer"] = [t for t, _ in texts]
    ex["no_visible_change"] = [hidden for _, hidden in texts]
    level_only = np.array([r.label == "attack" and answer(r, {t: scales[(r.version, t)] for t in r.tags})[1]
                           for r in ex.itertuples()], bool)  # the rule without the swing test, for the report
    ex["reacting"] = reacting(ex, scales, eligible)
    ex["train_use"] = training_sample(ex)

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
                 wins=wins, ex=ex, base=base, eligibility=eligibility, level_only=level_only)
    print("done")


if __name__ == "__main__":
    main()
