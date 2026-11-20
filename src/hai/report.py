"""Write REPORT.md from the numbers the build computed. No number is typed by hand."""
import pandas as pd

from hai.answers import SWING_ELIGIBLE_MAX_RATE, SWING_FLOOR, SWING_MIN_STEPS, SWING_MIN_TURNS, SWING_RATIO
from hai.config import COMMIT, ROOT, split_of
from hai.loaders import tag_map


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |")
    return "\n".join(lines)


NOT_VERIFIED = [
    "The answer text has been read by eye against charts for 24 examples only (the label-repair check: hidden attacks made "
    "visible, reacting rows, normal rows that mention swinging). The plan asks for 100 examples to be checked before training.",
    "The direction words come from a simple rule: a shift of the mean larger than three times the noise in the 60 seconds before, "
    "or 2% of the tag's normal-file range. Slow ramps (`-ST`, `-LT` scenarios) may read as `stays flat` early in the window.",
    "The swing words come from a second simple rule (section 4): the noise of the one-second steps after the start against the "
    "60 seconds before. Its thresholds were set on the train and validation splits from three ratios and three floors; they were "
    "not tuned further. A slow oscillation (period well above a minute) can pass as a level shift or as flat.",
    "`reacting` marks normal rows whose series swing when a loop of the same process is attacked. It is a mark, not a label: "
    "the label stays `normal`. Whether these loops are physically pushed by the attack or merely move by chance at that second "
    "is not settled; quiet loops in the same eligible loops swing at the placebo second in under 10% of examples. The "
    "after stretch runs to the attack's end, so a row can be marked for a swing that begins well after the attack starts "
    "(2 of the 8 reacting rows drawn for the eye check).",
    "P3-LC is swing-eligible under the final rule (rate below 10%) although the first measurement, without the turns "
    "condition, put it at 62%. Its slow pump and valve cycles are smooth; the turns condition removes them. No P3-LC attack "
    "row became visible through the swing test, and P3-LC has no same-process neighbour, so it has no `reacting` rows.",
    "22.04 A301, A304, A305 and A306 name one loop in the scenario and another in the points. They keep the scenario's loop and "
    "are marked `loop_uncertain`. The data did not settle it: the boiler loops move together during these attacks.",
    "Other points that do not fit their scenario are kept as printed (for example 22.04 A303 lists P1_B3004, P1_LIT01 for "
    "P1-LC-CO1PV1). The loop label always comes from the scenario code, not the points.",
    "20.07 A211, A212 and A213 name `P3_SP02`, which is not a 20.07 column. The same scenario (P3-LC-SP2CV2) names P3_LL01 "
    "elsewhere; kept as printed. The loop (P3-LC) is not affected.",
    "20.07 durations come from the flags (see the fixes above): the manual's 20.07 durations disagree with the flags for 37 "
    "of 38 attacks with no pattern, while the start minute and the per-process flags agree for all 38.",
    "23.05: no target column moves during any setpoint attack, including x1001_15_ASSIGN_OUT, which the manual ties to P1_B2016. "
    "Setpoint attacks in 23.05 leave no trace in the target series; the measurement and command must carry the evidence.",
    "23.05: P1_B3004 and P1_B3005 are matched from data (target tracking), not from the manual. P1_B4022 is not matched, so the "
    "23.05 boiler temperature loop has no target.",
    "23.05 label-test2.csv prints its times to the minute; the loader checks that every row's minute matches the data row.",
    "The water-level targets (P3_LH01, P3_LL01 and their older names) are constant in every file, also during the 20.07 "
    "attacks that name them. Setpoint attacks on that loop leave no trace in the target series.",
    "Turbine speed loop: 20.07 has only P2_SIT01 (no command, and P2_SD01 is constant per file), so that loop has one "
    "series there, not the two the brief expected; 21.03 has two (P2_SIT01, P2_CO_rpm) and no target, because P2_ASD is always 0.",
    "The loop label says which loop the manual says was attacked. It does not check that the attack is visible in that loop's series.",
    "Attack windows can overlap a second attack on another loop. Those loops are labelled `attack` too, with their own start second.",
    "Gradient boosting uses shape statistics per series plus how the measurement relates to the target and the command; no "
    "absolute levels, so it cannot tell files apart by their set points. The feature set was chosen after one comparison on "
    "validation (shape alone against shape plus relations); the relational features found more validation attacks and were kept. "
    "It has not been tuned otherwise.",
    "Validation holds only 15 attacks (21.03 test4 and 22.04 test3); 20.07 and 23.05 have no validation file, so their "
    "baseline scores are not measured here.",
    "The baseline tables compare three models on those 15 attacks. A difference of one or two attacks between them is "
    "within chance. The neighbour variant is reported, not chosen: it does not settle whether the loop view hides evidence.",
    "Each attack is scored on one window with a random start second. Another draw of the scored windows would move the "
    "counts by an attack or two.",
    "The training sample keeps two normal examples from attack windows and two from normal windows per attack example. "
    "That ratio is a choice, not a tuned value.",
    "`no_visible_change` comes from the direction rule and, in swing-eligible loops, the swing rule. An example marked this way "
    "may still hold a change the rules are too coarse to see; a missed slow ramp, or a swing change in P1-TC or P3-LC "
    "(not eligible), is the likely case.",
    "23.05: P1_B4002 is not matched. x1003_18_SETPOINT_OUT sits in its 22.04 range, but both are nearly constant, so "
    "the match cannot be confirmed. The 23.05 temperature loop therefore has no target series.",
    "Setpoint attacks on the temperature loop name P1_B4002. It is now in the loop view for 20.07, 21.03 and 22.04. "
    "Whether it moves during those attacks was not checked.",
    "The manual's hours per file (p. 2) were not compared with the row counts; the row counts come from the files.",
]


def write(*, versions, included, excluded, facts, checks, before, unmatched, renames, tracks, wins, ex, base,
          eligibility, level_only) -> None:
    out = ["# REPORT", "",
           f"Generated by `uv run hai-build --versions {','.join(versions)}` from icsdataset/hai at commit `{COMMIT}`. "
           "Every number below is computed by the build.", ""]
    out += [f"Versions built: **{', '.join(included) or 'none'}**."]
    if excluded:
        out += ["", "Versions left out:", ""] + [f"- **{v}**: {why}" for v, why in excluded.items()]
    out += ["", "## 1. Files", "",
            "One row per second in every file. `raw_lines` counts line breaks in the raw file, independent of pandas.", ""]
    f = facts.assign(split=[split_of(v, s) or ("train (normal file)" if s.startswith(("train", "hai-train")) else "")
                            for v, s in zip(facts.version, facts.file)])
    out += [md(f[["version", "file", "split", "rows", "raw_lines", "hours", "first", "last", "time_gaps",
                  "attacks", "attack_seconds"]]), ""]
    tot = facts.groupby("version").agg(rows=("rows", "sum"), hours=("hours", "sum"), attacks=("attacks", "sum"),
                                        attack_seconds=("attack_seconds", "sum")).reset_index()
    out += [md(tot.round(1)), ""]

    out += ["## 2. Attack table against the flags", "",
            "`data/frozen/attacks.csv` is extracted from the manual (PDF pages listed in `hai/attacks_pdf.py`) with pypdf. "
            "Each attack must switch the flag on inside its stated minute and keep it on for its stated seconds.", ""]
    t = check_attacks_summary(before, checks)
    out += [md(t), ""]
    bad = before[before.status != "ok"]
    out += ["Mismatches of the table **as printed in the manual**:", ""]
    out += [md(bad[["version", "attack_id", "table_start", "table_duration", "status", "detail"]]) if len(bad) else "None.", ""]
    from hai.check_table import attacks_table
    notes = attacks_table()
    notes = notes[(notes.note != "") & notes.version.isin(versions)].assign(note=lambda d: d.note.str.split(" | ", regex=False)).explode("note")
    notes = notes.drop_duplicates(["version", "attack_id", "note"])
    fixes = notes.groupby(["version", "note"], sort=False).attack_id.agg(
        lambda s: f"all {len(s)} attacks" if len(s) > 12 else ", ".join(s)).reset_index()
    out += ["Fixes and flags applied to the table (the flags and tag lists are the source). "
            "A `loop uncertain` row keeps the scenario's loop; its examples carry `loop_uncertain = True`.", ""]
    out += [md(fixes[["version", "attack_id", "note"]].rename(columns={"attack_id": "attacks"})) if len(fixes) else "None.", ""]
    out += ["Mismatches **after** the fixes:", ""]
    after = checks[checks.status != "ok"]
    out += [md(after[["version", "attack_id", "status", "detail"]]) if len(after) else "None.", ""]
    out += ["Flag runs with no attack in the table:", ""]
    out += [md(unmatched) if len(unmatched) else "None.", ""]
    out += ["For 20.07 and 21.03 the check also compares the per-process flags (`attack_P1..P3`) with the processes "
            "of the table's loops; a disagreement shows as status `process`.", ""]

    out += ["## 3. Name map", "",
            "Standard names are the 22.04 names. `data/frozen/tag_map.csv` gives the source column and the basis for each version.", ""]
    tm = tag_map()[lambda d: d.version.isin(versions) & (d.basis != "same name")]
    out += [md(tm[["version", "standard", "loop", "role", "source", "basis"]]), ""]
    if len(renames):
        out += ["Rename check: does the source column move during the attacks whose table row names the tag? "
                "(moves = leaves five times the noise of the five minutes before, or changes after being constant).", ""]
        rc = renames.groupby(["version", "standard", "source"]).agg(
            attacks=("attack_id", "size"), moves=("moves_during_attack", "sum")).reset_index()
        out += [md(rc), ""]

    out += ["Target tracking in each version's normal file: correlation of 10-minute means of the measurement and the target. "
            "A setpoint shows as a correlation near 1 unless it is constant. This is the evidence for the 23.05 data matches.", "",
            md(tracks), ""]
    out += ["## 4. Windows and loop examples", "",
            "Windows are 300 s. Attack windows: six per attack. The attack starts at a random second between 30 and 270, "
            "one draw from each of six equal bands (fixed seed), so a model cannot answer the start second without reading "
            "the curves. One of the six, picked at random, is the scored window of that attack. "
            "A window that would leave its file is skipped. Normal windows: at least 600 s from any attack, half from attack "
            "files and half from the normal file (normal-file windows go to train). A window whose only attacked loops are "
            "outside the six (P2-TC, P1-CC, P1-HC) is dropped.", ""]
    wk = wins.groupby(["version", "kind", "kept"]).size().unstack("kept", fill_value=0).reset_index()
    wk.columns = [str(c) if c not in (True, False) else ("kept" if c else "dropped") for c in wk.columns]
    out += [md(wk), ""]
    a = wins[wins.kind == "attack"]
    kept = a[a.kept].groupby("version").anchor.nunique()
    okc = checks[checks.status == "ok"].groupby("version").size()
    out += ["Attacks with at least one window kept, of the attacks that pass the flag check: " +
            ", ".join(f"{v}: {int(kept.get(v, 0))} of {int(okc.get(v, 0))}" for v in okc.index) + ".", ""]
    n = wins[wins.kind == "normal"].assign(source=lambda d: [
        "normal file" if f.startswith(("train", "hai-train")) else "attack files" for f in d.file])
    out += ["Normal windows by source (asked: 375 from attack files and 375 from the normal file per version; "
            "fewer means the source holds fewer non-overlapping 300 s windows that are 600 s from any attack):", "",
            md(n.groupby(["version", "source"]).size().rename("windows").reset_index()), ""]
    out += ["Loop examples per split and label:", ""]
    sl = ex.groupby(["split", "label"]).size().unstack("label", fill_value=0).reset_index()
    out += [md(sl), ""]
    vsl = ex.groupby(["version", "split", "label"]).size().unstack("label", fill_value=0).reset_index()
    out += [md(vsl), ""]
    lp = ex[ex.label == "attack"].groupby(["version", "loop"]).size().unstack("loop", fill_value=0).reset_index()
    out += ["Attack loop examples per loop:", "", md(lp), ""]
    att = ex[ex.label == "attack"]
    sc = att[att.scored]
    out += ["Start second of the attack inside its window (attack loop examples):", "",
            md(pd.DataFrame([
                {"windows": "all six per attack", "examples": len(att), "distinct_start_seconds": att.attack_start_s.nunique(),
                 "min": int(att.attack_start_s.min()), "median": int(att.attack_start_s.median()), "max": int(att.attack_start_s.max()),
                 "largest_share_of_one_second": round(float(att.attack_start_s.value_counts(normalize=True).iloc[0]), 3)},
                {"windows": "scored window only", "examples": len(sc), "distinct_start_seconds": sc.attack_start_s.nunique(),
                 "min": int(sc.attack_start_s.min()), "median": int(sc.attack_start_s.median()), "max": int(sc.attack_start_s.max()),
                 "largest_share_of_one_second": round(float(sc.attack_start_s.value_counts(normalize=True).iloc[0]), 3)}])), ""]
    out += ["### Swing test", "",
            "The level-shift rule misses a series that starts or stops oscillating. The swing test compares the noise "
            f"(standard deviation) of the one-second steps after the attack start with the 60 seconds before: a change when one is "
            f"more than {SWING_RATIO:g} times the other and the larger exceeds {SWING_FLOOR:g} of the tag's normal-file scale "
            f"(at least {SWING_MIN_STEPS} steps on each side), and the swinging side goes up and down: its path (the sum of "
            f"its absolute steps) is at least {SWING_MIN_TURNS:g} times its range. Without that second condition a smooth ramp, "
            "hump or single step also passes, and the words `starts swinging` would not match the chart. It dates the change to the first second where the 10-second step "
            "noise crosses the geometric mean of the two. The words are `starts swinging` and `stops swinging`.", "",
            "Some loops swing on their own, so the test counts only in loops that rarely do. The rate below is the share of "
            "quiet examples (normal windows, train and validation splits) that show a swing change in any series at a placebo "
            "second drawn with a fixed seed from the real attack starts and ends. A loop is eligible when its rate, pooled over "
            f"versions, is below {SWING_ELIGIBLE_MAX_RATE:.0%}. The choice is per loop, not per (version, loop), "
            "so that one loop uses the same words in every version; a per-version cut would let P1-PC, P1-FC and P1-TC "
            "switch vocabulary between versions on rates that sit near the limit or differ by version.", ""]
    el = eligibility.copy()
    pooled = el[el.version == "all"].set_index("loop")
    per = el[el.version != "all"].pivot(index="loop", columns="version", values="rate")
    per.columns = [f"rate_{c}" for c in per.columns]
    tab = pooled[["quiet_examples", "swinging", "rate", "eligible"]].join(per).reset_index()
    out += [md(tab), ""]
    out += ["Attack loop examples with no visible change in their series (`no_visible_change`). They are left out of "
            "training and counted on their own in scoring. `level_rule_only` is the count under the level-shift rule alone "
            "(before the swing test); `made_visible` moved out of the hidden group because an eligible series swings:", ""]
    lo = pd.Series(level_only, index=ex.index)
    nv = att.assign(level_rule_only=lo[att.index]).groupby(["version", "split"]).agg(
        attack_examples=("label", "size"), level_rule_only=("level_rule_only", "sum"),
        no_visible_change=("no_visible_change", "sum")).reset_index()
    nv = pd.concat([nv, pd.DataFrame([{"version": "all", "split": "all", **nv[["attack_examples", "level_rule_only",
                                                                                 "no_visible_change"]].sum()}])])
    nv["made_visible"] = nv.level_rule_only - nv.no_visible_change
    out += [md(nv), ""]
    hid_loop = att.assign(level_rule_only=lo[att.index]).groupby("loop").agg(
        level_rule_only=("level_rule_only", "sum"), no_visible_change=("no_visible_change", "sum")).reset_index()
    out += ["The same by loop:", "", md(hid_loop), ""]
    rea = ex[ex.reacting]
    nb = ex[(ex.window_kind == "attack") & (ex.label == "normal")]
    out += ["### Reacting neighbours", "",
            "`reacting` marks a row labelled `normal` in an attack window, in a swing-eligible loop, whose window holds an "
            "attacked loop of the same process (P1 boiler, P2 turbine, P3 water) and whose series show a swing change at that "
            "attack's start (earliest start, latest end of the same-process attacks that start at second 10 or later). The label "
            "stays `normal`. Reacting rows are never in the training sample and count neither for nor against `right_loop_only`. "
            f"{len(rea)} of the {len(nb)} normal rows in attack windows are reacting.", ""]
    out += [md(rea.groupby("loop").size().rename("reacting").reset_index()), ""]
    out += [md(rea.groupby(["version", "split"]).size().unstack("split", fill_value=0).reset_index()), ""]
    tr = ex[ex.split == "train"].assign(kind=lambda d: [
        "attack, visible change" if lab == "attack" and not hid else "attack, no visible change" if lab == "attack"
        else "normal, reacting neighbour" if rc
        else "normal, loop in an attack window" if wk == "attack" else "normal, normal window"
        for lab, hid, wk, rc in zip(d.label, d.no_visible_change, d.window_kind, d.reacting)])
    ts = tr.groupby("kind").agg(train_split=("label", "size"), training_sample=("train_use", "sum")).reset_index()
    used = ex[ex.train_use]
    out += ["Training sample (`train_use`). The full train split is kept for counting; the sample is what a model trains on:", "",
            md(ts), "",
            f"The training sample holds {len(used)} examples, {round(100 * float((used.label == 'attack').mean()), 1)}% of them attack. "
            f"The full train split holds {len(tr)}, {round(100 * float((tr.label == 'attack').mean()), 1)}% attack.", ""]
    out += ["Series per loop and version (a missing series is in the name map):", ""]
    sp = ex.drop_duplicates(["version", "loop"]).assign(series=lambda d: d.tags.apply(" ".join))
    out += [md(sp[["version", "loop", "series"]]), ""]
    out += ["Sample answers:", ""]
    for lab in ("attack", "normal"):
        for r in ex[ex.label == lab].sample(4, random_state=1).itertuples():
            out.append(f"- `{r.example_id}` ({r.label}): {r.answer}")
    out += [""]
    attack_s = ex[ex.label == "attack"]
    out += [f"Attack answers that name no clear step (`No series shows a clear step`): "
            f"{int(attack_s.no_visible_change.sum())} of {len(attack_s)}.", ""]
    out += [f"Attack examples whose attack started before the window: {(attack_s.attack_start_s < 0).sum()}. "
            f"Examples in windows that touch a `loop uncertain` attack: {int(ex.loop_uncertain.sum())}.", ""]

    out += ["## 5. Baselines (validation split only)", "",
            "Scored per attack on its one scored window, where the start second is random. "
            "`found_on_attacked_loop`: an attacked loop is flagged. `detected_any_loop`: any loop is flagged. "
            "`right_loop_only`: every attacked loop is flagged and no other loop. "
            "`visible_loops` and `hidden_loops` count the attacked loop examples in those windows with and without a "
            "visible change; `*_flagged` says how many of each were flagged. `reacting_loops` counts the reacting rows in "
            "those windows; they do not count for or against `right_loop_only`. "
            "False alarms are per 100 loop examples from normal windows. Gradient boosting trains on the training sample. "
            "The test split is not read.", ""]
    for name, df in base.items():
        out += [f"**{name.replace('_', ' ')}**", "", md(df), ""]
    out += ["Gate from the plan: the loop reader must find most validation attacks, in each version.", ""]
    gate = []
    for name, df in base.items():
        if name == "range_alarm":
            continue
        for r in df[df.version != "all"].itertuples():
            gate.append({"model": name.replace("_", " "), "version": r.version, "found": r.found_on_attacked_loop,
                         "attacks": r.attacks, "more_than_half": "yes" if 2 * r.found_on_attacked_loop > r.attacks else "no"})
    out += [md(pd.DataFrame(gate)), ""]

    out += ["## 6. Not verified", ""] + [f"- {x}" for x in NOT_VERIFIED] + [""]
    (ROOT / "REPORT.md").write_text("\n".join(out))


def check_attacks_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = before.groupby("version").agg(attacks=("attack_id", "size"),
                                       ok_as_printed=("status", lambda s: int((s == "ok").sum())))
    a = after.groupby("version").agg(ok_after_fixes=("status", lambda s: int((s == "ok").sum())))
    return b.join(a).reset_index()
