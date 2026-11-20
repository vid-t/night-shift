"""Extract the attack tables from the HAI manual into data/frozen/attacks.csv.

The table text from pypdf is a stream of tokens: row number, attack ID,
one or two steps (primitive, scenario, points), a date that is printed only
when it changes, the start time to the minute, and the duration in seconds.
We walk the tokens with a small state machine, then apply CORRECTIONS:
each one is a documented fix of a typo in the manual, checked against the
data flags (see hai.check_table) or against the version's tag list.
"""
import re
import sys
from datetime import datetime

import pandas as pd
from pypdf import PdfReader

from hai.config import FROZEN, RAW

PAGES = {"20.07": [40, 41, 42], "21.03": [37, 38, 39], "22.04": [34, 35, 36], "23.05": [32, 33]}
EXPECTED_ATTACKS = {"20.07": 38, "21.03": 50, "22.04": 58, "23.05": 52}
COLUMNS = ["version", "attack_id", "start_time", "duration_s", "scenario", "loop", "points", "primitive", "note"]

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
RE_ID = re.compile(r"^A\d{3}$")
RE_PRIM = re.compile(r"^A[PE]\d{1,2}$")
RE_SCEN = re.compile(r"^P\d-[A-Z]{2}-")
RE_POINT = re.compile(r"^(P\d_\w+|\d{4}-\d{2}-[\w.]+|OCB\d+\.\w+)[,.]?$")
RE_MONTH = re.compile(r"^(" + "|".join(MONTHS) + r")\.$")
RE_DAY = re.compile(r"^\d{1,2},$")
RE_YEAR = re.compile(r"^20(19|2\d)$")
RE_TIME = re.compile(r"^\d{1,2}:\d{2}$")
RE_INT = re.compile(r"^\d{1,3}(,\d{3})*$|^\d+$")

# Fixes of the manual's typos. Key: (version, attack_id, primitive) -> fields to set, and why.
# Every fix here is listed in REPORT.md.
CORRECTIONS = {
    ("22.04", "A415", "AP3"): (
        {"primitive": "AP33"},
        "manual prints AP3; scenario P2-TC-SP2 on P2_VTR02 is AP33 everywhere else in 22.04",
    ),
}
# Attack-level fixes found by the flag check (hai.check_table). The flags are the source.
ATTACK_FIXES = {
    ("21.03", "A209"): ({"duration_s": 199}, "manual says 159 s; the flag is on for 199 s"),
    ("21.03", "A512"): ({"start_time": "2020-07-31 11:20"}, "manual says 11:21; the flag switches on at 11:20:59"),
}
# Point names in the manual that do not exist in that version's files, and what they mean.
POINT_FIXES = {
    ("21.03", "P2_LCV01D"): ("P3_LCV01D", "manual prints P2_LCV01D for scenario P3-LC-CO2; no such tag, the P3 valve is P3_LCV01D"),
    ("22.04", "P2_LCV01D"): ("P3_LCV01D", "manual prints P2_LCV01D for scenario P3-LC-CO2; no such tag, the P3 valve is P3_LCV01D"),
}


def tokens(pdf_path, pages: list[int]) -> list[str]:
    reader = PdfReader(pdf_path)
    text = "\n".join(reader.pages[p - 1].extract_text() for p in pages)
    text = text.split("ATTACK OPERATION", 1)[1]
    toks = text.split()
    out: list[str] = []
    for t in toks:
        # "P1-PC- CO1PV1" is one scenario broken by a space
        if out and out[-1].endswith("-") and RE_SCEN.match(out[-1]):
            out[-1] += t
        else:
            out.append(t)
    return out


def parse(version: str, pdf_path) -> list[dict]:
    toks = tokens(pdf_path, PAGES[version])
    attacks: list[dict] = []
    cur: dict | None = None
    expect = None  # "scenario", "points" or "duration"
    for t in toks:
        if RE_ID.match(t):
            cur = {"attack_id": t, "steps": [], "time": None, "duration": None}
            attacks.append(cur)
            expect = None
            continue
        if cur is None:
            continue
        if RE_PRIM.match(t):
            cur["steps"].append({"primitive": t, "scenario": None, "points": []})
            expect = "scenario"
            continue
        if expect == "scenario":
            if not RE_SCEN.match(t):
                raise ValueError(f"{version} {cur['attack_id']}: expected scenario, got {t!r}")
            cur["steps"][-1]["scenario"] = t
            expect = "points"
            continue
        if expect == "points":
            m = RE_POINT.match(t)
            if m:
                cur["steps"][-1]["points"].append(m.group(1))
                continue
            expect = None
        if RE_TIME.match(t):
            cur["time"] = t
            expect = "duration"
            continue
        if expect == "duration":
            if not RE_INT.match(t):
                raise ValueError(f"{version} {cur['attack_id']}: expected duration, got {t!r}")
            cur["duration"] = int(t.replace(",", ""))
            expect = None
        # anything else: dates (read below), row numbers, page numbers, column headers
    # dates carry forward; a year printed after the time still belongs to that date
    rows = []
    for a, (m, d, y) in zip(attacks, _dates(toks), strict=True):
        if a["time"] is None or a["duration"] is None or not a["steps"]:
            raise ValueError(f"{version} {a['attack_id']}: incomplete row {a}")
        hh, mm = map(int, a["time"].split(":"))
        start = datetime(y, m, d, hh, mm)
        for s in a["steps"]:
            rows.append({
                "version": version,
                "attack_id": a["attack_id"],
                "start_time": start.strftime("%Y-%m-%d %H:%M"),
                "duration_s": a["duration"],
                "scenario": s["scenario"],
                "loop": s["scenario"][:5],
                "points": " ".join(s["points"]),
                "primitive": s["primitive"],
                "note": "",
            })
    return rows


def _dates(toks: list[str]) -> list[tuple[int, int, int]]:
    """Date in force for each attack, in order. A date printed inside an attack's
    rows belongs to that attack; the year may come a few tokens after the day."""
    out = []
    month = day = year = None
    pending = False
    ids = [k for k, t in enumerate(toks) if RE_ID.match(t)] + [len(toks)]
    for a, b in zip(ids, ids[1:]):
        seg = toks[a:b]
        for k, t in enumerate(seg):
            if RE_MONTH.match(t) and k + 1 < len(seg) and RE_DAY.match(seg[k + 1]):
                month, day = MONTHS[t[:-1]], int(seg[k + 1][:-1])
                year = None
                pending = True
            elif pending and RE_YEAR.match(t) and not (k > 0 and RE_TIME.match(seg[k - 1])):
                year = int(t)
                pending = False
        if pending:
            raise ValueError(f"no year after date at {seg[:6]}")
        out.append((month, day, year))
    return out


def apply_fixes(rows: list[dict]) -> list[dict]:
    for r in rows:
        key = (r["version"], r["attack_id"], r["primitive"])
        notes = []
        if key[:2] in ATTACK_FIXES:
            fields, why = ATTACK_FIXES[key[:2]]
            r.update(fields)
            notes.append(why)
        if key in CORRECTIONS:
            fields, why = CORRECTIONS[key]
            r.update(fields)
            notes.append(why)
        pts = r["points"].split()
        fixed = []
        for p in pts:
            if (r["version"], p) in POINT_FIXES:
                new, why = POINT_FIXES[(r["version"], p)]
                fixed.append(new)
                notes.append(why)
            else:
                fixed.append(p)
        r["points"] = " ".join(fixed)
        r["note"] = "; ".join(dict.fromkeys(notes))
    return rows


def build(pdf_path=RAW / "hai_dataset_technical_details.pdf", fixes: bool = True) -> pd.DataFrame:
    """The attack table from the PDF; fixes=False gives the manual exactly as printed."""
    rows = []
    for v in PAGES:
        rows += parse(v, pdf_path)
    if fixes:
        rows = apply_fixes(rows)
    df = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    for v, n in EXPECTED_ATTACKS.items():
        got = df.loc[df.version == v, "attack_id"].nunique()
        if got != n:
            raise ValueError(f"{v}: parsed {got} attacks, manual says {n}")
    return df


def main() -> None:
    df = build()
    out = FROZEN / "attacks.csv"
    if "--check" in sys.argv:
        frozen = pd.read_csv(out, dtype=str, keep_default_na=False)
        fresh = df.astype(str)
        if not frozen.equals(fresh):
            sys.exit("data/frozen/attacks.csv differs from a fresh extraction of the PDF")
        print("attacks.csv matches the PDF extraction")
        return
    FROZEN.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("version").agg(attacks=("attack_id", "nunique"), steps=("attack_id", "size")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
