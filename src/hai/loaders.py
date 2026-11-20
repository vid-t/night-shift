"""One loader per HAI version. Each returns the same tidy table:
`time`, the standard (22.04) loop tags, and one `attack` flag (0/1).
A standard tag with no source column in that version is all NaN.
"""
import hashlib
from functools import lru_cache

import pandas as pd

from hai.config import FROZEN, PROCESSED, raw_path


@lru_cache
def tag_map() -> pd.DataFrame:
    return pd.read_csv(FROZEN / "tag_map.csv", dtype=str, keep_default_na=False)


def standard_tags() -> list[str]:
    return list(dict.fromkeys(tag_map()["standard"]))


def _tidy(df: pd.DataFrame, version: str, time_col: str, attack: pd.Series) -> pd.DataFrame:
    tm = tag_map()[lambda d: d.version == version]
    out = pd.DataFrame({"time": pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S")})
    for std, src in zip(tm.standard, tm.source):
        out[std] = df[src].astype("float64") if src else float("nan")
    out["attack"] = attack.astype("int8").to_numpy()
    return out


def read_raw(version: str, stem: str) -> pd.DataFrame:
    """The source file as published, all columns."""
    if version == "20.07":
        return pd.read_csv(raw_path(version, f"{stem}.csv.gz"), sep=";")
    if version == "21.03":
        return pd.read_csv(raw_path(version, f"{stem}.csv.gz"))
    return pd.read_csv(raw_path(version, f"{stem}.csv"))


def load_2007(stem: str) -> pd.DataFrame:
    df = read_raw("20.07", stem)
    return _tidy(df, "20.07", "time", df["attack"])


def load_2103(stem: str) -> pd.DataFrame:
    df = read_raw("21.03", stem)
    return _tidy(df, "21.03", "time", df["attack"])


def load_2204(stem: str) -> pd.DataFrame:
    df = read_raw("22.04", stem)
    return _tidy(df, "22.04", "timestamp", df["Attack"])


def load_2305(stem: str) -> pd.DataFrame:
    df = read_raw("23.05", stem)
    if stem.startswith("hai-test"):
        labels = pd.read_csv(raw_path("23.05", f"label-{stem.removeprefix('hai-')}.csv"))
        if len(labels) != len(df) or not (labels["timestamp"].to_numpy() == df["timestamp"].to_numpy()).all():
            raise ValueError(f"23.05 {stem}: label file does not line up with the data file")
        attack = labels["label"]
    else:
        attack = pd.Series(0, index=df.index)
    return _tidy(df, "23.05", "timestamp", attack)


LOADERS = {"20.07": load_2007, "21.03": load_2103, "22.04": load_2204, "23.05": load_2305}


def load(version: str, stem: str, cache: bool = True) -> pd.DataFrame:
    """Tidy table for one file, cached as parquet in data/processed/tidy/<tag map hash>."""
    key = hashlib.sha256((FROZEN / "tag_map.csv").read_bytes()).hexdigest()[:8]
    path = PROCESSED / "tidy" / key / version / f"{stem}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    df = LOADERS[version](stem)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def count_rows(version: str, stem: str) -> int:
    """Data rows in the raw file, counted from line breaks, independent of pandas."""
    import gzip

    name = f"{stem}.csv.gz" if version in ("20.07", "21.03") else f"{stem}.csv"
    opener = gzip.open if name.endswith(".gz") else open
    lines, last = 0, b"\n"
    with opener(raw_path(version, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1  # last line without a line break
    return lines - 1  # header
