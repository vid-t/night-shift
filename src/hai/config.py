"""Pinned source, file list, paths and splits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FROZEN = DATA / "frozen"

COMMIT = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
RAW_URL = f"https://raw.githubusercontent.com/icsdataset/hai/{COMMIT}"
LFS_URL = f"https://media.githubusercontent.com/media/icsdataset/hai/{COMMIT}"

VERSIONS = ["21.03", "22.04", "20.07", "23.05"]  # build order
ENABLED = ["21.03", "22.04", "20.07", "23.05"]  # versions added to the build so far

# (version, stem, url base, file name)
FILES = (
    [("20.07", f, RAW_URL, f"{f}.csv.gz") for f in ("test1", "test2", "train1")]
    + [("21.03", f, RAW_URL, f"{f}.csv.gz") for f in ("test1", "test2", "test3", "test4", "test5", "train1")]
    + [("22.04", f, LFS_URL, f"{f}.csv") for f in ("test1", "test2", "test3", "test4", "train1")]
    + [("23.05", f, LFS_URL, f"{f}.csv") for f in ("hai-test1", "hai-test2", "label-test1", "label-test2", "hai-train3")]
)
MANUAL = ("manual", RAW_URL, "hai_dataset_technical_details.pdf")

# data files per version (label files of 23.05 are not data files)
DATA_FILES = {
    "20.07": ["test1", "test2", "train1"],
    "21.03": ["test1", "test2", "test3", "test4", "test5", "train1"],
    "22.04": ["test1", "test2", "test3", "test4", "train1"],
    "23.05": ["hai-test1", "hai-test2", "hai-train3"],
}
NORMAL_FILES = {"20.07": "train1", "21.03": "train1", "22.04": "train1", "23.05": "hai-train3"}

SPLITS = {
    "train": {"20.07": ["test1"], "21.03": ["test1", "test2", "test3"], "22.04": ["test1", "test2"], "23.05": ["hai-test2"]},
    "validation": {"21.03": ["test4"], "22.04": ["test3"]},
    "test": {"20.07": ["test2"], "21.03": ["test5"], "22.04": ["test4"], "23.05": ["hai-test1"]},
}


# Table versions. v1 is the table Night Shift was trained and scored on; `hai-build` and `hai-export`
# write it by default, to the same paths as always. v2 answers the TimeNet review of PR #841: labels and
# intervals from the per-process flags where HAI has them (20.07, 21.03), all four 23.05 boiler setpoints
# matched (marked as data matches), no rationale for no_visible_change rows, exact window copies dropped,
# and an audit of the written table. v2 writes everything under data/processed/v2/.
TABLES = ("v1", "v2")
PROCESS_FLAG_VERSIONS = ("20.07", "21.03")  # versions with attack_P1, attack_P2, attack_P3
_state = {"table": "v1", "out": None}


def use_table(table: str, out: Path | None = None) -> None:
    """Select the table version (and optionally the output folder) for every later step of this process."""
    if table not in TABLES:
        raise ValueError(f"table {table!r} is not one of {TABLES}")
    _state["table"], _state["out"] = table, out
    from hai import loaders  # late import: loaders imports config

    loaders.tag_map.cache_clear()


def table_version() -> str:
    return _state["table"]


def out_dir() -> Path:
    """Where the build and the export write: data/processed for v1, data/processed/v2 for v2, or --out."""
    if _state["out"] is not None:
        return Path(_state["out"])
    return PROCESSED if _state["table"] == "v1" else PROCESSED / _state["table"]


def tidy_dir() -> Path:
    """The tidy parquet cache. v1 keeps its cache in data/processed/tidy; v2 has its own (it adds process flags)."""
    return PROCESSED / "tidy" if _state["table"] == "v1" else PROCESSED / _state["table"] / "tidy"


def tag_map_file() -> Path:
    return FROZEN / ("tag_map.csv" if _state["table"] == "v1" else f"tag_map_{_state['table']}.csv")


def split_of(version: str, stem: str) -> str | None:
    for split, by_version in SPLITS.items():
        if stem in by_version.get(version, []):
            return split
    return None


def raw_path(version: str, name: str) -> Path:
    return RAW / f"hai-{version}" / name
