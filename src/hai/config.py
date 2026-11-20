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


def split_of(version: str, stem: str) -> str | None:
    for split, by_version in SPLITS.items():
        if stem in by_version.get(version, []):
            return split
    return None


def raw_path(version: str, name: str) -> Path:
    return RAW / f"hai-{version}" / name
