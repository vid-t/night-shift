import pandas as pd
import pytest

from hai.config import ENABLED, PROCESSED, RAW

PDF = RAW / "hai_dataset_technical_details.pdf"


def need(path):
    if not path.exists():
        pytest.skip(f"{path} missing: run `uv run hai-build` first")


@pytest.fixture(scope="session")
def examples():
    need(PROCESSED / "examples.parquet")
    return pd.read_parquet(PROCESSED / "examples.parquet")


@pytest.fixture(scope="session")
def windows():
    need(PROCESSED / "windows.parquet")
    return pd.read_parquet(PROCESSED / "windows.parquet")


@pytest.fixture(scope="session")
def checks():
    need(PROCESSED / "table_check.csv")
    return pd.read_csv(PROCESSED / "table_check.csv", dtype={"version": str})


def enabled():
    return ENABLED
