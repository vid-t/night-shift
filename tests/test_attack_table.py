import pandas as pd
import pytest

from hai import attacks_pdf
from hai.check_table import attacks_table
from hai.config import ENABLED, FROZEN
from tests.conftest import PDF, need


def test_frozen_table_matches_pdf_extraction():
    need(PDF)
    frozen = pd.read_csv(FROZEN / "attacks.csv", dtype=str, keep_default_na=False)
    assert frozen.equals(attacks_pdf.build().astype(str))


def test_attack_counts_per_version():
    t = attacks_table()
    assert t.groupby("version").attack_id.nunique().to_dict() == attacks_pdf.EXPECTED_ATTACKS


def test_steps_share_start_and_duration():
    t = attacks_table()
    per = t.groupby(["version", "attack_id"]).agg(n=("loop", "size"), s=("start_time", "nunique"),
                                                   d=("duration_s", "nunique"))
    assert per.n.between(1, 2).all() and (per.s == 1).all() and (per.d == 1).all()


@pytest.mark.parametrize("version", ENABLED)
def test_every_attack_matches_the_flags(checks, version):
    c = checks[checks.version == version]
    assert len(c) == attacks_pdf.EXPECTED_ATTACKS[version]
    assert (c.status == "ok").all(), c[c.status != "ok"].to_string()
