import pandas as pd
import pytest

from hai.check_table import attacks_table
from hai.config import DATA_FILES, ENABLED, NORMAL_FILES, raw_path
from hai.loaders import count_rows, load, standard_tags, tag_map, read_raw

FILES = [(v, s) for v in ENABLED for s in DATA_FILES[v]]

# check values for 21.03 from the task brief: rows, attacks, attack seconds
CHECK_2103 = {"train1": (216001, 0, 0), "test1": (43201, 5, 629), "test2": (118801, 20, 3449),
              "test3": (108001, 8, 1535), "test4": (39601, 5, 1157), "test5": (92401, 12, 2177)}


def runs(a):
    return int(((a.diff() == 1).sum()) + (a.iloc[0] == 1))


@pytest.mark.parametrize("version,stem", FILES)
def test_loader_shape_rows_and_gaps(version, stem):
    d = load(version, stem)
    assert list(d.columns) == ["time", *standard_tags(), "attack"]
    assert len(d) == count_rows(version, stem)
    assert (d.time.diff().dropna() == pd.Timedelta(seconds=1)).all()
    assert set(d.attack.unique()) <= {0, 1}


@pytest.mark.parametrize("version,stem", FILES)
def test_attack_seconds_equal_table_durations(version, stem, checks):
    d = load(version, stem)
    c = checks[(checks.version == version) & (checks.file == stem)]
    assert int(d.attack.sum()) == int(c.flag_duration.sum())
    t = attacks_table().drop_duplicates(["version", "attack_id"])
    t = t[(t.version == version) & t.attack_id.isin(c.attack_id)]
    assert int(d.attack.sum()) == int(t.duration_s.sum())
    assert runs(d.attack) == len(c)


@pytest.mark.skipif("21.03" not in ENABLED, reason="21.03 not built")
@pytest.mark.parametrize("stem", list(CHECK_2103))
def test_2103_check_values(stem):
    d = load("21.03", stem)
    assert (len(d), runs(d.attack), int(d.attack.sum())) == CHECK_2103[stem]


@pytest.mark.parametrize("version", ENABLED)
def test_normal_file_has_no_attacks(version):
    assert load(version, NORMAL_FILES[version]).attack.sum() == 0


@pytest.mark.parametrize("version,stem", FILES)
def test_tag_map_sources_exist(version, stem):
    cols = set(read_raw(version, stem).columns)
    tm = tag_map()[lambda d: (d.version == version) & (d.source != "")]
    assert set(tm.source) <= cols
