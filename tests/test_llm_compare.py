"""Tests of training/llm_compare.py. No network: a fake transport answers, or a local server on 127.0.0.1."""
import http.server
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tests.training_helpers import ROOT, TRAINING, load_script

lc = load_script("llm_compare")
score = load_script("score")

MODEL = "maker/some-model"
KEY = "sk-or-v1-SECRETKEY0123456789abcdef"
PRICING = {"prompt": "0.0000001", "completion": "0.0000004"}


# ---------------------------------------------------------------- fixtures

def series_row(eid, loop, label, start, kind, rng, attack_at=None):
    n = 300
    t = np.full(n, 50.0)
    m = 50 + rng.normal(0, 0.2, n)
    u = 30 + 0.5 * np.cumsum(t - m) * 0.01 + rng.normal(0, 0.05, n)
    if attack_at is not None:
        u[attack_at:] += 8.0
        m[attack_at + 5:] -= 2.0
    if label == "attack":
        answer = f"The command rises from second {start + 2} while the target stays flat. Change starts near second {start}. Answer: attack"
    else:
        answer = "The target and the measurement stay flat. The command stays flat. Answer: normal"
    return {"example_id": eid, "version": "22.04", "loop": loop, "loop_name": "boiler level", "split": "x",
            "label": label, "attack_start_s": start, "scored": label == "attack", "no_visible_change": False,
            "reacting": False, "window_kind": kind, "answer": answer,
            "series": [{"role": "target", "name": "target", "values": t.round(4).tolist(), "scale": 10.0},
                       {"role": "measurement", "name": "measurement", "values": m.round(4).tolist(), "scale": 8.0},
                       {"role": "command", "name": "command", "values": u.round(4).tolist(), "scale": 40.0}]}


@pytest.fixture(autouse=True)
def _own_compare_dir(tmp_path, monkeypatch):
    """The 5 USD limit counts spend under runs/llm_compare: point it at an empty folder so real runs never leak in."""
    monkeypatch.setattr(lc, "COMPARE_DIR", tmp_path / "compare_dir_of_this_test")


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(0)
    train = [series_row(f"22.04/tr/N{i:02d}/P1-LC", "P1-LC", "normal", -1, "normal", rng) for i in range(8)]
    train += [series_row(f"22.04/tr/A{i:02d}/P1-LC", "P1-LC", "attack", 100 + i, "attack", rng, 100 + i) for i in range(4)]
    val = [series_row(f"22.04/va/A{i:02d}/P1-LC", "P1-LC", "attack", 150, "attack", rng, 150) for i in range(3)]
    val += [series_row(f"22.04/va/N{i:02d}/P1-LC", "P1-LC", "normal", -1, "normal", rng) for i in range(5)]
    d = tmp_path / "data"
    d.mkdir()
    for name, rows in (("train.jsonl", train), ("validation.jsonl", val)):
        with open(d / name, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    key = tmp_path / "key.txt"
    key.write_text(KEY + "\n")
    return {"train": d / "train.jsonl", "rows": d / "validation.jsonl", "train_rows": train, "val_rows": val,
            "key": key, "out": tmp_path / "runs" / "llm_compare" / "run1.jsonl"}


class FakeTransport:
    """Answers the models list, the endpoints list and chat completions. `cost(body)` sets the reported cost."""

    def __init__(self, cost=lambda body: 0.0001, reply="The command rises. Change starts near second 150. Answer: attack",
                 delay=0.0, status=200, endpoints_pricing=None, ledger_probe=None):
        self.cost, self.reply, self.delay, self.status = cost, reply, delay, status
        self.endpoints_pricing = endpoints_pricing or [PRICING]
        self.gets, self.posts = [], []
        self.lock = threading.Lock()
        self.inflight = self.max_inflight = 0
        self.ledger_probe = ledger_probe

    def get(self, url, headers, timeout):
        with self.lock:
            self.gets.append((url, dict(headers)))
        if url.endswith("/models"):
            return 200, json.dumps({"data": [{"id": "other/model", "pricing": {"prompt": "1", "completion": "1"}},
                                             {"id": MODEL, "pricing": PRICING}]}).encode()
        if url.endswith("/endpoints"):
            return 200, json.dumps({"data": {"endpoints": [{"pricing": p} for p in self.endpoints_pricing]}}).encode()
        return 404, b""

    def post(self, url, headers, body, timeout):
        b = json.loads(body)
        with self.lock:
            self.posts.append((url, dict(headers), b))
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.ledger_probe:
                self.ledger_probe()
            time.sleep(self.delay)
            status = self.status(b) if callable(self.status) else self.status
            if status != 200:
                return status, json.dumps({"error": {"message": "nope"}}).encode()
            return 200, json.dumps({"choices": [{"message": {"content": self.reply}}],
                                    "usage": {"prompt_tokens": 1000, "completion_tokens": 30, "cost": self.cost(b)}}).encode()
        finally:
            with self.lock:
                self.inflight -= 1


def run(data, *args, transport=None, **kw):
    t = transport or FakeTransport()
    code = lc.main(["--model", MODEL, "--out", str(data["out"]), *args], transport=t, rows_path=data["rows"],
                   train_path=data["train"], sleep=lambda s: None, **kw)
    return code, t


def lines(path):
    return [json.loads(x) for x in open(path) if x.strip()]


# ---------------------------------------------------------------- prompts

def test_cold_prompt_has_loop_series_ranges_and_values(data):
    row = data["val_rows"][0]
    msgs = lc.build_messages(row, "cold", step=1)
    user = msgs[1]["content"]
    assert "boiler level" in user
    for s in row["series"]:
        assert f"{s['name']} ({s['role']}:" in user
        assert f"normal range {s['scale']:.4g}" in user
        assert ",".join(f"{v:.4g}" for v in s["values"]) in user
    assert "Solved window" not in user and "This is the" not in user
    step3 = lc.build_messages(row, "cold", step=3)[1]["content"]
    assert ",".join(f"{v:.4g}" for v in row["series"][1]["values"][::3]) + "\n" in step3 + "\n"
    assert "one value every 3 seconds, at seconds 0, 3, 6, ... 297" in step3


def test_examples_prompt_is_deterministic_and_balanced(data):
    train = data["train_rows"]
    a = lc.pick_examples(train, "P1-LC", 4)
    b = lc.pick_examples(list(reversed(train)), "P1-LC", 4)
    assert [e["example_id"] for e in a] == [e["example_id"] for e in b]
    assert [e["label"] for e in a] == ["attack", "normal", "attack", "normal"]
    row = data["val_rows"][0]
    user = lc.build_messages(row, "examples", step=1, examples=a, example_step=5)[1]["content"]
    assert user.count("Solved window") == 4
    for e in a:
        assert f"Right reply: {e['answer']}" in user
        assert ",".join(f"{v:.4g}" for v in e["series"][2]["values"][::5]) in user
    assert user.index("Solved window 4") < user.index("The window to judge")


def test_hint_lines_equal_the_dataset_text_lines_on_the_fixture(data, monkeypatch):
    pytest.importorskip("torch")
    rows, train = data["val_rows"], data["train_rows"]
    hints = lc.hint_lines_for(rows, train)
    # the dataset's own lines, built independently
    for k, v in (("HAI_SERIES_VIEW", "looprange"), ("HAI_TEXT_CLAUSES", "step,law"), ("HAI_ANSWER_ORDER", "label_last")):
        monkeypatch.setenv(k, v)
    hai = lc.load_dataset_module()
    cls = hai.HAIAttackQADataset
    cls.looprange_units, cls.step_units = cls._units(train), cls._step_units(train)
    cls.control_laws, cls.plant_models = cls._laws(train), None
    ds = cls.__new__(cls)
    ds.EOS_TOKEN = ""
    for r in rows:
        want = cls._format_sample(ds, r)["time_series_text"]
        assert hints[r["example_id"]] == want
        assert any("largest one-second step" in line for line in want)
        user = lc.build_messages(r, "hints", hint_lines=hints[r["example_id"]])[1]["content"]
        for line in want:
            assert line in user.split("\n")


def test_hint_lines_equal_curve_check_on_real_rows():
    pytest.importorskip("torch")
    val, train = lc.ROWS_PATH, lc.TRAIN_PATH
    cc_path = ROOT / "runs/nebius/2026-11-21/night/analysis/curvecheck/curve_check.py"
    if not (val.exists() and train.exists() and cc_path.exists()):
        pytest.skip("validation/train rows or curve_check.py missing")
    rows = lc.read_jsonl(val)[:12]
    ours = lc.hint_lines_for(rows, lc.read_jsonl(train))
    import importlib.util
    spec = importlib.util.spec_from_file_location("curve_check_for_test", cc_path)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    saved = {k: os.environ.get(k) for k in ("HAI_SERIES_VIEW", "HAI_TEXT_CLAUSES", "HAI_ANSWER_ORDER")}
    try:
        hai, ds = cc.setup_dataset("looprange", "step,law")
        theirs = {r["example_id"]: hai.HAIAttackQADataset._format_sample(ds, r)["time_series_text"] for r in rows}
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert ours == theirs
    assert any("opposite way to what the controller would do" in line for v in ours.values() for line in v)


def test_the_asked_format_is_what_score_parse_accepts(data):
    system = lc.build_messages(data["val_rows"][0], "cold")[0]["content"]
    assert lc.FORMAT_ATTACK in system and lc.FORMAT_NORMAL in system
    assert score.parse(lc.FORMAT_ATTACK) == ("attack", 118)
    assert score.parse(lc.FORMAT_NORMAL) == ("normal", None)
    assert score.parse("The change was there from the start. Change starts near second 0. Answer: attack") == ("attack", 0)
    for r in data["train_rows"]:
        assert score.parse(r["answer"])[0] == r["label"]


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("bad", ["rows", "ids", "out"])
def test_refuses_files_named_test(data, tmp_path, capsys, bad):
    t = FakeTransport()
    args = ["--mode", "cold", "--max-usd", "1"]
    rows_path = data["rows"]
    if bad == "rows":
        rows_path = tmp_path / "test.jsonl"
        rows_path.write_text(data["rows"].read_text())
    if bad == "ids":
        ids = tmp_path / "my_test_ids.txt"
        ids.write_text(data["val_rows"][0]["example_id"] + "\n")
        args += ["--ids", str(ids)]
    out = tmp_path / ("Test_run.jsonl" if bad == "out" else "run.jsonl")
    code = lc.main(["--model", MODEL, "--out", str(out), "--key-file", str(data["key"]), *args], transport=t,
                   rows_path=rows_path, train_path=data["train"], sleep=lambda s: None)
    assert code == 2
    assert "contains 'test'" in capsys.readouterr().err
    assert not t.posts and not out.exists()


@pytest.mark.parametrize("args", [["--max-usd", "5.01"], ["--max-usd", "0"], []])
def test_refuses_a_cap_above_5_or_none(data, capsys, args):
    code, t = run(data, "--mode", "cold", "--key-file", str(data["key"]), *args)
    assert code == 2
    assert "--max-usd" in capsys.readouterr().err
    assert not t.posts and not t.gets


def test_the_5_usd_limit_counts_earlier_runs_in_the_folder(data, capsys):
    data["out"].parent.mkdir(parents=True)
    (data["out"].parent / "earlier.jsonl").write_text(json.dumps({"example_id": "x", "cost_usd": 4.9999}) + "\n")
    code, t = run(data, "--mode", "cold", "--max-usd", "1", "--key-file", str(data["key"]))
    assert code == 3
    assert sum(o["cost_usd"] for o in lines(data["out"])) <= 0.0001 + 1e-12
    assert "4.9999 USD" in capsys.readouterr().out


def test_the_5_usd_limit_counts_runs_anywhere_under_the_compare_folder(tmp_path):
    compare = tmp_path / "llm_compare"
    (compare / "kimi").mkdir(parents=True)
    (compare / "kimi" / "a.jsonl").write_text(json.dumps({"cost_usd": 1.5}) + "\n")
    (compare / "b.jsonl").write_text(json.dumps({"cost_usd": 0.25}) + "\n")
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "c.jsonl").write_text(json.dumps({"cost_usd": 1.0}) + "\n")
    assert lc.spent_before(elsewhere, compare) == pytest.approx(2.75)
    assert lc.spent_before(compare, compare) == pytest.approx(1.75)
    (compare / "b.jsonl").write_text('{"cost_usd": NaN}\n')
    with pytest.raises(lc.Refused):
        lc.spent_before(elsewhere, compare)


def test_refuses_variable_prices():
    with pytest.raises(lc.Refused):
        lc.prices_from({"id": "m", "pricing": {"prompt": "-1", "completion": "-1"}}, [])


def test_worst_prices_take_overrides_providers_and_cache_writes_under_the_ceiling():
    entry = {"id": "m", "pricing": {"prompt": "0.0000001", "completion": "0.0000002", "input_cache_write": "0.0000003",
                                    "overrides": [{"prompt": "0.0000002", "completion": "0.0000005"}]}}
    eps = [{"pricing": {"prompt": "0.00000015", "completion": "0.0000007", "discount": 0.5}},
           {"pricing": {"prompt": "0.000003", "completion": "0.00001"}},          # 30 times dearer: left out
           {"pricing": {"prompt": "0.0000001", "completion": "0.0000002",
                        "overrides": [{"prompt": "0.0000009", "completion": "0.000002"}]}}]
    p = lc.prices_from(entry, eps, factor=4)
    assert p["list_in"] == 1e-7 and p["list_out"] == 2e-7 and p["providers"] == 2
    assert p["max_in"] == 3e-7 and p["max_prompt"] == 2e-7
    assert p["max_out"] == pytest.approx(7e-7) and p["max_completion"] == pytest.approx(7e-7)
    with pytest.raises(lc.Refused):
        lc.prices_from(entry, eps[1:2], factor=4)


# ---------------------------------------------------------------- cost control

def test_ledger_never_reserves_past_the_cap():
    led = lc.Ledger(1.0)
    assert led.reserve(0.4) == 0.4 and led.reserve(0.4) == 0.4
    assert led.reserve(0.4) is None              # 0.8 reserved + 0.4 > 1
    led.settle(0.4, 0.1)
    assert led.reserve(0.4) == 0.4               # 0.1 spent + 0.4 + 0.4 = 0.9
    assert led.reserve(0.2) is None
    led.settle(0.4, 0.8)                         # cost twice the reservation
    assert led.factor == pytest.approx(2.0)
    assert led.spent == pytest.approx(0.9)
    assert led.reserve(0.05) is None             # 0.9 + 0.4 + 0.1 > 1


def test_a_hard_bound_on_input_tokens():
    msgs = [{"role": "system", "content": "abc"}, {"role": "user", "content": "1.234,5.678,ü"}]
    assert lc.bound_tokens(msgs) >= len("abc") + len("1.234,5.678,ü".encode()) + lc.TEMPLATE_TOKENS
    assert lc.estimate_tokens(msgs) <= lc.bound_tokens(msgs)


def _probe(state):
    def check():
        led = state.get("ledger")
        if led is not None:
            with led.lock:
                assert led.spent + led.reserved <= led.cap + 1e-12
    return check


@pytest.mark.parametrize("concurrency", ["1", "8"])
def test_spend_never_exceeds_the_cap_with_concurrency(data, monkeypatch, concurrency):
    state = {}
    real = lc.Ledger

    class Watched(real):
        def __init__(self, cap):
            super().__init__(cap)
            state["ledger"] = self
    monkeypatch.setattr(lc, "Ledger", Watched)
    worst = lc.worst_cost(lc.build_messages(data["val_rows"][0], "cold"), 160,
                          lc.prices_from({"id": MODEL, "pricing": PRICING}, [{"pricing": PRICING}]))
    # each request reports 80% of its worst case
    t = FakeTransport(cost=lambda b: 0.8 * worst, delay=0.01, ledger_probe=_probe(state))
    cap = 4.5 * worst
    code, _ = run(data, "--mode", "cold", "--max-usd", f"{cap}", "--concurrency", concurrency,
                  "--key-file", str(data["key"]), transport=t)
    spent = sum(o["cost_usd"] for o in lines(data["out"]))
    assert code == 3
    assert spent <= cap
    assert 1 <= len(t.posts) < len(data["val_rows"])
    led = state["ledger"]
    assert led.reserved == pytest.approx(0.0) and led.spent == pytest.approx(spent)


def test_a_response_dearer_than_its_reservation_raises_later_reservations(data, monkeypatch):
    state = {}
    real = lc.Ledger

    class Watched(real):
        def __init__(self, cap):
            super().__init__(cap)
            state["ledger"] = self
    monkeypatch.setattr(lc, "Ledger", Watched)
    prices = lc.prices_from({"id": MODEL, "pricing": PRICING}, [{"pricing": PRICING}])
    worst = lc.worst_cost(lc.build_messages(data["val_rows"][0], "cold"), 160, prices)
    calls = []

    def cost(b):
        calls.append(1)
        return 1.6 * worst if len(calls) == 1 else 0.5 * worst
    t = FakeTransport(cost=cost, ledger_probe=_probe(state))
    cap = 4.0 * worst
    code, _ = run(data, "--mode", "cold", "--max-usd", f"{cap}", "--concurrency", "1",
                  "--key-file", str(data["key"]), transport=t)
    spent = sum(o["cost_usd"] for o in lines(data["out"]))
    assert code == 3
    assert spent <= cap
    assert state["ledger"].factor >= 1.6 - 1e-9
    # after the overrun each reservation is 1.6 worst: 1.6 + 0.5 + 0.5 spent, then 2.6 + 1.6 > 4
    assert len(t.posts) == 3


def test_a_timeout_books_the_whole_reservation(data):
    class Flaky(FakeTransport):
        def post(self, url, headers, body, timeout):
            with self.lock:
                self.posts.append((url, dict(headers), json.loads(body)))
            raise lc.TransportError("TimeoutError")
    t = Flaky()
    code, _ = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "1", "--key-file", str(data["key"]), transport=t)
    (rec,) = lines(data["out"])
    assert rec["error"].startswith("transport error") and rec["generated"] == ""
    assert rec["cost_source"] == "worst_case"
    assert len(t.posts) == lc.MAX_RETRIES + 1
    assert rec["cost_usd"] > 0


class Raw(FakeTransport):
    """Answers every chat completion with the given status and raw bytes."""

    def __init__(self, status, raw):
        super().__init__()
        self.raw_status, self.raw = status, raw

    def post(self, url, headers, body, timeout):
        with self.lock:
            self.posts.append((url, dict(headers), json.loads(body)))
        return self.raw_status, self.raw


@pytest.mark.parametrize("bad", ['NaN', 'Infinity', '-1.0', 'true'])
def test_a_bad_reported_cost_never_lifts_the_cap(data, bad):
    raw = ('{"choices": [{"message": {"content": "Flat. Answer: normal"}}], "usage": {"cost": %s, '
           '"prompt_tokens": -1000000, "completion_tokens": 5}}' % bad).encode()
    worst = lc.worst_cost(lc.build_messages(data["val_rows"][0], "cold"), 160,
                          lc.prices_from({"id": MODEL, "pricing": PRICING}, [{"pricing": PRICING}]))
    t = Raw(200, raw)
    code, _ = run(data, "--mode", "cold", "--max-usd", f"{2.5 * worst}", "--concurrency", "1",
                  "--key-file", str(data["key"]), transport=t)
    recs = lines(data["out"])
    assert code == 3 and len(t.posts) <= 3
    assert all(r["cost_source"] == "worst_case" and r["cost_usd"] > 0 for r in recs)


def test_an_unreadable_200_is_booked_at_its_reservation(data):
    t = Raw(200, json.dumps({"choices": "x", "usage": 5}).encode())
    code, _ = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "2", "--key-file", str(data["key"]), transport=t)
    recs = lines(data["out"])
    assert code == 0 and len(recs) == 2
    assert all(r["error"] and r["cost_source"] == "worst_case" and r["cost_usd"] > 0 for r in recs)


def test_no_key_prefix_survives_the_error_cut(data, capsys):
    t = Raw(400, ("x" * 285 + KEY).encode())
    run(data, "--mode", "cold", "--max-usd", "1", "--limit", "1", "--key-file", str(data["key"]), transport=t)
    text = data["out"].read_text() + capsys.readouterr().err
    assert KEY[:12] not in text and "***" in text


def test_a_missing_key_file_is_refused_before_any_post(data, tmp_path, capsys):
    code, t = run(data, "--mode", "cold", "--max-usd", "1", "--key-file", str(tmp_path / "absent.key"))
    assert code == 2 and not t.posts and "could not read the key file" in capsys.readouterr().err


def test_retries_on_429_and_5xx_then_answers(data):
    seen = []

    def status(b):
        seen.append(1)
        return [429, 503, 200][min(len(seen) - 1, 2)]
    t = FakeTransport(status=status)
    code, _ = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "1", "--key-file", str(data["key"]), transport=t)
    assert code == 0 and len(t.posts) == 3
    (rec,) = lines(data["out"])
    assert rec["error"] == "" and score.parse(rec["generated"]) == ("attack", 150)


def test_dry_run_makes_no_paid_call_and_writes_samples(data, capsys):
    code, t = run(data, "--mode", "examples", "--dry-run")
    assert code == 0 and not t.posts
    out = capsys.readouterr().out
    assert f"{len(data['val_rows'])} rows" in out and "estimated input tokens" in out and "USD" in out
    samples = data["out"].with_name("run1.samples.txt").read_text()
    assert samples.count("=====") == 3 and "Solved window 4" in samples
    assert not data["out"].exists()


# ---------------------------------------------------------------- output, resume, secrets, headers

def test_output_resume_and_results_table(data, capsys):
    code, t = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "5", "--key-file", str(data["key"]))
    assert code == 0 and len(t.posts) == 5
    recs = lines(data["out"])
    assert set(recs[0]) >= {"example_id", "generated", "model", "mode", "prompt_tokens", "completion_tokens",
                            "cost_usd", "seconds", "error"}
    code, t2 = run(data, "--mode", "cold", "--max-usd", "1", "--key-file", str(data["key"]))
    assert code == 0 and len(t2.posts) == len(data["val_rows"]) - 5
    ids = [r["example_id"] for r in lines(data["out"])]
    assert sorted(ids) == sorted(r["example_id"] for r in data["val_rows"])
    code, t3 = run(data, "--mode", "hints", "--max-usd", "1", "--key-file", str(data["key"]))
    assert code == 2 and not t3.posts   # another mode in the same file
    table = load_script("results_table")
    table.main(["--truth", str(data["rows"]), "--run", "big", str(data["out"])])
    assert "| big |" in capsys.readouterr().out


def test_errored_rows_are_retried_on_rerun(data):
    t = FakeTransport(status=lambda b: 400)
    run(data, "--mode", "cold", "--max-usd", "1", "--limit", "2", "--key-file", str(data["key"]), transport=t)
    assert all(r["error"] and r["generated"] == "" for r in lines(data["out"]))
    _, t2 = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "2", "--key-file", str(data["key"]))
    assert len(t2.posts) == 2


def test_the_key_never_shows(data, capsys):
    class Echo(FakeTransport):
        def post(self, url, headers, body, timeout):
            super().post(url, headers, body, timeout)
            if len(self.posts) == 1:
                raise lc.TransportError(f"boom {headers['Authorization']}")
            return 400, json.dumps({"error": {"message": f"bad key {KEY}"}}).encode()
    t = Echo()
    run(data, "--mode", "cold", "--max-usd", "1", "--limit", "2", "--concurrency", "1",
        "--key-file", str(data["key"]), transport=t)
    run(data, "--mode", "cold", "--dry-run")
    cap = capsys.readouterr()
    assert KEY not in cap.out and KEY not in cap.err
    for f in data["out"].parent.iterdir():
        assert KEY not in f.read_text()
    assert t.posts[0][1]["Authorization"] == f"Bearer {KEY}"


def test_requests_carry_only_authorization_and_content_type(data):
    code, t = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "2", "--key-file", str(data["key"]),
                  "--reasoning", "low")
    assert code == 0
    for _, headers, body in t.posts:
        assert set(headers) == {"Authorization", "Content-Type"}
        assert body["temperature"] == 0 and body["max_tokens"] == 160 and body["reasoning"] == {"effort": "low"}
        assert body["provider"]["require_parameters"] is True and body["provider"]["sort"] == "price"
        assert body["provider"]["max_price"] == {"prompt": pytest.approx(0.1), "completion": pytest.approx(0.4)}
        assert not {"user", "session_id", "metadata"} & set(body)
    assert all(h == {} for _, h in t.gets)


def test_the_real_http_client_sends_no_identifying_headers(data):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self, payload):
            seen.append((self.command, self.path, {k.lower() for k in self.headers.keys()}))
            data_ = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data_)))
            self.end_headers()
            self.wfile.write(data_)

        def do_GET(self):
            if self.path.endswith("/endpoints"):
                self._answer({"data": {"endpoints": [{"pricing": PRICING}]}})
            else:
                self._answer({"data": [{"id": MODEL, "pricing": PRICING}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._answer({"choices": [{"message": {"content": lc.FORMAT_NORMAL}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00001}})

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        code = lc.main(["--model", MODEL, "--mode", "cold", "--out", str(data["out"]), "--max-usd", "1", "--limit", "1",
                        "--key-file", str(data["key"]), "--base-url", f"http://127.0.0.1:{srv.server_address[1]}/api/v1"],
                       rows_path=data["rows"], train_path=data["train"])
    finally:
        srv.shutdown()
    assert code == 0
    protocol = {"host", "accept-encoding", "content-length"}
    gets = [h for m, _, h in seen if m == "GET"]
    posts = [h for m, _, h in seen if m == "POST"]
    assert len(gets) == 2 and all(h <= protocol for h in gets)
    assert len(posts) == 1 and posts[0] <= protocol | {"authorization", "content-type"}
    assert {"authorization", "content-type"} <= posts[0]
    assert lines(data["out"])[0]["generated"] == lc.FORMAT_NORMAL


def test_plain_http_is_refused_off_localhost():
    with pytest.raises(lc.Refused):
        lc.check_base_url("http://openrouter.ai/api/v1")
    assert lc.check_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------- test rows behind --rows test --allow-test

def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


@pytest.fixture()
def fake_test(data, tmp_path):
    """A small fake test.jsonl in its own temporary folder (never the real one)."""
    rng = np.random.default_rng(7)
    rows = [series_row(f"22.04/te/A{i:02d}/P1-LC", "P1-LC", "attack", 140, "attack", rng, 140) for i in range(2)]
    rows += [series_row(f"22.04/te/N{i:02d}/P1-LC", "P1-LC", "normal", -1, "normal", rng) for i in range(3)]
    folder = tmp_path / "faketest"
    return {"rows": rows, "path": _write_rows(folder / "test.jsonl", rows), "folder": folder}


def run_test_rows(data, test_path, *args, out=None, transport=None):
    t = transport or FakeTransport()
    code = lc.main(["--model", MODEL, "--out", str(out or data["out"]), "--key-file", str(data["key"]), *args],
                   transport=t, rows_path=data["rows"], train_path=data["train"], sleep=lambda s: None,
                   test_rows_path=test_path)
    return code, t


def test_test_rows_are_answered_with_both_flags(data, fake_test, capsys):
    code, t = run_test_rows(data, fake_test["path"], "--mode", "cold", "--max-usd", "1", "--rows", "test", "--allow-test")
    assert code == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == f"answering TEST rows: {len(fake_test['rows'])} rows"
    assert sorted(r["example_id"] for r in lines(data["out"])) == sorted(r["example_id"] for r in fake_test["rows"])
    assert len(t.posts) == len(fake_test["rows"])


def test_test_rows_with_ids_and_limit_print_the_selected_count(data, fake_test, tmp_path, capsys):
    ids = tmp_path / "final_ids.txt"
    ids.write_text("\n".join(r["example_id"] for r in fake_test["rows"][:3]) + "\n")
    code, _ = run_test_rows(data, fake_test["path"], "--mode", "cold", "--dry-run", "--rows", "test", "--allow-test",
                            "--ids", str(ids))
    assert code == 0
    assert "answering TEST rows: 3 rows" in capsys.readouterr().out


@pytest.mark.parametrize("flags, needle", [
    (["--rows", "test"], "needs --allow-test"),                    # no flag
    (["--allow-test"], "only valid together with --rows test"),     # flag without --rows test
    (["--rows", "validation", "--allow-test"], "only valid together with --rows test"),
])
def test_test_rows_need_both_flags(data, fake_test, capsys, flags, needle):
    code, t = run_test_rows(data, fake_test["path"], "--mode", "cold", "--max-usd", "1", *flags)
    assert code == 2
    assert needle in capsys.readouterr().err
    assert not t.posts and not t.gets and not data["out"].exists()


@pytest.mark.parametrize("name", ["test_shuffled.jsonl", "test_full.jsonl", "my_test.jsonl", "test.json", "TEST.jsonl"])
def test_other_test_files_stay_refused_even_with_both_flags(data, fake_test, capsys, name):
    other = _write_rows(fake_test["folder"] / name, fake_test["rows"])
    code, t = run_test_rows(data, other, "--mode", "cold", "--max-usd", "1", "--rows", "test", "--allow-test")
    assert code == 2
    assert f"refusing {name}" in capsys.readouterr().err
    assert not t.posts and not data["out"].exists()


@pytest.mark.parametrize("name", ["test_shuffled.jsonl", "test_full.jsonl", "test.jsonl"])
def test_test_files_as_validation_rows_stay_refused(data, fake_test, tmp_path, capsys, name):
    other = _write_rows(tmp_path / "elsewhere" / name, fake_test["rows"])
    t = FakeTransport()
    code = lc.main(["--model", MODEL, "--out", str(data["out"]), "--mode", "cold", "--max-usd", "1",
                    "--key-file", str(data["key"])], transport=t, rows_path=other, train_path=data["train"],
                   sleep=lambda s: None, test_rows_path=fake_test["path"])
    assert code == 2 and "contains 'test'" in capsys.readouterr().err and not t.posts


@pytest.mark.parametrize("bad", ["out", "ids"])
def test_out_and_ids_named_test_stay_refused_with_test_rows(data, fake_test, tmp_path, capsys, bad):
    args = ["--mode", "cold", "--max-usd", "1", "--rows", "test", "--allow-test"]
    out = tmp_path / "runs" / ("final_TEST.jsonl" if bad == "out" else "final.jsonl")
    if bad == "ids":
        ids = tmp_path / "test_ids.txt"
        ids.write_text(fake_test["rows"][0]["example_id"] + "\n")
        args += ["--ids", str(ids)]
    code, t = run_test_rows(data, fake_test["path"], *args, out=out)
    err = capsys.readouterr()
    assert code == 2 and "contains 'test'" in err.err
    assert "answering TEST rows" not in err.out    # refused before the rows are read
    assert not t.posts and not out.exists()


@pytest.mark.parametrize("via", ["out", "ids", "rows", "test_rows", "train"])
def test_symlinks_to_test_files_stay_refused(data, fake_test, tmp_path, capsys, via):
    target = _write_rows(fake_test["folder"] / "test_shuffled.jsonl", fake_test["rows"])
    links = tmp_path / "links"
    links.mkdir()
    link = links / {"out": "final.jsonl", "ids": "final_ids.txt", "rows": "validation.jsonl",
                    "test_rows": "test.jsonl", "train": "train.jsonl"}[via]
    link.symlink_to(target)
    before = target.read_text()
    args = ["--model", MODEL, "--mode", "examples", "--max-usd", "1", "--key-file", str(data["key"]),
            "--out", str(link if via == "out" else data["out"])]
    if via == "ids":
        args += ["--ids", str(link)]
    if via == "test_rows":
        args += ["--rows", "test", "--allow-test"]
    t = FakeTransport()
    code = lc.main(args, transport=t, rows_path=link if via == "rows" else data["rows"],
                   train_path=link if via == "train" else data["train"], sleep=lambda s: None,
                   test_rows_path=link if via == "test_rows" else fake_test["path"])
    err = capsys.readouterr().err
    assert code == 2 and "refusing test_shuffled.jsonl" in err
    assert fake_test["rows"][0]["example_id"] not in err
    assert not t.posts and target.read_text() == before


def test_test_rows_take_examples_and_statistics_from_train_only(data, fake_test, monkeypatch):
    seen = {}
    real_pick, real_hints = lc.pick_examples, lc.hint_lines_for

    def pick(train, loop, n, seed=lc.EXAMPLE_SEED):
        seen["pick"] = {r["example_id"] for r in train}
        return real_pick(train, loop, n, seed)

    def hints(rows, train):
        seen["stats"] = {r["example_id"] for r in train}
        return {r["example_id"]: [f"line {i} of {r['example_id']}:" for i in range(len(r["series"]))] for r in rows}
    monkeypatch.setattr(lc, "pick_examples", pick)
    monkeypatch.setattr(lc, "hint_lines_for", hints)
    code, _ = run_test_rows(data, fake_test["path"], "--mode", "hints_examples", "--dry-run", "--rows", "test",
                            "--allow-test")
    assert code == 0
    train_ids = {r["example_id"] for r in data["train_rows"]}
    assert seen["pick"] == train_ids and seen["stats"] == train_ids


# ---------------------------------------------------------------- mode hints_examples

EXAMPLE_LIST = __import__("re").compile(r"-?\d[\d.e+-]*(?:,-?\d[\d.e+-]*){3,}")   # four or more numbers in a list


def _hints_examples_prompt(data):
    pytest.importorskip("torch")
    train, row = data["train_rows"], data["val_rows"][0]
    shots = lc.pick_examples(train, row["loop"], 6)
    hints = lc.hint_lines_for([row] + shots, train)
    user = lc.build_messages(row, "hints_examples", step=1, examples=shots, hint_lines=hints[row["example_id"]],
                             example_hints=hints)[1]["content"]
    return user, shots, hints, row


def test_hints_examples_prompt_on_the_fixture(data):
    user, shots, hints, row = _hints_examples_prompt(data)
    assert sorted(e["label"] for e in shots) == ["attack"] * 3 + ["normal"] * 3
    assert user.count("Solved window") == 6
    lines_ = user.split("\n")
    head = user[: user.index("Now the window to judge")]
    for i, e in enumerate(shots, 1):
        block = head[head.index(f"Solved window {i}:"):].split("\n\n")[0].split("\n")
        assert block[1:-1] == hints[e["example_id"]]                    # the dataset's lines, exactly, in order
        assert block[-1] == f"Right reply: {e['answer']}"
        for line in hints[e["example_id"]]:
            assert line in lines_
        for s in e["series"]:                                           # no raw example numbers
            assert lc.values_text(s["values"][:4], 1) not in head
            assert lc.values_text(s["values"], 5)[:40] not in head
    assert not EXAMPLE_LIST.search(head)
    # the window to judge still has its own values
    for s in row["series"]:
        assert lc.values_text(s["values"], 1) in user


def test_hints_examples_is_deterministic(data):
    a, *_ = _hints_examples_prompt(data)
    data["train_rows"] = list(reversed(data["train_rows"]))
    b, *_ = _hints_examples_prompt(data)
    assert a == b


def test_hints_examples_default_six_examples_through_main(data, capsys):
    pytest.importorskip("torch")
    code, t = run(data, "--mode", "hints_examples", "--dry-run")
    assert code == 0 and not t.posts
    samples = data["out"].with_name("run1.samples.txt").read_text()
    assert "Solved window 6:" in samples and "Solved window 7:" not in samples
    code, _ = run(data, "--mode", "hints_examples", "--dry-run", "--examples", "2")
    samples = data["out"].with_name("run1.samples.txt").read_text()
    assert "Solved window 2:" in samples and "Solved window 3:" not in samples
    code, _ = run(data, "--mode", "hints_examples", "--dry-run", "--examples", "0")
    assert code == 2


def test_hints_examples_needs_hint_lines_for_every_example(data):
    row, shots = data["val_rows"][0], lc.pick_examples(data["train_rows"], "P1-LC", 6)
    fake = ["a:", "b:", "c:"]
    with pytest.raises(ValueError):
        lc.build_messages(row, "hints_examples", examples=shots, hint_lines=fake, example_hints={})


# ---------------------------------------------------------------- base rate, hint placement

@pytest.mark.parametrize("mode", lc.MODES)
def test_the_base_rate_sentence_is_in_every_mode(data, mode):
    row = data["val_rows"][0]
    shots = lc.pick_examples(data["train_rows"], "P1-LC", 6)
    fake = {e["example_id"]: [f"fact {i} of {e['example_id']}:" for i in range(3)] for e in shots + [row]}
    system = lc.build_messages(row, mode, examples=shots, hint_lines=fake[row["example_id"]], example_hints=fake)[0]["content"]
    assert lc.BASE_RATE in system
    assert "Most windows are normal" in system and "about one window in five shows an attack" in system


def test_the_base_rate_matches_the_fixture_counting_rule():
    # the share written into BASE_RATE is the share of train.jsonl (776 attack of 3880); the rule is a label count
    rows = [{"label": "attack"}] * 776 + [{"label": "normal"}] * 3104
    assert sum(r["label"] == "attack" for r in rows) / len(rows) == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["hints", "hints_examples"])
def test_each_hint_line_sits_directly_before_its_series(data, mode):
    row = data["val_rows"][0]
    shots = lc.pick_examples(data["train_rows"], "P1-LC", 6)
    fake = {e["example_id"]: [f"This is the {s['name']} of {e['example_id']}; fact:" for s in e["series"]]
            for e in shots + [row]}
    user = lc.build_messages(row, mode, examples=shots, hint_lines=fake[row["example_id"]], example_hints=fake)[1]["content"]
    ls = user.split("\n")
    for s, line in zip(row["series"], fake[row["example_id"]]):
        i = ls.index(line)
        assert ls[i + 1] == lc.values_text(s["values"], 1)
        assert ls[i - 1].startswith(f"{s['name']} ({s['role']}:")
    # no separate block of hint lines: every line of the row appears exactly once
    assert all(user.count(line) == 1 for line in fake[row["example_id"]])


def test_cold_and_examples_have_no_hint_lines(data):
    row = data["val_rows"][0]
    for mode in ("cold", "examples"):
        user = lc.build_messages(row, mode, examples=lc.pick_examples(data["train_rows"], "P1-LC", 4))[1]["content"]
        assert "its facts" not in user and "one line of facts" not in user


# ---------------------------------------------------------------- markdown around the label

@pytest.mark.parametrize("reply, want", [
    ("The command rises. Change starts near second 150. Answer: **attack**", ("attack", 150)),
    ("Flat.\n\n**Answer:** _normal_", ("normal", None)),
    ("`Answer: attack`", ("attack", None)),
    ("Change starts near second __0__. Answer: ***attack***", ("attack", 0)),
])
def test_markdown_is_removed_and_raw_is_kept(data, reply, want):
    t = FakeTransport(reply=reply)
    code, _ = run(data, "--mode", "cold", "--max-usd", "1", "--limit", "1", "--key-file", str(data["key"]), transport=t)
    assert code == 0
    (rec,) = lines(data["out"])
    assert rec["raw"] == reply
    assert rec["generated"] == reply.strip().replace("*", "").replace("_", "").replace("`", "")
    assert score.parse(rec["generated"]) == want


def test_unmark_changes_nothing_else():
    text = "  Ünïcode, digits 12.5 -3; P1-LC: Answer: normal\n"
    assert lc.unmark(text) == text
    assert lc.unmark("a*b_c`d") == "abcd"


def test_score_parse_rejects_the_bold_label():
    # why the cleanup is needed: score.py (unchanged) does not read a bold label
    assert score.parse("Flat. Answer: **attack**") == (None, None)
    assert score.parse(lc.unmark("Flat. Answer: **attack**")) == ("attack", None)


def test_an_errored_row_has_empty_raw(data):
    t = FakeTransport(status=lambda b: 400)
    run(data, "--mode", "cold", "--max-usd", "1", "--limit", "1", "--key-file", str(data["key"]), transport=t)
    (rec,) = lines(data["out"])
    assert rec["error"] and rec["raw"] == "" and rec["generated"] == ""
