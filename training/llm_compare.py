"""Ask a large general model, through an OpenAI-compatible API (OpenRouter), the question our fine-tuned model answers.

    uv run python training/llm_compare.py --model MODEL_ID --mode cold|examples|hints|hints_examples \
        --out runs/llm_compare/NAME.jsonl --max-usd 1.00 [--dry-run] [--ids FILE] [--limit N] \
        [--rows validation|test] [--allow-test] \
        [--step 1] [--example-step 5] [--examples N] [--concurrency 8] [--max-tokens 160] \
        [--reasoning none|low|default] [--key-file PATH] [--base-url https://openrouter.ai/api/v1] [--price-factor 4]

Rows: data/processed/opentslm/validation.jsonl (--rows validation, the default). With --rows test AND --allow-test,
data/processed/opentslm/test.jsonl, and only that file; --rows test without --allow-test, or --allow-test without
--rows test, is refused. Solved examples and the hint statistics always come from train.jsonl, never from the rows
answered. Every other file whose name contains "test" is refused (test_shuffled.jsonl, test_full.jsonl, any --out
or --ids file). The answer format is the one our model was trained on (label last), so `training/score.py` parses
both alike. Score with:

    uv run python training/results_table.py --truth data/processed/opentslm/validation.jsonl \
        --run NAME runs/llm_compare/NAME.jsonl

Modes
  cold      the task, the loop and each series as numbers: its normal range and its values (4 significant digits),
            every --step-th second.
  examples  cold plus --examples solved rows of the same loop from train.jsonl (half attack, half normal, fixed seed),
            their series every --example-step-th second, each with its reference answer.
  hints     cold plus, per series and directly before its values, the exact text line our model sees in the design
            "view looprange, clauses step,law", built by HAIAttackQADataset._format_sample with the class statistics
            of train.jsonl.
  hints_examples  hints plus --examples (default 6: 3 attack, 3 normal) solved windows of the same loop from
            train.jsonl, chosen as in mode examples, shown only by their hint lines (no values) and their
            reference answer.
Every mode's system prompt states the base rate: about one window in five of train.jsonl is an attack.

Cost control
  Before any request the tool reads the model's prices from the public models list and from its per-provider
  endpoints list. Providers differ a lot (up to 30 times for one model), so it keeps only the prices at most
  --price-factor (default 4) times the list price, and takes the highest of those of each kind (time-of-day
  overrides and cache writes included). Every request carries provider.max_price at those prices, sort by price
  and require_parameters, so OpenRouter routes it to the cheapest provider that honours max_tokens and never to a
  dearer one than the reservation assumes.
  Each request reserves its worst case before it is sent: UTF-8 bytes of the prompt plus a template allowance
  (a hard upper bound on input tokens) times the highest input price, plus --max-tokens times the highest output
  price, plus any per-request price. A request is sent only when spent + reserved in flight + its worst case fits
  under the cap. Spend is the `usage.cost` each response reports; without it, tokens times the highest prices;
  without tokens, or after a timeout or a dropped connection, the full reservation. A response that costs more
  than its reservation raises every later reservation by the same factor.
  The cap of a run is min(--max-usd, 5 USD minus everything already spent by the files next to --out). When the
  cap stops a run, the tool prints rows answered and spend, keeps the partial output and exits with code 3.

Output: one JSON line per row, appended and flushed: example_id, generated, raw, model, mode, prompt_tokens,
completion_tokens, cost_usd, cost_source, seconds, error. raw is the model's text unchanged; generated is that text
stripped of surrounding white space and of every *, _ and ` (so "Answer: **attack**" parses). A rerun skips rows
already answered without error.
"""
import argparse
import http.client
import json
import math
import os
import random
import sys
import threading
import time
import types
import urllib.parse
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed" / "opentslm"
ROWS_PATH = DATA / "validation.jsonl"
TEST_ROWS_PATH = DATA / "test.jsonl"   # read only with --rows test --allow-test
TEST_ROWS_NAME = "test.jsonl"          # the one file name with "test" the tool reads, and only as rows
TRAIN_PATH = DATA / "train.jsonl"
MODES = ("cold", "examples", "hints", "hints_examples")
DEFAULT_EXAMPLES = {"examples": 4, "hints_examples": 6}
DATASET_PY = REPO / "training" / "HAIAttackQADataset.py"
COMPARE_DIR = REPO / "runs" / "llm_compare"   # every comparison run's spend under here counts toward the 5 USD
OPENTSLM_SRC = REPO / "external" / "OpenTSLM" / "src" / "opentslm"

GLOBAL_LIMIT_USD = 5.0     # the owner's hard limit for the whole comparison
MAX_RETRIES = 4
TIMEOUT_S = 120
PROGRESS_EVERY = 50
EXAMPLE_SEED = 0
PRICE_FACTOR = 4.0   # default price ceiling: this many times the list price
HINT_VIEW, HINT_CLAUSES = "looprange", "step,law"
# Allowance for chat-template tokens around the messages, on top of the byte bound.
TEMPLATE_TOKENS = 64
TEMPLATE_TOKENS_PER_MESSAGE = 16

ROLE_WORDS = {
    "target": "the value the controller aims for",
    "measurement": "the measured value",
    "command": "what the controller sends to the actuator",
}
FORMAT_ATTACK = ("The measurement rises from second 120 while the target stays flat. The command falls after it. "
                 "Change starts near second 118. Answer: attack")
FORMAT_NORMAL = "The target and the measurement stay flat. The command rises. Answer: normal"
# train.jsonl: 776 attack and 3104 normal of 3880 rows, exactly one in five.
BASE_RATE = ("Most windows are normal: in the material the trained detector learned from, about one window in five "
             "shows an attack.")

SYSTEM = (
    "You read sensor data from one control loop of a power plant testbed and decide whether the loop behaves "
    "normally or is under attack. An attack makes a series move in a way the loop would not move on its own: "
    "for example the measurement or a command changes while the target stays flat, or a command moves against "
    "what the controller would do. " + BASE_RATE + "\n\n"
    "Reply in exactly this format, in plain text:\n"
    "1. One or two short sentences on how the series move.\n"
    "2. Only if the loop is under attack: the sentence \"Change starts near second S.\" where S is a whole number, "
    "the second where the change starts (the first reading is second 0, the last second 299). If the change is "
    "already under way when the readings begin, write \"Change starts near second 0.\"\n"
    "3. Last: \"Answer: attack\" or \"Answer: normal\".\n"
    "Do not begin your reply with the word \"attack\" or \"normal\". Write nothing after the answer.\n\n"
    f"Two replies in the right format:\n{FORMAT_ATTACK}\n{FORMAT_NORMAL}"
)


class Refused(SystemExit):
    """A refusal: exits with code 2 and a clear message."""

    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


class TransportError(Exception):
    """The request may or may not have reached the server (timeout, dropped connection)."""


# ---------------------------------------------------------------- files

def guard_name(path) -> Path:
    """The path, unless its file name, or the name of the file it points to through symlinks, contains "test"
    (any case): then refuse."""
    p = Path(path)
    for name in (p.name, p.resolve().name):
        if "test" in name.lower():
            raise Refused(f"refusing {name}: this tool never reads or writes a file whose name contains 'test'")
    return p


def read_jsonl(path) -> list[dict]:
    with open(guard_name(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_source(rows: str, allow_test: bool, rows_path, test_rows_path) -> tuple[Path, bool]:
    """(the rows file, whether it holds test rows). Test rows need --rows test and --allow-test together, and come
    only from a file named exactly test.jsonl; any other file whose name contains "test" stays refused."""
    if rows == "test" and not allow_test:
        raise Refused("--rows test needs --allow-test: test rows are answered only when both are given")
    if allow_test and rows != "test":
        raise Refused("--allow-test is only valid together with --rows test")
    if rows != "test":
        return guard_name(rows_path), False
    p = Path(test_rows_path)
    for name in (p.name, p.resolve().name):   # a symlink named test.jsonl to another file is refused too
        if name != TEST_ROWS_NAME:
            raise Refused(f"refusing {name}: with --rows test the tool reads {TEST_ROWS_NAME} and no other file")
    return p, True


def read_rows(path, is_test: bool) -> list[dict]:
    """The rows to answer. A test rows file is read only after rows_source has let it through."""
    p = Path(path)
    if not (is_test and p.name == TEST_ROWS_NAME and p.resolve().name == TEST_ROWS_NAME):
        return read_jsonl(p)
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_rows(rows: list[dict], ids_path=None, limit=None) -> list[dict]:
    if ids_path:
        by_id = {r["example_id"]: r for r in rows}
        with open(guard_name(ids_path), encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise Refused(f"{len(missing)} ids of {Path(ids_path).name} are not in the rows, e.g. {missing[0][:80]}")
        rows = [by_id[i] for i in ids]
    return rows[:limit] if limit is not None else rows


def previous(out: Path, model: str, mode: str) -> tuple[set, int]:
    """(ids answered without error, lines) of the out file; refuses a file of another model or mode."""
    if not out.exists():
        return set(), 0
    old = read_jsonl(out)
    other = {(o.get("model"), o.get("mode")) for o in old} - {(model, mode)}
    if other:
        raise Refused(f"{out.name} holds rows of {sorted(map(str, other))}, not ({model!r}, {mode!r}): use another out file")
    return {o["example_id"] for o in old if not o.get("error")}, len(old)


def spent_before(folder: Path, compare_dir: Path = None) -> float:
    """USD already spent: cost_usd summed over the *.jsonl files in `folder` and anywhere under runs/llm_compare
    (so an --out in another folder still counts the comparison's earlier spend)."""
    compare_dir = COMPARE_DIR if compare_dir is None else compare_dir
    files = set(folder.glob("*.jsonl")) if folder.exists() else set()
    if compare_dir.exists():
        files |= set(compare_dir.rglob("*.jsonl"))
    total = 0.0
    for f in sorted({p.resolve() for p in files}):
        if "test" in f.name.lower():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        v = float(json.loads(line).get("cost_usd") or 0)
                    except (ValueError, TypeError, AttributeError):
                        continue
                    if not math.isfinite(v) or v < 0:
                        raise Refused(f"{f.name} holds a cost of {v!r}: fix it before spending more")
                    total += v
    return total


# ---------------------------------------------------------------- prompts

def number(v) -> str:
    return f"{float(v):.4g}"


def values_text(values, step: int) -> str:
    return ",".join(number(v) for v in values[::step])


def every_text(step: int, n: int) -> str:
    if step == 1:
        return f"one value per second, seconds 0 to {n - 1}"
    last = (n - 1) // step * step
    return f"one value every {step} seconds, at seconds 0, {step}, {2 * step}, ... {last}"


def series_block(row: dict, step: int, hint_lines=None) -> str:
    """Each series: its header line, then (with hint_lines) its hint line exactly as the dataset writes it, then its
    values, so that the hint line sits directly before the numbers it describes."""
    if hint_lines is not None and len(hint_lines) != len(row["series"]):
        raise ValueError("one hint line per series is needed")
    lines = []
    for i, s in enumerate(row["series"]):
        head = f"{s['name']} ({s['role']}: {ROLE_WORDS.get(s['role'], s['role'])}); normal range {number(s['scale'])}"
        if hint_lines is None:
            lines.append(head + "; values:")
        else:
            lines.append(head + "; its facts, then its values:")
            lines.append(hint_lines[i])
        lines.append(values_text(s["values"], step))
    return "\n".join(lines)


def loop_text(row: dict) -> str:
    n = len(row["series"][0]["values"])
    names = "\n".join(f"- {s['name']}: {s['role']}, {ROLE_WORDS.get(s['role'], s['role'])}" for s in row["series"])
    return (f"The {row['loop_name']} control loop, {n} seconds, one reading per second. Its series:\n{names}\n"
            "Each series below states its normal range (how far its values spread in normal operation) and its "
            "values, rounded to 4 significant digits, comma-separated.")


def pick_examples(train: list[dict], loop: str, n: int, seed: int = EXAMPLE_SEED) -> list[dict]:
    """n solved rows of the loop: n // 2 attack then the rest normal, drawn with a fixed seed from the
    rows sorted by example_id, then interleaved attack, normal, attack, normal, ..."""
    rng = random.Random(f"{seed}/{loop}")
    attack = sorted((r for r in train if r["loop"] == loop and r["label"] == "attack"), key=lambda r: r["example_id"])
    normal = sorted((r for r in train if r["loop"] == loop and r["label"] == "normal"), key=lambda r: r["example_id"])
    a = rng.sample(attack, min(n // 2, len(attack)))
    b = rng.sample(normal, min(n - len(a), len(normal)))
    out = []
    for i in range(max(len(a), len(b))):
        out += ([a[i]] if i < len(a) else []) + ([b[i]] if i < len(b) else [])
    return out


HINT_EXPLAIN = ("Each series comes with one line of facts computed for its window. \"Usual\" means what the series "
                "does in a quiet window of this loop in normal operation; \"what the controller would do\" comes "
                "from the controller's law fitted on such windows.")


def build_messages(row: dict, mode: str, step: int = 1, examples=None, example_step: int = 5,
                   hint_lines=None, example_hints=None) -> list[dict]:
    """The chat messages for one row. examples: solved rows (modes examples, hints_examples). hint_lines: the
    dataset's text lines of the row's series, in series order (modes hints, hints_examples). example_hints:
    {example_id: its hint lines} for the solved rows (mode hints_examples)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    n = len(row["series"][0]["values"])
    hinted = mode in ("hints", "hints_examples")
    if hinted and (hint_lines is None or len(hint_lines) != len(row["series"])):
        raise ValueError(f"mode {mode} needs one hint line per series")
    parts = [loop_text(row)]
    if hinted:
        parts.append(HINT_EXPLAIN)
    if mode == "examples":
        parts.append(f"Here are {len(examples)} solved windows of this same loop, each series with "
                     f"{every_text(example_step, n)}, and the right reply for each.")
        for i, e in enumerate(examples, 1):
            parts.append(f"Solved window {i}:\n{series_block(e, example_step)}\nRight reply: {e['answer']}")
        parts.append("Now the window to judge.")
    if mode == "hints_examples":
        parts.append(f"Here are {len(examples)} solved windows of this same loop, each shown only by the facts line "
                     "of each of its series (no values), and the right reply for each.")
        for i, e in enumerate(examples, 1):
            lines = (example_hints or {}).get(e["example_id"])
            if lines is None or len(lines) != len(e["series"]):
                raise ValueError(f"mode hints_examples needs one hint line per series of {e['example_id']}")
            parts.append(f"Solved window {i}:\n" + "\n".join(lines) + f"\nRight reply: {e['answer']}")
        parts.append("Now the window to judge.")
    parts.append(f"The window to judge, each series with {every_text(step, n)}:\n"
                 f"{series_block(row, step, hint_lines if hinted else None)}")
    parts.append("Is this loop under attack? Reply in the required format.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def prompt_text(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


# ---------------------------------------------------------------- hint lines from the dataset code

class _Env:
    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_dataset_module():
    """HAIAttackQADataset.py as a module, with `opentslm` taken from the checkout without its package __init__."""
    import importlib.util
    try:
        import opentslm.time_series_datasets.QADataset  # noqa: F401
    except Exception:
        if not OPENTSLM_SRC.exists():
            raise Refused("mode hints needs the OpenTSLM checkout in external/OpenTSLM (training/setup.sh)")
        for k in [k for k in sys.modules if k == "opentslm" or k.startswith("opentslm.")]:
            del sys.modules[k]
        pkg = types.ModuleType("opentslm")
        pkg.__path__ = [str(OPENTSLM_SRC)]
        sys.modules["opentslm"] = pkg
    spec = importlib.util.spec_from_file_location("hai_ds_llm_compare", DATASET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hint_lines_for(rows: list[dict], train: list[dict]) -> dict:
    """{example_id: the dataset's time_series_text of the row} in the design view looprange, clauses step,law, with
    the class statistics set from `train` exactly as curve_check.setup_dataset does."""
    with _Env(HAI_SERIES_VIEW=HINT_VIEW, HAI_TEXT_CLAUSES=HINT_CLAUSES, HAI_ANSWER_ORDER="label_last"):
        hai = load_dataset_module()
        cls = hai.HAIAttackQADataset
        cls.looprange_units = cls._units(train) if hai.series_view() in ("looprange", "zsize") else None
        cls.step_units = cls._step_units(train) if "step" in hai.text_clauses() else None
        cls.control_laws = cls._laws(train) if "law" in hai.text_clauses() else None
        cls.plant_models = cls._plants(train) if "plant" in hai.text_clauses() else None
        ds = cls.__new__(cls)
        ds.EOS_TOKEN = ""
        return {r["example_id"]: list(cls._format_sample(ds, r)["time_series_text"]) for r in rows}


# ---------------------------------------------------------------- tokens and prices

def estimate_tokens(messages: list[dict]) -> int:
    """A conservative estimate of input tokens: every digit and every punctuation mark one token, other characters
    three to a token, plus the template allowance. Number text is about one token per 1.2 to 1.5 characters."""
    total = TEMPLATE_TOKENS
    for m in messages:
        text = m["content"]
        digits = sum(c.isdigit() for c in text)
        punct = sum(not c.isalnum() and not c.isspace() for c in text)
        total += digits + punct + math.ceil((len(text) - digits - punct) / 3) + TEMPLATE_TOKENS_PER_MESSAGE
    return total


def bound_tokens(messages: list[dict]) -> int:
    """A hard upper bound on input tokens: a token holds at least one byte, plus the template allowance."""
    return TEMPLATE_TOKENS + sum(len(m["content"].encode("utf-8")) + TEMPLATE_TOKENS_PER_MESSAGE for m in messages)


def _price(v) -> float:
    p = float(v)
    if not math.isfinite(p) or p < 0:
        raise Refused(f"the model has a variable or unknown price ({v!r}): pick a model with fixed prices")
    return p


def _pricings(pricing: dict) -> list[dict]:
    return [pricing] + [o for o in pricing.get("overrides") or [] if isinstance(o, dict)]


def prices_from(model_entry: dict, endpoints: list[dict], factor: float = PRICE_FACTOR) -> dict:
    """{list_in, list_out, max_in, max_out, max_prompt, max_completion, max_request, providers} in USD per token
    (max_request per request).

    list_*: the price on the models list. The ceiling is the list price times `factor`. Eligible are the pricings
    (the models list entry, every provider endpoint, and each of their time-of-day overrides) whose prompt and
    completion prices are both at or under the ceiling: the request's provider.max_price keeps OpenRouter on those.
    max_*: the highest over the eligible pricings; input includes cache writes, output includes internal reasoning.
    providers: how many endpoints have at least one eligible pricing."""
    base = model_entry.get("pricing") or {}
    if "prompt" not in base or "completion" not in base:
        raise Refused(f"no prompt and completion price for {model_entry.get('id')!r} on the models list")
    list_in, list_out = _price(base["prompt"]), _price(base["completion"])
    ceil_in, ceil_out = list_in * factor, list_out * factor

    def ok(p):
        return ("prompt" in p and "completion" in p and _price(p["prompt"]) <= ceil_in * (1 + 1e-9)
                and _price(p["completion"]) <= ceil_out * (1 + 1e-9))
    eligible = [p for p in _pricings(base) if ok(p)]
    providers = 0
    for e in endpoints:
        mine = [p for p in _pricings(e.get("pricing") or {}) if ok(p)]
        providers += bool(mine)
        eligible += mine
    if not providers:
        raise Refused(f"no provider of {model_entry.get('id')!r} charges at most {factor:g} times the list price")

    def top(keys):
        return max([_price(p[k]) for p in eligible for k in keys if p.get(k) not in (None, "")] or [0.0])
    return {"list_in": list_in, "list_out": list_out, "factor": factor, "providers": providers,
            "max_prompt": top(["prompt"]), "max_completion": top(["completion"]),
            "max_in": top(["prompt", "input_cache_write", "input_cache_write_1h"]),
            "max_out": top(["completion", "internal_reasoning"]), "max_request": top(["request"])}


def fetch_prices(transport, base_url: str, model: str, factor: float = PRICE_FACTOR) -> dict:
    """Prices from GET {base}/models and GET {base}/models/{model}/endpoints (public, no key, no headers)."""
    status, body = transport.get(f"{base_url}/models", {}, TIMEOUT_S)
    if status != 200:
        raise Refused(f"could not read the models list (HTTP {status})")
    entry = next((m for m in json.loads(body).get("data", []) if m.get("id") == model), None)
    if entry is None:
        raise Refused(f"model {model!r} is not on the models list of {base_url}")
    status, body = transport.get(f"{base_url}/models/{urllib.parse.quote(model, safe='/:')}/endpoints", {}, TIMEOUT_S)
    if status != 200:
        raise Refused(f"could not read the provider endpoints of {model!r} (HTTP {status}); without them the "
                      "worst-case price is unknown")
    endpoints = (json.loads(body).get("data") or {}).get("endpoints") or []
    if not endpoints:
        raise Refused(f"no provider endpoints listed for {model!r}")
    return prices_from(entry, endpoints, factor)


def worst_cost(messages: list[dict], max_tokens: int, prices: dict) -> float:
    return bound_tokens(messages) * prices["max_in"] + max_tokens * prices["max_out"] + prices["max_request"]


# ---------------------------------------------------------------- the ledger

class Ledger:
    """Spend and reservations under one lock. Nothing is sent unless spent + reserved + its worst case fits."""

    def __init__(self, cap: float):
        self.cap, self.spent, self.reserved, self.factor = float(cap), 0.0, 0.0, 1.0
        self.lock = threading.Lock()

    def reserve(self, worst: float):
        """The amount reserved (worst times the overrun factor), or None if it does not fit under the cap."""
        with self.lock:
            amount = worst * self.factor
            if self.spent + self.reserved + amount > self.cap:
                return None
            self.reserved += amount
            return amount

    def settle(self, reserved: float, actual: float) -> None:
        """Release a reservation and book what the request cost."""
        if not (math.isfinite(actual) and actual >= 0):
            actual = reserved   # never let a bad number lower the spend or disable the cap
        with self.lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += actual
            if reserved > 0 and actual > reserved:
                self.factor = max(self.factor, actual / reserved * self.factor)


# ---------------------------------------------------------------- HTTP

class HttpTransport:
    """http.client, so that nothing but the given headers (and Host, Content-Length, Accept-Encoding) is sent:
    no User-Agent, no Referer, no app or user identifiers."""

    def _request(self, method, url, headers, body, timeout):
        u = urllib.parse.urlsplit(url)
        conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(u.hostname, u.port, timeout=timeout)
        path = u.path + (f"?{u.query}" if u.query else "")
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read(), dict(resp.getheaders())
        except (OSError, http.client.HTTPException) as e:
            raise TransportError(type(e).__name__) from None
        finally:
            conn.close()

    def get(self, url, headers, timeout):
        status, body, _ = self._request("GET", url, headers, None, timeout)
        return status, body

    def post(self, url, headers, body, timeout):
        status, body, _ = self._request("POST", url, headers, body, timeout)
        return status, body


def check_base_url(base_url: str) -> str:
    u = urllib.parse.urlsplit(base_url)
    if u.scheme == "https" or (u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost")):
        return base_url.rstrip("/")
    raise Refused("--base-url must be https (plain http only for 127.0.0.1 or localhost)")


def read_key(path) -> str:
    if not path:
        raise Refused("no key file: pass --key-file or set OPENROUTER_API_KEY_FILE")
    try:
        key = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise Refused("could not read the key file") from None
    if not key or any(c.isspace() for c in key):
        raise Refused("the key file must hold one key on one line")
    return key


def request_body(model, messages, max_tokens, reasoning, prices) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "provider": {"require_parameters": True, "sort": "price",
                         "max_price": {"prompt": prices["max_prompt"] * 1e6, "completion": prices["max_completion"] * 1e6}}}
    if prices["max_request"] > 0:
        body["provider"]["max_price"]["request"] = prices["max_request"]
    if reasoning == "none":
        body["reasoning"] = {"effort": "none"}
    elif reasoning == "low":
        body["reasoning"] = {"effort": "low"}
    return body


def clean(text: str, key: str) -> str:
    text = str(text)
    return text.replace(key, "***") if key else text


MARKDOWN = str.maketrans("", "", "*_`")


def unmark(text: str) -> str:
    """The text with every asterisk, underscore and backtick removed, nothing else changed."""
    return text.translate(MARKDOWN)


def _count(v) -> bool:
    """A token count we can trust: a non-negative int (not a bool)."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def ask(transport, url, key, body, worst, ledger, prices, sleep) -> dict:
    """One row: the request with retries. The caller has reserved `worst`; this settles it, always."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8")
    held, t0, error, fatal = worst, time.time(), "", False
    result = {"generated": "", "raw": "", "prompt_tokens": None, "completion_tokens": None, "cost_usd": 0.0, "cost_source": "none"}
    try:
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                sleep(min(30.0, 2.0 ** (attempt - 1)))
                if held == 0:   # the last attempt may have been billed: reserve afresh
                    held = ledger.reserve(worst) or 0.0
                    if not held:
                        error = "spending cap reached before a retry"
                        break
            try:
                status, raw = transport.post(url, headers, data, TIMEOUT_S)
            except TransportError as e:
                # unknown whether it was billed: book the full reservation
                ledger.settle(held, held)
                result["cost_usd"] += held
                result["cost_source"] = "worst_case"
                held = 0.0
                error = clean(f"transport error: {e}", key)
                continue
            if status == 429 or 500 <= status < 600:
                error = f"HTTP {status}"
                continue
            if status != 200:
                error = clean(f"HTTP {status}: {raw.decode('utf-8', 'replace')}", key)[:300]
                fatal = status in (401, 402, 403, 404)
                break
            try:
                resp = json.loads(raw)
                if not isinstance(resp, dict):
                    raise ValueError("not an object")
                usage = resp.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError("usage is not an object")
                choices = resp.get("choices") or []
                content = ((choices[0].get("message") or {}).get("content") if choices else None) or ""
                if not isinstance(content, str):
                    raise ValueError("content is not text")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                # a 200 we cannot read may still have been billed: book the full reservation
                ledger.settle(held, held)
                result["cost_usd"] += held
                result["cost_source"] = "worst_case"
                held = 0.0
                error = "response is not the expected JSON"
                break
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if not (_count(pt) and _count(ct)):
                pt = ct = None
            rep = usage.get("cost")
            if isinstance(rep, (int, float)) and not isinstance(rep, bool) and math.isfinite(rep) and rep >= 0:
                cost, source = float(rep), "reported"
            elif pt is not None:
                cost = pt * prices["max_in"] + ct * prices["max_out"] + prices["max_request"]
                source = "tokens"
            else:
                cost, source = held, "worst_case"
            ledger.settle(held, cost)
            held = 0.0
            result.update(prompt_tokens=pt, completion_tokens=ct, cost_source=source)
            result["cost_usd"] += cost
            if resp.get("error") or not choices:
                error = clean(f"error in response: {json.dumps(resp.get('error'))}", key)[:300]
            else:
                error = ""
                result["raw"] = content
                result["generated"] = unmark(content.strip())
            break
    finally:
        if held:
            ledger.settle(held, 0.0)   # HTTP errors are not billed
    result.update(seconds=round(time.time() - t0, 2), error=error, fatal=fatal)
    if error:
        result["generated"] = result["raw"] = ""
    return result


# ---------------------------------------------------------------- main

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--out", required=True)
    p.add_argument("--ids")
    p.add_argument("--rows", choices=("validation", "test"), default="validation")
    p.add_argument("--allow-test", action="store_true", help="needed, with --rows test, to answer the test rows")
    p.add_argument("--limit", type=int)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--example-step", type=int, default=5)
    p.add_argument("--examples", type=int, help="solved windows (default 4 in mode examples, 6 in hints_examples)")
    p.add_argument("--max-usd", type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--reasoning", choices=("none", "low", "default"), default="none")
    p.add_argument("--key-file")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--price-factor", type=float, default=PRICE_FACTOR,
                   help="use only providers that charge at most this many times the list price (default 4)")
    a = p.parse_args(argv)
    if a.max_usd is None and not a.dry_run:
        raise Refused("--max-usd is required (at most 5) unless --dry-run is given")
    if a.max_usd is not None and not (0 < a.max_usd <= GLOBAL_LIMIT_USD):
        raise Refused(f"--max-usd must be above 0 and at most {GLOBAL_LIMIT_USD:g}, got {a.max_usd:g}")
    for name in ("step", "example_step", "concurrency", "max_tokens"):
        if getattr(a, name) < 1:
            raise Refused(f"--{name.replace('_', '-')} must be at least 1")
    if not (1 <= a.price_factor <= 100):
        raise Refused("--price-factor must be between 1 and 100")
    if a.limit is not None and a.limit < 0:
        raise Refused("--limit must be at least 0")
    if a.examples is None:
        a.examples = DEFAULT_EXAMPLES.get(a.mode, 0)
    if a.mode in DEFAULT_EXAMPLES and a.examples < 1:
        raise Refused(f"--examples must be at least 1 in mode {a.mode}")
    return a


def build_all(rows, a, train_path):
    """[(row, messages)] for the rows."""
    train = None
    if a.mode != "cold":
        train = read_jsonl(train_path)   # examples and statistics: train.jsonl only, never the rows answered
    shots = {}
    if a.mode in ("examples", "hints_examples"):
        for r in rows:
            if r["loop"] not in shots:
                shots[r["loop"]] = pick_examples(train, r["loop"], a.examples)
    hints = {}
    if a.mode in ("hints", "hints_examples"):
        shot_rows = list({e["example_id"]: e for v in shots.values() for e in v}.values())
        hints = hint_lines_for(rows + shot_rows, train)
    out = []
    for r in rows:
        out.append((r, build_messages(r, a.mode, a.step, shots.get(r["loop"]), a.example_step,
                                      hints.get(r["example_id"]), hints)))
    return out


def main(argv=None, transport=None, rows_path=ROWS_PATH, train_path=TRAIN_PATH, sleep=time.sleep,
         test_rows_path=TEST_ROWS_PATH) -> int:
    try:
        return _main(argv, transport or HttpTransport(), rows_path, train_path, sleep, test_rows_path)
    except Refused as e:
        print(f"llm_compare: {e.message}", file=sys.stderr)
        return 2


def _main(argv, transport, rows_path, train_path, sleep, test_rows_path=TEST_ROWS_PATH) -> int:
    a = parse_args(argv)
    base_url = check_base_url(a.base_url)
    out = guard_name(a.out)
    if out.suffix != ".jsonl":
        raise Refused("--out must end in .jsonl")
    if a.ids:
        guard_name(a.ids)
    guard_name(train_path)
    src, is_test = rows_source(a.rows, a.allow_test, rows_path, test_rows_path)
    rows = select_rows(read_rows(src, is_test), a.ids, a.limit)
    if is_test:
        print(f"answering TEST rows: {len(rows)} rows", flush=True)
    done, lines = previous(out, a.model, a.mode)
    todo = [r for r in rows if r["example_id"] not in done]
    print(f"{a.model} {a.mode}: {len(rows)} rows, {len(rows) - len(todo)} done, {len(todo)} to answer", flush=True)
    built = build_all(todo, a, train_path)
    prices = fetch_prices(transport, base_url, a.model, a.price_factor)
    worst = [worst_cost(m, a.max_tokens, prices) for _, m in built]
    prior = spent_before(out.parent)
    cap = None if a.max_usd is None else min(a.max_usd, GLOBAL_LIMIT_USD - prior)
    print(f"prices per million tokens: list input {prices['list_in'] * 1e6:.4g}, output {prices['list_out'] * 1e6:.4g}; "
          f"highest of the {prices['providers']} providers at most {prices['factor']:g} times the list price: "
          f"input {prices['max_in'] * 1e6:.4g}, output {prices['max_out'] * 1e6:.4g}"
          + (f", per request {prices['max_request']:.4g}" if prices["max_request"] else ""), flush=True)
    print(f"spent before in {out.parent}: {prior:.4f} USD of the {GLOBAL_LIMIT_USD:g} USD limit"
          + ("" if cap is None else f"; cap of this run {max(cap, 0):.4f} USD"), flush=True)

    if a.dry_run:
        return dry_run(a, out, built, worst, prices, cap)
    if cap is None or cap <= 0:
        print(f"nothing left under the {GLOBAL_LIMIT_USD:g} USD limit: 0 rows answered, 0.0000 USD spent", flush=True)
        return 3
    if not todo:
        return 0
    key = read_key(a.key_file or os.environ.get("OPENROUTER_API_KEY_FILE"))
    return run(a, out, built, worst, prices, Ledger(cap), transport, base_url, key, sleep)


def dry_run(a, out, built, worst, prices, cap) -> int:
    est = [estimate_tokens(m) for _, m in built]
    bound = [bound_tokens(m) for _, m in built]
    n = len(built)
    per = (sum(est) / n) if n else 0
    list_cost = sum(e * prices["list_in"] for e in est) + n * a.max_tokens * prices["list_out"]
    print(f"dry run, no paid call: {n} rows", flush=True)
    if n:
        print(f"estimated input tokens: total {sum(est)}, per row {per:.0f} (min {min(est)}, max {max(est)}); "
              f"hard bound per row {sum(bound) / n:.0f}", flush=True)
    print(f"estimated cost at list prices (estimated input, all {a.max_tokens} output tokens): {list_cost:.4f} USD", flush=True)
    print(f"worst case reserved (bound input at the highest prices, {a.max_tokens} output tokens): {sum(worst):.4f} USD", flush=True)
    if cap is not None:
        print(f"the worst case {'fits under' if sum(worst) <= cap else 'does NOT fit under'} the cap of {max(cap, 0):.4f} USD",
              flush=True)
    samples = out.with_name(out.stem + ".samples.txt")
    samples.parent.mkdir(parents=True, exist_ok=True)
    with open(samples, "w", encoding="utf-8") as f:
        for r, m in built[:3]:
            f.write(f"===== {r['example_id']} ({estimate_tokens(m)} estimated input tokens)\n{prompt_text(m)}\n\n")
    print(f"sample prompts: {samples}", flush=True)
    return 0


def run(a, out, built, worst, prices, ledger, transport, base_url, key, sleep) -> int:
    url = f"{base_url}/chat/completions"
    out.parent.mkdir(parents=True, exist_ok=True)
    answered = valid = finished = 0
    cap_hit = stop = False
    score_parse = _load_parse()

    def write(row, res, f):
        nonlocal answered, valid, finished, stop
        rec = {"example_id": row["example_id"], "generated": res["generated"], "raw": res["raw"],
               "model": a.model, "mode": a.mode,
               "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
               "cost_usd": res["cost_usd"], "cost_source": res["cost_source"], "seconds": res["seconds"],
               "error": res["error"]}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        finished += 1
        answered += not res["error"]
        valid += score_parse(res["generated"])[0] is not None
        if res["fatal"]:
            stop = True
            print(f"stopping: {res['error'][:200]}", file=sys.stderr, flush=True)
        if finished % PROGRESS_EVERY == 0:
            print(f"{finished} rows done, {valid} valid answers, {ledger.spent:.4f} USD spent", flush=True)

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        inflight = {}

        def collect(block_all=False):
            nonlocal inflight
            if not inflight:
                return
            done_, _ = wait(list(inflight), return_when=ALL_COMPLETED if block_all else FIRST_COMPLETED)
            for fut in done_:
                write(inflight.pop(fut), fut.result(), f)

        try:
            for (row, messages), w in zip(built, worst):
                if stop:
                    break
                while True:
                    amount = ledger.reserve(w) if len(inflight) < a.concurrency else None
                    if amount is not None:
                        break
                    if not inflight:
                        cap_hit = True
                        break
                    collect()
                if cap_hit or stop:
                    if amount is not None:
                        ledger.settle(amount, 0.0)
                    break
                body = request_body(a.model, messages, a.max_tokens, a.reasoning, prices)
                fut = pool.submit(ask, transport, url, key, body, amount, ledger, prices, sleep)
                inflight[fut] = row
            collect(block_all=True)
        except KeyboardInterrupt:
            stop = True
            collect(block_all=True)
            print("interrupted", file=sys.stderr, flush=True)
    print(f"{finished} rows sent, {answered} answered, {valid} valid answers, {ledger.spent:.4f} USD spent "
          f"(cap {ledger.cap:.4f} USD)", flush=True)
    if cap_hit:
        print(f"stopped by the spending cap: {answered} rows answered, {ledger.spent:.4f} USD spent; "
              f"partial output kept in {out}", flush=True)
        return 3
    return 2 if stop else 0


def _load_parse():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hai_score_llm_compare", REPO / "training" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


if __name__ == "__main__":
    sys.exit(main())
