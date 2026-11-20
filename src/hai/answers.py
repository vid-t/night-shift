"""Step 8: the answer text, written by code from computed facts.

Words only (rises, falls, stays flat, follows, starts swinging, stops swinging)
and seconds. No measured values. Every answer ends with `Answer: attack` or
`Answer: normal`.

Two tests read a series. `direction` sees a shift of the mean. `swing` sees a
change in how much the series moves from one second to the next: a command
that starts or stops oscillating. The swing test counts only in loops that
rarely swing on their own (`swing_ok`); the build picks those loops from quiet
windows (`hai.build.swing_eligibility`).

`answer` also says whether an attack example shows no visible change in its
series. Those examples are left out of training: their text could only teach
a model to guess.
"""
import numpy as np

from hai.dataset import SERIES_LABEL

ROLE_LABEL = {"target": "target", "measurement": "measurement", "command": "command"}

SWING_RATIO = 3.0  # the step noise after is this many times the noise before, or the other way round
SWING_FLOOR = 0.002  # and the larger of the two exceeds this share of the tag's normal-file scale
SWING_MIN_STEPS = 5  # fewer one-second steps on either side: no verdict
SWING_MIN_TURNS = 3.0  # the swinging side's path (sum of |steps|) must be this many times its range: it goes up and down
SWING_SPAN_S = 10  # seconds of steps in the rolling noise that dates a swing change
SWING_ELIGIBLE_MAX_RATE = 0.10  # a loop is swing-eligible when fewer quiet examples than this swing on their own


def label_of(tag: str, role: str, n_in_role: int) -> str:
    return SERIES_LABEL[tag] if n_in_role > 1 else ROLE_LABEL[role]


def direction(before: np.ndarray, after: np.ndarray, scale: float) -> str:
    """rises / falls / stays flat, from the shift of the mean against the noise before."""
    if len(before) == 0 or len(after) == 0:
        return "stays flat"
    thr = max(3 * float(np.std(before)), 0.02 * scale)
    delta = float(np.mean(after)) - float(np.mean(before))
    if delta > thr:
        return "rises"
    if delta < -thr:
        return "falls"
    return "stays flat"


def turns(x: np.ndarray) -> float:
    """Path length over range: about 1 for a ramp or a step, 2 for a hump, more for a series that goes up and down."""
    x = np.asarray(x, "float64")
    r = float(x.max() - x.min()) if len(x) else 0.0
    return float(np.abs(np.diff(x)).sum()) / r if r > 0 else 0.0


def swing(before: np.ndarray, after: np.ndarray, scale: float) -> str | None:
    """starts swinging / stops swinging / None.

    A change when the noise of the one-second steps on one side is more than SWING_RATIO times the other's
    and above the floor, and the swinging side goes up and down (`turns` at least SWING_MIN_TURNS). The
    second condition keeps a smooth ramp, hump or single step, which also raises the step noise, out.
    """
    before, after = np.asarray(before, "float64"), np.asarray(after, "float64")
    pre, post = np.diff(before), np.diff(after)
    if len(pre) < SWING_MIN_STEPS or len(post) < SWING_MIN_STEPS:
        return None
    a, b = float(np.std(pre)), float(np.std(post))
    floor = SWING_FLOOR * scale
    if b > SWING_RATIO * a and b > floor and turns(after) >= SWING_MIN_TURNS:
        return "starts swinging"
    if a > SWING_RATIO * b and a > floor and turns(before) >= SWING_MIN_TURNS:
        return "stops swinging"
    return None


def swing_onset(x: np.ndarray, start: int, base: np.ndarray, stop: int, scale: float, word: str) -> int:
    """The second a swing change is dated to.

    First second t at or after `start` where the noise of the SWING_SPAN_S steps from t crosses the
    geometric mean of the step noise before and after `start`: upwards for `starts swinging`, downwards
    for `stops swinging`. `start` when it never crosses.
    """
    x = np.asarray(x, "float64")
    a = float(np.std(np.diff(np.asarray(base, "float64"))))
    b = float(np.std(np.diff(x[start:stop])))
    thr = max(float(np.sqrt(a * b)), SWING_FLOOR * scale)
    d = np.diff(x)
    for t in range(start, len(d) - SWING_SPAN_S + 1):
        sd = float(np.std(d[t:t + SWING_SPAN_S]))
        if (sd > thr) if word == "starts swinging" else (sd < thr):
            return t
    return start


def onset(x: np.ndarray, start: int, base: np.ndarray, scale: float) -> int | None:
    """First second at or after `start` where the series leaves its band before the change."""
    thr = max(3 * float(np.std(base)), 0.02 * scale)
    away = np.nonzero(np.abs(x[start:] - float(np.mean(base))) > thr)[0]
    return start + int(away[0]) if len(away) else None


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def attack_text(tags, roles, values, start: int, end: int, scales: dict, swing_ok: bool = False) -> tuple[str, bool]:
    """The text, and True when no series shows a clear step (no visible change).

    A series with no level shift but a swing change is a mover too, when `swing_ok`."""
    if start < 0:
        return "The change is already under way when the window opens. Answer: attack", False
    if start < 10:
        return f"Change starts near second {start}. Answer: attack", False
    counts = {r: roles.count(r) for r in set(roles)}
    stop = min(end + 1, len(values[0]))
    movers, flat = [], []
    for tag, role, x in zip(tags, roles, values):
        base = x[max(0, start - 60):start]
        d = direction(base, x[start:stop], scales[tag])
        name = label_of(tag, role, counts[role])
        w = swing(base, x[start:stop], scales[tag]) if swing_ok and d == "stays flat" else None
        if w:
            movers.append((swing_onset(x, start, base, stop, scales[tag], w), name, w))
        elif d == "stays flat":
            flat.append(name)
        else:
            movers.append((onset(x, start, base, scales[tag]) or start, name, d))
    movers.sort()
    if not movers:
        return f"No series shows a clear step. Change starts near second {start}. Answer: attack", True
    t0, name, d = movers[0]
    s = f"The {name} {d} from second {t0}"
    if flat:
        s += f" while the {_join(flat)} {'stays' if len(flat) == 1 else 'stay'} flat"
    s += "."
    for t, name, d in movers[1:]:
        s += f" The {name} {d} {'with it' if t - t0 <= 2 else 'after it'}."
    return f"{s} Change starts near second {start}. Answer: attack", False


def normal_text(tags, roles, values, scales: dict, swing_ok: bool = False) -> str:
    """First half against second half. A series `direction` calls flat but whose swinging changes
    starts or stops swinging, when `swing_ok`."""
    half = len(values[0]) // 2
    dirs: dict[str, list[str]] = {}
    for tag, role, x in zip(tags, roles, values):
        d = direction(x[:half], x[half:], scales[tag])
        if swing_ok and d == "stays flat":
            d = swing(x[:half], x[half:], scales[tag]) or d
        dirs.setdefault(role, []).append(d)

    def summary(role):
        ds = [d for d in dirs.get(role, []) if d != "stays flat"]
        if role not in dirs:
            return None
        if not ds:
            return "stays flat"
        return ds[0] if len(set(ds)) == 1 else "moves"

    t, m, c = summary("target"), summary("measurement"), summary("command")
    parts = []
    if t is None:
        parts.append(f"The measurement {m}." if m != "moves" else "The measurement moves.")
    elif t == "stays flat" and m == "stays flat":
        parts.append("The target and the measurement stay flat.")
    elif t != "stays flat" and m == t:
        parts.append(f"The measurement follows the target as it {t}.")
    elif t == "stays flat":
        parts.append(f"The target stays flat while the measurement {m}.")
    else:
        parts.append(f"The target {t} and the measurement {m}.")
    if c is not None:
        parts.append(f"The command {c}." if c != "moves" else "The commands move.")
    return " ".join(parts) + " Answer: normal"


def answer(row, scales: dict, swing_ok: bool = False) -> tuple[str, bool]:
    """(answer text, no_visible_change). The second is True only for an attack example whose series show
    no level shift and, when `swing_ok`, no swing change."""
    tags, roles, values = list(row.tags), list(row.roles), [np.asarray(v) for v in row.values]
    if row.label == "attack":
        return attack_text(tags, roles, values, int(row.attack_start_s), int(row.attack_end_s), scales, swing_ok)
    return normal_text(tags, roles, values, scales, swing_ok), False
