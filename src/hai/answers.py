"""Step 8: the answer text, written by code from computed facts.

Words only (rises, falls, stays flat, follows) and a start second. No measured
values. Every answer ends with `Answer: attack` or `Answer: normal`.
"""
import numpy as np

from hai.dataset import SERIES_LABEL

ROLE_LABEL = {"target": "target", "measurement": "measurement", "command": "command"}


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


def onset(x: np.ndarray, start: int, base: np.ndarray, scale: float) -> int | None:
    """First second at or after `start` where the series leaves its band before the change."""
    thr = max(3 * float(np.std(base)), 0.02 * scale)
    away = np.nonzero(np.abs(x[start:] - float(np.mean(base))) > thr)[0]
    return start + int(away[0]) if len(away) else None


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def attack_text(tags, roles, values, start: int, end: int, scales: dict) -> str:
    if start < 0:
        return "The change is already under way when the window opens. Answer: attack"
    if start < 10:
        return f"Change starts near second {start}. Answer: attack"
    counts = {r: roles.count(r) for r in set(roles)}
    stop = min(end + 1, len(values[0]))
    movers, flat = [], []
    for tag, role, x in zip(tags, roles, values):
        base = x[max(0, start - 60):start]
        d = direction(base, x[start:stop], scales[tag])
        name = label_of(tag, role, counts[role])
        if d == "stays flat":
            flat.append(name)
        else:
            movers.append((onset(x, start, base, scales[tag]) or start, name, d))
    movers.sort()
    if not movers:
        return f"No series shows a clear step. Change starts near second {start}. Answer: attack"
    t0, name, d = movers[0]
    s = f"The {name} {d} from second {t0}"
    if flat:
        s += f" while the {_join(flat)} {'stays' if len(flat) == 1 else 'stay'} flat"
    s += "."
    for t, name, d in movers[1:]:
        s += f" The {name} {d} {'with it' if t - t0 <= 2 else 'after it'}."
    return f"{s} Change starts near second {start}. Answer: attack"


def normal_text(tags, roles, values, scales: dict) -> str:
    half = len(values[0]) // 2
    dirs: dict[str, list[str]] = {}
    for tag, role, x in zip(tags, roles, values):
        dirs.setdefault(role, []).append(direction(x[:half], x[half:], scales[tag]))

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


def answer(row, scales: dict) -> str:
    tags, roles, values = list(row.tags), list(row.roles), [np.asarray(v) for v in row.values]
    if row.label == "attack":
        return attack_text(tags, roles, values, int(row.attack_start_s), int(row.attack_end_s), scales)
    return normal_text(tags, roles, values, scales)
