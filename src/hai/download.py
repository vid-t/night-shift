"""Download the pinned HAI files and record or verify their SHA-256."""
import hashlib
import sys
import urllib.request
from pathlib import Path

from hai.config import FILES, MANUAL, RAW

SUMS = RAW / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def targets() -> list[tuple[str, Path]]:
    out = [(f"{base}/hai-{v}/{name}", RAW / f"hai-{v}" / name) for v, _, base, name in FILES]
    _, base, name = MANUAL
    out.append((f"{base}/{name}", RAW / name))
    return out


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "night-shift/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def read_sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    out = {}
    for line in SUMS.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        out[rel.strip()] = digest
    return out


def main() -> None:
    expected = read_sums()
    got = {}
    for url, dest in targets():
        rel = str(dest.relative_to(RAW))
        if not dest.exists():
            print(f"download {rel}", flush=True)
            fetch(url, dest)
        if dest.stat().st_size < 1000:
            sys.exit(f"{rel} is {dest.stat().st_size} bytes: an LFS pointer, not data")
        got[rel] = sha256(dest)
        if rel in expected and expected[rel] != got[rel]:
            sys.exit(f"SHA-256 mismatch for {rel}: expected {expected[rel]}, got {got[rel]}")
    if not expected:
        SUMS.write_text("".join(f"{d}  {r}\n" for r, d in sorted(got.items())))
        print(f"wrote {SUMS}")
    else:
        missing = set(expected) - set(got)
        if missing:
            sys.exit(f"files in SHA256SUMS not downloaded: {sorted(missing)}")
        print(f"all {len(got)} files match {SUMS.name}")


if __name__ == "__main__":
    main()
