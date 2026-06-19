#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".mplconfig", "__pycache__", "generated"}
EXCLUDE_FILES = {"MANIFEST.csv", "checksums.sha256"}

def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.name in EXCLUDE_FILES:
            continue
        yield rel

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for rel in iter_files():
    p = ROOT / rel
    rows.append({"path": rel.as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})

with (ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)

with (ROOT / "checksums.sha256").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(f"{row['sha256']}  {row['path']}\n")

print(f"Wrote {len(rows)} entries")
