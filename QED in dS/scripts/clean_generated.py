#!/usr/bin/env python3
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "generated"
for item in out.iterdir():
    if item.name == ".gitkeep":
        continue
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()
print(out)
