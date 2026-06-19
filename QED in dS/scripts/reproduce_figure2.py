#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from plot_figure2 import make_figure
from file_utils import clean_figure_metadata

def main() -> int:
    pdf, png = make_figure(ROOT / "data" / "figure_2", ROOT / "generated", stem="figure2")
    clean_figure_metadata([pdf, png])
    print(pdf)
    print(png)
    return 0

if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _code = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
