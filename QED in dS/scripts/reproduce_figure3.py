#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from plot_figure3 import make_figure
from file_utils import clean_figure_metadata

def main() -> int:
    outputs = make_figure(ROOT / "data" / "figure_3", ROOT / "generated")
    clean_figure_metadata([outputs["figure_pdf"], outputs["figure_png"]])
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0

if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _code = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
