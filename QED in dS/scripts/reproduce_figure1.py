#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from plot_figure1 import make_onepage_summary
from file_utils import clean_figure_metadata

def main() -> int:
    out = ROOT / "generated" / "figure1.pdf"
    make_onepage_summary(ROOT / "data" / "figure_1" / "transition_data.zip", out_path=out)
    clean_figure_metadata([out])
    print(out)
    return 0

if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _code = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
