#!/usr/bin/env python3
"""Regenerate the thermodynamic-limit Figure 2 from a source archive."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from file_utils import clean_figure_metadata
from plot_thermodynamic_limit import analyze_summary, load_summary_from_source, make_figure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the thermodynamic-limit Figure 2 from a run archive, extracted run directory, or summary .npz file."
    )
    parser.add_argument("source", help="Input source archive, extracted run directory, or summary .npz file")
    parser.add_argument("--outdir", default=str(ROOT / "generated"), help="Directory for the output figure")
    parser.add_argument("--basename", default="figure2_thermodynamic_limit", help="Base filename for the saved figure")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    meta, arrays, _summary_path = load_summary_from_source(args.source)
    analysis = analyze_summary(meta, arrays, thermodynamic_axis="invLphys")
    out_pdf, out_png = make_figure(
        analysis,
        outdir / f"{args.basename}.pdf",
        outdir / f"{args.basename}.png",
    )
    clean_figure_metadata([out_pdf, out_png])
    print(out_pdf)
    print(out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
