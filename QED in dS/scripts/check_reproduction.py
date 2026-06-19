#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plot_figure2 import load_summary, fit_line
from plot_figure3 import load_summary_data


def figure1_checks() -> dict:
    from plot_figure1 import load_runs, pick_run, gap_metrics, choose_tau_star, nearest_index
    runs = load_runs(ROOT / "data" / "figure_1" / "transition_data.zip")
    r = pick_run(runs, L=1.0, scale=True, run_type="coarse")
    arr = r.load()
    met = gap_metrics(arr, L=1.0)
    tau_star = choose_tau_star(met)
    it = nearest_index(met["tau"], tau_star)
    return {
        "m_star": float(met["mcrit"][it]),
        "tau_star": float(met["tau"][it]),
        "gap_minimum": float(met["gapcrit"][it]),
        "transition_runs": int(sum(1 for x in runs if x.run_type == "coarse")),
    }


def figure2_checks() -> dict:
    meta, N, a, tau_grid, gap_curves, tau_ref, gap_ref = load_summary(ROOT / "data" / "figure_2")
    coef50, _ = fit_line(a[N >= 50], tau_ref[N >= 50])
    coef60, _ = fit_line(a[N >= 60], tau_ref[N >= 60])
    return {
        "N_values": [int(x) for x in N.tolist()],
        "tau_limit_N_ge_50": float(coef50[0]),
        "tau_limit_N_ge_60": float(coef60[0]),
        "mass": float(meta["mass"]),
    }


def figure3_checks() -> dict:
    heatmap, beta_fronts, completion_bundle, finite_size_df = load_summary_data(ROOT / "data" / "figure_3", betas=(0.5, 1, 2, 5, 10), beta_heatmap=10.0)
    return {
        "heatmap_shape": [int(x) for x in np.asarray(heatmap["Z_dense"]).shape],
        "beta_values": [float(x) for x in sorted(beta_fronts)],
        "finite_size_N": [int(x) for x in finite_size_df["N"].tolist()],
        "locc_rows": int(len(completion_bundle["csv"])),
    }


def main() -> int:
    generated = ROOT / "generated"
    required = ["figure1.pdf", "figure2.pdf", "figure2.png", "figure3.pdf", "figure3.png"]
    missing = [name for name in required if not (generated / name).exists()]
    if missing:
        raise FileNotFoundError("Missing generated files: " + ", ".join(missing))
    checks = {
        "figure1": figure1_checks(),
        "figure2": figure2_checks(),
        "figure3": figure3_checks(),
    }
    out = generated / "reproduction_checks.json"
    out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
