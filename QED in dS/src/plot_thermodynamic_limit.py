"""Thermodynamic-limit plotting utilities for Figure 2."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return (len(digits), digits + path.name)


def find_summary_file(root: Path) -> Path:
    files = [p for p in root.glob("**/npz/*.npz") if "__MACOSX" not in p.as_posix()]
    if not files:
        raise FileNotFoundError(f"No summary .npz files found under {root}")
    return sorted(files, key=_sort_key)[-1]


def load_summary(npz_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    npz_path = Path(npz_path).resolve()
    if npz_path.suffix != ".npz":
        raise ValueError(f"Expected a .npz file, got: {npz_path}")
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    return meta, arrays, json_path


def load_summary_from_source(source: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    source = Path(source).resolve()
    if source.suffix == ".npz":
        return load_summary(source)
    if source.is_dir():
        return load_summary(find_summary_file(source))
    if source.suffix == ".zip":
        with tempfile.TemporaryDirectory(prefix="thermodynamic_limit_") as workdir:
            workdir_path = Path(workdir)
            with zipfile.ZipFile(source) as archive:
                archive.extractall(workdir_path)
            return load_summary(find_summary_file(workdir_path))
    raise ValueError(f"Unsupported input source: {source}")


def polynomial_fit(x: Sequence[float], y: Sequence[float], *, quadratic: bool = False, xname: str = "x") -> dict[str, Any]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    cols = [np.ones_like(x), x]
    names = ["const", xname]
    if quadratic:
        cols.append(x**2)
        names.append(f"{xname}^2")
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    resid = y - pred
    return {
        "coef_names": names,
        "coef": coef,
        "pred": pred,
        "resid": resid,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "rss": float(np.sum(resid**2)),
        "limit": float(coef[0]),
        "quadratic": bool(quadratic),
    }


def predict_curve(fit: dict[str, Any], xgrid: np.ndarray) -> np.ndarray:
    coef = np.asarray(fit["coef"], dtype=float)
    xgrid = np.asarray(xgrid, dtype=float)
    y = coef[0] + coef[1] * xgrid
    if len(coef) >= 3:
        y = y + coef[2] * xgrid**2
    return y


def weighted_fit(
    x: Sequence[float],
    y: Sequence[float],
    sigma: Sequence[float] | None = None,
    *,
    quadratic: bool = False,
    xname: str = "x",
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if sigma is None:
        sigma = np.ones_like(y)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    valid = np.isfinite(sigma) & (sigma > 0)
    fallback = float(np.nanmedian(sigma[valid])) if np.any(valid) else 1.0
    sigma = np.where(valid, sigma, fallback)

    cols = [np.ones_like(x), x]
    names = ["const", xname]
    if quadratic:
        cols.append(x**2)
        names.append(f"{xname}^2")
    design = np.column_stack(cols)
    weights = 1.0 / sigma
    design_w = design * weights[:, None]
    y_w = y * weights
    coef, *_ = np.linalg.lstsq(design_w, y_w, rcond=None)
    pred = design @ coef
    resid = y - pred
    dof = max(1, len(y) - design.shape[1])
    chi2 = float(np.sum((resid / sigma) ** 2))
    normal = design_w.T @ design_w
    try:
        cov = (chi2 / dof) * np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        cov = (chi2 / dof) * np.linalg.pinv(normal)
    return {
        "coef_names": names,
        "coef": coef,
        "pred": pred,
        "resid": resid,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "rss": float(np.sum(resid**2)),
        "limit": float(coef[0]),
        "limit_se": float(np.sqrt(max(cov[0, 0], 0.0))),
        "cov": cov,
        "quadratic": bool(quadratic),
    }


def tau_uncertainty(tau: Sequence[float], tau_coarse: Sequence[float], dtau_local: Sequence[float], dtau_full: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    tau_coarse = np.asarray(tau_coarse, dtype=float)
    dtau_local = np.asarray(dtau_local, dtype=float)
    dtau_local = np.where(np.isfinite(dtau_local) & (dtau_local > 0), dtau_local, dtau_full)
    sigma = np.maximum(0.5 * dtau_local, 0.5 * np.abs(tau - tau_coarse))
    return np.maximum(sigma, max(0.10 * dtau_full, 1e-6))


def gap_uncertainty(gap: Sequence[float], gap_coarse: Sequence[float]) -> np.ndarray:
    gap = np.asarray(gap, dtype=float)
    gap_coarse = np.asarray(gap_coarse, dtype=float)
    return np.maximum(0.5 * np.abs(gap - gap_coarse), 1e-4)


def _best_fit(linear: dict[str, Any], quadratic: dict[str, Any] | None) -> dict[str, Any]:
    if quadratic is not None and quadratic["rmse"] < 0.85 * linear["rmse"]:
        return quadratic
    return linear


def analyze_summary(meta: dict[str, Any], arrays: dict[str, np.ndarray], *, thermodynamic_axis: str = "invLphys") -> dict[str, Any]:
    required = ["a_lat", "N", "ell_phys", "tau_vals", "gap_curves", "tau_star", "gap_star"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing arrays: {missing}")

    a_all = np.asarray(arrays["a_lat"], dtype=float).reshape(-1)
    n_all = np.asarray(arrays["N"], dtype=int).reshape(-1)
    ell_all = np.asarray(arrays["ell_phys"], dtype=float).reshape(-1)
    tau_vals = np.asarray(arrays["tau_vals"], dtype=float).reshape(-1)
    gap_curves = np.asarray(arrays["gap_curves"], dtype=float)
    tau_star_all = np.asarray(arrays["tau_star"], dtype=float).reshape(-1)
    gap_star_all = np.asarray(arrays["gap_star"], dtype=float).reshape(-1)
    tau_star_coarse_all = np.asarray(arrays.get("tau_star_coarse", tau_star_all), dtype=float).reshape(-1)
    gap_star_coarse_all = np.asarray(arrays.get("gap_star_coarse", gap_star_all), dtype=float).reshape(-1)
    dtau_local_all = np.asarray(arrays.get("dtau_local", np.full_like(gap_star_all, np.nan)), dtype=float).reshape(-1)
    dtau_full_arr = np.asarray(arrays.get("dtau_full", np.full_like(gap_star_all, np.nan)), dtype=float).reshape(-1)
    dtau_full = float(np.nanmedian(dtau_full_arr)) if np.any(np.isfinite(dtau_full_arr)) else float(np.median(np.diff(tau_vals)))

    branches: list[dict[str, Any]] = []
    for aval in sorted(set(map(float, a_all)), reverse=True):
        idx = np.where(np.isclose(a_all, aval, atol=1e-12, rtol=0.0))[0]
        idx = idx[np.argsort(ell_all[idx])]
        ell = ell_all[idx]
        nvals = n_all[idx]
        x = 1.0 / ell if thermodynamic_axis == "invLphys" else 1.0 / nvals.astype(float)
        tau = tau_star_all[idx]
        gap = gap_star_all[idx]
        tau_coarse = tau_star_coarse_all[idx]
        gap_coarse = gap_star_coarse_all[idx]
        dtau_local = dtau_local_all[idx]

        tau_linear = polynomial_fit(x, tau, quadratic=False, xname=thermodynamic_axis)
        tau_quad = polynomial_fit(x, tau, quadratic=True, xname=thermodynamic_axis) if len(idx) >= 3 else None
        tau_fit = _best_fit(tau_linear, tau_quad)

        gap_linear = polynomial_fit(x, gap, quadratic=False, xname=thermodynamic_axis)
        gap_quad = polynomial_fit(x, gap, quadratic=True, xname=thermodynamic_axis) if len(idx) >= 3 else None
        gap_fit = _best_fit(gap_linear, gap_quad)

        tau_err = tau_uncertainty(tau, tau_coarse, dtau_local, dtau_full)
        gap_err = gap_uncertainty(gap, gap_coarse)

        wt_tau_linear = weighted_fit(x, tau, tau_err, quadratic=False, xname=thermodynamic_axis)
        wt_tau_quad = weighted_fit(x, tau, tau_err, quadratic=True, xname=thermodynamic_axis) if len(idx) >= 3 else None
        wt_tau = wt_tau_quad if (tau_fit["quadratic"] and wt_tau_quad is not None) else wt_tau_linear
        wt_tau_alt = wt_tau_linear if (tau_fit["quadratic"] and wt_tau_quad is not None) else wt_tau_quad
        tau_spread = 0.5 * abs(wt_tau["limit"] - wt_tau_alt["limit"]) if wt_tau_alt is not None else 0.0

        wt_gap_linear = weighted_fit(x, gap, gap_err, quadratic=False, xname=thermodynamic_axis)
        wt_gap_quad = weighted_fit(x, gap, gap_err, quadratic=True, xname=thermodynamic_axis) if len(idx) >= 3 else None
        wt_gap = wt_gap_quad if (gap_fit["quadratic"] and wt_gap_quad is not None) else wt_gap_linear
        wt_gap_alt = wt_gap_linear if (gap_fit["quadratic"] and wt_gap_quad is not None) else wt_gap_quad
        gap_spread = 0.5 * abs(wt_gap["limit"] - wt_gap_alt["limit"]) if wt_gap_alt is not None else 0.0

        branches.append(
            {
                "a_lat": float(aval),
                "N": nvals,
                "ell_phys": ell,
                "x": x,
                "tau_star": tau,
                "gap_star": gap,
                "tau_star_coarse": tau_coarse,
                "gap_star_coarse": gap_coarse,
                "gap_curves": gap_curves[idx, :],
                "dtau_local": dtau_local,
                "tau_err": tau_err,
                "gap_err": gap_err,
                "fit_tau_best": tau_fit,
                "fit_gap_best": gap_fit,
                "tau_limit_err": float(math.hypot(wt_tau["limit_se"], tau_spread)),
                "gap_limit_err": float(math.hypot(wt_gap["limit_se"], gap_spread)),
            }
        )

    a_values = np.array([branch["a_lat"] for branch in branches], dtype=float)
    tau_limits = np.array([branch["fit_tau_best"]["limit"] for branch in branches], dtype=float)
    tau_limit_err = np.array([branch["tau_limit_err"] for branch in branches], dtype=float)
    return {
        "meta": meta,
        "tau_vals": tau_vals,
        "branches": branches,
        "common_ell": float(min(np.max(branch["ell_phys"]) for branch in branches)),
        "fit_cont_a": polynomial_fit(a_values, tau_limits, quadratic=False, xname="a"),
        "fit_cont_a2": polynomial_fit(a_values**2, tau_limits, quadratic=False, xname="a^2"),
        "a": a_values,
        "tau_limits": tau_limits,
        "tau_limit_err": tau_limit_err,
        "thermodynamic_axis": thermodynamic_axis,
    }


def save_figure(fig: plt.Figure, out_path: str | Path, **kwargs: Any) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **kwargs)
    return out_path


def make_figure(analysis: dict[str, Any], out_pdf: str | Path, out_png: str | Path | None = None) -> tuple[Path, Path | None]:
    set_style()
    branches = analysis["branches"]
    tau_vals = np.asarray(analysis["tau_vals"], dtype=float)
    common_ell = float(analysis["common_ell"])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0, 0]
    selected = []
    for index, branch in enumerate(branches):
        row = int(np.argmin(np.abs(np.asarray(branch["ell_phys"], dtype=float) - common_ell)))
        selected.append((branch, row))
        color = colors[index % len(colors)]
        ax.plot(tau_vals, np.asarray(branch["gap_curves"], dtype=float)[row], color=color, label=rf"$a_{{\rm latt}}={branch['a_lat']:.3g}$")
        ax.plot([float(branch["tau_star"][row])], [float(branch["gap_star"][row])], marker="o", ms=3.3, color=color, lw=0)
    tau_pick = np.array([float(branch["tau_star"][row]) for branch, row in selected], dtype=float)
    ax.set_xlim(float(np.min(tau_pick) - 0.45), float(np.max(tau_pick) + 0.28))
    ax.set_ylim(-0.02, 2.90)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta(\tau)$")
    ax.set_title(rf"(a) Largest-volume dip traces, $\ell_{{\rm phys}}={common_ell:.0f}$", pad=4.0, fontweight="normal", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=2.2)

    ax = axes[0, 1]
    from matplotlib.lines import Line2D

    handles = []
    for index, branch in enumerate(branches):
        color = colors[index % len(colors)]
        x = 1.0 / np.asarray(branch["ell_phys"], dtype=float)
        y = np.asarray(branch["tau_star"], dtype=float)
        ax.plot(x, y, "o", color=color, ms=4.0)
        xx = np.linspace(0.0, np.max(x) * 1.03, 200)
        ax.plot(xx, predict_curve(branch["fit_tau_best"], xx), "-", color=color, lw=1.1)
        handles.append(Line2D([0], [0], marker="o", color=color, lw=1.1, markersize=4, label=rf"$a={branch['a_lat']:.3g}$"))
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right", handlelength=1.2)
    ax.set_xlabel(r"$1/\ell_{\rm phys}$")
    ax.set_ylabel(r"$\tau_\ast$")
    ax.set_title("(b) Fixed-cutoff thermodynamic extrapolation", pad=4.0, fontweight="normal", loc="left")

    ax = axes[1, 0]
    a_values = np.asarray(analysis["a"], dtype=float)
    tau_limits = np.asarray(analysis["tau_limits"], dtype=float)
    for index, (aval, tlim) in enumerate(zip(a_values, tau_limits)):
        ax.plot([aval], [tlim], "o", ms=6.0, color=colors[index % len(colors)])
    aa = np.linspace(0.0, np.max(a_values) * 1.02, 300)
    fit_a = analysis["fit_cont_a"]
    fit_a2 = analysis["fit_cont_a2"]
    ax.plot(aa, predict_curve(fit_a, aa), "-", lw=1.4, label=rf"linear in $a$: $\tau_{{\infty,0}}\!\approx {fit_a['limit']:.2f}$")
    ax.plot(aa, fit_a2["coef"][0] + fit_a2["coef"][1] * aa**2, "--", lw=1.4, label=rf"linear in $a^2$: $\tau_{{\infty,0}}\!\approx {fit_a2['limit']:.2f}$")
    ax.legend(frameon=False, loc="lower left", handlelength=2.0)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel(r"$a_{\rm latt}$")
    ax.set_ylabel(r"$\tau_\ast^{(\infty)}(a_{\rm latt})$")
    ax.set_title("(c) Continuum drift of the thermodynamic dip time", pad=4.0, fontweight="normal", loc="left")

    ax = axes[1, 1]
    gap_limit = np.array([branch["fit_gap_best"]["limit"] for branch in branches], dtype=float)
    gap_limit_err = np.array([branch["gap_limit_err"] for branch in branches], dtype=float)
    ax.errorbar(a_values, gap_limit, yerr=gap_limit_err, fmt="o-", ms=5.0, capsize=2.5, lw=1.2, color=colors[0])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlim(0.45, 1.02)
    ymin = min(float(np.min(gap_limit - gap_limit_err)), -0.020)
    ymax = max(float(np.max(gap_limit + gap_limit_err)), 0.020)
    ax.set_ylim(ymin * 1.02, ymax * 1.02)
    ax.set_xlabel(r"$a_{\rm latt}$")
    ax.set_ylabel(r"$\Delta_\ast^{(\infty)}(a_{\rm latt})$")
    ax.set_title("(d) Thermodynamic dip-depth limit", pad=4.0, fontweight="normal", loc="left")

    for axis in axes.flat:
        axis.set_facecolor("white")
    fig.tight_layout()
    out_pdf = save_figure(fig, out_pdf)
    if out_png is not None:
        out_png = save_figure(fig, out_png, dpi=220)
    plt.close(fig)
    return out_pdf, out_png


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the thermodynamic-limit Figure 2.")
    parser.add_argument("source", help="Input source archive, extracted run directory, or summary .npz file")
    parser.add_argument("--outdir", default="generated", help="Directory for the output figure")
    parser.add_argument("--basename", default="figure2_thermodynamic_limit", help="Base filename for the saved figure")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    meta, arrays, _ = load_summary_from_source(args.source)
    analysis = analyze_summary(meta, arrays, thermodynamic_axis="invLphys")
    out_pdf = outdir / f"{args.basename}.pdf"
    out_png = outdir / f"{args.basename}.png"
    make_figure(analysis, out_pdf, out_png)
    print(out_pdf)
    print(out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
