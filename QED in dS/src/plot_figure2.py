"""Plotting utilities for Figure 2."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.gridspec import GridSpec


def quadratic_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    j = int(np.argmin(y))
    if 0 < j < len(x) - 1:
        xs = x[j - 1 : j + 2]
        ys = y[j - 1 : j + 2]
        p = np.polyfit(xs, ys, 2)
        if p[0] > 0:
            x0 = -p[1] / (2 * p[0])
            if xs[0] <= x0 <= xs[-1]:
                return float(x0), float(np.polyval(p, x0))
    return float(x[j]), float(y[j])


def load_summary(data_dir: str | Path):
    data_dir = Path(data_dir)
    npz_path = data_dir / "fixed_mass_gap_summary.npz"
    json_path = data_dir / "fixed_mass_gap_summary.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    arr = np.load(npz_path)
    N = arr["N"].astype(int)
    a = arr["a_lat"].astype(float)
    tau_grid = arr["tau_vals"].astype(float)
    gap_curves = arr["gap_curves"].astype(float)
    tau_ref, gap_ref = [], []
    for row in gap_curves:
        t0, g0 = quadratic_minimum(tau_grid, row)
        tau_ref.append(t0)
        gap_ref.append(g0)
    return meta, N, a, tau_grid, gap_curves, np.array(tau_ref), np.array(gap_ref)


def fit_line(x: np.ndarray, y: np.ndarray):
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ coef
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return coef, rmse


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def title_with_panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", pad=3.5, fontsize=9, fontweight="normal")


def make_figure(data_dir: str | Path, outdir: str | Path, stem: str = "figure2"):
    set_style()
    meta, N, a, tau_grid, gap_curves, tau_ref, gap_ref = load_summary(data_dir)
    sel_traces = [20, 40, 60, 80, 100]
    sel_fit = N >= 50
    sel_fit2 = N >= 60
    coef, _ = fit_line(a[sel_fit], tau_ref[sel_fit])
    coef2, _ = fit_line(a[sel_fit2], tau_ref[sel_fit2])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(7.25, 5.9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.08, 0.92], hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for n in sel_traces:
        i = np.where(N == n)[0][0]
        f = PchipInterpolator(tau_grid, gap_curves[i])
        xf = np.linspace(tau_grid.min(), tau_grid.max(), 400)
        ax1.plot(xf, f(xf), lw=1.6, label=fr"$N={n}$")
        ax1.scatter([tau_ref[i]], [gap_ref[i]], s=22, zorder=4)
    ax1.set_yscale("log")
    ax1.set_xlim(0, 3.75)
    ax1.set_ylim(3e-3, 2e1)
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$\Delta(\tau)$")
    title_with_panel(ax1, "a", "Gap flow")
    ax1.legend(frameon=False, loc="lower left", handlelength=2.2)

    for n in sel_traces:
        i = np.where(N == n)[0][0]
        f = PchipInterpolator(tau_grid, gap_curves[i])
        xf = np.linspace(1.35, 3.35, 400)
        ax2.plot(xf, f(xf), lw=1.6)
        ax2.scatter([tau_ref[i]], [gap_ref[i]], s=22, zorder=4)
    ax2.set_xlim(1.35, 3.35)
    ax2.set_ylim(0, 0.62)
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"$\Delta(\tau)$")
    title_with_panel(ax2, "b", "Late-time dip")

    ax3.scatter(a, tau_ref, s=30, facecolors="none", edgecolors="0.6", linewidths=1.0, label="all runs")
    ax3.scatter(a[sel_fit], tau_ref[sel_fit], s=36, color=colors[0], zorder=3, label=r"fit set ($N\geq 50$)")
    for i, n in enumerate(N):
        if n in [20, 30, 40, 50, 60, 80, 100]:
            ax3.annotate(f"{n}", (a[i], tau_ref[i]), textcoords="offset points", xytext=(3, 3), fontsize=7, color="0.35")
    xx = np.linspace(0, 0.52, 200)
    ax3.plot(xx, coef[0] + coef[1] * xx, lw=1.6, color=colors[0], label=fr"linear fit: $\tau_c\!\approx\!{coef[0]:.2f}$")
    ax3.plot(xx, coef2[0] + coef2[1] * xx, lw=1.2, color=colors[1], ls="--", label=fr"$N\geq 60$: $\tau_c\!\approx\!{coef2[0]:.2f}$")
    ax3.scatter([0], [coef[0]], marker="*", s=85, color=colors[0], zorder=5)
    ax3.set_xlim(-0.02, 1.26)
    ax3.set_ylim(1.4, 4.1)
    ax3.set_xlabel(r"$a_{\rm latt}$")
    ax3.set_ylabel(r"$\tau_\ast$")
    title_with_panel(ax3, "c", "Continuum drift")
    ax3.legend(frameon=False, loc="lower left", fontsize=8)

    order = np.argsort(N)
    N_ord = N[order]
    g_ord = gap_curves[order]
    x = tau_grid
    dx0 = (x[1] - x[0]) / 2
    im = ax4.imshow(g_ord, extent=[x[0] - dx0, x[-1] + dx0, 0, len(N_ord)], origin="lower", aspect="auto", interpolation="nearest", rasterized=True)
    ax4.set_yticks(np.arange(len(N_ord)) + 0.5)
    ax4.set_yticklabels([fr"$N={n}$" for n in N_ord])
    ax4.set_xlim(0, 3.75)
    ax4.set_xlabel(r"$\tau$")
    title_with_panel(ax4, "d", "All runs")
    for row, i in enumerate(order):
        ax4.scatter([tau_ref[i]], [row + 0.5], s=23, facecolors="none", edgecolors="white", linewidths=1.0)
    cb = fig.colorbar(im, ax=ax4, fraction=0.047, pad=0.04)
    cb.set_label(r"$\Delta$")

    fig.text(0.50, 0.985, r"$m=-1.5,\ L_{\rm phys}=24$", ha="center", va="top", fontsize=10)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / stem
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", dpi=600)
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(base) + ".pdf", str(base) + ".png"


__all__ = ["make_figure", "load_summary", "quadratic_minimum"]
