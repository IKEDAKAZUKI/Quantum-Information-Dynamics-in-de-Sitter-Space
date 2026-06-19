"""Plotting utilities for Figure 1.

The routines read the bundled transition-summary data and generate the
four-panel dynamical-transition figure.
"""

from __future__ import annotations

import json
import logging
import warnings
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ----------------------------
# Style
# ----------------------------

def set_pub_style() -> None:
    # Suppress benign font timestamp warnings from fontTools
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.labelpad": 12.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Times-like (falls back if missing)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class Run:
    json_path: Path
    npz_path: Path
    kind: str
    L: float
    g: float
    scale: bool
    n_t: int
    n_m: int
    m_min: float
    m_max: float
    init_state: str

    @property
    def run_type(self) -> str:
        k = (self.kind or "").lower()
        if k == "structure_factor":
            return "sk"
        if k == "transition":
            return "coarse"
        return k or "other"

    def load(self) -> Dict[str, np.ndarray]:
        return dict(np.load(self.npz_path, allow_pickle=True))


# ----------------------------
# IO helpers
# ----------------------------

PathLike = Union[str, Path]


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def _find_data_root(extracted_root: Path) -> Path:
    extracted_root = Path(extracted_root)
    counts: Dict[Path, int] = {}
    for p in extracted_root.rglob("*.npz"):
        counts[p.parent] = counts.get(p.parent, 0) + 1
    if not counts:
        raise FileNotFoundError(f"No .npz files found under {extracted_root}")
    return max(counts.items(), key=lambda kv: kv[1])[0]


def load_runs(root: Path) -> List[Run]:
    root = Path(root)
    # If user passes the zip path, extract.
    if root.is_file() and root.suffix.lower() == ".zip":
        root = extract_zip(root, out_dir=root.with_suffix("").parent / (root.stem + "_extracted"))
    # If user passes the extracted top directory, find the folder with most npz files.
    if root.is_dir():
        data_root = _find_data_root(root)
    else:
        raise FileNotFoundError(root)

    runs: List[Run] = []
    for j in sorted(data_root.glob("*.json")):
        meta = json.loads(j.read_text())
        npz = j.with_suffix(".npz")
        if not npz.exists():
            continue
        runs.append(
            Run(
                json_path=j,
                npz_path=npz,
                kind=str(meta.get("kind", meta.get("tag", ""))),
                L=float(meta.get("L")),
                g=float(meta.get("g")),
                scale=bool(meta.get("scale_gauge_with_a")),
                n_t=int(meta.get("n_t")),
                n_m=int(meta.get("n_m")),
                m_min=float(meta.get("m_min")),
                m_max=float(meta.get("m_max")),
                init_state=str(meta.get("init_state", "")),
            )
        )
    if not runs:
        raise FileNotFoundError(f"No runs found under {data_root}")
    return runs


def pick_run(
    runs: List[Run],
    *,
    L: float,
    scale: bool,
    run_type: str,
    prefer_nt: Optional[int] = None,
) -> Optional[Run]:
    cands = [
        r
        for r in runs
        if abs(float(r.L) - float(L)) < 1e-12
        and bool(r.scale) == bool(scale)
        and r.run_type == run_type
    ]
    if not cands:
        return None
    if prefer_nt is not None:
        exact = [r for r in cands if int(r.n_t) == int(prefer_nt)]
        if exact:
            cands = exact
    cands = sorted(cands, key=lambda r: (int(r.n_t), r.json_path.name))
    return cands[-1]


# ----------------------------
# Analysis helpers
# ----------------------------

def gap_metrics(arr: Dict[str, np.ndarray], L: float) -> Dict[str, np.ndarray]:
    m = np.asarray(arr["m_vals"], float)
    t = np.asarray(arr["t_vals"], float)
    tau = t / float(L)
    gap = np.asarray(arr["gap"], float)  # (nm, nt)

    idx_m = np.argmin(gap, axis=0)  # per time
    mcrit = m[idx_m]
    gapcrit = gap[idx_m, np.arange(gap.shape[1])]

    return dict(m=m, t=t, tau=tau, gap=gap, mcrit=mcrit, gapcrit=gapcrit)


def choose_tau_star(met: Dict[str, np.ndarray], tau_cut: Optional[float] = None) -> float:
    tau = met["tau"]
    gapcrit = met["gapcrit"]
    if tau_cut is None:
        tau_cut = 1.8
    mask = tau <= float(tau_cut)
    if not np.any(mask):
        return float(tau[np.argmin(gapcrit)])
    i = np.argmin(gapcrit[mask])
    return float(tau[mask][i])


def nearest_index(x: np.ndarray, x0: float) -> int:
    return int(np.argmin(np.abs(np.asarray(x, float) - float(x0))))


def get_nonadiab(arr: Dict[str, np.ndarray]) -> np.ndarray:
    if "nonadiab" in arr:
        return np.asarray(arr["nonadiab"], float)
    if "one_minus_Fgs" in arr:
        return np.asarray(arr["one_minus_Fgs"], float)
    if "Fgs" in arr:
        return 1.0 - np.asarray(arr["Fgs"], float)
    raise KeyError("non-adiabaticity array not found (expected one of: nonadiab, one_minus_Fgs, Fgs).")


def edges_from_centers(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return np.array([0.0, 1.0])
    if x.size == 1:
        dx = 1.0
        return np.array([x[0] - 0.5 * dx, x[0] + 0.5 * dx])
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


# ----------------------------
# Plot helpers
# ----------------------------

def _imshow_heat(ax, data2d, extent, cmap, norm=None, vmin=None, vmax=None):
    """Heatmap (imshow) without custom clip boxes (robust across viewers)."""
    ax.set_facecolor("white")
    ax.patch.set_zorder(0)
    im = ax.imshow(
        data2d,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    im.set_zorder(1)
    ax.set_xlim(float(extent[0]), float(extent[1]))
    ax.set_ylim(float(extent[2]), float(extent[3]))
    return im


def _vector_colorbar_vertical(
    cax,
    *,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    title: str = "",
    ticks: Optional[List[float]] = None,
    ticklabels: Optional[List[str]] = None,
) -> None:
    """Draw a vertical colorbar in cax."""
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    n = 256
    y_edges = np.linspace(vmin, vmax, n + 1)
    x_edges = np.array([0.0, 1.0])
    vals = np.linspace(vmin, vmax, n, dtype=float).reshape(n, 1)

    cax.pcolormesh(
        x_edges,
        y_edges,
        vals,
        cmap=cmap_obj,
        norm=norm,
        shading="flat",
        edgecolors="face",
        linewidth=0.0,
        antialiased=False,
    )

    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(vmin, vmax)
    cax.set_xticks([])

    if ticks is None:
        locator = MaxNLocator(nbins=5)
        ticks = [float(t) for t in locator.tick_values(vmin, vmax) if (t >= vmin - 1e-12 and t <= vmax + 1e-12)]

    cax.set_yticks(ticks)
    if ticklabels is not None:
        cax.set_yticklabels(ticklabels)

    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    if title:
        cax.set_title(title, pad=2, fontsize=8)

    cax.tick_params(pad=1.2)
    for sp in cax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)


# ----------------------------
# Main figure
# ----------------------------

def make_onepage_summary(
    root_or_zip: PathLike,
    *,
    out_path: PathLike = "figure1.pdf",
    L_list=(0.5, 1.0, 2.0),
    L_ref: float = 1.0,
    scale: bool = True,
    prefer_nt: Optional[int] = None,
    tau_cut: Optional[float] = None,
) -> Path:

    set_pub_style()

    runs = load_runs(Path(root_or_zip))

    coarse_ref = pick_run(runs, L=float(L_ref), scale=bool(scale), run_type="coarse", prefer_nt=prefer_nt)
    if coarse_ref is None:
        raise ValueError(f"No coarse run found at L={L_ref}, scale={scale}")

    arrC = coarse_ref.load()
    met = gap_metrics(arrC, L=float(L_ref))
    tau = met["tau"]
    m = met["m"]
    gap = met["gap"]

    tau_star = choose_tau_star(met, tau_cut=tau_cut)
    it_star = nearest_index(tau, tau_star)
    m_star = float(met["mcrit"][it_star])

    # coarse runs for each L
    line_runs: Dict[float, Run] = {}
    for L in L_list:
        r = pick_run(runs, L=float(L), scale=bool(scale), run_type="coarse", prefer_nt=prefer_nt)
        if r is not None:
            line_runs[float(L)] = r

    # structure factor run at L_ref
    sk_cands = [
        r
        for r in runs
        if r.run_type == "sk" and abs(float(r.L) - float(L_ref)) < 1e-12 and bool(r.scale) == bool(scale)
    ]
    sk_run = None
    if sk_cands:
        sk_run = sorted(sk_cands, key=lambda r: abs(float(r.m_min) - float(m_star)))[0]

    # ----------------------------
    # Figure layout
    # ----------------------------
    fig = plt.figure(figsize=(8.3, 5.7))
    #gs = fig.add_gridspec(2, 2, left=0.10, right=0.86, bottom=0.19, top=0.965, wspace=0.32, hspace=0.44)
    gs = fig.add_gridspec(
    2, 2, 
    left=0.08,    # 左の余白を減らす (0.10 -> 0.05)
    right=0.95,   # 右の余白を減らす (0.86 -> 0.98)
    bottom=0.2,  # 下の余白を減らす (0.19 -> 0.05)
    top=0.97,     # 上の余白を減らす (0.965 -> 0.98)
    wspace=0.32, 
    hspace=0.44)

    # (a)
    gsA = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.065], wspace=0.08)
    axA = fig.add_subplot(gsA[0, 0])
    caxA = fig.add_subplot(gsA[0, 1])

    # (b)
    axB = fig.add_subplot(gs[0, 1])

    # (c)
    axC = fig.add_subplot(gs[1, 0])

    # (d)
    gsD = gs[1, 1].subgridspec(1, 2, width_ratios=[1.0, 0.065], wspace=0.08)
    axD = fig.add_subplot(gsD[0, 0])
    caxD = fig.add_subplot(gsD[0, 1])

    # ----------------------------
    # (a) Gap landscape
    # ----------------------------
    m_edges = edges_from_centers(m)
    tau_edges = edges_from_centers(tau)
    extentA = [float(m_edges[0]), float(m_edges[-1]), float(tau_edges[0]), float(tau_edges[-1])]
    imA = _imshow_heat(axA, gap.T, extent=extentA, cmap="magma")

    axA.plot(met["mcrit"], tau, color="w", lw=1.0, alpha=0.95)
    axA.axhline(tau_star, color="w", lw=0.8, ls=":", alpha=0.9)
    axA.axvline(m_star, color="w", lw=0.8, ls="--", alpha=0.9)
    axA.text(0.03, 0.04, rf"$m_*\simeq {m_star:.2f}$", transform=axA.transAxes, color="k", fontsize=8, ha="left", va="bottom")

    axA.set_xlabel(r"mass $m$",fontsize=10)
    axA.set_ylabel(r"$\tau=t/L$", labelpad=14)
    axA.set_title(r"(a) Energy gap", loc="left", pad=2)

    vminA, vmaxA = [float(x) for x in imA.get_clim()]
    _vector_colorbar_vertical(caxA, cmap=imA.get_cmap(), norm=imA.norm, vmin=vminA, vmax=vmaxA, title=r"$\Delta$")

    # ----------------------------
    # (b) Non-adiabaticity
    # ----------------------------
    Ls_present = sorted(line_runs.keys())
    L_to_color = {0.5: "C0", 1.0: "C1", 2.0: "C2", 4.0: "C3", 8.0: "C4"}

    for L in Ls_present:
        r = line_runs[L]
        arr = r.load()
        tL = np.asarray(arr["t_vals"], float)
        tauL = tL / float(L)
        mvals = np.asarray(arr["m_vals"], float)
        im = nearest_index(mvals, m_star)
        non = np.asarray(get_nonadiab(arr), float)[im, :]
        axB.plot(tauL, non, color=L_to_color.get(L, None), lw=1.3)

    axB.axvline(tau_star, color="k", lw=0.8, ls=":", alpha=0.6)
    axB.set_xlabel(r"$\tau=t/L$",fontsize=10)
    axB.set_ylabel(r"$1-F_{\rm GS}$", labelpad=14,fontsize=10)
    axB.set_title(r"(b) Non-adiabaticity", loc="left", pad=2)

    # ----------------------------
    # (c) Excitation energy density
    # ----------------------------
    for L in Ls_present:
        r = line_runs[L]
        arr = r.load()
        tL = np.asarray(arr["t_vals"], float)
        tauL = tL / float(L)
        mvals = np.asarray(arr["m_vals"], float)
        im = nearest_index(mvals, m_star)
        eps = np.asarray(arr["eps_exc"], float)[im, :]
        axC.plot(tauL, eps, color=L_to_color.get(L, None), lw=1.3)

    axC.axvline(tau_star, color="k", lw=0.8, ls=":", alpha=0.6)
    axC.set_xlabel(r"$\tau=t/L$",fontsize=10)
    axC.set_ylabel(r"$\epsilon_{\rm exc}$", labelpad=14,fontsize=10)
    axC.set_title(r"(c) Excitation energy density", loc="left", pad=2)

    # ----------------------------
    # (d) Structure-factor response
    # ----------------------------
    if sk_run is not None:
        arrS = sk_run.load()
        tS = np.asarray(arrS["t_vals"], float)
        tauS = tS / float(sk_run.L)
        kS = np.asarray(arrS["k_vals"], float)

        Sk = np.asarray(arrS["Sk_q"][0], float)
        Sk0 = np.asarray(arrS["Sk_q0"][0], float)
        dSk = Sk - Sk0  # (nt, Nk)

        vmax = float(np.percentile(np.abs(dSk.reshape(-1)), 99))
        vmax = max(vmax, 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        k_edges = edges_from_centers(kS)
        tauS_edges = edges_from_centers(tauS)
        extentD = [float(k_edges[0]), float(k_edges[-1]), float(tauS_edges[0]), float(tauS_edges[-1])]
        imD = _imshow_heat(axD, dSk, extent=extentD, cmap="RdBu_r", norm=norm)

        axD.axvline(math.pi, color="k", lw=0.8, ls="--", alpha=0.7)
        axD.set_xlabel(r"comoving $k$",fontsize=10)
        axD.set_ylabel(r"$\tau=t/L$", labelpad=14,fontsize=10)
        axD.set_title(r"(d) Structure-factor response", loc="left", pad=2)

        _vector_colorbar_vertical(
            caxD,
            cmap=imD.get_cmap(),
            norm=norm,
            vmin=-vmax,
            vmax=+vmax,
            title=r"$\Delta S_q$",
            ticks=[-vmax, 0.0, vmax],
            ticklabels=[f"{-vmax:.2f}", "0", f"{vmax:.2f}"],
        )

        # Inset: peak momentum p_peak(τ) = k_peak/a(τ), plus reference π/a(τ)
        axins = inset_axes(
            axD,
            width="34%",
            height="26%",
            loc="upper left",
            borderpad=0.55,
            #bbox_to_anchor=(0.28, 0.0, 1.0, 1.0),
            bbox_to_anchor=(0.09, 0, 1.2, 1.0),
            bbox_transform=axD.transAxes,
        )
        axins.set_facecolor("white")

        win = (kS > 0.6 * math.pi) & (kS < 1.4 * math.pi)
        if np.any(win):
            kW = kS[win]
            dW = dSk[:, win]
            k_peak = kW[np.argmax(np.abs(dW), axis=1)]
        else:
            k_peak = kS[np.argmax(np.abs(dSk), axis=1)]

        p_peak = k_peak * np.exp(-tauS)  # since a(τ)=e^{τ}
        p_pi = math.pi * np.exp(-tauS)

        axins.plot(tauS, p_peak, lw=1.0, label=r"$p_{\rm peak}(\tau)$")
        axins.plot(tauS, p_pi, lw=1.0, ls="--", alpha=0.9, label=r"$\pi/a(\tau)$")
        axins.set_xlabel(r"$\tau$", labelpad=0,fontsize=9)
        axins.set_ylabel(r"$p$", labelpad=1,fontsize=9)
        axins.tick_params(labelsize=5)
        axins.legend(loc="upper right", fontsize=7, frameon=False, handlelength=2.0)
        #axins.set_title("Peak momentum", fontsize=8, pad=1)

    else:
        axD.text(0.5, 0.5, "No structure-factor data", transform=axD.transAxes, ha="center", va="center")
        caxD.axis("off")

    # ----------------------------
    # Shared legend
    # ----------------------------
    handles_L = [
        Line2D([0], [0], color=L_to_color.get(L, "k"), lw=1.6, label=fr"$L={L:g}$") for L in Ls_present
    ]
    fig.legend(
        handles=handles_L,
        loc="lower center",
        ncol=max(1, len(handles_L)),
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.2,
        handlelength=2.2,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


__all__ = ["make_onepage_summary", "load_runs", "Run"]
