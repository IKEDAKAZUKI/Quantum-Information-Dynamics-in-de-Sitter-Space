"""Plotting utilities for Figure 3."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14.5,
        "legend.fontsize": 8.3,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": False,
        "ytick.right": False,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_summary_data(data_dir: str | Path, betas: Iterable[float], beta_heatmap: float, ell_values: Iterable[int] = (2, 3, 4)):
    data_dir = Path(data_dir)
    d = np.load(data_dir / "irreversibility_front_data.npz", allow_pickle=True)
    heatmap = {
        "tau_dense": d["heat_tau_dense"],
        "m_dense": d["heat_m_dense"],
        "Z_dense": d["heat_Z_dense"],
        "mcrit_dense": d["heat_mcrit_dense"],
    }

    beta_csv = pd.read_csv(data_dir / "temperature_fronts.csv")
    beta_fronts: Dict[float, Dict[str, np.ndarray]] = {}
    for beta in [float(b) for b in betas]:
        sub = beta_csv[np.isclose(beta_csv["beta"], beta)].sort_values("tau")
        if sub.empty:
            raise FileNotFoundError(f"Missing beta={beta:g} in temperature_fronts.csv")
        beta_fronts[beta] = {
            "tau": sub["tau"].to_numpy(dtype=float),
            "mcrit": sub["mcrit"].to_numpy(dtype=float),
            "x0": sub["m_front"].to_numpy(dtype=float) - sub["mcrit"].to_numpy(dtype=float),
            "width": sub["width"].to_numpy(dtype=float),
            "valid": sub["valid"].astype(bool).to_numpy(),
            "tau_onset": float(sub["tau_onset"].iloc[0]),
        }

    locc_csv = pd.read_csv(data_dir / "locc_fronts.csv")
    beta_ref = float(beta_heatmap if float(beta_heatmap) in beta_fronts else max(beta_fronts))
    full_ref = beta_fronts[beta_ref]
    full_bundle = {
        "fit_tau": np.asarray(full_ref["tau"], dtype=float),
        "fit_x0": np.asarray(full_ref["x0"], dtype=float),
        "fit_width": np.asarray(full_ref["width"], dtype=float),
        "fit_valid": np.asarray(full_ref["valid"], dtype=bool),
        "gap_mcrit": np.asarray(full_ref["mcrit"], dtype=float),
        "fit_tau_onset": float(full_ref["tau_onset"]),
    }

    completion_bundle: Dict[str, Dict] = {"full": full_bundle, "meas": {}, "tomo": {}}
    for method_csv, key in [("measurement", "meas"), ("tomography", "tomo")]:
        for ell in ell_values:
            sub = locc_csv[(locc_csv["method"] == method_csv) & (locc_csv["ell"] == int(ell))].sort_values("tau")
            if sub.empty:
                continue
            completion_bundle[key][int(ell)] = {
                "fit_tau": sub["tau"].to_numpy(dtype=float),
                "fit_x0": sub["m_front"].to_numpy(dtype=float) - sub["mcrit"].to_numpy(dtype=float),
                "fit_width": sub["width"].to_numpy(dtype=float),
                "fit_valid": np.ones(len(sub), dtype=bool),
                "gap_mcrit": sub["mcrit"].to_numpy(dtype=float),
                "fit_tau_onset": float(sub["tau"].min()),
            }
    completion_bundle["csv"] = pd.read_csv(data_dir / "locc_summary.csv")
    profile_path = data_dir / "locc_profile_overlap.csv"
    completion_bundle["profile_csv"] = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame()
    finite_size_df = pd.read_csv(data_dir / "finite_size_summary.csv")
    return heatmap, beta_fronts, completion_bundle, finite_size_df


def front_curve_split(front_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    tau = np.asarray(front_npz["tau"])
    mcrit = np.asarray(front_npz["mcrit"])
    x0 = np.asarray(front_npz["x0"])
    width = np.asarray(front_npz["width"])
    finite = np.isfinite(x0) & np.isfinite(width)
    valid = np.asarray(front_npz["valid"]).astype(bool)
    return {
        "tau": tau,
        "mcrit": mcrit,
        "mfront": mcrit + x0,
        "width": width,
        "finite": finite,
        "valid": valid,
        "tau_onset": float(front_npz.get("tau_onset", np.nan)),
    }


def bundle_curve_split(bundle_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    tau = np.asarray(bundle_npz["fit_tau"])
    x0 = np.asarray(bundle_npz["fit_x0"])
    width = np.asarray(bundle_npz["fit_width"])
    valid = np.asarray(bundle_npz["fit_valid"]).astype(bool)
    mcrit = np.asarray(bundle_npz["gap_mcrit"])
    finite = np.isfinite(x0) & np.isfinite(width)
    return {
        "tau": tau,
        "mcrit": mcrit,
        "mfront": mcrit + x0,
        "width": width,
        "finite": finite,
        "valid": valid,
        "tau_onset": float(bundle_npz.get("fit_tau_onset", np.nan)),
    }


def build_curve_tables(beta_fronts: Dict[float, Dict[str, np.ndarray]], completion_bundle: Dict[str, Dict], betas: Iterable[float], ell_values: Iterable[int] = (2, 3, 4)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    beta_rows = []
    for beta in betas:
        d = front_curve_split(beta_fronts[float(beta)])
        for i in range(len(d["tau"])):
            beta_rows.append({
                "beta": float(beta),
                "tau": float(d["tau"][i]),
                "mcrit": float(d["mcrit"][i]),
                "m_front": float(d["mfront"][i]) if np.isfinite(d["mfront"][i]) else np.nan,
                "width": float(d["width"][i]) if np.isfinite(d["width"][i]) else np.nan,
                "finite": bool(d["finite"][i]),
                "valid": bool(d["valid"][i]),
                "tau_onset": float(d["tau_onset"]),
                "delta_tau_from_onset": float(d["tau"][i] - d["tau_onset"]) if np.isfinite(d["tau_onset"]) else np.nan,
                "abs_offset": abs(float(d["mfront"][i] - d["mcrit"][i])) if np.isfinite(d["mfront"][i]) else np.nan,
            })
    beta_df = pd.DataFrame(beta_rows)

    full = bundle_curve_split(completion_bundle["full"])
    locc_rows = []
    for method_key, label in [("meas", "measurement"), ("tomo", "tomography")]:
        for ell in ell_values:
            dat = bundle_curve_split(completion_bundle[method_key][ell])
            common_tau = dat["tau"][dat["finite"]]
            full_interp = np.interp(common_tau, full["tau"][full["finite"]], full["mfront"][full["finite"]])
            for t, mf, mc, w, ref in zip(common_tau, dat["mfront"][dat["finite"]], dat["mcrit"][dat["finite"]], dat["width"][dat["finite"]], full_interp):
                locc_rows.append({"method": label, "ell": int(ell), "tau": float(t), "m_front": float(mf), "mcrit": float(mc), "width": float(w), "abs_err_to_full": abs(float(mf - ref))})
    for t, mf, mc, w in zip(full["tau"][full["finite"]], full["mfront"][full["finite"]], full["mcrit"][full["finite"]], full["width"][full["finite"]]):
        locc_rows.append({"method": "full", "ell": 0, "tau": float(t), "m_front": float(mf), "mcrit": float(mc), "width": float(w), "abs_err_to_full": 0.0})
    locc_df = pd.DataFrame(locc_rows)
    return beta_df, locc_df


def _plot_front_with_confidence(ax, curve: Dict[str, np.ndarray], color: str, label: str, lw_solid: float = 2.2) -> None:
    tau = curve["tau"]
    mfront = curve["mfront"]
    finite = curve["finite"]
    valid = curve["valid"]
    if finite.any():
        ax.plot(tau[finite], mfront[finite], color=color, lw=1.4, ls=(0, (3, 2)), alpha=0.45, zorder=2)
        i0 = np.where(finite)[0][0]
        ax.scatter([tau[i0]], [mfront[i0]], s=18, color=color, ec="white", lw=0.4, zorder=4)
    if valid.any():
        ax.plot(tau[valid], mfront[valid], color=color, lw=lw_solid, zorder=3, label=label)
    elif finite.any():
        ax.plot(tau[finite], mfront[finite], color=color, lw=lw_solid * 0.9, alpha=0.85, zorder=3, label=label)


def _panel_title(label: str, title: str) -> str:
    return f"{label} {title}"


def _set_white_legend_text(leg) -> None:
    if leg is None:
        return
    for txt in leg.get_texts():
        txt.set_color("white")


def make_summary_figure(heatmap: Dict[str, np.ndarray], beta_fronts: Dict[float, Dict[str, np.ndarray]], completion_bundle: Dict[str, Dict], n12_df: pd.DataFrame, out_path: Path, beta_heatmap: float = 10.0, betas: Iterable[float] = (0.5, 1.0, 2.0, 5.0, 10.0), ell_focus: int = 4) -> Path:
    set_pub_style()
    betas = [float(b) for b in betas]
    tau_dense = np.asarray(heatmap["tau_dense"])
    m_dense = np.asarray(heatmap["m_dense"])
    Z_dense = np.asarray(heatmap["Z_dense"])
    mcrit_dense = np.asarray(heatmap["mcrit_dense"])

    beta_colors = {0.5: "#4db6ac", 1.0: "#2b83ba", 2.0: "#80bf5a", 5.0: "#f39c34", 10.0: "#d73027"}
    fig = plt.figure(figsize=(17.2, 8.6), dpi=230)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.04, 1.26], wspace=0.24)

    axH = fig.add_subplot(outer[0, 0])
    im = axH.imshow(Z_dense, origin="lower", aspect="auto", extent=[m_dense.min(), m_dense.max(), tau_dense.min(), tau_dense.max()], cmap="viridis", interpolation="bicubic", zorder=1)
    axH.plot(mcrit_dense, tau_dense, color="white", lw=2.3, alpha=0.98, label=r"$m_c(\tau)$", zorder=3)

    beta_curve = front_curve_split(beta_fronts[float(beta_heatmap)])
    tau_H = beta_curve["tau"]
    mfront_H = beta_curve["mfront"]
    finite_H = beta_curve["finite"]
    valid_H = beta_curve["valid"]
    if finite_H.any():
        axH.plot(mfront_H[finite_H], tau_H[finite_H], color="#ff9f1c", lw=1.5, ls=(0, (3, 2)), alpha=0.65, zorder=4)
        i0 = np.where(finite_H)[0][0]
        axH.scatter([mfront_H[i0]], [tau_H[i0]], s=22, color="#ff9f1c", ec="white", lw=0.5, zorder=5)
    if valid_H.any():
        axH.plot(mfront_H[valid_H], tau_H[valid_H], color="#ff9f1c", lw=2.6, zorder=5, label=rf"$m_{{\rm DPT}}(\tau,\beta={beta_heatmap:g})$")
    elif finite_H.any():
        axH.plot(mfront_H[finite_H], tau_H[finite_H], color="#ff9f1c", lw=2.3, alpha=0.9, zorder=5, label=rf"$m_{{\rm DPT}}(\tau,\beta={beta_heatmap:g})$")

    finite_cols = np.any(np.isfinite(Z_dense), axis=0)
    m_right = float(m_dense[finite_cols][-1]) if finite_cols.any() else float(m_dense.max())
    m_left = max(float(m_dense.min()), -4.6)
    axH.set_xlim(m_left, m_right)
    axH.set_ylim(0.0, float(tau_dense.max()))
    axH.set_xlabel(r"mass $m$")
    axH.set_ylabel(r"$\tau=t/L$")
    axH.set_title(_panel_title("(a)", rf"Front at $\beta={beta_heatmap:g}$"), loc="left", pad=10)
    legH = axH.legend(loc="lower left", frameon=False, handlelength=2.5)
    _set_white_legend_text(legH)
    cbar = fig.colorbar(im, ax=axH, fraction=0.043, pad=0.025)
    cbar.set_label(r"cleaned $\log_{10}(1+\Sigma/\beta)$", rotation=90)

    right = GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0, 1], hspace=0.42, wspace=0.30, height_ratios=[1.0, 1.0, 1.15], width_ratios=[1.15, 0.95])

    axb = fig.add_subplot(right[0, 0])
    ref = front_curve_split(beta_fronts[float(beta_heatmap)])
    axb.plot(ref["tau"], ref["mcrit"], color="black", lw=1.4, label=r"$m_c(\tau)$", zorder=1)
    for beta in betas:
        d = front_curve_split(beta_fronts[beta])
        _plot_front_with_confidence(axb, d, color=beta_colors.get(beta, None), label=rf"$\beta={beta:g}$", lw_solid=2.0)
    axb.set_xlim(0.0, float(np.nanmax(ref["tau"])))
    axb.set_ylim(-4.9, -0.15)
    axb.set_ylabel(r"critical mass")
    axb.set_title(_panel_title("(b)", r"DPT mass vs time"), loc="left", pad=9)
    axb.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.9, columnspacing=0.8)

    axe = fig.add_subplot(right[0, 1])
    ndf = n12_df.sort_values("N").copy()
    for col in ["width_mean", "abs_x0_mean", "chi_mean"]:
        ndf[f"{col}_norm"] = ndf[col] if (ndf[col] <= 0).any() else ndf[col] / float(ndf[col].iloc[0])
    axe.plot(ndf["N"], ndf["width_mean_norm"], marker="o", ms=4.6, lw=2.0, color="#2b8cbe", label=r"$\langle \Delta m\rangle$")
    axe.plot(ndf["N"], ndf["abs_x0_mean_norm"], marker="s", ms=4.4, lw=2.0, color="#d95f0e", label=r"$\langle |m_{\rm DPT}-m_c|\rangle$")
    axe.plot(ndf["N"], ndf["chi_mean_norm"], marker="^", ms=4.8, lw=2.0, color="#31a354", label=r"$\langle \chi_{\rm fit}\rangle$")
    axe.axhline(1.0, color="0.85", lw=1.0, zorder=0)
    axe.set_xlabel(r"system size $N$")
    axe.set_ylabel("normalized metric")
    axe.set_title(_panel_title("(e)", r"Finite-size sharpening"), loc="left", pad=9)
    axe.set_xlim(ndf["N"].min() - 0.4, ndf["N"].max() + 0.4)
    axe.legend(loc="best", frameon=False, fontsize=7.7)

    axc = fig.add_subplot(right[1, 0])
    max_dt = 0.0
    for beta in betas:
        d = front_curve_split(beta_fronts[beta])
        finite = d["finite"]
        if finite.any():
            tau0 = d["tau"][finite][0]
            dt = d["tau"][finite] - tau0
            w = d["width"][finite]
            max_dt = max(max_dt, float(dt.max()))
            axc.plot(dt, w, lw=2.05, color=beta_colors.get(beta, None), label=rf"$\beta={beta:g}$")
    axc.axvline(0.0, color="0.7", lw=1.0, ls=":")
    axc.set_xlim(0.0, max_dt * 1.02 if max_dt > 0 else 1.0)
    axc.set_ylim(bottom=0.0)
    axc.set_xlabel(r"post-onset time $\Delta\tau=\tau-\tau_{\rm onset}$")
    axc.set_ylabel(r"front width $\Delta m_{25\to75}$")
    axc.set_title(_panel_title("(c)", r"Width after onset"), loc="left", pad=9)

    axf = fig.add_subplot(right[1, 1])
    csv = completion_bundle["csv"].copy()
    prof = completion_bundle.get("profile_csv", pd.DataFrame()).copy()
    merged = csv.copy()
    if not prof.empty and {"category", "label", "profile_mae_to_ref"}.issubset(prof.columns):
        prof = prof[prof["category"].isin(["LOCC_meas", "LOCC_tomo"])].copy()
        prof["method"] = prof["category"].map({"LOCC_meas": "meas", "LOCC_tomo": "tomo"})
        prof["ell"] = prof["label"].astype(str).str.extract(r"(\d+)").astype(int)
        merged = csv.merge(prof[["method", "ell", "profile_mae_to_ref"]], on=["method", "ell"], how="left")
    method_colors = {"meas": "#6a51a3", "tomo": "#1b9e77"}
    method_labels = {"meas": "measurement", "tomo": "tomography"}
    for method in ["meas", "tomo"]:
        sub = merged[merged["method"] == method].sort_values("ell")
        axf.plot(sub["ell"], sub["center_err_to_full"], marker="o", ms=4.8, lw=2.0, color=method_colors[method], label=method_labels[method] + r" center err.")
        if "profile_mae_to_ref" in sub.columns and np.isfinite(sub["profile_mae_to_ref"]).any():
            axf.plot(sub["ell"], sub["profile_mae_to_ref"], marker="s", ms=4.2, lw=1.8, ls="--", color=method_colors[method], alpha=0.9, label=method_labels[method] + r" profile MAE")
        else:
            axf.plot(sub["ell"], sub["width_err_to_full"], marker="s", ms=4.2, lw=1.8, ls="--", color=method_colors[method], alpha=0.9, label=method_labels[method] + r" width err.")
    axf.set_xlabel(r"LOCC block size $\ell$")
    axf.set_ylabel("error to full front")
    axf.set_title(_panel_title("(f)", r"LOCC improves with $\ell$"), loc="left", pad=9)
    axf.set_xlim(1.85, 4.15)
    axf.legend(loc="best", frameon=False, fontsize=7.4)

    axd = fig.add_subplot(right[2, :])
    full = bundle_curve_split(completion_bundle["full"])
    axd.plot(full["tau"], full["mcrit"], color="black", lw=1.4, label=r"$m_c(\tau)$")
    _plot_front_with_confidence(axd, full, color="#d73027", label="full", lw_solid=2.25)
    meas = bundle_curve_split(completion_bundle["meas"][ell_focus])
    tomo = bundle_curve_split(completion_bundle["tomo"][ell_focus])
    _plot_front_with_confidence(axd, meas, color="#6a51a3", label=rf"measurement LOCC ($\ell={ell_focus}$)", lw_solid=2.0)
    _plot_front_with_confidence(axd, tomo, color="#1b9e77", label=rf"tomography LOCC ($\ell={ell_focus}$)", lw_solid=2.0)
    axd.set_xlim(0.0, float(np.nanmax(full["tau"])))
    axd.set_ylim(-4.9, -0.15)
    axd.set_xlabel(r"$\tau=t/L$")
    axd.set_ylabel(r"critical mass")
    axd.set_title(_panel_title("(d)", r"LOCC tracks the front"), loc="left", pad=9)
    axd.legend(loc="upper right", frameon=False, ncol=2, fontsize=8.0)
    csv_ell = csv[csv["ell"] == ell_focus]
    meas_row = csv_ell[csv_ell["method"] == "meas"]
    tomo_row = csv_ell[csv_ell["method"] == "tomo"]
    msg = []
    if len(meas_row):
        msg.append(rf"meas: $\langle |m_{{\rm LOCC}}-m_{{\rm full}}|\rangle_\tau \approx {float(meas_row['center_err_to_full'].iloc[0]):.3f}$")
    if len(tomo_row):
        msg.append(rf"tomo: $\langle |m_{{\rm LOCC}}-m_{{\rm full}}|\rangle_\tau \approx {float(tomo_row['center_err_to_full'].iloc[0]):.3f}$")
    if msg:
        axd.text(0.03, 0.05, "\n".join(msg), transform=axd.transAxes, fontsize=8.0, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.8", alpha=0.92))

    for ax in [axH, axb, axc, axd, axe, axf]:
        ax.tick_params(length=5, width=1.05)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def make_figure(data_dir: str | Path, outdir: str | Path, beta_heatmap: float = 10.0, betas: Iterable[float] = (0.5, 1.0, 2.0, 5.0, 10.0), ell_focus: int = 4):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    heatmap, beta_fronts, completion_bundle, n12_df = load_summary_data(data_dir, betas=betas, beta_heatmap=beta_heatmap, ell_values=(2, 3, 4))
    fig_path = outdir / "figure3.png"
    make_summary_figure(heatmap, beta_fronts, completion_bundle, n12_df, fig_path, beta_heatmap=beta_heatmap, betas=betas, ell_focus=ell_focus)

    derived_dir = outdir / "figure3_data"
    derived_dir.mkdir(parents=True, exist_ok=True)
    beta_df, locc_df = build_curve_tables(beta_fronts, completion_bundle, betas=betas, ell_values=(2, 3, 4))
    beta_csv = derived_dir / "temperature_fronts.csv"
    locc_csv = derived_dir / "locc_fronts.csv"
    beta_df.to_csv(beta_csv, index=False)
    locc_df.to_csv(locc_csv, index=False)
    n12_df.to_csv(derived_dir / "finite_size_summary.csv", index=False)
    completion_bundle["csv"].to_csv(derived_dir / "locc_summary.csv", index=False)
    np.savez(
        derived_dir / "irreversibility_front_data.npz",
        heat_tau_dense=np.asarray(heatmap["tau_dense"]),
        heat_m_dense=np.asarray(heatmap["m_dense"]),
        heat_Z_dense=np.asarray(heatmap["Z_dense"]),
        heat_mcrit_dense=np.asarray(heatmap["mcrit_dense"]),
        betas=np.asarray(list(betas), dtype=float),
        n_values=np.asarray(n12_df["N"], dtype=float),
        n_width=np.asarray(n12_df["width_mean"], dtype=float),
        n_abs_x0=np.asarray(n12_df["abs_x0_mean"], dtype=float),
        n_chi=np.asarray(n12_df["chi_mean"], dtype=float),
    )
    return {
        "figure_png": fig_path,
        "figure_pdf": fig_path.with_suffix(".pdf"),
        "temperature_csv": beta_csv,
        "locc_csv": locc_csv,
        "data_npz": derived_dir / "irreversibility_front_data.npz",
    }


__all__ = ["make_figure", "load_summary_data", "make_summary_figure"]
