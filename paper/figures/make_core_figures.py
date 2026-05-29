#!/usr/bin/env python
"""
Regenerate the four core result figures (C1-C4) at publication quality.
Reads source CSVs from ../../paper_materials/tables/ and writes PNGs into
the current directory (paper/figures/).

Figures:
  fig_c1_universal.png  -- method-level split-regret vs AUROC (C1)
  fig_c2_sldlaw.png     -- within-dataset SLD trend / Simpson's paradox (C2)
  fig_c3_coverage.png   -- coverage vs AUROC trade-off (C3)
  fig_c4_decomp.png     -- structure vs leaf-estimate failure modes (C4)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# ----------------------------------------------------------------------
# Global publication style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.3,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SRC = os.path.join("..", "..", "paper_materials", "tables")

# Colour-blind-safe palette (Okabe-Ito)
CB = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#7F7F7F",
}

FAMILY = {
    "gain_path_herding": "Gain / landscape",
    "gain_path_refined": "Gain / landscape",
    "histdistill_density": "Gain / landscape",
    "histdistill_refined": "Gain / landscape",
    "gain_herding": "Gain / landscape",
    "histdistill_greedy": "Gain / landscape",
    "gradmatch_coreset": "Gradient / score",
    "gradient_sampling": "Gradient / score",
    "el2n_coreset": "Gradient / score",
    "grand_coreset": "Gradient / score",
    "craig_coreset": "Gradient / score",
    "mvs_coreset": "GBDT-native",
    "goss_coreset": "GBDT-native",
    "distribution_matching": "Distribution-matching",
    "random": "Geometry / classical",
    "herding": "Geometry / classical",
    "k_center": "Geometry / classical",
}
FAM_COLOR = {
    "Gain / landscape": CB["blue"],
    "Gradient / score": CB["orange"],
    "GBDT-native": CB["green"],
    "Distribution-matching": CB["purple"],
    "Geometry / classical": CB["grey"],
}
FAM_MARKER = {
    "Gain / landscape": "o",
    "Gradient / score": "s",
    "GBDT-native": "^",
    "Distribution-matching": "D",
    "Geometry / classical": "v",
}


def savefig(fig, name):
    out = os.path.abspath(name)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# C1: method-level split-regret vs AUROC
# ----------------------------------------------------------------------
def fig_c1():
    df = pd.read_csv(os.path.join(SRC, "phase2_e19_universal_sld_map.csv"))
    x = df["split_regret_norm"].values
    y = df["auroc"].values
    rho, _ = stats.spearmanr(x, y)
    slope, intercept, r, _, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(3.45, 2.75))

    for fam in ["Gain / landscape", "Gradient / score", "GBDT-native",
                "Distribution-matching", "Geometry / classical"]:
        sub = df[df["method"].map(FAMILY) == fam]
        ax.scatter(sub["split_regret_norm"], sub["auroc"],
                   s=34, c=FAM_COLOR[fam], marker=FAM_MARKER[fam],
                   edgecolor="white", linewidth=0.5, label=fam, zorder=3)

    # trend line
    xs = np.linspace(x.min() - 0.005, x.max() + 0.005, 50)
    ax.plot(xs, intercept + slope * xs, color="0.25", lw=1.1, ls="--",
            zorder=2)

    # annotate the two interpretable extremes
    def label(method, text, dx, dy, ha="left"):
        row = df[df["method"] == method].iloc[0]
        ax.annotate(text, (row["split_regret_norm"], row["auroc"]),
                    xytext=(row["split_regret_norm"] + dx, row["auroc"] + dy),
                    fontsize=7, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

    label("histdistill_greedy", "HistDistill-Greedy\n(coverage-optimal,\nhighest split-regret)",
          -0.012, 0.028, ha="right")
    label("gradient_sampling", "Gradient sampling\n(leaf-estimate failure)",
          0.006, 0.045, ha="left")
    label("histdistill_refined", "Refined family\n(most faithful)",
          0.004, -0.075, ha="left")

    ax.set_xlabel(r"Split-regret  (no candidate retraining)")
    ax.set_ylabel("Downstream AUROC")
    ax.text(0.97, 0.93,
            r"Spearman $\rho=-0.87$" + "\n" + r"OLS slope $=-0.91$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    ax.legend(loc="lower left", handletextpad=0.2, borderpad=0.2,
              labelspacing=0.25)
    ax.set_ylim(0.33, 0.79)
    savefig(fig, "fig_c1_universal.png")


# ----------------------------------------------------------------------
# C2: within-dataset SLD law -- Simpson's paradox
# ----------------------------------------------------------------------
def fig_c2():
    cells = pd.read_csv(os.path.join(SRC, "icdm_sld_law_cells.csv"))
    cells = cells.dropna(subset=["sld_inf_norm", "auroc"])

    # choose six representative datasets spanning the SLD range, each with a
    # clear within-dataset trend (>= 8 cells, > 2 distinct SLD values)
    stats_by_ds = []
    for d, g in cells.groupby("dataset"):
        if len(g) >= 8 and g["sld_inf_norm"].nunique() > 2:
            stats_by_ds.append((d, np.median(g["sld_inf_norm"])))
    stats_by_ds.sort(key=lambda t: t[1])
    if len(stats_by_ds) >= 6:
        idx = np.linspace(0, len(stats_by_ds) - 1, 6).round().astype(int)
        sel = [stats_by_ds[i][0] for i in idx]
    else:
        sel = [d for d, _ in stats_by_ds]

    fig, ax = plt.subplots(figsize=(3.45, 2.75))

    # faint scatter of every cell (full-data context)
    ax.scatter(cells["sld_inf_norm"], cells["auroc"], s=5, c="0.82",
               linewidth=0, zorder=1, rasterized=True)

    line_colors = [CB["blue"], CB["green"], CB["purple"], CB["sky"],
                   CB["orange"], CB["yellow"]]
    for d, col in zip(sel, line_colors):
        g = cells[cells["dataset"] == d]
        s, b, _, _, _ = stats.linregress(np.log10(g["sld_inf_norm"]), g["auroc"])
        xs = np.array([g["sld_inf_norm"].min(), g["sld_inf_norm"].max()])
        ax.scatter(g["sld_inf_norm"], g["auroc"], s=10, c=col, linewidth=0,
                   alpha=0.65, zorder=3)
        ax.plot(xs, b + s * np.log10(xs), color=col, lw=1.4, zorder=4)

    # within-dataset reference label (only one legend entry needed)
    ax.plot([], [], color=CB["blue"], lw=1.4,
            label="within-dataset trends")

    # pooled regression line (the misleading aggregate)
    s, b, _, _, _ = stats.linregress(np.log10(cells["sld_inf_norm"]), cells["auroc"])
    xs = np.logspace(np.log10(cells["sld_inf_norm"].min()),
                     np.log10(cells["sld_inf_norm"].max()), 50)
    ax.plot(xs, b + s * np.log10(xs), color=CB["red"], lw=2.4, zorder=5,
            label="pooled trend (positive)")

    ax.set_xlabel(r"$\mathrm{SLD}_\infty$  (lower = more faithful)")
    ax.set_ylabel("Downstream AUROC")
    ax.set_xscale("log")
    ax.set_ylim(0.45, 0.95)
    ax.text(0.03, 0.04,
            "within dataset: " + r"$\rho \approx -0.56$" +
            "\npooled: " + r"$\rho = +0.12$" + "  (Simpson's paradox)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    ax.legend(loc="upper center", handletextpad=0.5, ncol=1,
              bbox_to_anchor=(0.5, 1.02))
    savefig(fig, "fig_c2_sldlaw.png")


# ----------------------------------------------------------------------
# C3: coverage vs AUROC
# ----------------------------------------------------------------------
def fig_c3():
    cov = pd.read_csv(os.path.join(SRC, "icdm_selector_coverage_summary.csv"))
    perf = pd.read_csv(os.path.join(SRC, "phase2_e19_universal_sld_map.csv"))
    aur = dict(zip(perf["method"], perf["auroc"]))

    # mean coverage ratio across budgets per method
    cov_mean = cov.groupby(["method", "method_label"])["coverage_ratio"].mean().reset_index()
    cov_mean["auroc"] = cov_mean["method"].map(aur)
    cov_mean = cov_mean.dropna(subset=["auroc"])

    # merge the near-identical Refined/Density points into one label
    label_override = {
        "histdistill_refined": "HistDistill-Refined / Density",
        "histdistill_density": None,  # suppressed (coincident with Refined)
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for _, row in cov_mean.iterrows():
        is_greedy = row["method"] == "histdistill_greedy"
        ax.scatter(row["coverage_ratio"], row["auroc"],
                   s=70 if is_greedy else 46,
                   c=CB["red"] if is_greedy else CB["blue"],
                   marker="*" if is_greedy else "o",
                   edgecolor="white", linewidth=0.5, zorder=3)
        txt = label_override.get(row["method"], row["method_label"])
        if txt is None:
            continue
        ha = "right" if row["coverage_ratio"] > 0.2 else "left"
        dx = -0.014 if ha == "right" else 0.014
        ax.annotate(txt, (row["coverage_ratio"], row["auroc"]),
                    xytext=(row["coverage_ratio"] + dx, row["auroc"]),
                    fontsize=7, ha=ha, va="center")

    ax.set_xlabel("Split-coverage ratio  (higher = more thresholds)")
    ax.set_ylabel("Downstream AUROC")
    ax.annotate("high coverage,\nlow AUROC",
                xy=(cov_mean[cov_mean.method == "histdistill_greedy"]["coverage_ratio"].iloc[0],
                    aur["histdistill_greedy"]),
                xytext=(0.30, 0.66), fontsize=7, color=CB["red"], ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=CB["red"]))
    ax.set_xlim(0, 0.52)
    savefig(fig, "fig_c3_coverage.png")


# ----------------------------------------------------------------------
# C4: structure vs leaf-estimate failure modes
# ----------------------------------------------------------------------
def fig_c4():
    dec = pd.read_csv(os.path.join(SRC, "phase2_e16_error_decomposition_summary.csv"))

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for _, row in dec.iterrows():
        m = row["method"]
        if m == "histdistill_greedy":
            c, mk, s = CB["red"], "*", 90
        elif m == "gradient_sampling":
            c, mk, s = CB["orange"], "s", 55
        elif m in ("histdistill_refined", "histdistill_density",
                   "gain_path_refined", "gain_path_herding", "gain_herding"):
            c, mk, s = CB["blue"], "o", 42
        else:
            c, mk, s = CB["grey"], "o", 36
        ax.scatter(row["structure_error"], row["leaf_estimate_recovery"],
                   s=s, c=c, marker=mk, edgecolor="white", linewidth=0.5,
                   zorder=3)

    def lab(m, text, dx, dy, ha="left", color="0.2"):
        row = dec[dec["method"] == m].iloc[0]
        ax.annotate(text, (row["structure_error"], row["leaf_estimate_recovery"]),
                    xytext=(row["structure_error"] + dx,
                            row["leaf_estimate_recovery"] + dy),
                    fontsize=7, ha=ha, va="center", color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))

    lab("histdistill_greedy", "HistDistill-Greedy\n(structure failure)",
        -0.004, 0.10, ha="right", color=CB["red"])
    lab("gradient_sampling", "Gradient sampling\n(leaf failure)",
        0.004, -0.04, ha="left", color=CB["orange"])
    lab("histdistill_refined", "faithful gain methods",
        0.004, 0.03, ha="left", color=CB["blue"])

    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("Structure error")
    ax.set_ylabel("Leaf-estimate gap  (higher = worse)")
    savefig(fig, "fig_c4_decomp.png")


if __name__ == "__main__":
    fig_c1()
    fig_c2()
    fig_c3()
    fig_c4()
    print("done")
