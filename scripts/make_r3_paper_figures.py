"""Generate R3 paper figures from Stage B/C output.

Codex 2026-06-01 paper skeleton priority:
  1. Fixed-K K-sweep curve/heatmap     -> F_r3_K_sweep.pdf
  2. Stage C adaptive deployment       -> F_r3_adapt_deploy.pdf
  3. K=0 diagnostic + controller-health-> F_r3_k0_diagnostic.pdf
  4. Field-wise realised K heatmap     -> F_r3_field_k.pdf
  (5. System diagram is reused as-is.)

Outputs PDF + PNG to figures/ alongside the existing main-text figures.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Stage B + Stage C eval IDs
B6 = REPO / "runs/closed_loop/2026-05-27T11-09-25Z"
C1 = REPO / "runs/closed_loop/2026-06-01T00-43-36Z"  # default reliability
C2 = REPO / "runs/closed_loop/2026-06-01T02-00-24Z"  # single-cell SPSA
C3 = REPO / "runs/closed_loop/2026-06-01T02-22-56Z"  # global SPSA
PAIRED_CI = REPO / "runs/diag/r3_stage_c_paired_ci.csv"

KS = ["K=0.00", "K=0.25", "K=0.50", "K=0.75", "K=1.00"]
K_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
CELLS = [("none", 0), ("none", 18), ("high", 0), ("high", 18),
         ("xhigh", 0), ("xhigh", 18)]
NOISES = ["none", "high", "xhigh"]
DELAYS = [0, 18]
FIELDS = ["qpos", "qvel", "act", "tip_pos", "reach_err"]

# A single consistent palette across figures.
NOISE_COLOR = {"none": "#1f77b4", "high": "#ff7f0e", "xhigh": "#d62728"}
NOISE_LABEL = {"none": "no noise", "high": "high", "xhigh": "xhigh"}

# rcParams for paper-quality output.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  saved: {pdf}  +  {png}")


def _load_b6_groups():
    """Return {(noise, delay, estimator): group_dict}."""
    s = json.load(open(B6 / "summary.json"))
    by = {}
    for g in s["groups"]:
        by[(g["noise_condition"], g["delay_steps"], g["estimator"])] = g
    return by


def _load_c_groups(eval_dir):
    s = json.load(open(eval_dir / "summary.json"))
    by = {}
    for g in s["groups"]:
        by[(g["noise_condition"], g["delay_steps"], g["estimator"])] = g
    return by


def _load_c4_perlcell():
    """C4 batch 出力は 6 eval dirs に分散。estimator name == c4_feature_conditioned_<cell>"""
    out = {}
    for cl in REPO.glob("runs/closed_loop/2026-06-01T*/summary.json"):
        s = json.load(open(cl))
        for g in s["groups"]:
            est = g["estimator"]
            if est.startswith("c4_feature_conditioned_"):
                out[(g["noise_condition"], g["delay_steps"])] = g
    return out


def _load_per_cell_deploy():
    out = {}
    for cl in REPO.glob("runs/closed_loop/2026-06-01T*/summary.json"):
        s = json.load(open(cl))
        for g in s["groups"]:
            est = g["estimator"]
            if est.startswith("per_cell_beta_"):
                out[(g["noise_condition"], g["delay_steps"])] = g
    return out


def _load_paired_ci():
    rows = []
    with open(PAIRED_CI) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ---------------------------------------------------------------------
# Figure 1: K-sweep delay-dependent structure
# ---------------------------------------------------------------------
def make_F_r3_K_sweep():
    b6 = _load_b6_groups()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, d in zip(axes, DELAYS):
        for noise in NOISES:
            means = [b6[(noise, d, k)]["min_tip_error_mean"] for k in KS]
            best_idx = int(np.argmin(means))
            best_mean = means[best_idx]
            # Plot within-cell regret rather than absolute min-tip. This
            # prevents readers from interpreting the between-delay offset as
            # "delay improves accuracy"; the figure is about which K is best
            # inside each noise/delay cell.
            excess_cm = [(m - best_mean) * 100.0 for m in means]
            color = NOISE_COLOR[noise]
            ax.plot(
                K_VALUES, excess_cm, marker="o", label=NOISE_LABEL[noise],
                color=color, lw=1.8, ms=4,
            )
            ax.scatter([K_VALUES[best_idx]], [0.0],
                       marker="*", s=120, color=color, edgecolor="black",
                       zorder=10)
        ax.set_xlabel("Correction gain $K$")
        ax.set_title(f"delay = {d}")
        ax.set_xticks(K_VALUES)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("excess min-tip vs cell best (cm)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=3, frameon=False, fontsize=7,
    )
    fig.suptitle(
        "Swept fixed-$K$ oracle: within-cell penalty relative to best $K$",
        y=1.13, fontsize=10,
    )
    _save(fig, "F_r3_K_sweep")


# ---------------------------------------------------------------------
# Figure 2: K=0 diagnostic + controller-health
# ---------------------------------------------------------------------
def make_F_r3_k0_diagnostic():
    b6 = _load_b6_groups()
    ci_rows = _load_paired_ci()

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(6.2, 5.4),
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.65},
    )

    # Panel A: K0 vs bestK paired mean diff with 95% CI per cell.
    k0_rows = [r for r in ci_rows if r["contrast"] == "K0_vs_bestK"]
    labels = []
    means_, los, his = [], [], []
    for r in k0_rows:
        labels.append(f"{NOISE_LABEL[r['cell_noise']]}, d={r['cell_delay']}")
        means_.append(float(r["mean_diff"]) * 100)  # cm
        los.append(float(r["ci_lo"]) * 100)
        his.append(float(r["ci_hi"]) * 100)
    y = np.arange(len(labels))
    errs = [[m - lo for m, lo in zip(means_, los)],
            [hi - m for m, hi in zip(means_, his)]]
    axA.errorbar(means_, y, xerr=errs, fmt="o", color="#d62728",
                 ecolor="#d62728", capsize=3)
    axA.axvline(0, color="black", lw=0.7)
    axA.set_yticks(y)
    axA.set_yticklabels(labels)
    axA.set_xlabel("$\\min$-tip diff (cm), $K{=}0$ minus best $K$")
    axA.set_title("A. $K{=}0$ vs swept best $K$")
    axA.grid(axis="x", alpha=0.3)

    # Panel B: NNLS residual mean heatmap (6 cells × 5 K).
    nnls = np.array([
        [b6[(noise, d, k)].get("nnls_residual_mean_mean", np.nan)
         for k in KS]
        for noise, d in CELLS
    ])
    im = axB.imshow(nnls, aspect="auto", cmap="magma")
    axB.set_xticks(range(len(KS)))
    axB.set_xticklabels([k.replace("K=", "") for k in KS])
    axB.set_yticks(range(len(CELLS)))
    axB.set_yticklabels([f"{NOISE_LABEL[n]}, d={d}" for n, d in CELLS], fontsize=8)
    axB.set_xlabel("$K$")
    axB.set_title("B. NNLS residual mean (controller health)")
    for i in range(len(CELLS)):
        for j in range(len(KS)):
            v = nnls[i, j]
            color = "white" if v > 50 else "black"
            bbox_fc = (0, 0, 0, 0.30) if color == "white" else (1, 1, 1, 0.60)
            axB.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=8, color=color,
                     bbox={"boxstyle": "round,pad=0.12", "facecolor": bbox_fc,
                           "edgecolor": "none"})
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    _save(fig, "F_r3_k0_diagnostic")


# ---------------------------------------------------------------------
# Figure 3: Stage C adaptive deployment comparison
# ---------------------------------------------------------------------
def make_F_r3_adapt_deploy():
    b6 = _load_b6_groups()
    c1 = _load_c_groups(C1)
    c3 = _load_c_groups(C3)
    c4 = _load_c4_perlcell()
    pc = _load_per_cell_deploy()

    # Per-cell best K from B.6
    bestK_per_cell = {}
    for (noise, d) in CELLS:
        means = [b6[(noise, d, k)]["min_tip_error_mean"] for k in KS]
        idx = int(np.argmin(means))
        bestK_per_cell[(noise, d)] = (KS[idx], means[idx])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    width = 0.13
    estimator_order = [
        ("default", c1, "default_reliability", "#7f7f7f"),
        ("global β", c3, "c3_global_beta", "#2ca02c"),
        ("feat-cond β", c4, None, "#1f77b4"),
        ("per-cell β", pc, None, "#ff7f0e"),
        ("best K", None, None, "#d62728"),
        ("K=0", b6, "K=0.00", "#bcbd22"),
    ]

    for ax, d in zip(axes, DELAYS):
        cells_d = [(n, d) for n in NOISES]
        x = np.arange(len(cells_d))
        for i, (label, src, est_name, color) in enumerate(estimator_order):
            ys = []
            for (n, dd) in cells_d:
                if label == "best K":
                    ys.append(bestK_per_cell[(n, dd)][1])
                elif est_name is None:
                    g = src.get((n, dd))
                    ys.append(g["min_tip_error_mean"] if g else np.nan)
                else:
                    g = src.get((n, dd, est_name))
                    ys.append(g["min_tip_error_mean"] if g else np.nan)
            ax.bar(x + (i - 2.5) * width, ys, width=width, label=label,
                   color=color, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([NOISE_LABEL[n] for (n, _) in cells_d])
        ax.set_xlabel("noise")
        ax.set_title(f"delay = {d}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("min-tip error mean (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               frameon=False, fontsize=7, ncol=3)
    fig.suptitle(
        "Adaptive deployment vs swept oracle and $K{=}0$",
        y=1.15, fontsize=10,
    )
    _save(fig, "F_r3_adapt_deploy")


# ---------------------------------------------------------------------
# Figure 4: Field-wise realised K
# ---------------------------------------------------------------------
def make_F_r3_field_k():
    c1 = _load_c_groups(C1)
    c3 = _load_c_groups(C3)
    c4 = _load_c4_perlcell()
    pc = _load_per_cell_deploy()

    sources = [
        ("default", c1, "default_reliability"),
        ("global β", c3, "c3_global_beta"),
        ("feat-cond β", c4, None),
        ("per-cell β", pc, None),
    ]

    rows = []
    row_labels = []
    for est_label, src, est_name in sources:
        for field in FIELDS:
            row_labels.append(f"{est_label} · {field}")
            row = []
            for (noise, d) in CELLS:
                if est_name is None:
                    g = src.get((noise, d))
                else:
                    g = src.get((noise, d, est_name))
                v = g.get(f"k_{field}_second_half_mean_mean") if g else None
                row.append(v if v is not None else np.nan)
            rows.append(row)
    mat = np.array(rows)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([f"{NOISE_LABEL[n]}, d={d}" for n, d in CELLS], rotation=45,
                       ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if v < 0.5 else "black")
    # Horizontal lines between estimators (every 5 rows).
    for k in range(1, len(sources)):
        ax.axhline(k * 5 - 0.5, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="realised $K_f$")
    ax.set_title("Field-wise realised correction gain (second-half mean)")
    _save(fig, "F_r3_field_k")


if __name__ == "__main__":
    print("Generating R3 paper figures into", OUT)
    make_F_r3_K_sweep()
    make_F_r3_k0_diagnostic()
    make_F_r3_adapt_deploy()
    make_F_r3_field_k()
    print("done")
