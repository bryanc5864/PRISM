#!/usr/bin/env python
"""Generate all 6 publication figures for the PRISM presentation.

Outputs PNG + PDF at 300 DPI to figures/publication/.

Figures:
  1. UMAP Grid (Hero) — 6 systems × 4 methods
  2. Grouped Bar Chart — RF_AUROC, 15 systems × 7 methods
  3. Multi-Metric Heatmap + Radar
  4. Pre-training Benefit — Slope/dot plot
  5. PCP Training Curves
  6. Architecture Diagram
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from math import pi

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = "figures/publication"
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,  # TrueType for editable text in PDF
    "ps.fonttype": 42,
})

# Method colors (consistent across all figures)
METHOD_COLORS = {
    "PRISM": "#1E88E5",
    "PRISM_pretrained": "#0D47A1",
    "PCA": "#FF9800",
    "Harmony": "#AB47BC",
    "DiffMap": "#78909C",
    "UMAP": "#8D6E63",
    "scGPT": "#26A69A",
    "Geneformer": "#EF5350",
}

# System display names
SYSTEM_DISPLAY = {
    "skin": "Skin",
    "pancreas": "Pancreas",
    "cortex": "Cortex",
    "hsc": "HSC",
    "cardiac": "Cardiac",
    "intestine": "Intestine",
    "lung": "Lung",
    "neural_crest": "Neural Crest",
    "thcell": "T Helper",
    "oligo": "Oligo",
    "tirosh_melanoma": "Melanoma",
    "neftel_gbm": "GBM",
    "paul": "Paul HSC",
    "nestorowa": "Nestorowa",
    "sadefeldman": "Sade-Feldman",
}

# ---------------------------------------------------------------------------
# Figure 1: UMAP Grid (Hero) — 6 systems × 4 methods
# ---------------------------------------------------------------------------
UMAP_SYSTEMS = ["skin", "pancreas", "hsc", "cardiac", "neftel_gbm", "thcell"]
UMAP_METHODS = ["PRISM", "PCA", "scGPT", "Geneformer"]

SYSTEM_DATA = {
    "skin": {
        "adata": "data/processed/adata_processed.h5ad",
        "scgpt": "data/processed/scgpt_embeddings.npy",
        "geneformer": "data/processed/geneformer_embeddings.npy",
    },
}
# Auto-populate for subdirectory systems
for sys_name in UMAP_SYSTEMS:
    if sys_name not in SYSTEM_DATA:
        SYSTEM_DATA[sys_name] = {
            "adata": f"data/processed/{sys_name}/adata_processed.h5ad",
            "scgpt": f"data/processed/{sys_name}/scgpt_embeddings.npy",
            "geneformer": f"data/processed/{sys_name}/geneformer_embeddings.npy",
        }

FATE_COLORS = {
    "skin": {
        "eccrine": "#2196F3", "hair": "#E91E63",
        "non_appendage": "#9E9E9E", "undetermined": "#E0E0E0",
    },
    "pancreas": {
        "alpha": "#4CAF50", "beta": "#FF9800", "delta": "#9C27B0",
        "non_endocrine": "#9E9E9E", "undetermined": "#E0E0E0",
    },
    "cortex": {
        "upper_layer": "#F44336", "deep_layer": "#3F51B5",
        "non_neuronal": "#9E9E9E", "undetermined": "#E0E0E0",
    },
    "hsc": {
        "erythroid": "#D32F2F", "myeloid": "#1976D2", "lymphoid": "#388E3C",
        "undetermined": "#E0E0E0",
    },
    "cardiac": {
        "FHF": "#E53935", "SHF": "#1565C0",
        "non_cardiac": "#9E9E9E", "undetermined": "#E0E0E0",
    },
    "neftel_gbm": {
        "MES": "#E91E63", "AC": "#4CAF50", "OPC": "#FF9800", "NPC": "#3F51B5",
        "undetermined": "#E0E0E0",
    },
    "thcell": {
        "Th1": "#D32F2F", "Th17": "#1976D2",
        "non_thcell": "#9E9E9E", "undetermined": "#E0E0E0",
    },
}

FATE_DISPLAY = {
    "skin": {"eccrine": "Eccrine", "hair": "Hair follicle",
             "non_appendage": "Non-appendage", "undetermined": "Undetermined"},
    "pancreas": {"alpha": "Alpha", "beta": "Beta", "delta": "Delta",
                 "non_endocrine": "Non-endocrine", "undetermined": "Undetermined"},
    "cortex": {"upper_layer": "Upper layer", "deep_layer": "Deep layer",
               "non_neuronal": "Non-neuronal", "undetermined": "Undetermined"},
    "hsc": {"erythroid": "Erythroid", "myeloid": "Myeloid", "lymphoid": "Lymphoid",
            "undetermined": "Undetermined"},
    "cardiac": {"FHF": "FHF", "SHF": "SHF",
                "non_cardiac": "Non-cardiac", "undetermined": "Undetermined"},
    "neftel_gbm": {"MES": "Mesenchymal", "AC": "Astrocyte-like",
                   "OPC": "OPC-like", "NPC": "NPC-like", "undetermined": "Undetermined"},
    "thcell": {"Th1": "Th1", "Th17": "Th17",
               "non_thcell": "Non-T helper", "undetermined": "Undetermined"},
}

DRAW_ORDER = {
    "skin": ["undetermined", "non_appendage", "eccrine", "hair"],
    "pancreas": ["undetermined", "non_endocrine", "delta", "alpha", "beta"],
    "cortex": ["undetermined", "non_neuronal", "deep_layer", "upper_layer"],
    "hsc": ["undetermined", "lymphoid", "myeloid", "erythroid"],
    "cardiac": ["undetermined", "non_cardiac", "SHF", "FHF"],
    "neftel_gbm": ["undetermined", "NPC", "OPC", "AC", "MES"],
    "thcell": ["undetermined", "non_thcell", "Th17", "Th1"],
}


def _get_draw_order(system, labels):
    """Get draw order for a system, falling back to auto-detected order."""
    if system in DRAW_ORDER:
        # Filter to only labels that exist in the data
        known = DRAW_ORDER[system]
        unique_labels = set(labels)
        order = [f for f in known if f in unique_labels]
        # Append any labels not in the predefined order
        for lab in sorted(unique_labels):
            if lab not in order:
                order.append(lab)
        return order
    # Fallback: undetermined first, then alphabetical
    unique_labels = sorted(set(labels))
    order = []
    for bg in ["undetermined", "unknown"]:
        if bg in unique_labels:
            order.append(bg)
    for lab in unique_labels:
        if lab not in order:
            order.append(lab)
    return order


def _get_colors(system, labels):
    """Get color dict for a system, falling back to a tab10 palette."""
    if system in FATE_COLORS:
        colors = dict(FATE_COLORS[system])
        # Fill in any missing labels
        tab10 = plt.cm.tab10.colors
        idx = 0
        for lab in sorted(set(labels)):
            if lab not in colors:
                colors[lab] = matplotlib.colors.to_hex(tab10[idx % 10])
                idx += 1
        return colors
    # Auto-generate
    unique = sorted(set(labels))
    tab10 = plt.cm.tab10.colors
    colors = {}
    idx = 0
    for lab in unique:
        if lab in ("undetermined", "unknown"):
            colors[lab] = "#E0E0E0"
        else:
            colors[lab] = matplotlib.colors.to_hex(tab10[idx % 10])
            idx += 1
    return colors


def generate_fig1_umap_grid():
    """Figure 1: Composite grid from existing per-system UMAP images.

    Uses the pre-computed 4×4 grid (skin/pancreas/cortex/hsc) as the top half
    and assembles cardiac/neftel_gbm/thcell per-system strips as bottom rows.
    """
    print("\n" + "=" * 60)
    print("Figure 1: UMAP Grid (composite from existing figures)")
    print("=" * 60)

    # Use the existing 4-system grid as the hero figure directly
    # and also create a 6-row composite with additional systems
    systems = ["skin", "pancreas", "cortex", "hsc", "cardiac", "neftel_gbm", "thcell"]
    per_system_paths = {
        "skin": "figures/umap_comparison.png",  # top-level skin
    }
    for s in systems:
        if s not in per_system_paths:
            per_system_paths[s] = f"figures/{s}/umap_comparison.png"

    # Load all per-system images
    images = {}
    for s in systems:
        path = per_system_paths[s]
        if os.path.exists(path):
            images[s] = plt.imread(path)
            print(f"  Loaded {s}: {images[s].shape}")
        else:
            print(f"  WARNING: {path} not found, skipping {s}")

    available = [s for s in systems if s in images]
    if not available:
        print("  No UMAP images found, skipping Figure 1.")
        return

    n_rows = len(available)
    # Determine uniform width (resize all to same width)
    max_w = max(img.shape[1] for img in images.values())

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, system in enumerate(available):
        ax = axes[i]
        ax.imshow(images[system])
        ax.axis("off")
        ax.set_ylabel(SYSTEM_DISPLAY.get(system, system),
                       fontsize=13, fontweight="bold", labelpad=15, rotation=0,
                       va="center", ha="right")
        # Force ylabel visible even with axis off
        ax.yaxis.set_visible(True)
        ax.yaxis.label.set_visible(True)

    fig.suptitle("UMAP Embeddings: PRISM vs Baselines Across Biological Systems",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig1_umap_grid")

    # Also copy the existing 4×4 grid as an alternative hero
    import shutil
    src = "figures/umap_comparison_grid.png"
    if os.path.exists(src):
        dst = os.path.join(OUT_DIR, "fig1_umap_grid_4x4.png")
        shutil.copy2(src, dst)
        print(f"  Copied existing grid: {dst}")


# ---------------------------------------------------------------------------
# Figure 2: Grouped Bar Chart — RF_AUROC
# ---------------------------------------------------------------------------
def generate_fig2_bar_chart():
    """Figure 2: RF_AUROC grouped bar chart, 15 systems × 7 methods."""
    print("\n" + "=" * 60)
    print("Figure 2: RF_AUROC Grouped Bar Chart")
    print("=" * 60)

    df = pd.read_csv("data/processed/full_evaluation_results.csv")

    # Use 7 main methods (exclude PRISM_pretrained for cleaner comparison)
    methods = ["PRISM", "PCA", "Harmony", "DiffMap", "UMAP", "scGPT", "Geneformer"]
    systems_order = [
        "skin", "pancreas", "cortex", "hsc", "cardiac", "lung", "thcell",
        "neftel_gbm", "paul", "sadefeldman", "tirosh_melanoma",
        "neural_crest", "intestine", "oligo", "nestorowa",
    ]

    # Pivot to get systems × methods
    pivot = df.pivot(index="system", columns="method", values="RF_AUROC")

    fig, ax = plt.subplots(figsize=(14, 7))
    n_systems = len(systems_order)
    n_methods = len(methods)
    bar_width = 0.11
    x = np.arange(n_systems)

    for i, method in enumerate(methods):
        vals = []
        for sys in systems_order:
            if sys in pivot.index and method in pivot.columns:
                v = pivot.loc[sys, method]
                vals.append(v if pd.notna(v) else 0)
            else:
                vals.append(0)

        color = METHOD_COLORS.get(method, "#999999")
        edgecolor = "black" if method == "PRISM" else "none"
        linewidth = 1.5 if method == "PRISM" else 0
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                       label=method, color=color,
                       edgecolor=edgecolor, linewidth=linewidth, zorder=3)

    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels([SYSTEM_DISPLAY.get(s, s) for s in systems_order],
                        rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RF AUROC", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(-0.3, n_systems - 0.3)
    ax.axhline(y=1.0, color="#CCCCCC", linestyle="--", linewidth=0.5, zorder=1)
    ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=True,
              framealpha=0.9, edgecolor="#CCC")
    ax.set_title("RF AUROC Across 15 Biological Systems", fontsize=14,
                 fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "fig2_rf_auroc_bar")


# ---------------------------------------------------------------------------
# Figure 3: Multi-Metric Heatmap + Radar
# ---------------------------------------------------------------------------
def generate_fig3_heatmap_radar():
    """Figure 3: Panel A = Heatmap, Panel B = Radar chart."""
    print("\n" + "=" * 60)
    print("Figure 3: Multi-Metric Heatmap + Radar")
    print("=" * 60)

    df = pd.read_csv("data/processed/full_evaluation_results.csv")
    methods = ["PRISM", "PCA", "Harmony", "scGPT", "Geneformer"]
    metrics = ["RF_AUROC", "ARI", "NMI", "ASW", "kNN_purity@50"]
    metric_display = {
        "RF_AUROC": "RF AUROC", "ARI": "ARI", "NMI": "NMI",
        "ASW": "ASW", "kNN_purity@50": "kNN Purity",
    }

    # Compute mean across all 15 systems for each method × metric
    mean_vals = pd.DataFrame(index=methods, columns=metrics, dtype=float)
    for method in methods:
        sub = df[df["method"] == method]
        for metric in metrics:
            mean_vals.loc[method, metric] = sub[metric].mean()

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # --- Panel A: Heatmap ---
    ax_heat = fig.add_subplot(gs[0])
    data = mean_vals.values.astype(float)
    im = ax_heat.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(metrics)))
    ax_heat.set_xticklabels([metric_display.get(m, m) for m in metrics],
                             fontsize=10, fontweight="bold")
    ax_heat.set_yticks(range(len(methods)))
    ax_heat.set_yticklabels(methods, fontsize=10, fontweight="bold")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax_heat.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score (15 systems)", fontsize=9)
    ax_heat.set_title("A. Mean Metrics by Method", fontsize=12, fontweight="bold", pad=10)

    # --- Panel B: Radar ---
    ax_radar = fig.add_subplot(gs[1], polar=True)
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    # PRISM values
    prism_vals = mean_vals.loc["PRISM"].values.tolist()
    prism_vals += prism_vals[:1]

    # Best-of-baselines (max of non-PRISM methods per metric)
    baselines = mean_vals.loc[mean_vals.index != "PRISM"]
    best_baseline = baselines.max(axis=0).values.tolist()
    best_baseline += best_baseline[:1]

    ax_radar.plot(angles, prism_vals, "o-", linewidth=2, color="#1E88E5", label="PRISM")
    ax_radar.fill(angles, prism_vals, alpha=0.15, color="#1E88E5")
    ax_radar.plot(angles, best_baseline, "s--", linewidth=2, color="#FF9800",
                  label="Best Baseline")
    ax_radar.fill(angles, best_baseline, alpha=0.1, color="#FF9800")

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([metric_display.get(m, m) for m in metrics], fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#666")
    ax_radar.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9, frameon=True)
    ax_radar.set_title("B. PRISM vs Best Baseline", fontsize=12, fontweight="bold", pad=20)

    _save(fig, "fig3_heatmap_radar")


# ---------------------------------------------------------------------------
# Figure 4: Pre-training Benefit — Slope/Dot plot
# ---------------------------------------------------------------------------
def generate_fig4_pretraining():
    """Figure 4: Paired dot/slope plot showing pre-training benefit."""
    print("\n" + "=" * 60)
    print("Figure 4: Pre-training Benefit")
    print("=" * 60)

    df = pd.read_csv("data/processed/full_evaluation_results.csv")
    with open("checkpoints/pretrain_scgpt/pretrained_results.json") as f:
        pretrained = json.load(f)

    # Get PRISM (no pretrain) and PRISM_pretrained RF_AUROC per system
    systems = []
    no_pt = []
    pt = []
    for system in sorted(SYSTEM_DISPLAY.keys()):
        row_nopt = df[(df["system"] == system) & (df["method"] == "PRISM")]
        if row_nopt.empty:
            continue
        nopt_val = row_nopt["RF_AUROC"].values[0]

        pt_val = None
        if system in pretrained and pretrained[system].get("RF_AUROC") is not None:
            pt_val = pretrained[system]["RF_AUROC"]
        else:
            # Check CSV for PRISM_pretrained
            row_pt = df[(df["system"] == system) & (df["method"] == "PRISM_pretrained")]
            if not row_pt.empty:
                pt_val = row_pt["RF_AUROC"].values[0]

        if pt_val is not None:
            systems.append(system)
            no_pt.append(nopt_val)
            pt.append(pt_val)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by gain (descending)
    gains = [p - n for p, n in zip(pt, no_pt)]
    sorted_idx = np.argsort(gains)[::-1]
    systems = [systems[i] for i in sorted_idx]
    no_pt = [no_pt[i] for i in sorted_idx]
    pt = [pt[i] for i in sorted_idx]
    gains = [gains[i] for i in sorted_idx]

    y = np.arange(len(systems))

    for i in range(len(systems)):
        color = "#4CAF50" if gains[i] > 0 else "#F44336"
        ax.plot([no_pt[i], pt[i]], [y[i], y[i]], color=color, linewidth=2, zorder=2)

    ax.scatter(no_pt, y, c="#FF9800", s=80, zorder=3, label="No pre-training", edgecolors="white", linewidths=0.5)
    ax.scatter(pt, y, c="#1E88E5", s=80, zorder=3, label="Pre-trained (PCP)", edgecolors="white", linewidths=0.5)

    # Annotate gains
    for i in range(len(systems)):
        xpos = max(no_pt[i], pt[i]) + 0.005
        sign = "+" if gains[i] >= 0 else ""
        ax.text(xpos, y[i], f"{sign}{gains[i]:.3f}", va="center", fontsize=8,
                color="#4CAF50" if gains[i] > 0 else "#F44336", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([SYSTEM_DISPLAY.get(s, s) for s in systems], fontsize=10)
    ax.set_xlabel("RF AUROC", fontsize=12, fontweight="bold")
    ax.set_xlim(0.5, 1.05)
    ax.axvline(x=1.0, color="#CCCCCC", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_title("Pre-training Benefit: PCP Initialization", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Mean gain annotation
    mean_gain = np.mean(gains)
    ax.text(0.52, len(systems) - 0.5, f"Mean gain: +{mean_gain:.3f}",
            fontsize=11, fontweight="bold", color="#1E88E5",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1E88E5", alpha=0.9))

    fig.tight_layout()
    _save(fig, "fig4_pretraining_benefit")


# ---------------------------------------------------------------------------
# Figure 5: PCP Training Curves
# ---------------------------------------------------------------------------
def generate_fig5_training_curves():
    """Figure 5: Training loss and MLM accuracy curves."""
    print("\n" + "=" * 60)
    print("Figure 5: PCP Training Curves")
    print("=" * 60)

    with open("checkpoints/pretrain_scgpt/training_history.json") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    total_loss = [h["loss"] for h in history]
    contrastive_loss = [h["contrastive_loss"] for h in history]
    mlm_loss = [h["mlm_loss"] for h in history]
    mlm_accuracy = [h["mlm_accuracy"] for h in history]
    lr = [h["lr"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Panel A: Losses ---
    ax1.plot(epochs, total_loss, "o-", color="#1E88E5", linewidth=2, markersize=6, label="Total Loss")
    ax1.plot(epochs, contrastive_loss, "s-", color="#FF9800", linewidth=2, markersize=5, label="Contrastive Loss")
    ax1.plot(epochs, mlm_loss, "^-", color="#4CAF50", linewidth=2, markersize=5, label="MLM Loss")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True, loc="upper right")
    ax1.set_title("A. Training Loss", fontsize=12, fontweight="bold", pad=8)
    ax1.grid(alpha=0.3)

    # Learning rate on secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(epochs, [l * 1e5 for l in lr], "--", color="#9E9E9E", linewidth=1, alpha=0.6)
    ax1b.set_ylabel("LR (×10⁻⁵)", fontsize=9, color="#9E9E9E")
    ax1b.tick_params(axis="y", labelcolor="#9E9E9E", labelsize=8)

    # --- Panel B: MLM Accuracy ---
    ax2.plot(epochs, mlm_accuracy, "o-", color="#E91E63", linewidth=2, markersize=6)
    ax2.fill_between(epochs, mlm_accuracy, alpha=0.15, color="#E91E63")
    ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MLM Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("B. Masked Language Model Accuracy", fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylim(0.64, 0.66)
    ax2.grid(alpha=0.3)

    # Annotate final values
    ax2.annotate(f"{mlm_accuracy[-1]:.4f}", xy=(epochs[-1], mlm_accuracy[-1]),
                 xytext=(epochs[-1] - 1, mlm_accuracy[-1] + 0.002),
                 fontsize=10, fontweight="bold", color="#E91E63",
                 arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.2))

    fig.tight_layout()
    _save(fig, "fig5_training_curves")


# ---------------------------------------------------------------------------
# Figure 6: Architecture Diagram
# ---------------------------------------------------------------------------
def generate_fig6_architecture():
    """Figure 6: PRISM architecture flow diagram."""
    print("\n" + "=" * 60)
    print("Figure 6: Architecture Diagram")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def add_box(ax, x, y, w, h, text, color, fontsize=10, text_color="white"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color)

    def add_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#424242",
                                     lw=2, mutation_scale=15))

    # Boxes
    bh = 1.2  # box height
    y_main = 1.9  # main row y

    # Input
    add_box(ax, 0.2, y_main, 2.0, bh, "scRNA-seq\nCount Matrix", "#78909C")

    # Pre-processing
    add_box(ax, 3.0, y_main, 2.0, bh, "QC + HVG\nSelection", "#AB47BC")

    # Encoder
    add_box(ax, 5.8, y_main, 2.4, bh, "Transformer\nEncoder (12L)", "#1E88E5", fontsize=11)

    # Embeddings
    add_box(ax, 9.0, y_main, 2.0, bh, "PRISM\nEmbeddings", "#4CAF50")

    # Downstream
    add_box(ax, 11.8, y_main, 2.0, bh, "Fate\nResolution", "#E91E63")

    # Arrows
    add_arrow(ax, 2.2, y_main + bh / 2, 3.0, y_main + bh / 2)
    add_arrow(ax, 5.0, y_main + bh / 2, 5.8, y_main + bh / 2)
    add_arrow(ax, 8.2, y_main + bh / 2, 9.0, y_main + bh / 2)
    add_arrow(ax, 11.0, y_main + bh / 2, 11.8, y_main + bh / 2)

    # Training components (below main row)
    y_train = 0.2
    add_box(ax, 3.5, y_train, 2.2, 0.9, "PCP Pre-training\n(7.5M cells)", "#0D47A1", fontsize=9)
    add_box(ax, 6.5, y_train, 2.2, 0.9, "Contrastive +\nRecon Loss", "#FF6F00", fontsize=9)
    add_box(ax, 9.5, y_train, 2.2, 0.9, "Condition-aware\nSampling", "#00695C", fontsize=9)

    # Arrows from training row to encoder
    add_arrow(ax, 4.6, 1.1, 6.5, y_main)
    add_arrow(ax, 7.6, 1.1, 7.0, y_main)
    add_arrow(ax, 10.6, 1.1, 8.0, y_main)

    # Key innovations (above main row)
    y_top = 3.6
    innovations = [
        (1.0, "Multi-GPU\nDataParallel"),
        (5.0, "scGPT-initialized\nWeights"),
        (10.0, "Fate Probability\nEstimation"),
    ]
    for x, text in innovations:
        ax.text(x + 1, y_top + 0.5, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#37474F",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                         edgecolor="#90CAF9", alpha=0.9))

    ax.set_title("PRISM Architecture", fontsize=16, fontweight="bold", pad=15)

    _save(fig, "fig6_architecture")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name):
    """Save figure as PNG and PDF."""
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate PRISM publication figures")
    parser.add_argument("--figures", nargs="*", default=None,
                        help="Specific figures to generate (1-6). Default: all")
    args = parser.parse_args()

    generators = {
        "1": generate_fig1_umap_grid,
        "2": generate_fig2_bar_chart,
        "3": generate_fig3_heatmap_radar,
        "4": generate_fig4_pretraining,
        "5": generate_fig5_training_curves,
        "6": generate_fig6_architecture,
    }

    if args.figures:
        to_run = [f for f in args.figures if f in generators]
    else:
        to_run = list(generators.keys())

    print(f"Working directory: {os.getcwd()}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Generating figures: {', '.join(to_run)}")

    for fig_num in to_run:
        generators[fig_num]()

    print(f"\nAll done! Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
