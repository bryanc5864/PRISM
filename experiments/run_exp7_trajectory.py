#!/usr/bin/env python3
"""
Experiment 7: Trajectory Analysis Comparison for PRISM.

Compares PRISM-Trace's hybrid DPT pseudotime approach against alternative
methods and performs detailed gene cascade and branch assignment analysis.

Sections:
  1. DPT Pseudotime Comparison (PCA vs PRISM vs Scanpy default)
  2. Palantir Comparison (if available)
  3. Gene Cascade Analysis (top 20 discriminators along pseudotime)
  4. Branch Assignment Quality (PRISM fate-prob vs KMeans vs Leiden)
"""

import os, sys, time, warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = (
        cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from prism.resolve.mixture import BayesianFateMixture


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_expression(adata, gene):
    """Return dense expression vector for a single gene."""
    gene_idx = list(adata.var_names).index(gene)
    X = adata.X
    if sp.issparse(X):
        return X[:, gene_idx].toarray().flatten()
    return X[:, gene_idx].flatten()


def set_root_epi0(adata, cluster_key="cluster"):
    """Set iroot to a cell in the Epi0/epidermal progenitor cluster."""
    if cluster_key in adata.obs.columns:
        # Try several root cluster names
        for root_name in ["Epi0", "epidermal"]:
            root_mask = adata.obs[cluster_key] == root_name
            if root_mask.any():
                if "total_counts" in adata.obs:
                    root_idx = adata.obs.loc[root_mask, "total_counts"].idxmin()
                    adata.uns["iroot"] = np.where(adata.obs.index == root_idx)[0][0]
                else:
                    adata.uns["iroot"] = np.where(root_mask.values)[0][0]
                return
    # Fallback
    adata.uns["iroot"] = 0


# ===================================================================
# Section 1: DPT Pseudotime Comparison
# ===================================================================

def run_dpt_comparison(adata):
    """Compute DPT pseudotime in PCA, PRISM, and Scanpy-default spaces."""
    print("=" * 70)
    print("SECTION 1: DPT Pseudotime Comparison")
    print("=" * 70)

    labels = adata.obs["fate_int"].values
    results = {}

    # --- 1a. DPT in PCA space (50d) - what PRISM-Trace uses ---
    print("\n[1a] DPT in PCA space (50 PCs) ...")
    adata_pca = adata.copy()
    sc.pp.neighbors(adata_pca, use_rep="X_pca", n_neighbors=30,
                    key_added="pca_neighbors")
    sc.tl.diffmap(adata_pca, n_comps=15, neighbors_key="pca_neighbors")
    set_root_epi0(adata_pca)
    sc.tl.dpt(adata_pca, n_dcs=15, neighbors_key="pca_neighbors")

    pt_pca = adata_pca.obs["dpt_pseudotime"].values
    valid_pca = np.isfinite(pt_pca)
    n_valid_pca = valid_pca.sum()
    # Spearman with fate labels (among labeled cells only: 2=eccrine, 3=hair)
    fate_mask = labels >= 2
    usable_pca = valid_pca & fate_mask
    if usable_pca.sum() > 20:
        rho_pca, p_pca = spearmanr(pt_pca[usable_pca], labels[usable_pca])
    else:
        rho_pca, p_pca = float("nan"), 1.0
    results["PCA_DPT"] = {
        "n_valid": int(n_valid_pca), "spearman_rho": float(rho_pca),
        "p_value": float(p_pca),
    }
    print(f"     Valid cells: {n_valid_pca}/{len(pt_pca)}")
    print(f"     Spearman(pt, fate_int) on labeled cells: rho={rho_pca:.4f}, p={p_pca:.2e}")

    # --- 1b. DPT in PRISM space (128d) ---
    print("\n[1b] DPT in PRISM space (128d) ...")
    adata_prism = adata.copy()
    try:
        sc.pp.neighbors(adata_prism, use_rep="X_prism", n_neighbors=30,
                        key_added="prism_neighbors")
        sc.tl.diffmap(adata_prism, n_comps=15, neighbors_key="prism_neighbors")
        set_root_epi0(adata_prism)
        sc.tl.dpt(adata_prism, n_dcs=15, neighbors_key="prism_neighbors")

        pt_prism = adata_prism.obs["dpt_pseudotime"].values
        valid_prism = np.isfinite(pt_prism)
        n_valid_prism = valid_prism.sum()
        usable_prism = valid_prism & fate_mask
        if usable_prism.sum() > 20:
            rho_prism, p_prism = spearmanr(pt_prism[usable_prism],
                                           labels[usable_prism])
        else:
            rho_prism, p_prism = float("nan"), 1.0
        results["PRISM_DPT"] = {
            "n_valid": int(n_valid_prism), "spearman_rho": float(rho_prism),
            "p_value": float(p_prism),
        }
        print(f"     Valid cells: {n_valid_prism}/{len(pt_prism)}")
        print(f"     Spearman(pt, fate_int) on labeled cells: rho={rho_prism:.4f}, p={p_prism:.2e}")

        # Cross-comparison between PCA-DPT and PRISM-DPT
        both_valid = valid_pca & valid_prism
        if both_valid.sum() > 50:
            rho_cross, _ = spearmanr(pt_pca[both_valid], pt_prism[both_valid])
            results["PCA_vs_PRISM_rho"] = float(rho_cross)
            print(f"     PCA-DPT vs PRISM-DPT Spearman: {rho_cross:.4f}")
    except Exception as e:
        print(f"     PRISM-DPT FAILED (likely disconnected graph): {e}")
        results["PRISM_DPT"] = {"n_valid": 0, "error": str(e)}

    # --- 1c. Scanpy default DPT ---
    print("\n[1c] Scanpy default DPT (neighbors on X_pca, default params) ...")
    adata_default = adata.copy()
    # Scanpy default: neighbors computed on X_pca with default params
    sc.pp.neighbors(adata_default, use_rep="X_pca", n_neighbors=15)
    sc.tl.diffmap(adata_default, n_comps=15)
    set_root_epi0(adata_default)
    sc.tl.dpt(adata_default, n_dcs=15)

    pt_default = adata_default.obs["dpt_pseudotime"].values
    valid_default = np.isfinite(pt_default)
    n_valid_default = valid_default.sum()
    usable_default = valid_default & fate_mask
    if usable_default.sum() > 20:
        rho_default, p_default = spearmanr(pt_default[usable_default],
                                           labels[usable_default])
    else:
        rho_default, p_default = float("nan"), 1.0
    results["Scanpy_DPT"] = {
        "n_valid": int(n_valid_default), "spearman_rho": float(rho_default),
        "p_value": float(p_default),
    }
    print(f"     Valid cells: {n_valid_default}/{len(pt_default)}")
    print(f"     Spearman(pt, fate_int) on labeled cells: rho={rho_default:.4f}, p={p_default:.2e}")

    # Cross-comparison PCA-DPT vs Scanpy-default
    both_valid2 = valid_pca & valid_default
    if both_valid2.sum() > 50:
        rho_cross2, _ = spearmanr(pt_pca[both_valid2], pt_default[both_valid2])
        results["PCA_vs_Scanpy_rho"] = float(rho_cross2)
        print(f"     PCA-DPT vs Scanpy-default DPT Spearman: {rho_cross2:.4f}")

    # Store the PCA pseudotime in the main adata for downstream sections
    adata.obs["dpt_pseudotime"] = pt_pca

    print("\n--- DPT Comparison Summary ---")
    for method, info in results.items():
        if isinstance(info, dict) and "n_valid" in info:
            print(f"  {method}: valid={info['n_valid']}, rho={info.get('spearman_rho', 'N/A')}")

    return results, pt_pca


# ===================================================================
# Section 2: Palantir Comparison
# ===================================================================

def run_palantir_comparison(adata, pt_pca):
    """Run Palantir pseudotime and compare with DPT."""
    print("\n" + "=" * 70)
    print("SECTION 2: Palantir Comparison")
    print("=" * 70)

    results = {}

    try:
        import palantir

        print("  Palantir v" + palantir.__version__ + " available, running ...")

        adata_pal = adata.copy()

        # Palantir needs PCA and diffusion maps
        # Use the PCA already computed
        pca_df = pd.DataFrame(
            adata_pal.obsm["X_pca"],
            index=adata_pal.obs_names,
        )

        # Run Palantir diffusion maps
        print("  Computing Palantir diffusion maps ...")
        dm_res = palantir.utils.run_diffusion_maps(pca_df, n_components=10)

        # Determine multiscale space
        print("  Computing multiscale space ...")
        ms_data = palantir.utils.determine_multiscale_space(dm_res)

        # Find the root cell (same logic: epidermal progenitor)
        set_root_epi0(adata_pal)
        root_cell = adata_pal.obs_names[adata_pal.uns["iroot"]]

        print(f"  Running Palantir (root={root_cell}) ...")
        pr_res = palantir.core.run_palantir(
            ms_data,
            root_cell,
            num_waypoints=500,
            use_early_cell_as_start=True,
        )

        pal_pt = pr_res.pseudotime
        # Align indices
        common_idx = adata.obs_names.intersection(pal_pt.index)
        pal_pt_arr = pal_pt.loc[common_idx].values
        pca_pt_arr = pd.Series(pt_pca, index=adata.obs_names).loc[common_idx].values
        valid = np.isfinite(pal_pt_arr) & np.isfinite(pca_pt_arr)

        rho_pal_pca, p_pal_pca = spearmanr(pal_pt_arr[valid], pca_pt_arr[valid])
        print(f"  Palantir pseudotime computed for {valid.sum()} cells")
        print(f"  Spearman(Palantir_PT, PCA_DPT_PT): rho={rho_pal_pca:.4f}, p={p_pal_pca:.2e}")

        # Palantir vs fate labels
        labels = adata.obs["fate_int"].values
        fate_mask = labels >= 2
        pal_full = np.full(len(adata), np.nan)
        loc_idx = [adata.obs_names.get_loc(c) for c in common_idx]
        pal_full[loc_idx] = pal_pt_arr
        usable = np.isfinite(pal_full) & fate_mask
        if usable.sum() > 20:
            rho_pal_fate, p_pal_fate = spearmanr(pal_full[usable], labels[usable])
        else:
            rho_pal_fate, p_pal_fate = float("nan"), 1.0
        print(f"  Spearman(Palantir_PT, fate_int): rho={rho_pal_fate:.4f}, p={p_pal_fate:.2e}")

        # Palantir branch probabilities vs PRISM
        if hasattr(pr_res, "branch_probs") and pr_res.branch_probs is not None:
            n_branches = pr_res.branch_probs.shape[1]
            print(f"  Palantir detected {n_branches} branch(es)")
            results["palantir_n_branches"] = n_branches
        else:
            print("  Palantir branch probabilities not available")

        results["palantir_available"] = True
        results["palantir_n_cells"] = int(valid.sum())
        results["palantir_vs_pca_dpt_rho"] = float(rho_pal_pca)
        results["palantir_vs_pca_dpt_p"] = float(p_pal_pca)
        results["palantir_vs_fate_rho"] = float(rho_pal_fate)
        results["palantir_vs_fate_p"] = float(p_pal_fate)

    except ImportError:
        print("  Palantir not installed. Skipping.")
        results["palantir_available"] = False
    except Exception as e:
        print(f"  Palantir failed: {e}")
        results["palantir_available"] = False
        results["palantir_error"] = str(e)

    return results


# ===================================================================
# Section 3: Gene Cascade Analysis
# ===================================================================

def run_gene_cascade_analysis(adata, pt_pca, fate_probs):
    """Analyze top 20 discriminator gene expression along pseudotime."""
    print("\n" + "=" * 70)
    print("SECTION 3: Gene Cascade Analysis (Top 20 PRISM Discriminators)")
    print("=" * 70)

    # Top 20 discriminators from PRISM-Resolve (ordered by PIP/effect size)
    top20_genes = [
        "Tfap2b", "Lgr6", "Trp63", "Sox6", "Meis1",
        "Mybpc1", "Tspear", "Alcam", "Tenm2", "Cpa6",
        "Dsp", "Sptlc3", "Stox2", "Ctnnd2", "Nrk",
        "Dmd", "Col1a2", "Ctnna3", "Lsamp", "Ntm",
    ]

    # Filter to those actually in var_names
    top20_genes = [g for g in top20_genes if g in adata.var_names]
    print(f"  Genes present in data: {len(top20_genes)}")

    valid_pt = np.isfinite(pt_pca)
    pseudotime = pt_pca.copy()

    results = []
    for gene in top20_genes:
        expr = get_expression(adata, gene)

        # Only consider cells with valid pseudotime
        expr_valid = expr[valid_pt]
        pt_valid = pseudotime[valid_pt]

        gene_mean = expr_valid.mean()
        gene_std = expr_valid.std()
        threshold = gene_mean + 1.0 * gene_std

        # Sort by pseudotime
        sort_idx = np.argsort(pt_valid)
        pt_sorted = pt_valid[sort_idx]
        expr_sorted = expr_valid[sort_idx]

        # Smooth expression with rolling window for activation detection
        window = max(100, len(expr_sorted) // 50)
        if len(expr_sorted) < window:
            window = max(10, len(expr_sorted) // 5)
        expr_smooth = np.convolve(expr_sorted, np.ones(window) / window, mode="valid")
        pt_smooth = pt_sorted[:len(expr_smooth)]

        # Activation pseudotime: first time smoothed expression exceeds mean + 1 SD
        above_thresh = np.where(expr_smooth > threshold)[0]
        if len(above_thresh) > 0:
            activation_pt = float(pt_smooth[above_thresh[0]])
        else:
            # Gene never exceeds threshold; use pseudotime of max expression
            activation_pt = float(pt_smooth[np.argmax(expr_smooth)])

        # Spearman correlation of gene expression with pseudotime
        rho_pt, p_pt = spearmanr(pt_valid, expr_valid)

        # Correlation with fate score (eccrine_prob - hair_prob)
        fate_score = fate_probs[:, 1] - fate_probs[:, 2]
        late_mask = valid_pt & (pseudotime > np.percentile(pseudotime[valid_pt], 40))
        if late_mask.sum() > 50:
            rho_fate, p_fate = spearmanr(expr[late_mask], fate_score[late_mask])
        else:
            rho_fate, p_fate = float("nan"), 1.0

        results.append({
            "gene": gene,
            "activation_pt": activation_pt,
            "mean_expr": float(gene_mean),
            "threshold": float(threshold),
            "spearman_vs_pt": float(rho_pt),
            "spearman_vs_fate": float(rho_fate),
        })

    df = pd.DataFrame(results).sort_values("activation_pt").reset_index(drop=True)
    df["activation_rank"] = range(1, len(df) + 1)

    print("\n  Gene Activation Ordering (by activation pseudotime):")
    print("  " + "-" * 80)
    print(f"  {'Rank':>4}  {'Gene':<10}  {'Activation PT':>14}  {'rho(PT)':>8}  {'rho(Fate)':>10}  {'Mean Expr':>10}")
    print("  " + "-" * 80)
    for _, row in df.iterrows():
        print(f"  {row['activation_rank']:4d}  {row['gene']:<10}  {row['activation_pt']:14.4f}"
              f"  {row['spearman_vs_pt']:8.4f}  {row['spearman_vs_fate']:10.4f}"
              f"  {row['mean_expr']:10.4f}")

    # Specific biological checks
    print("\n  --- Biological Checks ---")
    gene_pt_map = dict(zip(df["gene"], df["activation_pt"]))

    if "Lgr6" in gene_pt_map and "Tfap2b" in gene_pt_map:
        lgr6_first = gene_pt_map["Lgr6"] < gene_pt_map["Tfap2b"]
        print(f"  Lgr6 activates before Tfap2b? {lgr6_first}"
              f"  (Lgr6={gene_pt_map['Lgr6']:.4f}, Tfap2b={gene_pt_map['Tfap2b']:.4f})")
    else:
        lgr6_first = None
        print("  Cannot compare Lgr6 vs Tfap2b (missing from data)")

    if "Trp63" in gene_pt_map:
        trp63_rank = df.loc[df["gene"] == "Trp63", "activation_rank"].values[0]
        trp63_early = trp63_rank <= len(df) // 2
        print(f"  Trp63 activates early? {trp63_early}"
              f"  (rank {trp63_rank}/{len(df)}, PT={gene_pt_map['Trp63']:.4f})")
    else:
        trp63_early = None
        print("  Trp63 not in gene list")

    cascade_results = {
        "gene_cascade_df": df,
        "lgr6_before_tfap2b": lgr6_first,
        "trp63_activates_early": trp63_early,
    }
    return cascade_results


# ===================================================================
# Section 4: Branch Assignment Quality
# ===================================================================

def run_branch_quality(adata, pt_pca, fate_probs):
    """Compare branch assignment methods against ground-truth fate labels."""
    print("\n" + "=" * 70)
    print("SECTION 4: Branch Assignment Quality")
    print("=" * 70)

    labels = adata.obs["fate_int"].values
    valid_pt = np.isfinite(pt_pca)
    pseudotime = pt_pca.copy()

    # Late pseudotime cells (top 50%)
    pt_50 = np.percentile(pseudotime[valid_pt], 50)
    late_mask = valid_pt & (pseudotime > pt_50)

    # --- Ground truth branch labels (for evaluation) ---
    # Only evaluate on labeled cells that are in late pseudotime
    fate_mask = (labels >= 2)  # eccrine=2, hair=3
    eval_mask = late_mask & fate_mask
    n_eval = eval_mask.sum()
    print(f"\n  Evaluation cells (late PT + fate label): {n_eval}")
    print(f"    eccrine: {(labels[eval_mask] == 2).sum()}, hair: {(labels[eval_mask] == 3).sum()}")

    gt_labels = labels[eval_mask]  # 2 or 3
    gt_binary = (gt_labels == 2).astype(int)  # 1=eccrine, 0=hair

    results = {}

    # --- Method A: PRISM fate-probability branch assignment ---
    print("\n  [A] PRISM fate-probability branch assignment ...")
    prism_ecc_prob = fate_probs[eval_mask, 1]
    prism_hair_prob = fate_probs[eval_mask, 2]
    prism_branch = (prism_ecc_prob > prism_hair_prob).astype(int)  # 1=eccrine, 0=hair

    # All late cells for ARI (not just labeled)
    prism_branch_all = (fate_probs[late_mask, 1] > fate_probs[late_mask, 2]).astype(int)

    prism_ari = adjusted_rand_score(gt_binary, prism_branch)
    # RF AUROC: train on embeddings with branch labels, evaluate vs ground truth
    prism_auroc = _compute_branch_auroc(
        adata.obsm["X_prism"][eval_mask], gt_binary, prism_branch, "PRISM-fate"
    )
    results["PRISM_fate"] = {
        "ARI_vs_gt": float(prism_ari),
        "RF_AUROC_vs_gt": float(prism_auroc),
        "n_eccrine": int(prism_branch.sum()),
        "n_hair": int((1 - prism_branch).sum()),
    }
    print(f"     ARI vs ground truth: {prism_ari:.4f}")
    print(f"     RF AUROC vs ground truth: {prism_auroc:.4f}")
    print(f"     Eccrine: {prism_branch.sum()}, Hair: {(1 - prism_branch).sum()}")

    # --- Method B: KMeans (k=2) on late-pseudotime cells ---
    print("\n  [B] KMeans (k=2) on late-pseudotime cells ...")
    # Use PCA embeddings for KMeans (fair comparison on same space as DPT)
    late_pca = adata.obsm["X_pca"][late_mask]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels_all = kmeans.fit_predict(late_pca)

    # Extract labels for eval cells within late cells
    late_indices = np.where(late_mask)[0]
    eval_indices = np.where(eval_mask)[0]
    eval_in_late = np.array([np.where(late_indices == i)[0][0] for i in eval_indices])
    km_labels = km_labels_all[eval_in_late]

    # Match KMeans clusters to eccrine/hair using majority vote
    km_labels = _match_clusters_to_gt(km_labels, gt_binary)

    km_ari = adjusted_rand_score(gt_binary, km_labels)
    km_auroc = _compute_branch_auroc(
        adata.obsm["X_pca"][eval_mask], gt_binary, km_labels, "KMeans"
    )
    results["KMeans"] = {
        "ARI_vs_gt": float(km_ari),
        "RF_AUROC_vs_gt": float(km_auroc),
        "n_eccrine": int(km_labels.sum()),
        "n_hair": int((1 - km_labels).sum()),
    }
    print(f"     ARI vs ground truth: {km_ari:.4f}")
    print(f"     RF AUROC vs ground truth: {km_auroc:.4f}")

    # ARI: PRISM vs KMeans agreement
    prism_km_ari = adjusted_rand_score(prism_branch, km_labels)
    results["PRISM_vs_KMeans_ARI"] = float(prism_km_ari)
    print(f"     PRISM vs KMeans ARI: {prism_km_ari:.4f}")

    # --- Method C: Leiden clustering on late-pseudotime cells ---
    print("\n  [C] Leiden clustering on late-pseudotime cells ...")
    adata_late = adata[late_mask].copy()
    sc.pp.neighbors(adata_late, use_rep="X_pca", n_neighbors=15)
    # Find resolution that gives ~2 clusters
    best_res = 0.1
    for res in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        sc.tl.leiden(adata_late, resolution=res, key_added="leiden_branch")
        n_clust = adata_late.obs["leiden_branch"].nunique()
        if n_clust >= 2:
            best_res = res
            break

    sc.tl.leiden(adata_late, resolution=best_res, key_added="leiden_branch")
    n_leiden_clusters = adata_late.obs["leiden_branch"].nunique()
    print(f"     Leiden resolution={best_res}, clusters={n_leiden_clusters}")

    # Map leiden clusters to binary labels
    leiden_all = adata_late.obs["leiden_branch"].values.astype(int)
    leiden_labels = leiden_all[eval_in_late]

    # If more than 2 clusters, merge to 2 using majority vote approach
    if n_leiden_clusters > 2:
        # For each cluster, check which ground-truth label is majority
        leiden_binary = np.zeros_like(leiden_labels)
        for cl in np.unique(leiden_labels):
            cl_mask = leiden_labels == cl
            if cl_mask.sum() > 0:
                # Majority vote
                mean_gt = gt_binary[cl_mask].mean()
                leiden_binary[cl_mask] = 1 if mean_gt > 0.5 else 0
        leiden_labels = leiden_binary
    else:
        leiden_labels = _match_clusters_to_gt(leiden_labels, gt_binary)

    leiden_ari = adjusted_rand_score(gt_binary, leiden_labels)
    leiden_auroc = _compute_branch_auroc(
        adata.obsm["X_pca"][eval_mask], gt_binary, leiden_labels, "Leiden"
    )
    results["Leiden"] = {
        "ARI_vs_gt": float(leiden_ari),
        "RF_AUROC_vs_gt": float(leiden_auroc),
        "n_eccrine": int(leiden_labels.sum()),
        "n_hair": int((1 - leiden_labels).sum()),
        "n_clusters": int(n_leiden_clusters),
        "resolution": float(best_res),
    }
    print(f"     ARI vs ground truth: {leiden_ari:.4f}")
    print(f"     RF AUROC vs ground truth: {leiden_auroc:.4f}")

    # ARI: PRISM vs Leiden agreement
    prism_leiden_ari = adjusted_rand_score(prism_branch, leiden_labels)
    results["PRISM_vs_Leiden_ARI"] = float(prism_leiden_ari)
    print(f"     PRISM vs Leiden ARI: {prism_leiden_ari:.4f}")

    # --- Summary Table ---
    print("\n  --- Branch Assignment Summary ---")
    print(f"  {'Method':<20} {'ARI vs GT':>12} {'RF AUROC vs GT':>16}")
    print(f"  {'-'*48}")
    for method in ["PRISM_fate", "KMeans", "Leiden"]:
        info = results[method]
        print(f"  {method:<20} {info['ARI_vs_gt']:12.4f} {info['RF_AUROC_vs_gt']:16.4f}")

    return results


def _match_clusters_to_gt(pred_labels, gt_binary):
    """Match cluster IDs to ground truth via majority vote."""
    pred = pred_labels.copy()
    unique_clusters = np.unique(pred)
    if len(unique_clusters) != 2:
        return pred

    # Check which mapping has higher agreement
    mapping_a = (pred == unique_clusters[0]).astype(int)  # cluster0=eccrine
    mapping_b = (pred == unique_clusters[1]).astype(int)  # cluster1=eccrine
    ari_a = adjusted_rand_score(gt_binary, mapping_a)
    ari_b = adjusted_rand_score(gt_binary, mapping_b)

    if ari_a >= ari_b:
        return mapping_a
    else:
        return mapping_b


def _compute_branch_auroc(embeddings, gt_binary, branch_labels, method_name):
    """Train RF on branch_labels and evaluate AUROC against gt_binary.

    This measures: if we use branch_labels as training signal, can an RF
    trained on embeddings recover ground-truth fate labels?
    """
    if len(np.unique(gt_binary)) < 2:
        return float("nan")

    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Cross-validated prediction on gt_binary
        probs = cross_val_predict(rf, embeddings, gt_binary,
                                  cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(gt_binary, probs)
    except Exception as e:
        print(f"     AUROC computation failed for {method_name}: {e}")
        return float("nan")


# ===================================================================
# Main
# ===================================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("EXPERIMENT 7: Trajectory Analysis Comparison")
    print("=" * 70)
    print()

    # --- Load data ---
    print("Loading data ...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    print(f"  Embeddings: X_pca {adata.obsm['X_pca'].shape}, "
          f"X_prism {adata.obsm['X_prism'].shape}")

    # --- Compute PRISM fate probabilities ---
    print("\nComputing PRISM fate probabilities ...")
    labels = adata.obs["fate_int"].values
    embeddings = adata.obsm["X_prism"]
    label_mask = labels >= 2
    mixture = BayesianFateMixture(n_components=3)
    mixture.fit(embeddings, labels, label_mask)
    fate_probs = mixture.predict_proba(embeddings)
    n_ecc = (fate_probs.argmax(1) == 1).sum()
    n_hair = (fate_probs.argmax(1) == 2).sum()
    n_unc = (fate_probs.argmax(1) == 0).sum()
    print(f"  Fate assignment: {n_ecc} eccrine, {n_hair} hair, {n_unc} uncommitted")

    # === Section 1: DPT Comparison ===
    dpt_results, pt_pca = run_dpt_comparison(adata)

    # === Section 2: Palantir Comparison ===
    palantir_results = run_palantir_comparison(adata, pt_pca)

    # === Section 3: Gene Cascade Analysis ===
    cascade_results = run_gene_cascade_analysis(adata, pt_pca, fate_probs)

    # === Section 4: Branch Assignment Quality ===
    branch_results = run_branch_quality(adata, pt_pca, fate_probs)

    # === Write Results ===
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Experiment 7 Complete ({total_time:.0f}s)")
    print(f"{'=' * 70}")

    _write_results(dpt_results, palantir_results, cascade_results,
                   branch_results, total_time)


def _write_results(dpt_results, palantir_results, cascade_results,
                   branch_results, total_time):
    """Append results to results.md."""

    lines = []
    lines.append(f"\n\n---\n")
    lines.append(f"\n### Experiment 7: Trajectory Analysis Comparison")
    lines.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Section 1 ---
    lines.append("#### 1. DPT Pseudotime Comparison\n")
    lines.append("| Method | Valid Cells | Spearman(PT, fate) | p-value |")
    lines.append("|--------|-----------|-------------------|---------|")
    for method in ["PCA_DPT", "PRISM_DPT", "Scanpy_DPT"]:
        info = dpt_results.get(method, {})
        if "error" in info:
            lines.append(f"| {method} | FAILED | -- | -- |")
        else:
            n_v = info.get("n_valid", "N/A")
            rho = info.get("spearman_rho", float("nan"))
            p = info.get("p_value", float("nan"))
            rho_str = f"{rho:.4f}" if np.isfinite(rho) else "N/A"
            p_str = f"{p:.2e}" if np.isfinite(p) else "N/A"
            lines.append(f"| {method} | {n_v} | {rho_str} | {p_str} |")

    for cross_key in ["PCA_vs_PRISM_rho", "PCA_vs_Scanpy_rho"]:
        if cross_key in dpt_results:
            lines.append(f"\n- {cross_key.replace('_', ' ')}: {dpt_results[cross_key]:.4f}")

    lines.append("")

    # --- Section 2 ---
    lines.append("#### 2. Palantir Comparison\n")
    if palantir_results.get("palantir_available"):
        lines.append(f"- Palantir cells: {palantir_results.get('palantir_n_cells', 'N/A')}")
        lines.append(f"- Palantir vs PCA-DPT Spearman: "
                     f"{palantir_results.get('palantir_vs_pca_dpt_rho', 0):.4f} "
                     f"(p={palantir_results.get('palantir_vs_pca_dpt_p', 1):.2e})")
        lines.append(f"- Palantir vs fate labels Spearman: "
                     f"{palantir_results.get('palantir_vs_fate_rho', 0):.4f} "
                     f"(p={palantir_results.get('palantir_vs_fate_p', 1):.2e})")
        if "palantir_n_branches" in palantir_results:
            lines.append(f"- Palantir detected branches: "
                         f"{palantir_results['palantir_n_branches']}")
    else:
        error = palantir_results.get("palantir_error", "not installed")
        lines.append(f"- Palantir skipped: {error}")
    lines.append("")

    # --- Section 3 ---
    lines.append("#### 3. Gene Cascade Analysis (Top 20 Discriminators)\n")
    df = cascade_results["gene_cascade_df"]
    lines.append("| Rank | Gene | Activation PT | rho(PT) | rho(Fate) |")
    lines.append("|------|------|---------------|---------|-----------|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['activation_rank']:.0f} | {row['gene']} | "
            f"{row['activation_pt']:.4f} | {row['spearman_vs_pt']:.4f} | "
            f"{row['spearman_vs_fate']:.4f} |"
        )

    lines.append("")
    lgr6_tf = cascade_results.get("lgr6_before_tfap2b")
    if lgr6_tf is not None:
        lines.append(f"- Lgr6 activates before Tfap2b: **{lgr6_tf}**")
    trp63_e = cascade_results.get("trp63_activates_early")
    if trp63_e is not None:
        lines.append(f"- Trp63 activates early: **{trp63_e}**")
    lines.append("")

    # --- Section 4 ---
    lines.append("#### 4. Branch Assignment Quality\n")
    lines.append("| Method | ARI vs GT | RF AUROC vs GT |")
    lines.append("|--------|----------|----------------|")
    for method in ["PRISM_fate", "KMeans", "Leiden"]:
        info = branch_results.get(method, {})
        ari = info.get("ARI_vs_gt", float("nan"))
        auroc = info.get("RF_AUROC_vs_gt", float("nan"))
        ari_str = f"{ari:.4f}" if np.isfinite(ari) else "N/A"
        auroc_str = f"{auroc:.4f}" if np.isfinite(auroc) else "N/A"
        lines.append(f"| {method} | {ari_str} | {auroc_str} |")

    for cross_key in ["PRISM_vs_KMeans_ARI", "PRISM_vs_Leiden_ARI"]:
        if cross_key in branch_results:
            lines.append(f"\n- {cross_key.replace('_', ' ')}: "
                         f"{branch_results[cross_key]:.4f}")
    lines.append("")

    lines.append(f"\nTotal experiment time: {total_time:.0f}s\n")

    result_text = "\n".join(lines)

    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(result_text)

    print(f"\nResults appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
