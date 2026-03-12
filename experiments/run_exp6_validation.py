#!/usr/bin/env python3
"""
Experiment 6: Discriminator Gene Validation

Validates PRISM's discriminator genes through three independent analyses:
1. GO Enrichment Analysis (gprofiler-official)
2. Marker Recovery Analysis (precision@K, recall@K, hypergeometric test)
3. Differential Expression Validation (Wilcoxon rank-sum comparison)

Reads the processed adata, re-runs PRISM-Resolve and PRISM-Trace analyses
to obtain gene lists, then performs validation.
"""

import os, sys, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.stats import mannwhitneyu, spearmanr, hypergeom
from gprofiler import GProfiler

# ============================================================
# Import PRISM modules
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from prism.resolve.mixture import BayesianFateMixture
from prism.resolve.horseshoe import HorseshoeDE


# ============================================================
# Helper functions (same as run_trace_final.py)
# ============================================================

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_DIR, "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
    flat["_structured"] = config
    return flat


def compute_pseudotime_pca(adata, n_neighbors=30, n_dcs=15):
    """DPT in PCA space."""
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors, key_added="pca_neighbors")
    sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key="pca_neighbors")
    if "cluster" in adata.obs.columns:
        root_mask = adata.obs["cluster"] == "Epi0"
        if root_mask.any() and "total_counts" in adata.obs:
            root_idx = adata.obs.loc[root_mask, "total_counts"].idxmin()
            adata.uns["iroot"] = np.where(adata.obs.index == root_idx)[0][0]
        elif root_mask.any():
            adata.uns["iroot"] = np.where(root_mask)[0][0]
        else:
            adata.uns["iroot"] = 0
    else:
        adata.uns["iroot"] = 0
    sc.tl.dpt(adata, n_dcs=n_dcs, neighbors_key="pca_neighbors")
    return adata


def get_expression(adata, gene):
    gene_idx = list(adata.var_names).index(gene)
    X = adata.X
    if sp.issparse(X):
        return X[:, gene_idx].toarray().flatten()
    return X[:, gene_idx].flatten()


def bh_correct(pvals):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    q = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        q[idx] = pvals[idx] * n / (i + 1)
    q_sorted = q[sorted_idx]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[sorted_idx] = q_sorted
    return np.clip(q, 0, 1)


def temporal_correlation(adata, pseudotime, fate_probs, gene_list):
    """Find genes whose expression correlates with fate commitment."""
    valid = np.isfinite(pseudotime)
    fate_score = fate_probs[:, 1] - fate_probs[:, 2]
    pt_thresh = np.percentile(pseudotime[valid], 40)
    late_valid = valid & (pseudotime > pt_thresh)
    results = []
    for gene in gene_list:
        if gene not in adata.var_names:
            continue
        expr = get_expression(adata, gene)
        expr_late = expr[late_valid]
        fate_late = fate_score[late_valid]
        if len(expr_late) < 50:
            continue
        rho, p = spearmanr(expr_late, fate_late)
        results.append({
            "gene": gene,
            "spearman_rho": float(rho),
            "p_value": p,
            "direction": "eccrine_corr" if rho > 0 else "hair_corr",
            "abs_rho": abs(rho),
        })
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["q_value"] = bh_correct(df["p_value"].values)
    return df.sort_values("q_value").reset_index(drop=True)


# ============================================================
# Known markers from literature
# ============================================================
KNOWN_MARKERS = [
    "En1", "Trpv6", "Dkk4", "Lgr6", "S100a4", "Trp63", "Tfap2b",
    "Sox9", "Lef1", "Wnt10b", "Edar", "Shh", "Krt14", "Krt15",
    "Krt17", "Krt5", "Bmp2", "Bmp4", "Sostdc1", "Foxd1", "Meis1",
    "Msx2", "Lhx2", "Fgf20", "Wnt3", "Wnt5a",
]


# ============================================================
# Part 1: GO Enrichment Analysis
# ============================================================
def run_go_enrichment(resolve_genes, trace_genes):
    """Run GO Biological Process enrichment using g:Profiler."""
    print("\n" + "=" * 60)
    print("Part 1: GO Enrichment Analysis")
    print("=" * 60)

    gp = GProfiler(return_dataframe=True)

    results = {}
    skin_keywords = [
        "skin", "epiderm", "hair", "follicle", "sweat", "gland",
        "appendage", "ectoderm", "keratinocyte", "morphogenesis",
        "epithelial", "Wnt", "BMP", "placode",
    ]

    for name, gene_list in [("PRISM-Resolve (PIP>0.5)", resolve_genes),
                             ("PRISM-Trace (FDR<0.05)", trace_genes)]:
        print(f"\n--- {name}: {len(gene_list)} genes ---")

        if len(gene_list) == 0:
            print("  No genes to test.")
            results[name] = {"df": pd.DataFrame(), "skin_terms": []}
            continue

        try:
            go_df = gp.profile(
                organism="mmusculus",
                query=gene_list,
                sources=["GO:BP"],
                user_threshold=0.05,
                significance_threshold_method="fdr",
                no_evidences=False,
            )
        except Exception as e:
            print(f"  g:Profiler query failed: {e}")
            # Retry with a smaller subset if too large
            try:
                go_df = gp.profile(
                    organism="mmusculus",
                    query=gene_list[:200],
                    sources=["GO:BP"],
                    user_threshold=0.05,
                    significance_threshold_method="fdr",
                    no_evidences=False,
                )
            except Exception as e2:
                print(f"  Retry also failed: {e2}")
                results[name] = {"df": pd.DataFrame(), "skin_terms": []}
                continue

        if go_df.empty:
            print("  No significant GO terms found.")
            results[name] = {"df": go_df, "skin_terms": []}
            continue

        # Sort by p-value
        go_df = go_df.sort_values("p_value").reset_index(drop=True)

        print(f"  Total significant GO:BP terms: {len(go_df)}")
        print(f"\n  Top 10 GO terms:")
        for i, row in go_df.head(10).iterrows():
            print(f"    {i+1}. {row['name']} (p={row['p_value']:.2e}, size={row['term_size']})")

        # Check for skin/appendage-related terms
        skin_terms = []
        for _, row in go_df.iterrows():
            term_name = row["name"].lower()
            for kw in skin_keywords:
                if kw.lower() in term_name:
                    skin_terms.append({
                        "term": row["name"],
                        "p_value": row["p_value"],
                        "keyword": kw,
                    })
                    break

        if skin_terms:
            print(f"\n  Skin/appendage-related terms ({len(skin_terms)}):")
            for st in skin_terms[:10]:
                print(f"    - {st['term']} (p={st['p_value']:.2e}, keyword: {st['keyword']})")
        else:
            print("  No explicitly skin/appendage-related terms found in enriched GO terms.")

        results[name] = {"df": go_df, "skin_terms": skin_terms}

    return results


# ============================================================
# Part 2: Marker Recovery Analysis
# ============================================================
def run_marker_recovery(resolve_ranked, trace_ranked, resolve_filtered, trace_filtered, all_tested_genes):
    """Compute precision@K, recall@K, and hypergeometric enrichment.

    Args:
        resolve_ranked: all genes ranked by PIP (for precision@K)
        trace_ranked: all genes ranked by q-value (for precision@K)
        resolve_filtered: genes with PIP > 0.5 (for hypergeometric test)
        trace_filtered: genes with FDR < 0.05 (for hypergeometric test)
        all_tested_genes: full gene universe (HVGs)
    """
    print("\n" + "=" * 60)
    print("Part 2: Marker Recovery Analysis")
    print("=" * 60)

    # Filter known markers to those actually in the tested gene universe
    markers_in_universe = [m for m in KNOWN_MARKERS if m in all_tested_genes]
    M = len(markers_in_universe)
    N_universe = len(all_tested_genes)
    marker_set = set(markers_in_universe)

    print(f"  Known markers in gene universe: {M}/{len(KNOWN_MARKERS)}")
    print(f"  Markers present: {', '.join(sorted(markers_in_universe))}")
    print(f"  Gene universe size: {N_universe}")

    results = {}
    K_values = [5, 10, 20, 50, 100]

    for name, ranked_list, filtered_list in [
        ("PRISM-Resolve (PIP>0.5)", resolve_ranked, resolve_filtered),
        ("PRISM-Trace (FDR<0.05)", trace_ranked, trace_filtered),
    ]:
        print(f"\n--- {name}: {len(filtered_list)} significant genes (ranked list: {len(ranked_list)}) ---")

        filtered_set = set(filtered_list)
        overlap = filtered_set & marker_set
        n_overlap = len(overlap)

        print(f"  Overlap with known markers: {n_overlap}/{M}")
        if overlap:
            print(f"  Recovered markers: {', '.join(sorted(overlap))}")

        # Precision@K and Recall@K (for ranked lists, use full list order)
        pr_at_k = {}
        for K in K_values:
            top_k = ranked_list[:K] if K <= len(ranked_list) else ranked_list
            top_k_set = set(top_k)
            hits = top_k_set & marker_set
            precision = len(hits) / K if K > 0 else 0
            recall = len(hits) / M if M > 0 else 0
            pr_at_k[K] = {"precision": precision, "recall": recall, "hits": len(hits)}
            print(f"  Precision@{K}: {precision:.3f}, Recall@{K}: {recall:.3f} ({len(hits)} hits)")

        # Hypergeometric test for enrichment of known markers in the FILTERED set
        # P(X >= n_overlap) where X ~ Hypergeom(N_universe, M, len(filtered_list))
        n_drawn = len(filtered_list)
        # sf gives P(X > k), so P(X >= k) = sf(k-1)
        if n_overlap > 0:
            pval = hypergeom.sf(n_overlap - 1, N_universe, M, n_drawn)
        else:
            pval = 1.0
        expected = n_drawn * M / N_universe if N_universe > 0 else 0
        fold_enrichment = n_overlap / expected if expected > 0 else float('inf')

        print(f"  Hypergeometric test (filtered set, n={n_drawn}): p={pval:.2e}")
        print(f"  Expected overlap by chance: {expected:.2f}")
        print(f"  Observed overlap: {n_overlap}")
        print(f"  Fold enrichment: {fold_enrichment:.2f}x")

        results[name] = {
            "n_genes": len(filtered_list),
            "n_overlap": n_overlap,
            "overlap_genes": sorted(overlap),
            "pr_at_k": pr_at_k,
            "hypergeom_pval": pval,
            "expected_overlap": expected,
            "fold_enrichment": fold_enrichment,
        }

    return results


# ============================================================
# Part 3: Differential Expression Validation
# ============================================================
def run_de_validation(adata, resolve_genes, trace_genes):
    """Run standard Wilcoxon DE and compare with PRISM genes."""
    print("\n" + "=" * 60)
    print("Part 3: Differential Expression Validation (Wilcoxon)")
    print("=" * 60)

    # Get eccrine and hair cells
    eccrine_mask = adata.obs["fate_int"].values == 2
    hair_mask = adata.obs["fate_int"].values == 3

    n_ecc = eccrine_mask.sum()
    n_hair = hair_mask.sum()
    print(f"  Eccrine cells: {n_ecc}, Hair cells: {n_hair}")

    if n_ecc < 10 or n_hair < 10:
        print("  WARNING: Too few cells for meaningful DE. Skipping.")
        return {}

    # Run Wilcoxon rank-sum on HVGs
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()

    print(f"  Testing {len(hvg_genes)} HVGs with Wilcoxon rank-sum...")

    wilcoxon_results = []
    for gene in hvg_genes:
        expr = get_expression(adata, gene)
        ecc_expr = expr[eccrine_mask]
        hair_expr = expr[hair_mask]

        # Skip genes with no variation
        if ecc_expr.std() == 0 and hair_expr.std() == 0:
            continue

        try:
            stat, p = mannwhitneyu(ecc_expr, hair_expr, alternative="two-sided")
        except ValueError:
            continue

        log2fc = float(np.log2((ecc_expr.mean() + 1e-3) / (hair_expr.mean() + 1e-3)))

        wilcoxon_results.append({
            "gene": gene,
            "p_value": p,
            "log2fc": log2fc,
            "mean_eccrine": float(ecc_expr.mean()),
            "mean_hair": float(hair_expr.mean()),
            "abs_log2fc": abs(log2fc),
        })

    if not wilcoxon_results:
        print("  No Wilcoxon results obtained.")
        return {}

    wilcox_df = pd.DataFrame(wilcoxon_results)
    wilcox_df["q_value"] = bh_correct(wilcox_df["p_value"].values)
    wilcox_df = wilcox_df.sort_values("q_value").reset_index(drop=True)

    sig_wilcox = wilcox_df[wilcox_df["q_value"] < 0.05]
    wilcox_gene_set = set(sig_wilcox["gene"].values)

    print(f"  Wilcoxon DE genes (FDR < 0.05): {len(sig_wilcox)}")
    print(f"\n  Top 15 Wilcoxon DE genes:")
    for i, row in sig_wilcox.head(15).iterrows():
        direction = "ecc_up" if row["log2fc"] > 0 else "hair_up"
        print(f"    {row['gene']:15s} log2FC={row['log2fc']:+.3f} ({direction}) q={row['q_value']:.2e}")

    # Overlap with PRISM gene sets
    resolve_set = set(resolve_genes)
    trace_set = set(trace_genes)

    # Jaccard similarity
    def jaccard(a, b):
        if len(a | b) == 0:
            return 0.0
        return len(a & b) / len(a | b)

    overlap_resolve = resolve_set & wilcox_gene_set
    overlap_trace = trace_set & wilcox_gene_set
    overlap_both = (resolve_set | trace_set) & wilcox_gene_set

    jaccard_resolve = jaccard(resolve_set, wilcox_gene_set)
    jaccard_trace = jaccard(trace_set, wilcox_gene_set)
    jaccard_union = jaccard(resolve_set | trace_set, wilcox_gene_set)

    print(f"\n--- Overlap with Wilcoxon DE ---")
    print(f"  PRISM-Resolve vs Wilcoxon:")
    print(f"    Overlap: {len(overlap_resolve)} genes")
    print(f"    Jaccard: {jaccard_resolve:.4f}")
    print(f"    Resolve-only: {len(resolve_set - wilcox_gene_set)}")
    print(f"    Wilcoxon-only: {len(wilcox_gene_set - resolve_set)}")

    print(f"\n  PRISM-Trace vs Wilcoxon:")
    print(f"    Overlap: {len(overlap_trace)} genes")
    print(f"    Jaccard: {jaccard_trace:.4f}")
    print(f"    Trace-only: {len(trace_set - wilcox_gene_set)}")
    print(f"    Wilcoxon-only: {len(wilcox_gene_set - trace_set)}")

    print(f"\n  PRISM-Union vs Wilcoxon:")
    print(f"    Overlap: {len(overlap_both)} genes")
    print(f"    Jaccard: {jaccard_union:.4f}")

    # Check PRISM-unique genes (found by PRISM but not Wilcoxon)
    prism_only_resolve = resolve_set - wilcox_gene_set
    prism_only_trace = trace_set - wilcox_gene_set
    prism_unique = (resolve_set | trace_set) - wilcox_gene_set

    # Check which known markers are found by each method
    marker_set = set(KNOWN_MARKERS)
    markers_in_wilcox = wilcox_gene_set & marker_set
    markers_in_resolve = resolve_set & marker_set
    markers_in_trace = trace_set & marker_set
    markers_prism_only = (markers_in_resolve | markers_in_trace) - markers_in_wilcox
    markers_wilcox_only = markers_in_wilcox - (markers_in_resolve | markers_in_trace)

    print(f"\n  Known markers recovered:")
    print(f"    Wilcoxon: {len(markers_in_wilcox)} - {', '.join(sorted(markers_in_wilcox)) if markers_in_wilcox else 'none'}")
    print(f"    PRISM-Resolve: {len(markers_in_resolve)} - {', '.join(sorted(markers_in_resolve)) if markers_in_resolve else 'none'}")
    print(f"    PRISM-Trace: {len(markers_in_trace)} - {', '.join(sorted(markers_in_trace)) if markers_in_trace else 'none'}")
    print(f"    PRISM-only markers: {', '.join(sorted(markers_prism_only)) if markers_prism_only else 'none'}")
    print(f"    Wilcoxon-only markers: {', '.join(sorted(markers_wilcox_only)) if markers_wilcox_only else 'none'}")

    # Overlap between resolve and trace (PRISM internal consistency)
    resolve_trace_overlap = resolve_set & trace_set
    resolve_trace_jaccard = jaccard(resolve_set, trace_set)
    print(f"\n  PRISM internal consistency (Resolve vs Trace):")
    print(f"    Overlap: {len(resolve_trace_overlap)} genes")
    print(f"    Jaccard: {resolve_trace_jaccard:.4f}")

    return {
        "wilcox_df": wilcox_df,
        "sig_wilcox": sig_wilcox,
        "n_wilcox_sig": len(sig_wilcox),
        "overlap_resolve": len(overlap_resolve),
        "overlap_trace": len(overlap_trace),
        "overlap_union": len(overlap_both),
        "jaccard_resolve": jaccard_resolve,
        "jaccard_trace": jaccard_trace,
        "jaccard_union": jaccard_union,
        "prism_only_count": len(prism_unique),
        "markers_wilcox": sorted(markers_in_wilcox),
        "markers_resolve": sorted(markers_in_resolve),
        "markers_trace": sorted(markers_in_trace),
        "markers_prism_only": sorted(markers_prism_only),
        "markers_wilcox_only": sorted(markers_wilcox_only),
        "resolve_trace_overlap": len(resolve_trace_overlap),
        "resolve_trace_jaccard": resolve_trace_jaccard,
    }


# ============================================================
# Main
# ============================================================
def main():
    start = time.time()
    config = load_config()

    print("=" * 60)
    print("Experiment 6: Discriminator Gene Validation")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("\nLoading data...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    embeddings = adata.obsm["X_prism"]
    print(f"  Loaded {adata.shape[0]} cells, {adata.shape[1]} genes")

    # ----------------------------------------------------------
    # Re-run PRISM-Resolve to get the 277 PIP>0.5 genes
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Re-running PRISM-Resolve (horseshoe DE)")
    print("=" * 60)

    labels = adata.obs["fate_int"].values
    label_mask = labels >= 2
    mixture = BayesianFateMixture(n_components=3, n_init=1, max_iter=100)
    mixture.fit(embeddings, labels, label_mask)
    fate_probs = mixture.predict_proba(embeddings)

    n_ecc = (fate_probs.argmax(1) == 1).sum()
    n_hair = (fate_probs.argmax(1) == 2).sum()
    n_unc = (fate_probs.argmax(1) == 0).sum()
    print(f"  Fate assignment: {n_ecc} eccrine, {n_hair} hair, {n_unc} uncommitted")

    # Horseshoe DE (fast)
    hvg_mask = adata.var["highly_variable"].values
    gene_names = adata.var_names[hvg_mask].tolist()
    X_hvg = adata[:, hvg_mask].X
    if sp.issparse(X_hvg):
        X_hvg = X_hvg.toarray()

    de = HorseshoeDE()
    eccrine_probs = fate_probs[:, 1]
    de_results = de.fit_fast(X_hvg, eccrine_probs, gene_names)

    resolve_all = de_results.sort_values("posterior_inclusion_prob", ascending=False)
    resolve_genes_pip05 = resolve_all[resolve_all["posterior_inclusion_prob"] > 0.5]["gene"].tolist()
    # For ranked list, use all genes sorted by PIP (for precision@K)
    resolve_ranked = resolve_all["gene"].tolist()
    print(f"  Resolve: {len(resolve_genes_pip05)} genes with PIP > 0.5")
    print(f"  Top 5: {resolve_genes_pip05[:5]}")

    # ----------------------------------------------------------
    # Re-run PRISM-Trace temporal correlation to get FDR<0.05 genes
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Re-running PRISM-Trace (temporal fate correlation)")
    print("=" * 60)

    adata = compute_pseudotime_pca(adata)
    pseudotime = adata.obs["dpt_pseudotime"].values

    corr_df = temporal_correlation(adata, pseudotime, fate_probs, gene_names)
    sig_corr = corr_df[corr_df["q_value"] < 0.05] if not corr_df.empty else corr_df
    trace_genes_fdr05 = sig_corr["gene"].tolist() if not sig_corr.empty else []
    # For ranked list, use all genes sorted by q-value
    trace_ranked = corr_df["gene"].tolist() if not corr_df.empty else []
    print(f"  Trace: {len(trace_genes_fdr05)} genes with FDR < 0.05")
    if trace_genes_fdr05:
        print(f"  Top 5: {trace_genes_fdr05[:5]}")

    # ----------------------------------------------------------
    # Part 1: GO Enrichment
    # ----------------------------------------------------------
    go_results = run_go_enrichment(resolve_genes_pip05, trace_genes_fdr05)

    # ----------------------------------------------------------
    # Part 2: Marker Recovery
    # ----------------------------------------------------------
    marker_results = run_marker_recovery(resolve_ranked, trace_ranked, resolve_genes_pip05, trace_genes_fdr05, gene_names)

    # ----------------------------------------------------------
    # Part 3: Differential Expression Validation
    # ----------------------------------------------------------
    de_validation = run_de_validation(adata, resolve_genes_pip05, trace_genes_fdr05)

    # ----------------------------------------------------------
    # Write results to results.md
    # ----------------------------------------------------------
    total_time = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Experiment 6 Complete ({total_time:.0f}s)")
    print(f"{'=' * 60}")

    result_text = f"""### Experiment 6: Discriminator Gene Validation
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}

**Overview**: Validates PRISM-identified discriminator genes through three
independent analyses: (1) GO enrichment, (2) marker recovery, (3) Wilcoxon DE comparison.

**Gene Sets**:
- PRISM-Resolve: {len(resolve_genes_pip05)} genes with PIP > 0.5
- PRISM-Trace: {len(trace_genes_fdr05)} genes with temporal fate correlation FDR < 0.05
- Gene universe: {len(gene_names)} HVGs

---

#### Part 1: GO Enrichment Analysis (g:Profiler, GO:BP)

"""

    for name in ["PRISM-Resolve (PIP>0.5)", "PRISM-Trace (FDR<0.05)"]:
        go_info = go_results.get(name, {})
        go_df = go_info.get("df", pd.DataFrame())
        skin_terms = go_info.get("skin_terms", [])

        result_text += f"**{name}**:\n"
        if go_df.empty:
            result_text += "- No significant GO:BP terms found.\n\n"
        else:
            result_text += f"- Total significant GO:BP terms: {len(go_df)}\n"
            result_text += f"- Skin/appendage-related terms: {len(skin_terms)}\n\n"

            result_text += "Top 10 GO:BP terms:\n\n"
            result_text += "| Rank | GO Term | p-value | Term Size |\n"
            result_text += "|------|---------|---------|----------|\n"
            for i, row in go_df.head(10).iterrows():
                result_text += f"| {i+1} | {row['name']} | {row['p_value']:.2e} | {row['term_size']} |\n"

            if skin_terms:
                result_text += f"\nSkin/appendage-related GO terms:\n\n"
                result_text += "| GO Term | p-value | Keyword |\n"
                result_text += "|---------|---------|--------|\n"
                for st in skin_terms[:10]:
                    result_text += f"| {st['term']} | {st['p_value']:.2e} | {st['keyword']} |\n"
            result_text += "\n"

    result_text += "---\n\n"
    result_text += "#### Part 2: Marker Recovery Analysis\n\n"
    result_text += f"Known markers tested: {', '.join(KNOWN_MARKERS)}\n\n"

    for name in ["PRISM-Resolve (PIP>0.5)", "PRISM-Trace (FDR<0.05)"]:
        mr = marker_results.get(name, {})
        if not mr:
            continue
        result_text += f"**{name}** ({mr['n_genes']} genes):\n"
        result_text += f"- Markers recovered: {mr['n_overlap']}/{len([m for m in KNOWN_MARKERS if m in gene_names])}\n"
        if mr['overlap_genes']:
            result_text += f"- Recovered: {', '.join(mr['overlap_genes'])}\n"

        result_text += "\n| K | Precision@K | Recall@K | Hits |\n"
        result_text += "|---|------------|----------|------|\n"
        for K in [5, 10, 20, 50, 100]:
            pk = mr['pr_at_k'].get(K, {})
            result_text += f"| {K} | {pk.get('precision', 0):.3f} | {pk.get('recall', 0):.3f} | {pk.get('hits', 0)} |\n"

        result_text += f"\n- Hypergeometric enrichment p-value: {mr['hypergeom_pval']:.2e}\n"
        result_text += f"- Expected overlap (by chance): {mr['expected_overlap']:.2f}\n"
        result_text += f"- Observed overlap: {mr['n_overlap']}\n"
        result_text += f"- Fold enrichment: {mr['fold_enrichment']:.2f}x\n\n"

    result_text += "---\n\n"
    result_text += "#### Part 3: Differential Expression Validation (Wilcoxon rank-sum)\n\n"

    if de_validation:
        result_text += f"Standard Wilcoxon rank-sum test: eccrine (n={(adata.obs['fate_int'].values == 2).sum()}) vs hair (n={(adata.obs['fate_int'].values == 3).sum()})\n"
        result_text += f"- Wilcoxon DE genes (FDR < 0.05): {de_validation['n_wilcox_sig']}\n\n"

        result_text += "**Overlap with PRISM gene sets**:\n\n"
        result_text += "| Comparison | Overlap | Jaccard |\n"
        result_text += "|-----------|---------|--------|\n"
        result_text += f"| PRISM-Resolve vs Wilcoxon | {de_validation['overlap_resolve']} | {de_validation['jaccard_resolve']:.4f} |\n"
        result_text += f"| PRISM-Trace vs Wilcoxon | {de_validation['overlap_trace']} | {de_validation['jaccard_trace']:.4f} |\n"
        result_text += f"| PRISM-Union vs Wilcoxon | {de_validation['overlap_union']} | {de_validation['jaccard_union']:.4f} |\n"
        result_text += f"| PRISM-Resolve vs PRISM-Trace | {de_validation['resolve_trace_overlap']} | {de_validation['resolve_trace_jaccard']:.4f} |\n"

        result_text += f"\n**Known marker recovery by method**:\n"
        result_text += f"- Wilcoxon: {len(de_validation['markers_wilcox'])} ({', '.join(de_validation['markers_wilcox']) if de_validation['markers_wilcox'] else 'none'})\n"
        result_text += f"- PRISM-Resolve: {len(de_validation['markers_resolve'])} ({', '.join(de_validation['markers_resolve']) if de_validation['markers_resolve'] else 'none'})\n"
        result_text += f"- PRISM-Trace: {len(de_validation['markers_trace'])} ({', '.join(de_validation['markers_trace']) if de_validation['markers_trace'] else 'none'})\n"
        result_text += f"- PRISM-only markers (not in Wilcoxon): {', '.join(de_validation['markers_prism_only']) if de_validation['markers_prism_only'] else 'none'}\n"
        result_text += f"- Wilcoxon-only markers (not in PRISM): {', '.join(de_validation['markers_wilcox_only']) if de_validation['markers_wilcox_only'] else 'none'}\n"
        result_text += f"- PRISM-unique genes (not in Wilcoxon): {de_validation['prism_only_count']}\n"

        # Top Wilcoxon DE genes
        sig_wilcox = de_validation.get("sig_wilcox", pd.DataFrame())
        if not sig_wilcox.empty:
            result_text += f"\nTop 15 Wilcoxon DE genes:\n\n"
            result_text += "| Gene | log2FC | Direction | q-value |\n"
            result_text += "|------|--------|-----------|--------|\n"
            for _, row in sig_wilcox.head(15).iterrows():
                direction = "eccrine_up" if row["log2fc"] > 0 else "hair_up"
                result_text += f"| {row['gene']} | {row['log2fc']:.3f} | {direction} | {row['q_value']:.2e} |\n"
    else:
        result_text += "Insufficient cells for Wilcoxon DE analysis.\n"

    result_text += f"\n---\n\nTotal Experiment 6 time: {total_time:.0f}s ({total_time/60:.1f} min)\n"

    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(f"\n\n---\n\n")
        f.write(result_text)

    print(f"\nResults appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
