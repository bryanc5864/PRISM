#!/usr/bin/env python
"""Preprocess corpus datasets for PCP pre-training.

Steps:
1. Build gene vocabulary (top N genes by dataset coverage)
2. For each dataset: select vocabulary genes, rank-value encode, save as .npy
3. Extract perturbation metadata and save
4. Build global corpus index

Usage:
    python scripts/preprocess_corpus.py
    python scripts/preprocess_corpus.py --n-genes 2000 --skip-lamanno --skip-cao
"""

import argparse
import os
import sys
import json
import re
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.pretrain.vocab import build_gene_vocabulary, load_gene_vocabulary


CORPUS_DIR = "data/corpus"
PROCESSED_DIR = "data/corpus/processed"

# Datasets to skip (non-h5ad or problematic)
DEFAULT_SKIP = [
    "cao_organogenesis.txt.gz",
    "lamanno_developing_brain.h5ad",
    "lamanno_developing_brain.loom",
]

# Patterns for identifying control cells
CONTROL_PATTERNS = re.compile(
    r"(^control$|^non-targeting$|^DMSO$|^unperturbed$|^vehicle$|^nt$|^NT$|"
    r"^nontargeting$|^negative.control$|^ctrl$|^Ctrl$)",
    re.IGNORECASE,
)


def rank_value_encode(X, n_bins=51):
    """Rank-value encode expression matrix.

    Args:
        X: (n_cells, n_genes) dense float expression matrix
        n_bins: Number of expression bins (0=unexpressed, 1-50=expression)

    Returns:
        encoded: (n_cells, n_genes) uint8 array
    """
    encoded = np.zeros(X.shape, dtype=np.uint8)

    for i in range(X.shape[0]):
        row = X[i]
        nonzero_mask = row > 0
        if nonzero_mask.sum() == 0:
            continue

        nonzero_vals = row[nonzero_mask]
        ranks = np.argsort(np.argsort(nonzero_vals)) + 1
        max_rank = ranks.max()
        if max_rank > 0:
            bins = np.clip(
                (ranks / max_rank * (n_bins - 1)).astype(np.int64) + 1,
                1, n_bins - 1,
            ).astype(np.uint8)
            encoded[i, nonzero_mask] = bins

    return encoded


def infer_control(perturbation_values):
    """Infer which cells are controls from perturbation labels."""
    is_control = np.zeros(len(perturbation_values), dtype=bool)
    for i, val in enumerate(perturbation_values):
        if CONTROL_PATTERNS.match(str(val)):
            is_control[i] = True
    return is_control


def get_perturbation_column(adata):
    """Find and return perturbation labels from an AnnData object."""
    # Try common column names
    candidates = ["perturbation", "gene", "guide_id", "condition", "treatment",
                   "perturbation_name", "sgRNA", "target_gene"]
    for col in candidates:
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).values

    # Fallback: use "unknown" for all cells
    print("    WARNING: No perturbation column found, using 'unknown'")
    return np.array(["unknown"] * adata.shape[0])


def process_dataset(h5ad_path, gene_vocab, gene_to_idx, n_genes, n_bins, output_dir):
    """Process a single h5ad dataset.

    Returns:
        dict with dataset_id, n_cells, n_perturbations, or None on failure
    """
    import anndata as ad

    fname = os.path.basename(h5ad_path)
    dataset_id = fname.replace(".h5ad", "")

    print(f"\n  Processing {dataset_id}...")

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"    FAILED to load: {e}")
        return None

    n_cells = adata.shape[0]
    print(f"    {n_cells:,} cells, {adata.shape[1]} genes")

    # Map dataset genes to vocabulary
    dataset_genes = adata.var_names.tolist()
    vocab_gene_indices = []  # indices into adata's gene axis
    vocab_positions = []  # positions in our vocabulary

    for i, gene in enumerate(gene_vocab):
        if gene in dataset_genes:
            vocab_gene_indices.append(dataset_genes.index(gene))
            vocab_positions.append(i)

    n_matched = len(vocab_positions)
    print(f"    Matched {n_matched}/{n_genes} vocabulary genes ({n_matched/n_genes*100:.1f}%)")

    if n_matched < n_genes * 0.3:
        print(f"    SKIP: Too few genes matched (<30%)")
        return None

    # Extract expression for vocabulary genes only (memory-efficient for sparse)
    X = adata.X
    aligned = np.zeros((n_cells, n_genes), dtype=np.float32)

    if sp.issparse(X):
        # Extract only vocabulary columns from sparse matrix
        vocab_gene_indices_arr = np.array(vocab_gene_indices)
        X_subset = X[:, vocab_gene_indices_arr].toarray().astype(np.float32)
        for i, vocab_pos in enumerate(vocab_positions):
            aligned[:, vocab_pos] = X_subset[:, i]
        del X_subset
    else:
        for vocab_pos, adata_idx in zip(vocab_positions, vocab_gene_indices):
            aligned[:, vocab_pos] = X[:, adata_idx]

    # Rank-value encode
    print(f"    Rank-value encoding...", end=" ", flush=True)
    encoded = rank_value_encode(aligned, n_bins=n_bins)
    print(f"done. Shape: {encoded.shape}, dtype: {encoded.dtype}")

    # Extract perturbation metadata
    perturbations = get_perturbation_column(adata)
    unique_perts = sorted(set(perturbations))
    pert_to_id = {p: i for i, p in enumerate(unique_perts)}
    perturbation_ids = np.array([pert_to_id[p] for p in perturbations], dtype=np.int32)

    is_control = infer_control(perturbations)
    n_control = is_control.sum()
    print(f"    {len(unique_perts)} unique perturbations, {n_control} control cells "
          f"({n_control/n_cells*100:.1f}%)")

    # Save
    enc_path = os.path.join(output_dir, f"{dataset_id}_encoded.npy")
    np.save(enc_path, encoded)

    meta_path = os.path.join(output_dir, f"{dataset_id}_meta.npz")
    np.savez(
        meta_path,
        perturbation_ids=perturbation_ids,
        is_control=is_control,
        perturbation_vocab=np.array(unique_perts),
    )

    size_mb = os.path.getsize(enc_path) / (1024**2)
    print(f"    Saved: {enc_path} ({size_mb:.1f} MB)")

    del adata, X, aligned
    return {
        "dataset_id": dataset_id,
        "n_cells": n_cells,
        "n_genes_matched": n_matched,
        "n_perturbations": len(unique_perts),
        "n_control": int(n_control),
        "encoded_path": enc_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess corpus for PCP pre-training")
    parser.add_argument("--n-genes", type=int, default=2000, help="Vocabulary size")
    parser.add_argument("--n-bins", type=int, default=51, help="Expression bins")
    parser.add_argument("--skip-lamanno", action="store_true", default=True,
                        help="Skip La Manno (62GB dense)")
    parser.add_argument("--skip-cao", action="store_true", default=True,
                        help="Skip Cao (text format)")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Build skip list
    skip_files = list(DEFAULT_SKIP)

    # Step 1: Build gene vocabulary
    print("=" * 60)
    print("  STEP 1: Building gene vocabulary")
    print("=" * 60)

    vocab_path = os.path.join(PROCESSED_DIR, "gene_vocab.json")
    if os.path.exists(vocab_path):
        print(f"Loading existing vocabulary from {vocab_path}")
        vocab_data = load_gene_vocabulary(vocab_path)
        gene_vocab = vocab_data["genes"][:args.n_genes]
    else:
        gene_vocab = build_gene_vocabulary(
            CORPUS_DIR,
            n_genes=args.n_genes,
            exclude_datasets=skip_files,
            output_path=vocab_path,
        )

    gene_to_idx = {g: i for i, g in enumerate(gene_vocab)}
    n_genes = len(gene_vocab)
    print(f"Vocabulary: {n_genes} genes")

    # Step 2: Process each dataset
    print("\n" + "=" * 60)
    print("  STEP 2: Processing datasets")
    print("=" * 60)

    dataset_infos = []

    for tier_dir in ["tier1", "tier2", "tier3"]:
        tier_path = os.path.join(CORPUS_DIR, tier_dir)
        if not os.path.isdir(tier_path):
            continue

        print(f"\n--- {tier_dir.upper()} ---")

        for fname in sorted(os.listdir(tier_path)):
            if not fname.endswith(".h5ad"):
                continue
            if fname in skip_files:
                print(f"  Skipping {fname}")
                continue

            # Check if already processed
            dataset_id = fname.replace(".h5ad", "")
            enc_path = os.path.join(PROCESSED_DIR, f"{dataset_id}_encoded.npy")
            if os.path.exists(enc_path):
                # Load existing info
                meta_path = os.path.join(PROCESSED_DIR, f"{dataset_id}_meta.npz")
                if os.path.exists(meta_path):
                    enc = np.load(enc_path, mmap_mode="r")
                    meta = np.load(meta_path, allow_pickle=True)
                    info = {
                        "dataset_id": dataset_id,
                        "n_cells": enc.shape[0],
                        "n_genes_matched": n_genes,
                        "n_perturbations": len(meta["perturbation_vocab"]),
                        "n_control": int(meta["is_control"].sum()),
                        "encoded_path": enc_path,
                    }
                    dataset_infos.append(info)
                    print(f"  {dataset_id}: already processed ({info['n_cells']:,} cells)")
                    continue

            fpath = os.path.join(tier_path, fname)
            info = process_dataset(
                fpath, gene_vocab, gene_to_idx, n_genes, args.n_bins, PROCESSED_DIR
            )
            if info:
                dataset_infos.append(info)

    # Step 3: Build corpus index
    print("\n" + "=" * 60)
    print("  STEP 3: Building corpus index")
    print("=" * 60)

    total_cells = sum(d["n_cells"] for d in dataset_infos)
    total_perts = sum(d["n_perturbations"] for d in dataset_infos)

    index = {
        "n_genes": n_genes,
        "n_bins": args.n_bins,
        "total_cells": total_cells,
        "total_perturbations": total_perts,
        "n_datasets": len(dataset_infos),
        "datasets": dataset_infos,
    }

    index_path = os.path.join(PROCESSED_DIR, "corpus_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nCorpus ready:")
    print(f"  Datasets: {len(dataset_infos)}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total perturbations: {total_perts}")
    print(f"  Gene vocabulary: {n_genes}")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    main()
