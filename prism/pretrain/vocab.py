"""Gene vocabulary builder for PCP pre-training.

Builds a common gene vocabulary across all corpus datasets by selecting
genes that appear in the most datasets. This ensures broad coverage and
consistent gene indexing during training.
"""

import os
import json
from collections import Counter
from typing import List, Dict, Optional
import anndata as ad


def build_gene_vocabulary(
    corpus_dir: str,
    n_genes: int = 2000,
    exclude_datasets: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> List[str]:
    """Build a common gene vocabulary from corpus h5ad files.

    Strategy: select genes that appear in the most datasets, then
    rank by total cell coverage within those datasets.

    Args:
        corpus_dir: Path to corpus directory with tier1/tier2/tier3 subdirs
        n_genes: Number of genes to include in vocabulary
        exclude_datasets: Dataset filenames to skip
        output_path: Path to save vocabulary JSON

    Returns:
        Ordered list of gene names (vocabulary)
    """
    exclude = set(exclude_datasets or [])

    # Scan all h5ad files
    gene_dataset_count = Counter()  # gene -> number of datasets it appears in
    gene_cell_count = Counter()  # gene -> total cells across datasets
    dataset_count = 0

    for tier_dir in ["tier1", "tier2", "tier3"]:
        tier_path = os.path.join(corpus_dir, tier_dir)
        if not os.path.isdir(tier_path):
            continue

        for fname in sorted(os.listdir(tier_path)):
            if not fname.endswith(".h5ad"):
                continue
            if fname in exclude:
                continue

            fpath = os.path.join(tier_path, fname)
            print(f"  Scanning {fname}...", end=" ", flush=True)

            try:
                adata = ad.read_h5ad(fpath, backed="r")
                genes = adata.var_names.tolist()
                n_cells = adata.shape[0]
                adata.file.close()

                for g in set(genes):  # unique genes per dataset
                    gene_dataset_count[g] += 1
                    gene_cell_count[g] += n_cells

                dataset_count += 1
                print(f"{len(genes)} genes, {n_cells:,} cells")

            except Exception as e:
                print(f"SKIP ({e})")
                continue

    print(f"\nScanned {dataset_count} datasets, {len(gene_dataset_count)} unique genes")

    # Select top genes: primary sort by dataset count, secondary by cell count
    ranked_genes = sorted(
        gene_dataset_count.keys(),
        key=lambda g: (gene_dataset_count[g], gene_cell_count[g]),
        reverse=True,
    )

    vocab = ranked_genes[:n_genes]

    # Report coverage
    min_datasets = gene_dataset_count[vocab[-1]] if vocab else 0
    print(f"Selected {len(vocab)} genes (min dataset coverage: {min_datasets}/{dataset_count})")

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vocab_data = {
            "genes": vocab,
            "n_genes": len(vocab),
            "n_datasets_scanned": dataset_count,
            "min_dataset_coverage": min_datasets,
            "gene_to_idx": {g: i for i, g in enumerate(vocab)},
        }
        with open(output_path, "w") as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Saved vocabulary to {output_path}")

    return vocab


def load_gene_vocabulary(vocab_path: str) -> Dict:
    """Load a saved gene vocabulary."""
    with open(vocab_path) as f:
        return json.load(f)
