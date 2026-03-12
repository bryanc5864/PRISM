# PRISM: Progenitor Resolution via Invariance-Sensitive Modeling

PRISM is a contrastive-learning framework for resolving transcriptionally cryptic cell-fate decisions in single-cell RNA-seq data. Validated across **15 biological systems** spanning developmental, hematopoietic, and cancer contexts, it separates progenitor populations that are indistinguishable by standard methods (PCA, Harmony, scGPT, Geneformer).

**Author:** Bryan Cheng

## Three-Stage Pipeline

1. **PRISM-Encode** -- Hard-negative contrastive learning with a Transformer encoder (12L/512d), scGPT-initialized gene vocabulary, condition-aware sampling, and reconstruction regularization. Produces cell embeddings in a discriminative latent space.
2. **PRISM-Resolve** -- Bayesian Gaussian mixture model with semi-supervised anchors assigns fate probabilities; horseshoe-prior differential expression ranks discriminator genes by posterior inclusion probability.
3. **PRISM-Trace** -- Hybrid DPT pseudotime (computed in PCA space, branch assignment via PRISM fate probabilities) followed by GAM-based gene-cascade analysis.

## Repository Layout

```
PRISM/
  prism/                 # Core library (encoder, data, resolve, trace, training)
  configs/               # Per-system YAML configs (15 systems + default)
  scripts/               # Evaluation, figure generation, corpus download
  data/                  # Raw + processed AnnData objects
  checkpoints/           # Trained model weights + PCP pre-training
  figures/               # Generated plots + publication figures
  presentation/          # Reveal.js presentation + PDF + speaker notes
  experiments/           # Experiment runner scripts
  benchmarks/            # Baseline comparison scripts
  run_prism.py           # Main entry point
```

## Quick Start

```bash
pip install -e .

# Run on a specific system
prism-run --system configs/skin.yaml --stage all

# Or via Python API
import prism
model = prism.PRISM(adata, condition_key='genotype')
model.fit()
model.resolve()
```

## Key Results

### 15-System Evaluation (7 Methods, 19 Metrics)

| System | GEO | Cells | RF AUROC | Winner |
|--------|-----|------:|:--------:|--------|
| Skin | GSE220977 | 25,344 | **0.908** | PRISM |
| Pancreas | GSE132188 | 33,896 | **0.989** | PRISM |
| Cortex | GSE153164 | 78,129 | **0.991** | PRISM |
| HSC | GSE140802 | 75,339 | **0.999** | PRISM |
| Cardiac | GSE126128 | 28,479 | **0.999** | PRISM |
| Lung | GSE149563 | 82,040 | **0.997** | PRISM |
| T Helper | GSE160055 | 44,175 | **0.998** | PRISM |
| GBM | GSE131928 | 19,855 | **0.997** | PRISM |
| Paul HSC | paul15 | 8,944 | **0.990** | PRISM |
| Sade-Feldman | GSE120575 | 16,291 | **0.987** | PRISM |
| Melanoma | GSE72056 | 4,645 | **0.939** | PRISM |
| Intestine | GSE92332 | 7,216 | **0.834** | PRISM |
| Neural Crest | GSE129845 | 6,043 | 0.767 | PCA |
| Oligo | GSE75330 | 4,997 | 0.784 | PCA |
| Nestorowa | nestorowa16 | 1,165 | 0.573 | PCA |

**PRISM wins RF AUROC on 12/15 systems.** Mean RF AUROC = 0.917 across all 15 systems.

### Pre-training (PCP)

scGPT-initialized Perturbation-Contrastive Pre-training on 23 datasets (~7.5M cells) improves mean RF AUROC from 0.830 to 0.962 (+0.131), with largest gains on weaker systems (cardiac +0.358, intestine +0.427).

### Foundation Model Comparison

PRISM outperforms scGPT (33M params) and Geneformer (10M params) zero-shot embeddings on 13/15 systems. Mean advantage: +0.086 RF AUROC over best foundation model.

### Top Discriminator Genes

- **Skin**: Tfap2b, Trp63, Lgr6 (eccrine vs hair, 252 genes)
- **Pancreas**: Nkx6-1, Pdx1, Arx (endocrine fates, 1,282 genes)
- **HSC**: Hba-a2, Prdx2, Fcer1g (lineage priming, 703 genes)
- **Cortex**: Fezf2, Tbr1, Cux1, Satb2 (cortical layers, 1,159 genes)

## Requirements

Python >= 3.9, PyTorch >= 2.0, 4x NVIDIA A100 GPUs (for training).
See `requirements.txt` for the full dependency list.
