#!/usr/bin/env python3
"""Benchmark scGPT zero-shot embeddings on PRISM data.

Implements a minimal scGPT model loader that bypasses broken torchtext
dependency. Loads pretrained weights directly and extracts cell embeddings.
"""

import os, sys, time, json, math
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LD_LIBRARY_PATH
import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Mock flash_attn
import types
for mod_name in ["flash_attn", "flash_attn.flash_attn_func",
                  "flash_attn.flash_attn_interface",
                  "flash_attn.bert_padding", "flash_attn.flash_attn_triton"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

# Mock torchtext so scgpt can import
torchtext_mock = types.ModuleType("torchtext")
torchtext_vocab_mock = types.ModuleType("torchtext.vocab")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class SimpleVocab:
    """Minimal vocabulary replacement for torchtext.vocab.Vocab."""
    def __init__(self, token2idx):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}

    def __getitem__(self, token):
        return self.token2idx.get(token, 0)

    def __contains__(self, token):
        return token in self.token2idx

    def __len__(self):
        return len(self.token2idx)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(json.load(f))


def get_mouse_to_human_orthologs(mouse_genes):
    """Map mouse gene symbols to human gene symbols."""
    import mygene
    mg = mygene.MyGeneInfo()

    print(f"  Querying mygene for {len(mouse_genes)} mouse genes...")
    results = mg.querymany(
        mouse_genes, scopes="symbol", fields="homologene",
        species="mouse", returnall=True,
    )

    mouse_to_human_entrez = {}
    human_entrez_ids = set()
    for hit in results.get("out", []):
        query = hit.get("query", "")
        if "homologene" in hit and "genes" in hit["homologene"]:
            for entry in hit["homologene"]["genes"]:
                if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 9606:
                    eid = str(entry[1])
                    mouse_to_human_entrez[query] = eid
                    human_entrez_ids.add(eid)
                    break

    print(f"  Found {len(mouse_to_human_entrez)} homologs")

    if human_entrez_ids:
        human_results = mg.querymany(
            list(human_entrez_ids), scopes="entrezgene", fields="symbol",
            species="human", returnall=True,
        )
        entrez_to_sym = {}
        for hit in human_results.get("out", []):
            if "symbol" in hit:
                entrez_to_sym[hit.get("query", "")] = hit["symbol"]
        print(f"  Resolved {len(entrez_to_sym)} to symbols")
    else:
        entrez_to_sym = {}

    mapping = {}
    for mg, eid in mouse_to_human_entrez.items():
        if eid in entrez_to_sym:
            mapping[mg] = entrez_to_sym[eid]
    for gene in mouse_genes:
        if gene not in mapping:
            mapping[gene] = gene.upper()

    return mapping


# ===== Minimal scGPT TransformerModel =====

class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_value=512):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return self.norm(x)


class ScGPTModel(nn.Module):
    """Minimal scGPT model for embedding extraction."""

    def __init__(self, ntoken, d_model=512, nhead=8, d_hid=512,
                 nlayers=12, dropout=0.2, pad_token_id=0, pad_value=-2):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value

        # Gene token embedding
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=pad_token_id)

        # Value encoder (continuous)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, gene_ids, values, src_key_padding_mask=None):
        """
        Args:
            gene_ids: (B, seq_len) gene token IDs
            values: (B, seq_len) expression values
        Returns:
            (B, d_model) cell embeddings (mean pooled)
        """
        gene_emb = self.encoder(gene_ids) * math.sqrt(self.d_model)
        val_emb = self.value_encoder(values)
        total_emb = gene_emb + val_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = values.eq(self.pad_value)

        output = self.transformer_encoder(total_emb, src_key_padding_mask=src_key_padding_mask)

        # Mean pool (excluding padding)
        mask = ~src_key_padding_mask
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (output * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        cell_emb = summed / counts

        return cell_emb


def load_scgpt_model(model_dir, device):
    """Load scGPT pretrained model with proper key mapping."""
    with open(os.path.join(model_dir, "args.json")) as f:
        args = json.load(f)

    vocab = SimpleVocab.from_file(os.path.join(model_dir, "vocab.json"))

    model = ScGPTModel(
        ntoken=len(vocab),
        d_model=args.get("embsize", 512),
        nhead=args.get("nheads", 8),
        d_hid=args.get("d_hid", 512),
        nlayers=args.get("nlayers", 12),
        dropout=0.0,
        pad_token_id=vocab["<pad>"],
        pad_value=args.get("pad_value", -2),
    )

    state_dict = torch.load(
        os.path.join(model_dir, "best_model.pt"),
        map_location="cpu", weights_only=False,
    )

    # Map scGPT key names to our model's key names
    mapped_state = {}
    for k, v in state_dict.items():
        new_k = k
        # encoder.embedding.weight → encoder.weight
        new_k = new_k.replace("encoder.embedding.weight", "encoder.weight")
        # self_attn.Wqkv → self_attn.in_proj (PyTorch fused QKV)
        new_k = new_k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
        new_k = new_k.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
        mapped_state[new_k] = v

    matched = model.load_state_dict(mapped_state, strict=False)
    n_loaded = len(state_dict) - len(matched.unexpected_keys)
    print(f"  Loaded {n_loaded}/{len(state_dict)} weights "
          f"({len(matched.unexpected_keys)} unexpected, {len(matched.missing_keys)} missing)")
    if matched.missing_keys:
        print(f"  Missing: {matched.missing_keys[:5]}...")

    model.eval()
    model = model.to(device)
    return model, vocab, args


def binning(row, n_bins=51):
    """Bin expression values into discrete tokens (scGPT style)."""
    nonzero = row > 0
    if nonzero.sum() == 0:
        return np.zeros_like(row)
    result = np.zeros_like(row)
    vals = row[nonzero]
    # Rank-based binning
    ranks = np.argsort(np.argsort(vals)) + 1
    max_rank = ranks.max()
    bins = np.clip((ranks / max_rank * (n_bins - 1)).astype(int) + 1, 1, n_bins - 1)
    result[nonzero] = bins
    return result


def main():
    start = time.time()

    print("Loading data...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    labels = adata.obs["fate_int"].values
    print(f"  {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Load vocab
    model_dir = os.path.join(PROJECT_DIR, "models", "scGPT_human", "scGPT_human")
    vocab = SimpleVocab.from_file(os.path.join(model_dir, "vocab.json"))
    print(f"  scGPT vocab: {len(vocab)} genes")

    # Mouse → human mapping
    mouse_to_human = get_mouse_to_human_orthologs(adata.var_names.tolist())

    # Map to vocab
    gene_to_vid = {}
    for gene in adata.var_names:
        human = mouse_to_human.get(gene, gene.upper())
        if human in vocab:
            gene_to_vid[gene] = vocab[human]

    print(f"  Genes mapped: {len(gene_to_vid)}/{len(adata.var_names)}")
    for g in ["Trp63", "Lgr6", "Sox9", "Tfap2b", "Krt14", "Edar"]:
        human = mouse_to_human.get(g, g.upper())
        print(f"    {g} → {human}: {'YES' if human in vocab else 'NO'}")

    # Prepare expression matrix
    mapped_genes = [g for g in adata.var_names if g in gene_to_vid]
    mask = [g in gene_to_vid for g in adata.var_names]
    vid_array = np.array([gene_to_vid[g] for g in mapped_genes])

    X = adata[:, mask].X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Limit to top 1200 genes per cell (scGPT max_seq_len)
    max_seq_len = 1200
    if X.shape[1] > max_seq_len:
        # Keep top expressed genes per cell (global selection for batch efficiency)
        gene_means = X.mean(axis=0)
        top_idx = np.argsort(gene_means)[-max_seq_len:]
        X = X[:, top_idx]
        vid_array = vid_array[top_idx]
        print(f"  Trimmed to top {max_seq_len} genes")

    # Bin expression values
    print("  Binning expression values...")
    X_binned = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        X_binned[i] = binning(X[i])

    print(f"  Final matrix: {X_binned.shape}")

    # Load model
    print("\nLoading scGPT model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, model_args = load_scgpt_model(model_dir, device)
    print(f"  Model on {device}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    batch_size = 64
    gene_ids_base = torch.tensor(vid_array, dtype=torch.long)
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(X_binned), batch_size):
            end = min(i + batch_size, len(X_binned))
            batch_vals = torch.tensor(X_binned[i:end], dtype=torch.float32).to(device)
            batch_genes = gene_ids_base.unsqueeze(0).expand(end - i, -1).to(device)

            cell_emb = model(batch_genes, batch_vals)
            all_embeddings.append(cell_emb.cpu().numpy())

            if i % (batch_size * 100) == 0:
                print(f"  {i}/{len(X_binned)} cells...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Embedding shape: {embeddings.shape}")

    # Check for NaN/Inf
    valid_mask = np.isfinite(embeddings).all(axis=1)
    if not valid_mask.all():
        print(f"  WARNING: {(~valid_mask).sum()} cells have NaN/Inf embeddings")
        embeddings[~valid_mask] = 0

    # Compute metrics
    print("\nComputing metrics...")
    eval_mask = labels >= 2
    X_eval = embeddings[eval_mask]
    y_eval = (labels[eval_mask] == 2).astype(int)
    print(f"  Eval set: {eval_mask.sum()} cells ({(y_eval==1).sum()} eccrine, {(y_eval==0).sum()} hair)")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_eval, y_eval, cv=5, scoring="roc_auc")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_eval, y_eval, cv=5, scoring="roc_auc")

    total_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"scGPT Zero-Shot Results")
    print(f"{'='*60}")
    print(f"  Genes mapped: {len(gene_to_vid)}")
    print(f"  RF AUROC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  LR AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  Time: {total_time:.0f}s")

    # Write results
    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(f"\n\n---\n\n### scGPT Zero-Shot Benchmark\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Model: scGPT whole-human pretrained (512d, 12 layers)\n")
        f.write(f"- Genes mapped (mouse→human): {len(gene_to_vid)}\n")
        f.write(f"- **RF AUROC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}**\n")
        f.write(f"- **LR AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}**\n")
        f.write(f"- Time: {total_time:.0f}s\n")

    print(f"Results appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
