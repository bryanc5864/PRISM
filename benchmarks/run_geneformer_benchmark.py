#!/usr/bin/env python3
"""Benchmark Geneformer zero-shot embeddings on PRISM data.

Minimal BERT implementation bypassing broken transformers BertModel import.
Loads pretrained weights directly from safetensors/pytorch_model.bin.
"""

import os, sys, time, pickle, math, json
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# ===== Minimal BERT Model =====

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position, type_vocab_size=2,
                 layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_len = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.02):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        return context


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="relu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.ReLU() if hidden_act == "relu" else nn.GELU()

    def forward(self, hidden_states):
        return self.act(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size,
                 hidden_act="relu", layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.attention = nn.ModuleDict({
            "self": BertSelfAttention(hidden_size, num_heads, dropout),
            "output": BertSelfOutput(hidden_size, layer_norm_eps, dropout),
        })
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size, layer_norm_eps, dropout)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.attention["self"](hidden_states, attention_mask)
        attention_output = self.attention["output"](self_output, hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MinimalBert(nn.Module):
    """Minimal BERT encoder matching HuggingFace BertForMaskedLM layout."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = nn.ModuleDict({
            "embeddings": BertEmbeddings(
                config["vocab_size"], config["hidden_size"],
                config["max_position_embeddings"], config.get("type_vocab_size", 2),
                config.get("layer_norm_eps", 1e-12), config.get("hidden_dropout_prob", 0.02),
            ),
            "encoder": nn.ModuleDict({
                "layer": nn.ModuleList([
                    BertLayer(
                        config["hidden_size"], config["num_attention_heads"],
                        config["intermediate_size"], config.get("hidden_act", "relu"),
                        config.get("layer_norm_eps", 1e-12), config.get("hidden_dropout_prob", 0.02),
                    )
                    for _ in range(config["num_hidden_layers"])
                ])
            })
        })

    def forward(self, input_ids, attention_mask=None):
        """
        Returns: (B, hidden_size) mean-pooled cell embeddings
        """
        # Prepare attention mask: (B, 1, 1, seq_len) with -10000 for padding
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :].float()
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        hidden = self.bert["embeddings"](input_ids)

        for layer in self.bert["encoder"]["layer"]:
            hidden = layer(hidden, extended_mask)

        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = hidden.mean(dim=1)

        return pooled, hidden


def load_geneformer_model(model_dir, device):
    """Load Geneformer pretrained weights into MinimalBert."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    print(f"  Config: {config['hidden_size']}d, {config['num_hidden_layers']} layers, "
          f"{config['num_attention_heads']} heads, vocab={config['vocab_size']}")

    model = MinimalBert(config)

    # Load weights
    model_file = os.path.join(model_dir, "pytorch_model.bin")
    safetensors_file = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    elif os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file)
    else:
        raise FileNotFoundError(f"No model weights in {model_dir}")

    # Map HuggingFace BERT keys to our structure
    mapped_state = {}
    for k, v in state_dict.items():
        # Skip cls/predictions heads (we only need encoder)
        if k.startswith("cls.") or k.startswith("predictions."):
            continue
        mapped_state[k] = v

    result = model.load_state_dict(mapped_state, strict=False)
    n_loaded = len(mapped_state) - len(result.unexpected_keys)
    print(f"  Loaded {n_loaded}/{len(mapped_state)} weights "
          f"({len(result.unexpected_keys)} unexpected, {len(result.missing_keys)} missing)")
    if result.missing_keys:
        print(f"  Missing: {result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"  Unexpected: {result.unexpected_keys[:5]}...")

    model.eval()
    model = model.to(device)
    return model


def get_mouse_to_human_ensembl(mouse_genes, gf_token_dict, gf_gene_name_id):
    """Map mouse gene symbols to human Ensembl IDs in Geneformer vocab."""
    human_upper_to_ensembl = {}
    for name, ens_id in gf_gene_name_id.items():
        if ens_id in gf_token_dict:
            human_upper_to_ensembl[name.upper()] = ens_id

    mapping = {}

    # Direct uppercase match
    for gene in mouse_genes:
        upper = gene.upper()
        if upper in human_upper_to_ensembl:
            mapping[gene] = human_upper_to_ensembl[upper]

    remaining = [g for g in mouse_genes if g not in mapping]
    print(f"  Direct match: {len(mapping)}/{len(mouse_genes)}")

    # mygene ortholog lookup
    if remaining:
        import mygene
        mg = mygene.MyGeneInfo()

        print(f"  Querying mygene for {len(remaining)} remaining mouse genes...")
        results = mg.querymany(
            remaining, scopes="symbol", fields="homologene",
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

        if human_entrez_ids:
            human_results = mg.querymany(
                list(human_entrez_ids), scopes="entrezgene", fields="symbol",
                species="human", returnall=True,
            )
            entrez_to_sym = {}
            for hit in human_results.get("out", []):
                if "symbol" in hit:
                    entrez_to_sym[hit.get("query", "")] = hit["symbol"]

            for gene, eid in mouse_to_human_entrez.items():
                if eid in entrez_to_sym:
                    sym = entrez_to_sym[eid]
                    if sym.upper() in human_upper_to_ensembl:
                        mapping[gene] = human_upper_to_ensembl[sym.upper()]

    print(f"  Total mapped: {len(mapping)}/{len(mouse_genes)}")
    return mapping


def tokenize_cell(expression, gene_ensembl_ids, gene_medians, token_dict, max_len=2048):
    """Tokenize one cell: normalize by median, rank, convert to token IDs."""
    nonzero = expression > 0
    if nonzero.sum() == 0:
        return np.array([token_dict.get("<pad>", 0)], dtype=np.int64)

    nz_expr = expression[nonzero]
    nz_ensembl = gene_ensembl_ids[nonzero]

    medians = np.array([gene_medians.get(eid, 1.0) for eid in nz_ensembl])
    normalized = nz_expr / np.clip(medians, 1e-6, None)

    rank_order = np.argsort(-normalized)[:max_len]

    token_ids = np.array([token_dict[nz_ensembl[i]] for i in rank_order], dtype=np.int64)
    return token_ids


def main():
    start = time.time()

    print("Loading data...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    labels = adata.obs["fate_int"].values
    print(f"  {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Load Geneformer dictionaries
    gene_dict_dir = os.path.join(PROJECT_DIR, "models", "Geneformer", "geneformer", "gene_dictionaries_30m")

    with open(os.path.join(gene_dict_dir, "token_dictionary_gc30M.pkl"), "rb") as f:
        token_dict = pickle.load(f)
    with open(os.path.join(gene_dict_dir, "gene_name_id_dict_gc30M.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)
    with open(os.path.join(gene_dict_dir, "gene_median_dictionary_gc30M.pkl"), "rb") as f:
        gene_medians = pickle.load(f)

    print(f"  Geneformer vocab: {len(token_dict)} tokens")

    # Map mouse genes to human Ensembl IDs
    mouse_to_ensembl = get_mouse_to_human_ensembl(
        adata.var_names.tolist(), token_dict, gene_name_id
    )

    for g in ["Trp63", "Lgr6", "Sox9", "Tfap2b", "Krt14", "Edar"]:
        ens = mouse_to_ensembl.get(g, None)
        tid = token_dict.get(ens, None) if ens else None
        print(f"    {g} → {ens}: token={tid}")

    # Filter to mapped genes
    mapped_genes = [g for g in adata.var_names if g in mouse_to_ensembl]
    mask = np.array([g in mouse_to_ensembl for g in adata.var_names])
    ensembl_ids = np.array([mouse_to_ensembl[g] for g in mapped_genes])
    print(f"  Using {len(mapped_genes)} mapped genes")

    X = adata[:, mask].X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Tokenize all cells
    print("\nTokenizing cells...")
    max_len = 2048
    all_tokens = []
    token_lengths = []

    for i in range(X.shape[0]):
        tokens = tokenize_cell(X[i], ensembl_ids, gene_medians, token_dict, max_len)
        all_tokens.append(tokens)
        token_lengths.append(len(tokens))

    median_len = int(np.median(token_lengths))
    print(f"  Token lengths: median={median_len}, min={min(token_lengths)}, max={max(token_lengths)}")

    # Pad to uniform length
    pad_id = token_dict.get("<pad>", 0)
    padded_len = min(max(token_lengths), max_len)
    token_matrix = np.full((len(all_tokens), padded_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(all_tokens), padded_len), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        length = min(len(tokens), padded_len)
        token_matrix[i, :length] = tokens[:length]
        attention_mask[i, :length] = 1

    print(f"  Token matrix: {token_matrix.shape}")

    # Load model
    print("\nLoading Geneformer model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_geneformer_model(os.path.join(PROJECT_DIR, "models", "Geneformer", "Geneformer-V1-10M"), device)
    print(f"  Model on {device}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    batch_size = 64
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(token_matrix), batch_size):
            end = min(i + batch_size, len(token_matrix))
            input_ids = torch.tensor(token_matrix[i:end], dtype=torch.long).to(device)
            attn_mask = torch.tensor(attention_mask[i:end], dtype=torch.long).to(device)

            pooled, _ = model(input_ids, attn_mask)
            all_embeddings.append(pooled.cpu().numpy())

            if i % (batch_size * 50) == 0:
                print(f"  {i}/{len(token_matrix)} cells...")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    valid_mask = np.isfinite(embeddings).all(axis=1)
    if not valid_mask.all():
        print(f"  WARNING: {(~valid_mask).sum()} cells have NaN/Inf embeddings")
        embeddings[~valid_mask] = 0

    # Compute metrics
    print("\nComputing metrics...")
    eval_mask = labels >= 2
    X_eval = embeddings[eval_mask]
    y_eval = (labels[eval_mask] == 2).astype(int)
    print(f"  Eval: {eval_mask.sum()} cells ({(y_eval==1).sum()} ecc, {(y_eval==0).sum()} hair)")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_eval, y_eval, cv=5, scoring="roc_auc")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_eval, y_eval, cv=5, scoring="roc_auc")

    total_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"Geneformer Zero-Shot Results")
    print(f"{'='*60}")
    print(f"  Genes mapped: {len(mapped_genes)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  RF AUROC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  LR AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
    print(f"  Time: {total_time:.0f}s")

    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(f"\n\n---\n\n### Geneformer Zero-Shot Benchmark\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- Model: Geneformer V1 (10M params, 256d, 6 layers)\n")
        f.write(f"- Genes mapped (mouse→human Ensembl): {len(mapped_genes)}\n")
        f.write(f"- **RF AUROC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}**\n")
        f.write(f"- **LR AUROC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}**\n")
        f.write(f"- Time: {total_time:.0f}s\n")

    print(f"Results appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
