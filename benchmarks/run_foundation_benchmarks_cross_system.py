#!/usr/bin/env python3
"""Cross-system foundation model benchmarks (scGPT + Geneformer).

Generalizes the per-system scGPT and Geneformer benchmark scripts to
run on any of the 4 PRISM biological systems (skin, pancreas, cortex, hsc).

Usage:
    python benchmarks/run_foundation_benchmarks_cross_system.py --system skin
    python benchmarks/run_foundation_benchmarks_cross_system.py --system all

Each (system, model) runs in a subprocess to prevent OOM from crashing the driver.
"""

import os, sys, time, json, argparse, subprocess
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Data paths per system (skin uses legacy root path)
SYSTEM_DATA_PATHS = {
    "skin": os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"),
    "pancreas": os.path.join(PROJECT_DIR, "data", "processed", "pancreas", "adata_processed.h5ad"),
    "cortex": os.path.join(PROJECT_DIR, "data", "processed", "cortex", "adata_processed.h5ad"),
    "hsc": os.path.join(PROJECT_DIR, "data", "processed", "hsc", "adata_processed.h5ad"),
    "intestine": os.path.join(PROJECT_DIR, "data", "processed", "intestine", "adata_processed.h5ad"),
    "cardiac": os.path.join(PROJECT_DIR, "data", "processed", "cardiac", "adata_processed.h5ad"),
    "neural_crest": os.path.join(PROJECT_DIR, "data", "processed", "neural_crest", "adata_processed.h5ad"),
    "thcell": os.path.join(PROJECT_DIR, "data", "processed", "thcell", "adata_processed.h5ad"),
    "oligo": os.path.join(PROJECT_DIR, "data", "processed", "oligo", "adata_processed.h5ad"),
    "lung": os.path.join(PROJECT_DIR, "data", "processed", "lung", "adata_processed.h5ad"),
    "paul": os.path.join(PROJECT_DIR, "data", "processed", "paul", "adata_processed.h5ad"),
    "nestorowa": os.path.join(PROJECT_DIR, "data", "processed", "nestorowa", "adata_processed.h5ad"),
    "sadefeldman": os.path.join(PROJECT_DIR, "data", "processed", "sadefeldman", "adata_processed.h5ad"),
    "tirosh_melanoma": os.path.join(PROJECT_DIR, "data", "processed", "tirosh_melanoma", "adata_processed.h5ad"),
    "neftel_gbm": os.path.join(PROJECT_DIR, "data", "processed", "neftel_gbm", "adata_processed.h5ad"),
}

SCGPT_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "scGPT_human", "scGPT_human")
GENEFORMER_MODEL_DIR = os.path.join(PROJECT_DIR, "models", "Geneformer", "Geneformer-V1-10M")
GENEFORMER_DICT_DIR = os.path.join(PROJECT_DIR, "models", "Geneformer", "geneformer", "gene_dictionaries_30m")


def _get_output_dir(system):
    """Get output directory for a system."""
    if system == "skin":
        return os.path.join(PROJECT_DIR, "data", "processed")
    return os.path.join(PROJECT_DIR, "data", "processed", system)


def _get_env():
    """Build environment with LD_LIBRARY_PATH for torch."""
    env = dict(os.environ)
    try:
        sp = subprocess.check_output(
            [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
        ).decode().strip()
        extra = []
        cusparselt = os.path.join(sp, "nvidia", "cusparselt", "lib")
        if os.path.exists(cusparselt):
            extra.append(cusparselt)
        libstdcxx = "/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/"
        if os.path.exists(libstdcxx):
            extra.append(libstdcxx)
        if extra:
            env["LD_LIBRARY_PATH"] = ":".join(extra) + ":" + env.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass
    return env


def run_scgpt_subprocess(system, adata_path, output_dir, gpu_id="0"):
    """Run scGPT benchmark in a subprocess."""
    print(f"\n{'='*60}")
    print(f"scGPT Zero-Shot: {system}")
    print(f"{'='*60}")

    emb_path = os.path.join(output_dir, "scgpt_embeddings.npy")
    metrics_path = os.path.join(output_dir, "scgpt_metrics.json")

    script = f'''
import os, sys, json, time, math, pickle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

PROJECT_DIR = "{PROJECT_DIR}"
sys.path.insert(0, PROJECT_DIR)

import warnings
warnings.filterwarnings("ignore")

# Mock flash_attn
import types
for mod_name in ["flash_attn", "flash_attn.flash_attn_func",
                  "flash_attn.flash_attn_interface",
                  "flash_attn.bert_padding", "flash_attn.flash_attn_triton"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

import torch
import anndata as ad
import scipy.sparse as sp

start = time.time()
print("Loading data: {system}...")
adata = ad.read_h5ad("{adata_path}")
labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else np.zeros(len(adata))
print(f"  {{adata.shape[0]}} cells, {{adata.shape[1]}} genes")

# ---- scGPT model loading (from run_scgpt_benchmark.py) ----
model_dir = "{SCGPT_MODEL_DIR}"

class SimpleVocab:
    def __init__(self, token2idx):
        self.token2idx = token2idx
        self.idx2token = {{v: k for k, v in token2idx.items()}}
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

import torch.nn as nn

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
    def __init__(self, ntoken, d_model=512, nhead=8, d_hid=512,
                 nlayers=12, dropout=0.2, pad_token_id=0, pad_value=-2):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, gene_ids, values, src_key_padding_mask=None):
        gene_emb = self.encoder(gene_ids) * math.sqrt(self.d_model)
        val_emb = self.value_encoder(values)
        total_emb = gene_emb + val_emb
        if src_key_padding_mask is None:
            src_key_padding_mask = values.eq(self.pad_value)
        output = self.transformer_encoder(total_emb, src_key_padding_mask=src_key_padding_mask)
        mask = ~src_key_padding_mask
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (output * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        return summed / counts

# Load vocab + model
vocab = SimpleVocab.from_file(os.path.join(model_dir, "vocab.json"))
print(f"  scGPT vocab: {{len(vocab)}} genes")

# Map genes to vocab (mouse→human: exact, upper, capitalize)
gene_to_vid = {{}}
for gene in adata.var_names:
    if gene in vocab:
        gene_to_vid[gene] = vocab[gene]
    elif gene.upper() in vocab:
        gene_to_vid[gene] = vocab[gene.upper()]
    elif gene.capitalize() in vocab:
        gene_to_vid[gene] = vocab[gene.capitalize()]

# Also try mygene ortholog lookup for remaining genes
remaining = [g for g in adata.var_names if g not in gene_to_vid]
if remaining and len(gene_to_vid) < 1000:
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            remaining[:5000], scopes="symbol", fields="homologene",
            species="mouse", returnall=True,
        )
        mouse_to_human_entrez = {{}}
        human_entrez_ids = set()
        for hit in results.get("out", []):
            query = hit.get("query", "")
            if "homologene" in hit and "genes" in hit["homologene"]:
                for entry in hit["homologene"]["genes"]:
                    if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 9606:
                        mouse_to_human_entrez[query] = str(entry[1])
                        human_entrez_ids.add(str(entry[1]))
                        break
        if human_entrez_ids:
            human_results = mg.querymany(
                list(human_entrez_ids), scopes="entrezgene", fields="symbol",
                species="human", returnall=True,
            )
            entrez_to_sym = {{}}
            for hit in human_results.get("out", []):
                if "symbol" in hit:
                    entrez_to_sym[hit.get("query", "")] = hit["symbol"]
            for gene, eid in mouse_to_human_entrez.items():
                if eid in entrez_to_sym:
                    sym = entrez_to_sym[eid]
                    if sym in vocab:
                        gene_to_vid[gene] = vocab[sym]
                    elif sym.upper() in vocab:
                        gene_to_vid[gene] = vocab[sym.upper()]
    except Exception as e:
        print(f"  mygene lookup failed: {{e}}")

print(f"  Genes mapped: {{len(gene_to_vid)}}/{{len(adata.var_names)}}")

# Prepare expression matrix
mapped_genes = [g for g in adata.var_names if g in gene_to_vid]
mask = [g in gene_to_vid for g in adata.var_names]
vid_array = np.array([gene_to_vid[g] for g in mapped_genes])
X = adata[:, mask].X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)

# Limit to top 1200 genes
max_seq_len = 1200
if X.shape[1] > max_seq_len:
    gene_means = X.mean(axis=0)
    top_idx = np.argsort(gene_means)[-max_seq_len:]
    X = X[:, top_idx]
    vid_array = vid_array[top_idx]

# Bin expression
def binning(row, n_bins=51):
    nonzero = row > 0
    if nonzero.sum() == 0:
        return np.zeros_like(row)
    result = np.zeros_like(row)
    vals = row[nonzero]
    ranks = np.argsort(np.argsort(vals)) + 1
    max_rank = ranks.max()
    bins = np.clip((ranks / max_rank * (n_bins - 1)).astype(int) + 1, 1, n_bins - 1)
    result[nonzero] = bins
    return result

X_binned = np.zeros_like(X, dtype=np.float32)
for i in range(X.shape[0]):
    X_binned[i] = binning(X[i])

# Load model weights
with open(os.path.join(model_dir, "args.json")) as f:
    model_args = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ScGPTModel(
    ntoken=len(vocab),
    d_model=model_args.get("embsize", 512),
    nhead=model_args.get("nheads", 8),
    d_hid=model_args.get("d_hid", 512),
    nlayers=model_args.get("nlayers", 12),
    dropout=0.0,
    pad_token_id=vocab["<pad>"],
    pad_value=model_args.get("pad_value", -2),
)

state_dict = torch.load(
    os.path.join(model_dir, "best_model.pt"),
    map_location="cpu", weights_only=False,
)
mapped_state = {{}}
for k, v in state_dict.items():
    new_k = k.replace("encoder.embedding.weight", "encoder.weight")
    new_k = new_k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
    new_k = new_k.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
    mapped_state[new_k] = v
model.load_state_dict(mapped_state, strict=False)
model.eval()
model = model.to(device)

# Extract embeddings
print("  Extracting scGPT embeddings...")
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

embeddings = np.concatenate(all_embeddings, axis=0)
valid_mask = np.isfinite(embeddings).all(axis=1)
if not valid_mask.all():
    embeddings[~valid_mask] = 0

# Save embeddings
np.save("{emb_path}", embeddings)
print(f"  Saved embeddings: {{embeddings.shape}}")

# Compute metrics
sys.path.insert(0, "{PROJECT_DIR}")
from prism.utils.metrics import compute_all_metrics
metrics = compute_all_metrics(embeddings, labels, method_name="scGPT")
metrics["n_genes_mapped"] = len(gene_to_vid)
metrics["embedding_dim"] = int(embeddings.shape[1])
metrics["time_seconds"] = time.time() - start

with open("{metrics_path}", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"  Metrics: ARI={{metrics.get('ARI', 'N/A')}}, RF_AUROC={{metrics.get('RF_AUROC', 'N/A')}}")
print("SUCCESS")
'''

    env = _get_env()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=3600, env=env,
    )

    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        # Filter out common warnings
        stderr_lines = [l for l in result.stderr.split('\n')
                       if l.strip() and not any(w in l for w in ['FutureWarning', 'UserWarning', 'DeprecationWarning'])]
        if stderr_lines:
            print("STDERR (last 500 chars):", result.stderr[-500:])

    if "SUCCESS" in result.stdout:
        with open(metrics_path) as f:
            return json.load(f)
    else:
        return {"method": "scGPT", "error": f"subprocess_failed_{system}"}


def run_geneformer_subprocess(system, adata_path, output_dir, gpu_id="0"):
    """Run Geneformer benchmark in a subprocess."""
    print(f"\n{'='*60}")
    print(f"Geneformer Zero-Shot: {system}")
    print(f"{'='*60}")

    emb_path = os.path.join(output_dir, "geneformer_embeddings.npy")
    metrics_path = os.path.join(output_dir, "geneformer_metrics.json")

    script = f'''
import os, sys, json, time, math, pickle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

PROJECT_DIR = "{PROJECT_DIR}"
sys.path.insert(0, PROJECT_DIR)

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scipy.sparse as sp

start = time.time()
print("Loading data: {system}...")
adata = ad.read_h5ad("{adata_path}")
labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else np.zeros(len(adata))
print(f"  {{adata.shape[0]}} cells, {{adata.shape[1]}} genes")

# ---- Minimal BERT model (from run_geneformer_benchmark.py) ----

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
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(embeddings))

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
        return x.view(B, S, self.num_heads, self.head_size).permute(0, 2, 1, 3)
    def forward(self, hidden_states, attention_mask=None):
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(probs, V).permute(0, 2, 1, 3).contiguous()
        return context.view(context.size(0), context.size(1), self.all_head_size)

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="relu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.ReLU() if hidden_act == "relu" else nn.GELU()
    def forward(self, x):
        return self.act(self.dense(x))

class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size,
                 hidden_act="relu", layer_norm_eps=1e-12, dropout=0.02):
        super().__init__()
        self.attention = nn.ModuleDict({{
            "self": BertSelfAttention(hidden_size, num_heads, dropout),
            "output": BertSelfOutput(hidden_size, layer_norm_eps, dropout),
        }})
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size, layer_norm_eps, dropout)
    def forward(self, hidden_states, attention_mask=None):
        self_output = self.attention["self"](hidden_states, attention_mask)
        attention_output = self.attention["output"](self_output, hidden_states)
        return self.output(self.intermediate(attention_output), attention_output)

class MinimalBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = nn.ModuleDict({{
            "embeddings": BertEmbeddings(
                config["vocab_size"], config["hidden_size"],
                config["max_position_embeddings"], config.get("type_vocab_size", 2),
                config.get("layer_norm_eps", 1e-12), config.get("hidden_dropout_prob", 0.02),
            ),
            "encoder": nn.ModuleDict({{
                "layer": nn.ModuleList([
                    BertLayer(
                        config["hidden_size"], config["num_attention_heads"],
                        config["intermediate_size"], config.get("hidden_act", "relu"),
                        config.get("layer_norm_eps", 1e-12), config.get("hidden_dropout_prob", 0.02),
                    )
                    for _ in range(config["num_hidden_layers"])
                ])
            }})
        }})
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        else:
            extended_mask = None
        hidden = self.bert["embeddings"](input_ids)
        for layer in self.bert["encoder"]["layer"]:
            hidden = layer(hidden, extended_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return pooled, hidden

# Load Geneformer dictionaries
gene_dict_dir = "{GENEFORMER_DICT_DIR}"
with open(os.path.join(gene_dict_dir, "token_dictionary_gc30M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
with open(os.path.join(gene_dict_dir, "gene_name_id_dict_gc30M.pkl"), "rb") as f:
    gene_name_id = pickle.load(f)
with open(os.path.join(gene_dict_dir, "gene_median_dictionary_gc30M.pkl"), "rb") as f:
    gene_medians = pickle.load(f)

print(f"  Geneformer vocab: {{len(token_dict)}} tokens")

# Map mouse genes to Geneformer token IDs
human_upper_to_ensembl = {{}}
for name, ens_id in gene_name_id.items():
    if ens_id in token_dict:
        human_upper_to_ensembl[name.upper()] = ens_id

mapping = {{}}
for gene in adata.var_names:
    upper = gene.upper()
    if upper in human_upper_to_ensembl:
        mapping[gene] = human_upper_to_ensembl[upper]

remaining = [g for g in adata.var_names if g not in mapping]
print(f"  Direct match: {{len(mapping)}}/{{len(adata.var_names)}}")

if remaining:
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            remaining[:5000], scopes="symbol", fields="homologene",
            species="mouse", returnall=True,
        )
        mouse_to_human_entrez = {{}}
        human_entrez_ids = set()
        for hit in results.get("out", []):
            query = hit.get("query", "")
            if "homologene" in hit and "genes" in hit["homologene"]:
                for entry in hit["homologene"]["genes"]:
                    if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 9606:
                        mouse_to_human_entrez[query] = str(entry[1])
                        human_entrez_ids.add(str(entry[1]))
                        break
        if human_entrez_ids:
            human_results = mg.querymany(
                list(human_entrez_ids), scopes="entrezgene", fields="symbol",
                species="human", returnall=True,
            )
            entrez_to_sym = {{}}
            for hit in human_results.get("out", []):
                if "symbol" in hit:
                    entrez_to_sym[hit.get("query", "")] = hit["symbol"]
            for gene, eid in mouse_to_human_entrez.items():
                if eid in entrez_to_sym:
                    sym = entrez_to_sym[eid]
                    if sym.upper() in human_upper_to_ensembl:
                        mapping[gene] = human_upper_to_ensembl[sym.upper()]
    except Exception as e:
        print(f"  mygene failed: {{e}}")

print(f"  Total mapped: {{len(mapping)}}/{{len(adata.var_names)}}")

# Build tokenized matrix
mapped_genes = [g for g in adata.var_names if g in mapping]
mask = np.array([g in mapping for g in adata.var_names])
ensembl_ids = np.array([mapping[g] for g in mapped_genes])
X = adata[:, mask].X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)

def tokenize_cell(expression, gene_ensembl_ids, gene_medians, token_dict, max_len=2048):
    nonzero = expression > 0
    if nonzero.sum() == 0:
        return np.array([token_dict.get("<pad>", 0)], dtype=np.int64)
    nz_expr = expression[nonzero]
    nz_ensembl = gene_ensembl_ids[nonzero]
    medians = np.array([gene_medians.get(eid, 1.0) for eid in nz_ensembl])
    normalized = nz_expr / np.clip(medians, 1e-6, None)
    rank_order = np.argsort(-normalized)[:max_len]
    return np.array([token_dict[nz_ensembl[i]] for i in rank_order], dtype=np.int64)

print("  Tokenizing cells...")
max_len = 2048
all_tokens = []
token_lengths = []
for i in range(X.shape[0]):
    tokens = tokenize_cell(X[i], ensembl_ids, gene_medians, token_dict, max_len)
    all_tokens.append(tokens)
    token_lengths.append(len(tokens))

pad_id = token_dict.get("<pad>", 0)
padded_len = min(max(token_lengths), max_len)
token_matrix = np.full((len(all_tokens), padded_len), pad_id, dtype=np.int64)
attention_mask = np.zeros((len(all_tokens), padded_len), dtype=np.int64)
for i, tokens in enumerate(all_tokens):
    length = min(len(tokens), padded_len)
    token_matrix[i, :length] = tokens[:length]
    attention_mask[i, :length] = 1

print(f"  Token matrix: {{token_matrix.shape}}")

# Load model
model_dir = "{GENEFORMER_MODEL_DIR}"
with open(os.path.join(model_dir, "config.json")) as f:
    config = json.load(f)
print(f"  Config: {{config['hidden_size']}}d, {{config['num_hidden_layers']}} layers")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MinimalBert(config)
model_file = os.path.join(model_dir, "pytorch_model.bin")
safetensors_file = os.path.join(model_dir, "model.safetensors")
if os.path.exists(model_file):
    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
elif os.path.exists(safetensors_file):
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_file)
mapped_sd = {{k: v for k, v in state_dict.items() if not k.startswith("cls.") and not k.startswith("predictions.")}}
model.load_state_dict(mapped_sd, strict=False)
model.eval()
model = model.to(device)

# Extract embeddings
print("  Extracting Geneformer embeddings...")
batch_size = 64
all_embeddings = []
with torch.no_grad():
    for i in range(0, len(token_matrix), batch_size):
        end = min(i + batch_size, len(token_matrix))
        input_ids = torch.tensor(token_matrix[i:end], dtype=torch.long).to(device)
        attn_mask = torch.tensor(attention_mask[i:end], dtype=torch.long).to(device)
        pooled, _ = model(input_ids, attn_mask)
        all_embeddings.append(pooled.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
valid_mask = np.isfinite(embeddings).all(axis=1)
if not valid_mask.all():
    embeddings[~valid_mask] = 0

np.save("{emb_path}", embeddings)
print(f"  Saved embeddings: {{embeddings.shape}}")

# Compute metrics
from prism.utils.metrics import compute_all_metrics
metrics = compute_all_metrics(embeddings, labels, method_name="Geneformer")
metrics["n_genes_mapped"] = len(mapping)
metrics["embedding_dim"] = int(embeddings.shape[1])
metrics["time_seconds"] = time.time() - start

with open("{metrics_path}", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"  Metrics: ARI={{metrics.get('ARI', 'N/A')}}, RF_AUROC={{metrics.get('RF_AUROC', 'N/A')}}")
print("SUCCESS")
'''

    env = _get_env()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=3600, env=env,
    )

    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        stderr_lines = [l for l in result.stderr.split('\n')
                       if l.strip() and not any(w in l for w in ['FutureWarning', 'UserWarning', 'DeprecationWarning'])]
        if stderr_lines:
            print("STDERR (last 500 chars):", result.stderr[-500:])

    if "SUCCESS" in result.stdout:
        with open(metrics_path) as f:
            return json.load(f)
    else:
        return {"method": "Geneformer", "error": f"subprocess_failed_{system}"}


def run_system(system, gpu_id="0"):
    """Run both foundation model benchmarks for a single system."""
    adata_path = SYSTEM_DATA_PATHS.get(system)
    if not adata_path or not os.path.exists(adata_path):
        print(f"  Data not found for {system}: {adata_path}")
        return {}

    output_dir = _get_output_dir(system)
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # scGPT
    results["scGPT"] = run_scgpt_subprocess(system, adata_path, output_dir, gpu_id)

    # Geneformer
    results["Geneformer"] = run_geneformer_subprocess(system, adata_path, output_dir, gpu_id)

    # Save combined results
    combined_path = os.path.join(output_dir, "foundation_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved foundation results to {combined_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-system foundation model benchmarks")
    parser.add_argument("--system", default="all",
                       help="System to benchmark (skin/pancreas/cortex/hsc/all)")
    parser.add_argument("--gpu", default="0", help="GPU ID to use")
    parser.add_argument("--model", default="all",
                       help="Model to run (scgpt/geneformer/all)")
    args = parser.parse_args()

    start = time.time()

    if args.system == "all":
        systems = list(SYSTEM_DATA_PATHS.keys())
    else:
        systems = [args.system]

    all_results = {}
    for system in systems:
        print(f"\n{'#'*60}")
        print(f"# System: {system}")
        print(f"{'#'*60}")

        adata_path = SYSTEM_DATA_PATHS.get(system)
        if not adata_path or not os.path.exists(adata_path):
            print(f"  Skipping {system}: data not found at {adata_path}")
            continue

        output_dir = _get_output_dir(system)
        os.makedirs(output_dir, exist_ok=True)
        system_results = {}

        if args.model in ["all", "scgpt"]:
            system_results["scGPT"] = run_scgpt_subprocess(
                system, adata_path, output_dir, args.gpu)

        if args.model in ["all", "geneformer"]:
            system_results["Geneformer"] = run_geneformer_subprocess(
                system, adata_path, output_dir, args.gpu)

        # Save combined foundation results
        combined_path = os.path.join(output_dir, "foundation_results.json")
        with open(combined_path, "w") as f:
            json.dump(system_results, f, indent=2, default=str)

        all_results[system] = system_results

    # Print summary
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"Foundation Model Benchmarks Complete ({total_time:.0f}s)")
    print(f"{'='*60}")
    print(f"\n{'System':<12} {'Model':<14} {'ARI':<8} {'RF_AUROC':<10} {'Genes':<8}")
    print("-" * 52)
    for system, results in all_results.items():
        for model_name, metrics in results.items():
            if "error" in metrics:
                print(f"{system:<12} {model_name:<14} {'FAILED':<8} {'':<10} {metrics.get('error', '')}")
            else:
                print(f"{system:<12} {model_name:<14} "
                      f"{metrics.get('ARI', 0):<8.3f} "
                      f"{metrics.get('RF_AUROC', 0):<10.3f} "
                      f"{metrics.get('n_genes_mapped', '?')}")


if __name__ == "__main__":
    main()
