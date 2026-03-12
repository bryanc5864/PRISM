#!/usr/bin/env python3
"""Benchmark PRISM against scGPT and Geneformer foundation models.

Both models are pretrained on human data, so we need mouse→human
ortholog mapping for scGPT. Geneformer uses Ensembl IDs.
"""

import os, sys, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fix LD_LIBRARY_PATH
import subprocess
site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"]
).decode().strip()
cusparselt_path = os.path.join(site_packages, "nvidia", "cusparselt", "lib")
if os.path.exists(cusparselt_path):
    os.environ["LD_LIBRARY_PATH"] = cusparselt_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Block flash_attn (causes bus error with running ablation)
import types
fa_mock = types.ModuleType("flash_attn")
fa_mock.flash_attn_func = types.ModuleType("flash_attn.flash_attn_func")
fa_mock.flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
sys.modules["flash_attn"] = fa_mock
sys.modules["flash_attn.flash_attn_func"] = fa_mock.flash_attn_func
sys.modules["flash_attn.flash_attn_interface"] = fa_mock.flash_attn_interface

import warnings
warnings.filterwarnings("ignore")

import torch
import anndata as ad
import scanpy as sc
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def compute_benchmark_metrics(embeddings, labels, method_name):
    """Compute RF and LR classification metrics for eccrine vs hair."""
    # Filter to labeled cells (eccrine=2, hair=3)
    mask = labels >= 2
    X = embeddings[mask]
    y = (labels[mask] == 2).astype(int)  # 1=eccrine, 0=hair

    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"method": method_name, "error": "too_few_labeled"}

    # RF with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")

    # LR
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=5, scoring="roc_auc")

    return {
        "method": method_name,
        "RF_AUROC": float(rf_scores.mean()),
        "RF_AUROC_std": float(rf_scores.std()),
        "LR_AUROC": float(lr_scores.mean()),
        "LR_AUROC_std": float(lr_scores.std()),
        "n_cells_evaluated": int(mask.sum()),
    }


def get_mouse_to_human_mapping():
    """Build mouse gene symbol → human gene symbol mapping using biomart-style lookup."""
    # Common mouse→human orthologs for skin/appendage genes
    # For a comprehensive mapping, we'd use pybiomart, but let's use a
    # simple heuristic: most mouse genes have the same name as human but
    # with different capitalization (Mouse: Trp63 → Human: TP63)
    # Mouse convention: First letter cap, rest lowercase
    # Human convention: ALL CAPS

    # For scGPT, gene names need to match their vocabulary
    # Strategy: try uppercase version of mouse gene name
    return None  # We'll do dynamic mapping against vocab


def run_scgpt_benchmark(adata, labels):
    """Run scGPT zero-shot embedding benchmark."""
    print("\n" + "="*60)
    print("scGPT Zero-Shot Embedding")
    print("="*60)

    try:
        # Check if model is downloaded
        model_dir = Path(PROJECT_DIR) / "models" / "scGPT_cp"
        if not model_dir.exists():
            model_dir = Path(PROJECT_DIR) / "models" / "scGPT_human"
        if not model_dir.exists():
            # Try to download
            print("  Downloading scGPT pretrained model...")
            os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
            # Use gdown for Google Drive
            try:
                import gdown
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
                import gdown

            # scGPT continual pretrained model (best for zero-shot)
            model_dir = Path(PROJECT_DIR) / "models" / "scGPT_cp"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download from Google Drive folder
            gdown.download_folder(
                "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
                output=str(model_dir),
                quiet=False,
            )

        if not (model_dir / "vocab.json").exists():
            # Try the whole-human model
            print("  Model files not found, trying alternative download...")
            model_dir = Path(PROJECT_DIR) / "models" / "scGPT_human"
            model_dir.mkdir(parents=True, exist_ok=True)
            import gdown
            gdown.download_folder(
                "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
                output=str(model_dir),
                quiet=False,
            )

        # Load vocabulary
        vocab_file = model_dir / "vocab.json"
        if not vocab_file.exists():
            print("  ERROR: Could not download scGPT model files")
            return {"method": "scGPT", "error": "model_not_found"}

        with open(vocab_file) as f:
            vocab = json.load(f)

        print(f"  Vocabulary size: {len(vocab)}")

        # Map mouse genes to scGPT vocabulary
        # Strategy: try exact match, then uppercase, then title case
        gene_names = adata.var_names.tolist()
        gene_to_vocab = {}
        matched = 0
        for gene in gene_names:
            if gene in vocab:
                gene_to_vocab[gene] = gene
                matched += 1
            elif gene.upper() in vocab:
                gene_to_vocab[gene] = gene.upper()
                matched += 1
            elif gene.capitalize() in vocab:
                gene_to_vocab[gene] = gene.capitalize()
                matched += 1

        print(f"  Gene mapping: {matched}/{len(gene_names)} genes matched to scGPT vocab")

        if matched < 500:
            print("  WARNING: Low gene overlap. Mouse→human mapping may be insufficient.")

        # Use scGPT's embed_data if available
        try:
            from scgpt.tasks import embed_data

            # Prepare adata with mapped gene names
            adata_mapped = adata.copy()
            # Rename var_names to human orthologs where possible
            new_names = []
            keep_mask = []
            for gene in adata_mapped.var_names:
                if gene in gene_to_vocab:
                    new_names.append(gene_to_vocab[gene])
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)

            adata_mapped = adata_mapped[:, keep_mask]
            adata_mapped.var_names = new_names
            adata_mapped.var_names_make_unique()

            # Add gene column for embed_data
            adata_mapped.var["gene_name"] = adata_mapped.var_names.values

            print(f"  Running scGPT embed_data on {adata_mapped.shape[0]} cells, {adata_mapped.shape[1]} genes...")
            embed_adata = embed_data(
                adata_mapped,
                str(model_dir),
                gene_col="gene_name",
                batch_size=64,
                return_new_adata=True,
            )

            embeddings = embed_adata.X
            if hasattr(embeddings, "toarray"):
                embeddings = embeddings.toarray()
            embeddings = np.array(embeddings, dtype=np.float32)

            print(f"  Embedding shape: {embeddings.shape}")
            metrics = compute_benchmark_metrics(embeddings, labels, "scGPT")
            print(f"  RF AUROC: {metrics.get('RF_AUROC', 'N/A')}")
            return metrics

        except Exception as e:
            print(f"  embed_data failed: {e}")
            # Fallback: manual forward pass
            return _scgpt_manual_embed(adata, labels, model_dir, vocab, gene_to_vocab)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"method": "scGPT", "error": str(e)}


def _scgpt_manual_embed(adata, labels, model_dir, vocab, gene_to_vocab):
    """Manual scGPT embedding extraction without embed_data."""
    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        # Load model config
        args_file = model_dir / "args.json"
        with open(args_file) as f:
            model_args = json.load(f)

        # Build vocabulary
        gene_vocab = GeneVocab.from_file(model_dir / "vocab.json")

        # Build model
        ntokens = len(gene_vocab)
        model = TransformerModel(
            ntoken=ntokens,
            d_model=model_args.get("embsize", 512),
            nhead=model_args.get("nheads", 8),
            d_hid=model_args.get("d_hid", 512),
            nlayers=model_args.get("nlayers", 12),
            nlayers_cls=0,
            n_cls=1,
            vocab=gene_vocab,
            dropout=0.0,
            pad_token=gene_vocab.get("<pad>", gene_vocab.get("[pad]", 0)),
            pad_value=-2,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            input_emb_style="continuous",
            explicit_zero_prob=False,
        )

        # Load weights
        model_file = model_dir / "best_model.pt"
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Tokenize and get embeddings
        import scipy.sparse as sp

        # Get gene indices in vocab
        gene_ids = []
        gene_mask = []
        for gene in adata.var_names:
            if gene in gene_to_vocab:
                mapped = gene_to_vocab[gene]
                if mapped in gene_vocab:
                    gene_ids.append(gene_vocab[mapped])
                    gene_mask.append(True)
                else:
                    gene_mask.append(False)
            else:
                gene_mask.append(False)

        gene_ids = np.array(gene_ids)
        X = adata[:, gene_mask].X
        if sp.issparse(X):
            X = X.toarray()

        print(f"  Manual embedding: {X.shape[1]} genes mapped, {X.shape[0]} cells")

        # Batch inference
        batch_size = 64
        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_expr = torch.tensor(X[start:end], dtype=torch.float32).to(device)
                batch_genes = torch.tensor(
                    np.tile(gene_ids, (end - start, 1)), dtype=torch.long
                ).to(device)

                # Forward pass - get cell embeddings
                output = model(batch_genes, batch_expr, src_key_padding_mask=None)
                # Use CLS token or mean pooling
                cell_emb = output["cell_emb"] if "cell_emb" in output else output[:, 0, :]
                all_embeddings.append(cell_emb.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  Embedding shape: {embeddings.shape}")

        metrics = compute_benchmark_metrics(embeddings, labels, "scGPT")
        print(f"  RF AUROC: {metrics.get('RF_AUROC', 'N/A')}")
        return metrics

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"method": "scGPT", "error": f"manual_embed_failed: {e}"}


def run_geneformer_benchmark(adata, labels):
    """Run Geneformer zero-shot embedding benchmark."""
    print("\n" + "="*60)
    print("Geneformer Zero-Shot Embedding")
    print("="*60)

    try:
        # Check if geneformer is installed
        gf_dir = Path(PROJECT_DIR) / "models" / "Geneformer"
        if not gf_dir.exists():
            print("  Cloning Geneformer from HuggingFace...")
            os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
            result = subprocess.run(
                ["git", "lfs", "install"],
                capture_output=True, text=True
            )
            result = subprocess.run(
                ["git", "clone", "https://huggingface.co/ctheodoris/Geneformer",
                 str(gf_dir)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"  Clone failed: {result.stderr[:200]}")
                # Try without LFS (just get code)
                result = subprocess.run(
                    ["GIT_LFS_SKIP_SMUDGE=1", "git", "clone",
                     "https://huggingface.co/ctheodoris/Geneformer", str(gf_dir)],
                    capture_output=True, text=True, timeout=300, shell=True,
                )

        # Try to install geneformer
        try:
            import geneformer
            print("  Geneformer already installed")
        except ImportError:
            if (gf_dir / "setup.py").exists() or (gf_dir / "pyproject.toml").exists():
                print("  Installing Geneformer...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", str(gf_dir), "-q"],
                    timeout=300,
                )
                import geneformer
            else:
                return {"method": "Geneformer", "error": "install_failed"}

        from geneformer import TranscriptomeTokenizer, EmbExtractor

        # Geneformer needs Ensembl IDs. Our data has mouse gene symbols.
        # Need to map mouse gene symbols → mouse Ensembl IDs
        print("  Mapping mouse gene symbols to Ensembl IDs...")
        ensembl_map = _get_mouse_ensembl_mapping(adata.var_names.tolist())
        print(f"  Mapped {len(ensembl_map)}/{len(adata.var_names)} genes to Ensembl IDs")

        if len(ensembl_map) < 500:
            return {"method": "Geneformer", "error": f"low_gene_overlap_{len(ensembl_map)}"}

        # Prepare AnnData for Geneformer
        adata_gf = adata.copy()
        # Keep only mapped genes
        mapped_mask = [g in ensembl_map for g in adata_gf.var_names]
        adata_gf = adata_gf[:, mapped_mask]

        # Set Ensembl IDs
        adata_gf.var["ensembl_id"] = [ensembl_map[g] for g in adata_gf.var_names]
        adata_gf.obs["n_counts"] = np.array(adata_gf.X.sum(axis=1)).flatten()
        adata_gf.obs["joinid"] = list(range(adata_gf.n_obs))

        # Save for tokenizer
        h5ad_dir = "/tmp/gf_input"
        token_dir = "/tmp/gf_tokens"
        output_dir = "/tmp/gf_output"
        os.makedirs(h5ad_dir, exist_ok=True)
        os.makedirs(token_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        adata_gf.write(os.path.join(h5ad_dir, "data.h5ad"))

        # Tokenize
        print("  Tokenizing...")
        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict={"joinid": "joinid"},
            nproc=4,
        )
        tokenizer.tokenize_data(
            data_directory=h5ad_dir,
            output_directory=token_dir,
            output_prefix="data",
            file_format="h5ad",
        )

        # Extract embeddings
        model_path = str(gf_dir / "geneformer-v2-316M")
        if not Path(model_path).exists():
            model_path = str(gf_dir)  # model might be in root

        print("  Extracting embeddings...")
        embex = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",
            emb_layer=-1,
            max_ncells=None,
            emb_label=["joinid"],
            forward_batch_size=64,
            nproc=4,
        )

        embs = embex.extract_embs(
            model_directory=model_path,
            input_data_file=os.path.join(token_dir, "data.dataset"),
            output_directory=output_dir,
            output_prefix="cell_embs",
        )

        embs = embs.sort_values("joinid")
        embeddings = embs.drop(columns=["joinid"]).to_numpy().astype(np.float32)
        print(f"  Embedding shape: {embeddings.shape}")

        metrics = compute_benchmark_metrics(embeddings, labels, "Geneformer")
        print(f"  RF AUROC: {metrics.get('RF_AUROC', 'N/A')}")
        return metrics

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"method": "Geneformer", "error": str(e)}


def _get_mouse_ensembl_mapping(gene_symbols):
    """Map mouse gene symbols to Ensembl IDs using pybiomart or fallback."""
    mapping = {}
    try:
        from pybiomart import Server
        server = Server(host="http://www.ensembl.org")
        dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["mmusculus_gene_ensembl"]
        result = dataset.query(
            attributes=["external_gene_name", "ensembl_gene_id"],
        )
        for _, row in result.iterrows():
            name = row["Gene name"]
            eid = row["Gene stable ID"]
            if name in gene_symbols:
                mapping[name] = eid
    except Exception as e:
        print(f"  pybiomart failed ({e}), trying mygene...")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                gene_symbols, scopes="symbol", fields="ensembl.gene",
                species="mouse", returnall=True
            )
            for hit in results.get("out", []):
                if "ensembl" in hit and "gene" in hit["ensembl"]:
                    symbol = hit.get("query", "")
                    eid = hit["ensembl"]["gene"]
                    if isinstance(eid, list):
                        eid = eid[0]
                    mapping[symbol] = eid
        except Exception as e2:
            print(f"  mygene also failed ({e2})")
            print("  Falling back to simple Ensembl ID generation (won't match Geneformer vocab)")

    return mapping


def main():
    start = time.time()

    print("Loading data...")
    adata = ad.read_h5ad(os.path.join(PROJECT_DIR, "data", "processed", "adata_processed.h5ad"))
    labels = adata.obs["fate_int"].values if "fate_int" in adata.obs else np.zeros(len(adata))
    print(f"  {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    results = {}

    # scGPT
    results["scGPT"] = run_scgpt_benchmark(adata, labels)

    # Geneformer
    results["Geneformer"] = run_geneformer_benchmark(adata, labels)

    # Summary
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"Foundation Model Benchmarks Complete ({total_time:.0f}s)")
    print(f"{'='*60}")

    result_text = "**Foundation Model Benchmarks**\n\n"
    result_text += "| Method | RF AUROC | LR AUROC | Genes Mapped | Notes |\n"
    result_text += "|--------|----------|----------|-------------|-------|\n"

    for name, m in results.items():
        if "error" in m:
            result_text += f"| {name} | FAILED | - | - | {m['error']} |\n"
        else:
            result_text += (
                f"| {name} | {m.get('RF_AUROC', 0):.3f}±{m.get('RF_AUROC_std', 0):.3f} | "
                f"{m.get('LR_AUROC', 0):.3f}±{m.get('LR_AUROC_std', 0):.3f} | "
                f"{m.get('n_genes_mapped', '?')} | |\n"
            )

    for name, m in results.items():
        print(f"  {name}: {m}")

    with open(os.path.join(PROJECT_DIR, "results.md"), "a") as f:
        f.write(f"\n\n---\n\n### Foundation Model Benchmarks\n")
        f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(result_text)

    print(f"\nResults appended to {os.path.join(PROJECT_DIR, 'results.md')}")


if __name__ == "__main__":
    main()
