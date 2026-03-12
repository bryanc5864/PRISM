"""Dataset and sampler for PCP pre-training.

Loads preprocessed corpus data (rank-encoded .npy files) and provides:
- Random gene masking for MLM
- Perturbation-aware batch sampling for contrastive learning
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from typing import Dict, List, Tuple, Optional


class CorpusDataset(Dataset):
    """Pre-training dataset from preprocessed corpus.

    Loads rank-encoded expression arrays and metadata from the processed
    corpus directory. Applies random masking for MLM during __getitem__.

    Expected files in processed_dir:
        {dataset_id}_encoded.npy  - (n_cells, n_genes) uint8
        {dataset_id}_meta.npz     - perturbation_ids, is_control arrays
        gene_vocab.json           - gene vocabulary
        corpus_index.json         - global cell index
    """

    def __init__(
        self,
        processed_dir: str,
        mask_ratio: float = 0.15,
        mask_token_id: int = 51,
        n_bins: int = 51,
    ):
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.n_bins = n_bins

        # Load corpus index
        index_path = os.path.join(processed_dir, "corpus_index.json")
        with open(index_path) as f:
            self.index = json.load(f)

        self.n_genes = self.index["n_genes"]
        self.total_cells = self.index["total_cells"]

        # Load all encoded datasets into memory
        self.data_arrays = {}  # dataset_id -> np.ndarray
        self.meta_arrays = {}  # dataset_id -> dict of arrays
        self.cell_map = []  # (dataset_id, local_idx) for global indexing
        self.dataset_ids = {}  # dataset_id -> integer ID

        # Assign global perturbation offsets so IDs are unique across datasets
        global_pert_offset = 0

        for ds_idx, ds_info in enumerate(self.index["datasets"]):
            ds_id = ds_info["dataset_id"]
            n_cells = ds_info["n_cells"]
            self.dataset_ids[ds_id] = ds_idx

            # Load encoded expression
            enc_path = os.path.join(processed_dir, f"{ds_id}_encoded.npy")
            self.data_arrays[ds_id] = np.load(enc_path, mmap_mode="r")

            # Load metadata
            meta_path = os.path.join(processed_dir, f"{ds_id}_meta.npz")
            meta = np.load(meta_path, allow_pickle=True)

            # Offset perturbation IDs to be globally unique
            pert_ids = meta["perturbation_ids"].astype(np.int64) + global_pert_offset
            n_perts = ds_info.get("n_perturbations", int(meta["perturbation_ids"].max()) + 1)
            global_pert_offset += n_perts

            self.meta_arrays[ds_id] = {
                "perturbation_ids": pert_ids,
                "is_control": meta["is_control"],
            }

            # Build global index
            for i in range(n_cells):
                self.cell_map.append((ds_id, i))

        print(f"Loaded corpus: {self.total_cells:,} cells, {self.n_genes} genes, "
              f"{len(self.data_arrays)} datasets")

    def __len__(self) -> int:
        return self.total_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_id, local_idx = self.cell_map[idx]

        # Get rank-encoded expression
        expression = self.data_arrays[ds_id][local_idx].copy().astype(np.int64)

        # Get metadata
        pert_id = int(self.meta_arrays[ds_id]["perturbation_ids"][local_idx])

        # Apply random masking for MLM
        mask = np.random.random(self.n_genes) < self.mask_ratio
        masked_expression = expression.copy()
        masked_expression[mask] = self.mask_token_id

        return {
            "expression": torch.from_numpy(expression),
            "masked_expression": torch.from_numpy(masked_expression),
            "mask": torch.from_numpy(mask),
            "perturbation_id": torch.tensor(pert_id, dtype=torch.long),
            "global_idx": torch.tensor(idx, dtype=torch.long),
        }


class PerturbationBatchSampler(Sampler):
    """Sampler that ensures positive pairs for contrastive learning.

    Constructs batches with P perturbations × K cells each, guaranteeing
    at least K cells per perturbation for meaningful contrastive pairs.
    """

    def __init__(
        self,
        dataset: CorpusDataset,
        perturbations_per_batch: int = 32,
        cells_per_perturbation: int = 8,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.P = perturbations_per_batch
        self.K = cells_per_perturbation
        self.batch_size = self.P * self.K
        self.rng = np.random.RandomState(seed)

        # Build perturbation -> global indices mapping
        self.pert_to_indices = {}
        for global_idx, (ds_id, local_idx) in enumerate(dataset.cell_map):
            pert_id = int(dataset.meta_arrays[ds_id]["perturbation_ids"][local_idx])
            # Use (ds_id, pert_id) as key to keep dataset-specific perturbations separate
            key = f"{ds_id}_{pert_id}"
            if key not in self.pert_to_indices:
                self.pert_to_indices[key] = []
            self.pert_to_indices[key].append(global_idx)

        # Filter perturbation groups with at least K cells
        self.valid_perts = [
            k for k, v in self.pert_to_indices.items() if len(v) >= self.K
        ]
        # Convert to arrays for faster sampling
        for k in self.valid_perts:
            self.pert_to_indices[k] = np.array(self.pert_to_indices[k])

        self.n_batches = max(1, len(dataset) // self.batch_size)
        self.base_seed = seed
        self.epoch = 0
        print(f"PerturbationBatchSampler: {len(self.valid_perts)} valid perturbation groups "
              f"(>= {self.K} cells), {self.n_batches} batches/epoch")

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic reseeding."""
        self.epoch = epoch

    def __iter__(self):
        # Reseed RNG each epoch for reproducibility and to avoid stale state
        self.rng = np.random.RandomState(self.base_seed + self.epoch)
        for _ in range(self.n_batches):
            # Sample P perturbation groups
            chosen_perts = self.rng.choice(
                self.valid_perts, size=min(self.P, len(self.valid_perts)), replace=False
            )

            batch_indices = []
            for pert_key in chosen_perts:
                indices = self.pert_to_indices[pert_key]
                chosen = self.rng.choice(indices, size=self.K, replace=len(indices) < self.K)
                batch_indices.extend(chosen.tolist())

            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches


def build_pretraining_dataloader(
    processed_dir: str,
    config,
    seed: int = 42,
) -> DataLoader:
    """Build DataLoader for PCP pre-training."""
    dataset = CorpusDataset(
        processed_dir=processed_dir,
        mask_ratio=config.mask_ratio,
        mask_token_id=config.mask_token_id,
        n_bins=config.n_bins,
    )

    sampler = PerturbationBatchSampler(
        dataset,
        perturbations_per_batch=config.perturbations_per_batch,
        cells_per_perturbation=config.cells_per_perturbation,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    return loader
