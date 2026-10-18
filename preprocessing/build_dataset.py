"""
Dataset Assembly Module
========================
Assembles final PyTorch datasets with stratified splitting.

Usage:
    python preprocessing/build_dataset.py --config configs/config.yaml
"""

import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir, save_pickle, load_pickle, compute_class_weights
from utils.seed import set_seed

# Windows multiprocessing fix: DataLoader with num_workers>0 requires spawn context
# which causes issues. Force num_workers=0 on Windows.
_SAFE_NUM_WORKERS = 0 if sys.platform == "win32" else None  # None = use config value

logger = get_logger(__name__)


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein function prediction.

    Provides sequence embeddings, graph indices, and multi-label targets.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        accessions: List[str],
        protein_to_graph_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            embeddings: Sequence embeddings (N, D).
            labels: Multi-label matrix (N, C).
            indices: Original dataset indices for this split.
            accessions: Protein accession IDs.
            protein_to_graph_idx: Mapping from accession to graph node index.
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.indices = indices
        self.accessions = accessions
        self.protein_to_graph_idx = protein_to_graph_idx or {}

        # Graph indices for each protein
        self.graph_indices = torch.tensor(
            [self.protein_to_graph_idx.get(acc, -1) for acc in accessions],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "graph_idx": self.graph_indices[idx],
            "dataset_idx": self.indices[idx],
        }


def stratified_multilabel_split(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform approximate stratified split for multi-label data.

    Uses iterative stratification based on the most frequent label.

    Args:
        labels: Binary label matrix (N, C).
        train_ratio: Training set proportion.
        val_ratio: Validation set proportion.
        test_ratio: Test set proportion.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    n_samples = labels.shape[0]
    indices = np.arange(n_samples)

    # Use label frequency as stratification key
    # Sum labels per sample as proxy for stratification
    label_counts = labels.sum(axis=1).astype(int)
    # Bin into quartiles to create a stratification variable
    bins = np.quantile(label_counts, [0.25, 0.5, 0.75])
    strat_key = np.digitize(label_counts, bins)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=strat_key,
        random_state=seed,
    )

    # Second split: val vs test
    val_relative = val_ratio / val_test_ratio
    strat_key_vt = strat_key[valtest_idx]
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=(1 - val_relative),
        stratify=strat_key_vt,
        random_state=seed,
    )

    logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    logger.info(f"  Ratios: {len(train_idx)/n_samples:.2f} / "
                f"{len(val_idx)/n_samples:.2f} / {len(test_idx)/n_samples:.2f}")

    # Verify no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"

    return train_idx, val_idx, test_idx


def create_data_loaders(
    train_dataset: ProteinDataset,
    val_dataset: ProteinDataset,
    test_dataset: ProteinDataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Only pin memory when CUDA is available (avoids warnings on CPU)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader


def build_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main dataset assembly pipeline.

    Args:
        config: Project configuration.

    Returns:
        Dictionary with datasets, loaders, and metadata.
    """
    data_cfg = config["data"]
    processed_dir = data_cfg["processed_dir"]
    embeddings_dir = data_cfg["embeddings_dir"]

    set_seed(data_cfg["split"]["seed"])

    # Load data
    proteins_df = pd.read_csv(os.path.join(processed_dir, "proteins_annotated.tsv"), sep="\t")
    label_matrix = np.load(os.path.join(processed_dir, "label_matrix.npy"), mmap_mode="r")
    embeddings = np.load(os.path.join(embeddings_dir, "esm2_embeddings.npy"))

    # Load graph data
    pyg_data_path = os.path.join(processed_dir, "pyg_graph_data.pt")
    if os.path.exists(pyg_data_path):
        import torch as _torch
        pyg_data = _torch.load(pyg_data_path, map_location="cpu", weights_only=False)
        protein_to_graph_idx = pyg_data["protein_to_idx"]
    else:
        protein_to_graph_idx = {acc: i for i, acc in enumerate(proteins_df["accession"])}

    accessions = list(proteins_df["accession"].values)  # plain Python list for safe numpy-int indexing

    logger.info(f"Assembling dataset: {len(accessions)} proteins, "
                f"{embeddings.shape[1]}D embeddings, {label_matrix.shape[1]} GO terms")

    # Verify alignment
    assert len(accessions) == embeddings.shape[0] == label_matrix.shape[0], \
        "Mismatched data dimensions!"

    # Stratified split
    train_idx, val_idx, test_idx = stratified_multilabel_split(
        label_matrix,
        train_ratio=data_cfg["split"]["train_ratio"],
        val_ratio=data_cfg["split"]["val_ratio"],
        test_ratio=data_cfg["split"]["test_ratio"],
        seed=data_cfg["split"]["seed"],
    )

    # Create datasets  -  copy slices explicitly and free memory between steps
    train_labels = np.array(label_matrix[train_idx], dtype=np.float32)
    train_dataset = ProteinDataset(
        embeddings[train_idx], train_labels, train_idx,
        [accessions[i] for i in train_idx], protein_to_graph_idx,
    )
    gc.collect()

    val_dataset = ProteinDataset(
        embeddings[val_idx], np.array(label_matrix[val_idx], dtype=np.float32), val_idx,
        [accessions[i] for i in val_idx], protein_to_graph_idx,
    )
    gc.collect()

    test_dataset = ProteinDataset(
        embeddings[test_idx], np.array(label_matrix[test_idx], dtype=np.float32), test_idx,
        [accessions[i] for i in test_idx], protein_to_graph_idx,
    )
    gc.collect()

    # Compute class weights from training set
    pos_weights = compute_class_weights(
        train_labels,
        cap=config["training"]["loss"].get("pos_weight_cap", 10.0),
    )
    del train_labels
    gc.collect()

    # Create data loaders  -  force num_workers=0 on Windows to avoid spawn errors
    batch_size = config["training"]["batch_size"]
    num_workers = config["project"].get("num_workers", 0)
    if _SAFE_NUM_WORKERS is not None:
        num_workers = _SAFE_NUM_WORKERS
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size, num_workers=num_workers,
    )

    # Save split indices and metadata
    split_data = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_accessions": [accessions[i] for i in train_idx],
        "val_accessions": [accessions[i] for i in val_idx],
        "test_accessions": [accessions[i] for i in test_idx],
        "pos_weights": pos_weights,
        "num_classes": label_matrix.shape[1],
        "embedding_dim": embeddings.shape[1],
    }
    save_pickle(split_data, os.path.join(processed_dir, "split_data.pkl"))

    logger.info("Dataset assembly complete!")
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    logger.info(f"  Num classes: {label_matrix.shape[1]}")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "pos_weights": pos_weights,
        "split_data": split_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Build final datasets")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    build_dataset(config)


if __name__ == "__main__":
    main()
