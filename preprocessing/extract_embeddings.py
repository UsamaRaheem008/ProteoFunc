"""
ESM-2 Embedding Extraction Module
===================================
Extracts protein sequence embeddings using the ESM-2 transformer model.

Usage:
    python preprocessing/extract_embeddings.py --config configs/config.yaml
"""

import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir
from utils.seed import get_device

logger = get_logger(__name__)


class ESM2EmbeddingExtractor:
    """
    Extracts protein sequence embeddings using the ESM-2 transformer model.

    Supports mean pooling, CLS token, and attention-weighted pooling strategies.
    Embeddings are cached to disk for reuse.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        max_length: int = 1024,
        pooling: str = "mean",
    ):
        """
        Initialize the ESM-2 embedding extractor.

        Args:
            model_name: HuggingFace model identifier for ESM-2.
            device: Compute device ("cuda", "cpu", or None for auto).
            max_length: Maximum sequence length for tokenization.
            pooling: Pooling strategy ("mean", "cls", "attention").
        """
        self.model_name = model_name
        self.device = get_device(device or "auto")
        self.max_length = max_length
        self.pooling = pooling

        logger.info(f"Loading ESM-2 model: {model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Pooling: {pooling}")

        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"  Embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def extract_single(self, sequence: str) -> np.ndarray:
        """
        Extract embedding for a single protein sequence.

        Args:
            sequence: Amino acid sequence string.

        Returns:
            Embedding vector as numpy array.
        """
        # Truncate if necessary
        if len(sequence) > self.max_length:
            sequence = sequence[: self.max_length]

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        embedding = self._pool(hidden_states, inputs["attention_mask"])
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(
        self, sequences: List[str], batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of sequences.

        Args:
            sequences: List of amino acid sequences.
            batch_size: Batch size for inference.

        Returns:
            Embedding matrix of shape (num_sequences, embedding_dim).
        """
        all_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
            batch = sequences[i : i + batch_size]

            # Truncate long sequences
            batch = [s[: self.max_length] if len(s) > self.max_length else s for s in batch]

            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state

                embeddings = self._pool(hidden_states, inputs["attention_mask"])
                all_embeddings.append(embeddings.cpu().numpy())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at batch {i}. Processing one-by-one...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    for seq in batch:
                        emb = self.extract_single(seq)
                        all_embeddings.append(emb.reshape(1, -1))
                else:
                    raise

            # Periodic memory cleanup
            if i % (batch_size * 10) == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()

        result = np.vstack(all_embeddings)
        logger.info(f"Extracted embeddings: shape {result.shape}")
        return result

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token-level representations into a single vector.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)

        Returns:
            Pooled embeddings of shape (batch, hidden_dim).
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        elif self.pooling == "mean":
            # Mask-aware mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        elif self.pooling == "attention":
            # Learned attention pooling (simple version)
            attn_weights = torch.softmax(
                hidden_states.mean(dim=-1) * attention_mask.float(), dim=-1
            )
            attn_weights = attn_weights.unsqueeze(-1)
            return (hidden_states * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


def extract_and_save_embeddings(config: Dict[str, Any]) -> np.ndarray:
    """
    Main embedding extraction pipeline.

    Args:
        config: Project configuration.

    Returns:
        Embedding matrix of shape (num_proteins, embedding_dim).
    """
    data_cfg = config["data"]
    model_cfg = config["model"]["sequence_encoder"]
    processed_dir = data_cfg["processed_dir"]
    embeddings_dir = data_cfg["embeddings_dir"]
    ensure_dir(embeddings_dir)

    # Check for cached embeddings
    embeddings_path = os.path.join(embeddings_dir, "esm2_embeddings.npy")
    accessions_path = os.path.join(embeddings_dir, "embedding_accessions.npy")

    if os.path.exists(embeddings_path) and os.path.exists(accessions_path):
        logger.info(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: shape {embeddings.shape}")
        return embeddings

    # Load proteins
    proteins_df = pd.read_csv(
        os.path.join(processed_dir, "proteins_annotated.tsv"), sep="\t"
    )
    sequences = proteins_df["sequence"].tolist()
    accessions = proteins_df["accession"].tolist()

    logger.info(f"Extracting ESM-2 embeddings for {len(sequences)} proteins")

    # Initialize extractor
    extractor = ESM2EmbeddingExtractor(
        model_name=model_cfg["model_name"],
        max_length=model_cfg["max_length"],
        pooling=model_cfg["pooling"],
    )

    # Extract embeddings
    embeddings = extractor.extract_batch(
        sequences, batch_size=model_cfg["batch_size"]
    )

    # Save
    np.save(embeddings_path, embeddings)
    np.save(accessions_path, np.array(accessions))
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Cleanup
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 protein embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    extract_and_save_embeddings(config)


if __name__ == "__main__":
    main()
