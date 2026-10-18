"""
Inference Module  -  Real-Time Protein Function Prediction
=========================================================
Provides ProteinFunctionPredictor for on-the-fly prediction
from raw amino acid sequences without needing pre-computed embeddings.

Pipeline:
    Raw sequence -> ESM-2 embedding -> SequenceOnlyModel -> GO predictions
                                   ↗
    (optional) PPI graph context -> MultimodalModel

Usage:
    from inference.predict import ProteinFunctionPredictor

    predictor = ProteinFunctionPredictor(
        model_type="sequence_only",
        checkpoint_path="outputs/checkpoints/sequence_only_best.pt",
        config_path="configs/config.yaml",
    )
    results = predictor.predict("MKTAYIAKQRQISFVKSHFSRQ...")
"""

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.helpers import load_config, load_pickle
from utils.seed import get_device

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# QuickGO API helpers
# ──────────────────────────────────────────────────────────────────────────────

QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
_go_name_cache: Dict[str, str] = {}


def fetch_go_name(go_id: str, timeout: int = 5) -> str:
    """
    Fetch GO term name from EBI QuickGO API with local cache.

    Args:
        go_id: GO term identifier (e.g. "GO:0003674").
        timeout: Request timeout in seconds.

    Returns:
        Human-readable GO term name.
    """
    if go_id in _go_name_cache:
        return _go_name_cache[go_id]

    try:
        url = QUICKGO_URL.format(go_id=go_id)
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                name = results[0].get("name", go_id)
                _go_name_cache[go_id] = name
                return name
    except Exception:
        pass

    _go_name_cache[go_id] = go_id
    return go_id


def fetch_go_names_batch(go_ids: List[str], timeout: int = 10) -> Dict[str, str]:
    """
    Batch-fetch GO term names. Falls back to individual calls.

    Args:
        go_ids: List of GO IDs.
        timeout: Request timeout.

    Returns:
        Dictionary mapping GO ID -> name.
    """
    names: Dict[str, str] = {}
    to_fetch = [gid for gid in go_ids if gid not in _go_name_cache]

    if to_fetch:
        # Try comma-separated batch endpoint
        try:
            ids_str = ",".join(to_fetch[:50])  # API limit
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids_str}"
            resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
            if resp.status_code == 200:
                data = resp.json()
                for result in data.get("results", []):
                    gid = result.get("id", "")
                    name = result.get("name", gid)
                    _go_name_cache[gid] = name
        except Exception:
            pass

        # Fill any remaining
        for gid in to_fetch:
            if gid not in _go_name_cache:
                fetch_go_name(gid)

    for gid in go_ids:
        names[gid] = _go_name_cache.get(gid, gid)

    return names


# ──────────────────────────────────────────────────────────────────────────────
# ESM-2 Embedding Extractor
# ──────────────────────────────────────────────────────────────────────────────

class ESM2Embedder:
    """
    On-the-fly ESM-2 sequence embedding extractor.

    Uses the smaller ESM-2 model (esm2_t6_8M_UR50D) for fast inference
    so predictions can be returned in seconds.
    """

    # Use lighter model for inference speed
    DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
    DEFAULT_DIM   = 320

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace ESM-2 model identifier.
            device: Compute device.
        """
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load model to avoid startup overhead."""
        if self._model is None:
            logger.info(f"Loading ESM-2 model: {self.model_name}")
            from transformers import EsmTokenizer, EsmModel
            self._tokenizer = EsmTokenizer.from_pretrained(self.model_name)
            self._model = EsmModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info("ESM-2 model loaded.")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension based on model name."""
        dims = {
            "facebook/esm2_t6_8M_UR50D":   320,
            "facebook/esm2_t12_35M_UR50D":  480,
            "facebook/esm2_t30_150M_UR50D": 640,
            "facebook/esm2_t33_650M_UR50D": 1280,
            "facebook/esm2_t36_3B_UR50D":   2560,
        }
        return dims.get(self.model_name, self.DEFAULT_DIM)

    @torch.no_grad()
    def embed(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """
        Embed a single protein sequence.

        Args:
            sequence: Amino acid sequence string.
            max_length: Maximum token length.

        Returns:
            Mean-pooled embedding (D,).
        """
        self._load()
        sequence = sequence.strip().upper()
        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0)
        return embedding.cpu().numpy()

    @torch.no_grad()
    def embed_batch(
        self, sequences: List[str], max_length: int = 1024, batch_size: int = 8
    ) -> np.ndarray:
        """
        Embed a list of protein sequences.

        Args:
            sequences: List of amino acid sequences.
            max_length: Maximum token length.
            batch_size: Batch size for embedding.

        Returns:
            Embedding matrix (N, D).
        """
        self._load()
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self._tokenizer(
                batch_seqs,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            # Mean-pool over non-padding tokens (exclude CLS/EOS)
            attention_mask = inputs["attention_mask"]
            token_embs = outputs.last_hidden_state
            for j in range(len(batch_seqs)):
                mask = attention_mask[j, 1:-1].bool()
                emb = token_embs[j, 1:-1][mask].mean(dim=0)
                embeddings.append(emb.cpu().numpy())
        return np.stack(embeddings)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Result
# ──────────────────────────────────────────────────────────────────────────────

class PredictionResult:
    """Container for a single protein's prediction results."""

    def __init__(
        self,
        sequence: str,
        predictions: List[Dict[str, Any]],
        embedding: Optional[np.ndarray] = None,
        inference_time: float = 0.0,
        model_type: str = "sequence_only",
    ):
        self.sequence = sequence
        self.predictions = predictions       # list of {go_id, go_name, namespace, probability}
        self.embedding = embedding
        self.inference_time = inference_time
        self.model_type = model_type

    @property
    def top_predictions(self) -> List[Dict[str, Any]]:
        """Return predictions sorted by probability descending."""
        return sorted(self.predictions, key=lambda x: x["probability"], reverse=True)

    def by_namespace(self) -> Dict[str, List[Dict]]:
        """Group predictions by GO namespace."""
        namespaces: Dict[str, List] = {"MF": [], "BP": [], "CC": [], "unknown": []}
        for pred in self.predictions:
            ns = pred.get("namespace", "unknown")
            namespaces.setdefault(ns, []).append(pred)
        return namespaces

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_length": len(self.sequence),
            "model_type": self.model_type,
            "inference_time_ms": round(self.inference_time * 1000, 1),
            "num_predictions": len(self.predictions),
            "predictions": self.top_predictions,
        }

    def __repr__(self) -> str:
        top = self.top_predictions[:3]
        top_str = ", ".join(f"{p['go_id']}({p['probability']:.2f})" for p in top)
        return (
            f"PredictionResult(len={len(self.sequence)}, "
            f"n_preds={len(self.predictions)}, top=[{top_str}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main Predictor Class
# ──────────────────────────────────────────────────────────────────────────────

class ProteinFunctionPredictor:
    """
    High-level interface for protein function prediction.

    Supports:
    - sequence_only: Uses pre-computed or on-the-fly ESM-2 embeddings.
    - multimodal:    Adds PPI graph context (requires graph data).

    Example:
        predictor = ProteinFunctionPredictor(
            model_type="sequence_only",
            checkpoint_path="outputs/checkpoints/sequence_only_best.pt",
        )
        result = predictor.predict("MKTAYIAKQRQISFVKSHFSRQ...")
        for pred in result.top_predictions[:5]:
            print(pred["go_id"], pred["go_name"], pred["probability"])
    """

    def __init__(
        self,
        model_type: str = "sequence_only",
        checkpoint_path: Optional[str] = None,
        config_path: str = "configs/config.yaml",
        go_term_list: Optional[List[str]] = None,
        go_metadata_path: Optional[str] = None,
        graph_data_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.3,
        top_k: int = 20,
        fetch_go_names: bool = True,
    ):
        """
        Args:
            model_type: "sequence_only" or "multimodal".
            checkpoint_path: Path to trained model checkpoint (.pt).
            config_path: Path to project config YAML.
            go_term_list: Ordered list of GO term IDs matching model output.
            go_metadata_path: Path to GO term metadata TSV.
            graph_data_path: Path to graph data pickle (for multimodal).
            device: "cuda", "cpu", or "auto".
            threshold: Probability threshold for positive predictions.
            top_k: Maximum number of predictions to return.
            fetch_go_names: Whether to fetch GO term names from QuickGO API.
        """
        self.model_type = model_type
        self.threshold = threshold
        self.top_k = top_k
        self.fetch_go_names_flag = fetch_go_names
        self.config_path = config_path

        # Load config
        self.config = load_config(config_path)

        # Device
        device_str = device or self.config["project"].get("device", "auto")
        self.device = get_device(device_str)
        logger.info(f"Predictor using device: {self.device}")

        # GO term metadata
        self.go_terms: List[str] = []
        self.go_namespace: Dict[str, str] = {}
        self._load_go_metadata(go_term_list, go_metadata_path)

        # ESM-2 embedder for on-the-fly inference (must match sequence_encoder model_name)
        esm_model = self.config["model"]["sequence_encoder"].get(
            "model_name", "facebook/esm2_t33_650M_UR50D"
        )
        self.embedder = ESM2Embedder(model_name=esm_model, device=self.device)

        # Graph data (multimodal or network_only)
        self.graph_data = None
        self.protein_to_idx: Dict[str, int] = {}
        if model_type in ["multimodal", "network_only"]:
            if not graph_data_path:
                graph_data_path = os.path.join(
                    self.config["data"]["processed_dir"], "pyg_graph_data.pt"
                )
            if os.path.exists(graph_data_path):
                self._load_graph_data(graph_data_path)
            else:
                logger.warning(f"Graph data not found at {graph_data_path}")

        # Load model
        self.model: Optional[nn.Module] = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
        else:
            logger.warning(
                f"No checkpoint at '{checkpoint_path}'. Model not loaded. "
                "Call load_checkpoint() before predicting."
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_go_metadata(
        self,
        go_term_list: Optional[List[str]],
        go_metadata_path: Optional[str],
    ) -> None:
        """Load GO term list and namespace mapping."""
        if go_term_list:
            self.go_terms = go_term_list
        elif go_metadata_path and os.path.exists(go_metadata_path):
            import pandas as pd
            df = pd.read_csv(go_metadata_path, sep="\t")
            self.go_terms = df["go_id"].tolist()
            if "namespace" in df.columns:
                self.go_namespace = dict(zip(df["go_id"], df["namespace"]))
        else:
            # Try default path
            default_path = os.path.join(
                self.config["data"]["processed_dir"], "go_term_metadata.tsv"
            )
            if os.path.exists(default_path):
                import pandas as pd
                df = pd.read_csv(default_path, sep="\t")
                self.go_terms = df["go_id"].tolist()
                if "namespace" in df.columns:
                    self.go_namespace = dict(zip(df["go_id"], df["namespace"]))
                logger.info(f"Loaded {len(self.go_terms)} GO terms from {default_path}")

    def _load_graph_data(self, path: str) -> None:
        """Load PPI graph data (PyG format) for multimodal/network inference."""
        try:
            self.graph_data = torch.load(path, map_location=self.device, weights_only=False)
            # Move tensors to device
            for key in ["edge_index", "edge_weight", "x"]:
                val = getattr(self.graph_data, key, None) if hasattr(self.graph_data, key) else self.graph_data.get(key) if isinstance(self.graph_data, dict) else None
                if val is not None and isinstance(val, torch.Tensor):
                    if isinstance(self.graph_data, dict):
                        self.graph_data[key] = val.to(self.device)
                    else:
                        setattr(self.graph_data, key, val.to(self.device))

            # Extract protein_to_idx mapping
            if isinstance(self.graph_data, dict):
                self.protein_to_idx = self.graph_data.get("protein_to_idx", {})
            else:
                self.protein_to_idx = getattr(self.graph_data, "protein_to_idx", {}) or {}

            n_nodes = self.graph_data.get("num_nodes", "?") if isinstance(self.graph_data, dict) else getattr(self.graph_data, "num_nodes", "?")
            logger.info(f"Graph data loaded: {n_nodes} nodes, {len(self.protein_to_idx)} mapped proteins")
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            import traceback
            traceback.print_exc()

    def _load_model(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        from training.train import build_model

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_config = checkpoint.get("config", self.config)
        num_classes = len(self.go_terms) if self.go_terms else \
            checkpoint.get("metrics", {}).get("num_classes", 100)

        self.model = build_model(self.model_type, ckpt_config, num_classes)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            logger.warning(f"Strict state_dict load failed: {e}. Retrying with strict=False...")
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from: {checkpoint_path}")

    # ── Public API ───────────────────────────────────────────────────────────

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file."""
        self._load_model(checkpoint_path)

    def set_threshold(self, threshold: float) -> None:
        """Update the probability threshold for predictions."""
        self.threshold = threshold

    @torch.no_grad()
    def predict(
        self,
        sequence: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        node_index: Optional[int] = None,
    ) -> PredictionResult:
        """
        Predict GO functions for a single protein sequence.

        Args:
            sequence: Amino acid sequence string.
            threshold: Override default probability threshold.
            top_k: Override default top-K.
            node_index: Graph node index for network/multimodal models.
                        If None and model needs graph, uses -1 (fallback embedding).

        Returns:
            PredictionResult with GO term probabilities.
        """
        t0 = time.time()
        thr = threshold if threshold is not None else self.threshold
        k   = top_k    if top_k    is not None else self.top_k

        # Validate sequence
        sequence = self._validate_sequence(sequence)

        # Extract embedding
        embedding_np = self.embedder.embed(sequence)

        # Adapt embedding dimension if needed
        embedding_np = self._adapt_embedding(embedding_np)

        probs = self._run_model_single(embedding_np, node_index=node_index)

        # Build prediction list
        predictions = self._build_predictions(probs, thr, k)

        elapsed = time.time() - t0
        return PredictionResult(
            sequence=sequence,
            predictions=predictions,
            embedding=embedding_np,
            inference_time=elapsed,
            model_type=self.model_type,
        )

    @torch.no_grad()
    def predict_by_accession(
        self,
        accession: str,
        sequence: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> PredictionResult:
        """
        Predict GO functions using a UniProt accession ID.

        For network-only and multimodal models, this looks up the protein
        in the PPI graph to get proper graph context. For sequence-only,
        it fetches the sequence from UniProt and runs standard prediction.

        Args:
            accession: UniProt accession ID (e.g. 'P04637').
            sequence: Pre-fetched sequence (optional, fetched if not provided).
            threshold: Override probability threshold.
            top_k: Override top-K.

        Returns:
            PredictionResult with GO term probabilities.
        """
        # Get sequence if not provided
        if not sequence:
            try:
                import requests
                url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    sequence = "".join(l.strip() for l in lines if not l.startswith(">"))
                else:
                    raise ValueError(f"Could not fetch sequence for {accession}")
            except Exception as e:
                raise ValueError(f"Failed to fetch sequence for {accession}: {e}")

        # Look up node index in the PPI graph
        node_index = None
        if self.model_type in ["multimodal", "network_only"] and self.protein_to_idx:
            node_index = self.protein_to_idx.get(accession, None)
            if node_index is not None:
                logger.info(f"Protein {accession} found in PPI graph at node index {node_index}")
            else:
                logger.warning(f"Protein {accession} not found in PPI graph  -  using fallback embedding")

        return self.predict(
            sequence=sequence,
            threshold=threshold,
            top_k=top_k,
            node_index=node_index if node_index is not None else -1,
        )

    def get_protein_graph_info(self, accession: str) -> Dict[str, Any]:
        """
        Check if a protein is in the PPI graph and return info.

        Args:
            accession: UniProt accession ID.

        Returns:
            Dict with 'in_graph', 'node_index', 'num_total_nodes'.
        """
        node_idx = self.protein_to_idx.get(accession, None)
        return {
            "in_graph": node_idx is not None,
            "node_index": node_idx,
            "num_total_nodes": len(self.protein_to_idx),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: List[str],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Predict GO functions for multiple sequences.

        Args:
            sequences: List of amino acid sequences.
            threshold: Override default probability threshold.
            top_k: Override default top-K.

        Returns:
            List of PredictionResult objects.
        """
        t0 = time.time()
        thr = threshold if threshold is not None else self.threshold
        k   = top_k    if top_k    is not None else self.top_k

        sequences = [self._validate_sequence(s) for s in sequences]
        embeddings_np = self.embedder.embed_batch(sequences)
        embeddings_np = np.stack([self._adapt_embedding(e) for e in embeddings_np])

        probs_batch = self._run_model_batch(embeddings_np)

        results = []
        for i, seq in enumerate(sequences):
            t_i = time.time()
            predictions = self._build_predictions(probs_batch[i], thr, k)
            results.append(PredictionResult(
                sequence=seq,
                predictions=predictions,
                embedding=embeddings_np[i],
                inference_time=time.time() - t_i,
                model_type=self.model_type,
            ))

        logger.info(
            f"Batch prediction: {len(sequences)} sequences in "
            f"{time.time() - t0:.2f}s"
        )
        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    def _validate_sequence(self, sequence: str) -> str:
        """Clean and validate amino acid sequence."""
        sequence = sequence.strip().upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYBZXOU")
        cleaned = "".join(c for c in sequence if c in valid_aa)
        if len(cleaned) < 10:
            raise ValueError(
                f"Sequence too short ({len(cleaned)} AAs). Minimum is 10."
            )
        if len(cleaned) != len(sequence):
            logger.warning(
                f"Removed {len(sequence) - len(cleaned)} invalid characters from sequence."
            )
        return cleaned

    def _adapt_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Adapt embedding dimension if model expects a different size.
        Uses zero-padding or linear interpolation.
        """
        model_dim = self.config["model"]["sequence_encoder"].get("embedding_dim", 1280)
        if embedding.shape[0] == model_dim:
            return embedding
        # Pad or truncate
        if embedding.shape[0] < model_dim:
            padded = np.zeros(model_dim, dtype=np.float32)
            padded[:embedding.shape[0]] = embedding
            return padded
        return embedding[:model_dim]

    def _get_graph_tensor(self, key: str):
        """Safely retrieve a tensor from graph_data (supports both dict and PyG Data)."""
        if isinstance(self.graph_data, dict):
            return self.graph_data.get(key)
        return getattr(self.graph_data, key, None)

    def _run_model_single(self, embedding_np: np.ndarray, node_index: Optional[int] = None) -> np.ndarray:
        """Run model forward pass for a single embedding."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        emb_tensor = torch.tensor(embedding_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.model_type == "sequence_only":
            output = self.model({"embedding": emb_tensor})
        elif self.model_type in ["multimodal", "network_only"]:
            if self.graph_data is None:
                raise RuntimeError(f"Graph data required for {self.model_type} inference.")

            idx_val = node_index if node_index is not None else -1
            node_idx = torch.tensor([idx_val], dtype=torch.long).to(self.device)

            x = self._get_graph_tensor("x")
            edge_index = self._get_graph_tensor("edge_index")
            edge_weight = self._get_graph_tensor("edge_weight")

            if self.model_type == "multimodal":
                output = self.model(
                    seq_embeddings=emb_tensor,
                    node_features=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    node_indices=node_idx,
                )
            else:
                output = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    node_indices=node_idx,
                )
        else:
            raise ValueError(f"Unsupported model type for inference: {self.model_type}")

        return output["probabilities"].squeeze(0).cpu().numpy()

    def _run_model_batch(self, embeddings_np: np.ndarray) -> np.ndarray:
        """Run model forward pass for a batch of embeddings."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        emb_tensor = torch.tensor(embeddings_np, dtype=torch.float32).to(self.device)

        if self.model_type == "sequence_only":
            output = self.model({"embedding": emb_tensor})
        elif self.model_type in ["multimodal", "network_only"]:
            if self.graph_data is None:
                raise RuntimeError(f"Graph data required for {self.model_type} inference.")
            n = emb_tensor.size(0)
            node_idx = torch.full((n,), -1, dtype=torch.long).to(self.device)
            x = self._get_graph_tensor("x")
            edge_index = self._get_graph_tensor("edge_index")
            edge_weight = self._get_graph_tensor("edge_weight")
            if self.model_type == "multimodal":
                output = self.model(
                    seq_embeddings=emb_tensor,
                    node_features=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    node_indices=node_idx,
                )
            else:
                output = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    node_indices=node_idx,
                )
        else:
            raise ValueError(f"Unsupported model type for inference: {self.model_type}")

        return output["probabilities"].cpu().numpy()

    def _build_predictions(
        self,
        probs: np.ndarray,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Build structured prediction list from probability array.

        Args:
            probs: Probability array (num_classes,).
            threshold: Minimum probability to include.
            top_k: Maximum predictions to return.

        Returns:
            List of prediction dicts sorted by probability descending.
        """
        # Sort by probability descending
        sorted_idx = np.argsort(probs)[::-1]

        predictions = []
        go_ids_to_fetch = []

        for idx in sorted_idx[:top_k]:
            p = float(probs[idx])
            if p < threshold:
                break

            go_id = self.go_terms[idx] if idx < len(self.go_terms) else f"GO:{idx:07d}"
            namespace = self.go_namespace.get(go_id, "unknown")

            predictions.append({
                "go_id":       go_id,
                "go_name":     go_id,   # filled in below
                "namespace":   namespace,
                "probability": round(p, 4),
                "rank":        len(predictions) + 1,
            })
            go_ids_to_fetch.append(go_id)

        # Fetch GO names in one batch call
        if self.fetch_go_names_flag and go_ids_to_fetch:
            try:
                names = fetch_go_names_batch(go_ids_to_fetch)
                for pred in predictions:
                    pred["go_name"] = names.get(pred["go_id"], pred["go_id"])
            except Exception:
                pass

        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for display."""
        info = {
            "model_type":  self.model_type,
            "device":      str(self.device),
            "num_go_terms": len(self.go_terms),
            "threshold":   self.threshold,
            "esm2_model":  self.embedder.model_name,
        }
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable    = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["total_params"]     = total_params
            info["trainable_params"] = trainable
        return info
