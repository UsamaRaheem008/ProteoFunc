"""
Main Training Script
=====================
Trains sequence-only, network-only, and multimodal models.

Usage:
    python training/train.py --config configs/config.yaml --model sequence_only
    python training/train.py --config configs/config.yaml --model network_only
    python training/train.py --config configs/config.yaml --model multimodal
    python training/train.py --config configs/config.yaml --model all
"""

import argparse
import gc
import os
import sys
from typing import Any, Dict

# Prevent PyTorch CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.helpers import load_config, load_pickle, ensure_dir, count_parameters, format_number
from utils.seed import set_seed, get_device
from training.losses import get_loss_function
from training.scheduler import get_scheduler
from training.trainer import Trainer
from preprocessing.build_dataset import build_dataset
from models.sequence_only import SequenceOnlyModel
from models.network_only import NetworkOnlyModel
from models.multimodal import MultimodalModel

logger = get_logger(__name__)


def build_model(
    model_type: str, config: Dict[str, Any], num_classes: int
) -> torch.nn.Module:
    """
    Build model based on type and config.

    Args:
        model_type: Model variant name.
        config: Full project config.
        num_classes: Number of GO term classes.

    Returns:
        Initialized model.
    """
    model_cfg = config["model"]
    seq_cfg = model_cfg["sequence_encoder"]
    graph_cfg = model_cfg["graph_encoder"]
    fusion_cfg = model_cfg["fusion"]
    cls_cfg = model_cfg["classifier"]

    if model_type == "sequence_only":
        model = SequenceOnlyModel(
            embedding_dim=seq_cfg["embedding_dim"],
            encoder_dim=fusion_cfg["hidden_dim"],
            num_classes=num_classes,
            hidden_dims=cls_cfg["hidden_dims"],
            dropout=cls_cfg["dropout"],
            activation=cls_cfg["activation"],
            norm=cls_cfg["norm"],
            residual=cls_cfg.get("residual", True),
        )
    elif model_type == "network_only":
        model = NetworkOnlyModel(
            input_dim=graph_cfg["input_dim"],
            hidden_dim=graph_cfg["hidden_dim"],
            output_dim=graph_cfg["output_dim"],
            num_layers=graph_cfg["num_layers"],
            num_classes=num_classes,
            aggregation=graph_cfg["aggregation"],
            graph_type=graph_cfg.get("type", "sage"),
            heads=graph_cfg.get("heads", 4),
            hidden_dims=cls_cfg["hidden_dims"],
            dropout=cls_cfg["dropout"],
            activation=cls_cfg["activation"],
            norm=cls_cfg["norm"],
            residual=cls_cfg.get("residual", True),
        )
    elif model_type == "multimodal":
        model = MultimodalModel(
            seq_input_dim=seq_cfg["embedding_dim"],
            seq_output_dim=fusion_cfg["hidden_dim"],
            graph_input_dim=graph_cfg["input_dim"],
            graph_hidden_dim=graph_cfg["hidden_dim"],
            graph_output_dim=graph_cfg["output_dim"],
            graph_num_layers=graph_cfg["num_layers"],
            graph_aggregation=graph_cfg["aggregation"],
            fusion_method=fusion_cfg["method"],
            fusion_dim=fusion_cfg["hidden_dim"],
            fusion_num_heads=fusion_cfg.get("num_heads", 4),
            num_classes=num_classes,
            classifier_hidden_dims=cls_cfg["hidden_dims"],
            dropout=cls_cfg["dropout"],
            activation=cls_cfg["activation"],
            norm=cls_cfg["norm"],
            residual=cls_cfg.get("residual", True),
            graph_type=graph_cfg.get("type", "sage"),
            heads=graph_cfg.get("heads", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Built {model_type} model: {format_number(count_parameters(model))} trainable params")
    return model


def train_model(model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a single model variant.

    Args:
        model_type: Model variant ("sequence_only", "network_only", "multimodal").
        config: Full project config.

    Returns:
        Training history.
    """
    logger.info("=" * 70)
    logger.info(f"TRAINING: {model_type.upper()}")
    logger.info("=" * 70)

    # Setup
    device = get_device(config["project"]["device"])
    set_seed(config["project"]["seed"])
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Build dataset
    dataset_result = build_dataset(config)
    train_loader = dataset_result["train_loader"]
    val_loader = dataset_result["val_loader"]
    pos_weights = dataset_result["pos_weights"].to(device)
    num_classes = dataset_result["split_data"]["num_classes"]

    # Load graph data for network/multimodal models
    graph_data = None
    if model_type in ("network_only", "multimodal"):
        pyg_path = os.path.join(data_cfg["processed_dir"], "pyg_graph_data.pt")
        if os.path.exists(pyg_path):
            graph_data = torch.load(pyg_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded graph data: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
        else:
            logger.error("Graph data not found! Run preprocessing/build_graph.py first.")
            return {}

    # Build model
    model = build_model(model_type, config, num_classes)
    model = model.to(device)

    # Multi-GPU support  -  DISABLED
    # DataParallel splits tensors along dim 0. Graph edge_index has shape [2, N],
    # so splitting gives each GPU 1 row instead of 2, breaking graph convolution.
    # Use DistributedDataParallel with proper graph handling if multi-GPU is needed.
    if torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs  -  using single GPU "
                     "(DataParallel is incompatible with graph edge_index shape [2, N])")

    # Loss function
    criterion = get_loss_function(
        name=train_cfg["loss"]["name"],
        pos_weight=pos_weights,
        focal_gamma=train_cfg["loss"].get("focal_gamma", 2.0),
        focal_alpha=train_cfg["loss"].get("focal_alpha", 0.25),
        gamma_neg=train_cfg["loss"].get("gamma_neg", 4.0),
        gamma_pos=train_cfg["loss"].get("gamma_pos", 1.0),
    )

    # Optimizer  -  use separate (lower) LR for graph encoder in multimodal
    # to prevent the randomly-initialized graph encoder from corrupting
    # the strong ESM-2 sequence signal during early fusion training.
    base_lr = train_cfg["optimizer"]["lr"]
    if model_type in ("multimodal",) and hasattr(model, "graph_encoder"):
        graph_params = list(model.graph_encoder.parameters())
        graph_param_ids = {id(p) for p in graph_params}
        other_params = [p for p in model.parameters() if id(p) not in graph_param_ids]
        param_groups = [
            {"params": other_params, "lr": base_lr},
            {"params": graph_params, "lr": base_lr * 0.1, "name": "graph_encoder"},
        ]
        logger.info(f"Using separate LR for graph encoder: {base_lr * 0.1:.2e} (10x lower)")
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        betas=tuple(train_cfg["optimizer"]["betas"]),
        eps=train_cfg["optimizer"]["eps"],
    )

    # Scheduler
    scheduler_dict = get_scheduler(
        name=train_cfg["scheduler"]["name"],
        optimizer=optimizer,
        total_epochs=train_cfg["epochs"],
        warmup_epochs=train_cfg["scheduler"]["warmup_epochs"],
        min_lr=train_cfg["scheduler"]["min_lr"],
        patience=train_cfg["scheduler"].get("patience", 10),
        factor=train_cfg["scheduler"].get("factor", 0.5),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler_dict=scheduler_dict,
        config=config,
        device=device,
        model_type=model_type,
        graph_data=graph_data,
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    # Clean up model & CUDA memory explicitly to avoid cross-stage memory retention
    del trainer, model, optimizer, criterion, train_loader, val_loader
    if graph_data is not None:
        del graph_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return history


def main():
    parser = argparse.ArgumentParser(description="Train protein function prediction models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["sequence_only", "network_only", "multimodal", "all"],
        help="Model type to train",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        for model_type in ["sequence_only", "network_only", "multimodal"]:
            train_model(model_type, config)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    else:
        train_model(args.model, config)


if __name__ == "__main__":
    main()
