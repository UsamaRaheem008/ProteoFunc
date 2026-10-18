"""
Trainer Module
===============
Unified trainer for all model variants with mixed-precision, gradient clipping,
early stopping, checkpoint management, and logging.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.0
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import ensure_dir, save_json

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.should_stop


class Trainer:
    """
    Unified trainer for sequence-only, network-only, and multimodal models.

    Supports:
    - Mixed precision (AMP) training
    - Gradient clipping
    - Early stopping
    - Checkpoint saving/loading
    - TensorBoard logging
    - Multi-GPU (DataParallel)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler_dict: Dict[str, Any],
        config: Dict[str, Any],
        device: torch.device,
        model_type: str = "sequence_only",
        graph_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            criterion: Loss function.
            scheduler_dict: Scheduler config dict with "scheduler" and "monitor" keys.
            config: Full project configuration.
            device: Compute device.
            model_type: One of "sequence_only", "network_only", "multimodal".
            graph_data: Graph data dict (for network/multimodal models).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler_dict["scheduler"]
        self.scheduler_monitor = scheduler_dict["monitor"]
        self.config = config
        self.device = device
        self.model_type = model_type
        self.graph_data = graph_data

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.max_grad_norm = train_cfg["gradient_clipping"]["max_norm"]
        self.accumulation_steps = train_cfg.get("accumulation_steps", 1)

        # Mixed precision
        self.use_amp = config["project"].get("mixed_precision", True) and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Early stopping
        es_cfg = train_cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            mode=es_cfg["mode"],
        )

        # Checkpointing
        self.save_dir = ensure_dir(train_cfg["checkpoint"]["save_dir"])
        self.save_best = train_cfg["checkpoint"]["save_best"]
        self.save_every = train_cfg["checkpoint"]["save_every"]
        self.best_metric = None

        # Move graph data to device
        if self.graph_data is not None:
            self._move_graph_to_device()

        # TensorBoard
        self.writer = None
        if config["logging"].get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                from pathlib import Path
                log_dir = ensure_dir(config["logging"]["log_dir"])
                self.writer = SummaryWriter(log_dir=str(Path(str(log_dir)) / model_type))
                logger.info(f"TensorBoard logging to {log_dir / model_type}")
            except ImportError:
                logger.warning("TensorBoard not available. Skipping.")

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_fmax": [],
            "val_auroc": [],
            "lr": [],
        }

    def _move_graph_to_device(self):
        """Move graph tensors to the compute device."""
        if self.graph_data is None:
            return
        for key in ["edge_index", "edge_weight", "x", "y"]:
            if key in self.graph_data and self.graph_data[key] is not None:
                self.graph_data[key] = self.graph_data[key].to(self.device)

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute one forward pass based on model type.

        Returns:
            Raw logits tensor.
        """
        if self.model_type == "sequence_only":
            batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch_gpu)
            return output["logits"]

        elif self.model_type == "network_only":
            node_indices = batch["graph_idx"].to(self.device)
            output = self.model(
                x=self.graph_data["x"],
                edge_index=self.graph_data["edge_index"],
                edge_weight=self.graph_data.get("edge_weight"),
                node_indices=node_indices,
            )
            return output["logits"]

        elif self.model_type == "multimodal":
            seq_emb = batch["embedding"].to(self.device)
            node_indices = batch["graph_idx"].to(self.device)
            output = self.model(
                seq_embeddings=seq_emb,
                node_features=self.graph_data["x"],
                edge_index=self.graph_data["edge_index"],
                edge_weight=self.graph_data.get("edge_weight"),
                node_indices=node_indices,
            )
            return output["logits"]

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(pbar):
            labels = batch["label"].to(self.device)

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast("cuda"):
                    logits = self._forward_step(batch)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self._forward_step(batch)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast("cuda"):
                    logits = self._forward_step(batch)
                    loss = self.criterion(logits, labels)
            else:
                logits = self._forward_step(batch)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.sigmoid(all_logits)

        # Compute metrics
        metrics = self._compute_quick_metrics(all_probs.numpy(), all_labels.numpy())
        metrics["val_loss"] = total_loss / max(num_batches, 1)

        return metrics

    def _compute_quick_metrics(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute quick validation metrics (F-max and AUROC)."""
        from sklearn.metrics import roc_auc_score

        # F-max computation
        best_fmax = 0.0
        best_threshold = 0.5

        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= t).astype(float)
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()

            precision = tp / max(tp + fp, 1e-10)
            recall = tp / max(tp + fn, 1e-10)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            if f1 > best_fmax:
                best_fmax = f1
                best_threshold = t

        # AUROC (macro)
        try:
            # Filter out classes with no positive samples
            valid_cols = labels.sum(axis=0) > 0
            if valid_cols.sum() > 0:
                auroc = roc_auc_score(
                    labels[:, valid_cols], probs[:, valid_cols], average="macro"
                )
            else:
                auroc = 0.0
        except Exception:
            auroc = 0.0

        return {
            "val_fmax": float(best_fmax),
            "val_auroc": float(auroc),
            "best_threshold": float(best_threshold),
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"{self.model_type}_epoch_{epoch}.pt"

        path = os.path.join(str(self.save_dir), filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "model_type": self.model_type,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training: {self.model_type}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        start_time = time.time()

        # Graph encoder warmup: freeze it for the first N epochs so the fusion
        # module learns to rely on the strong sequence backbone first, then
        # unfreeze so graph context can refine predictions.
        graph_warmup_epochs = 5
        has_graph_encoder = self.model_type == "multimodal" and hasattr(self.model, "graph_encoder")
        if has_graph_encoder:
            for param in self.model.graph_encoder.parameters():
                param.requires_grad = False
            logger.info(f"Graph encoder FROZEN for first {graph_warmup_epochs} warmup epochs")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Unfreeze graph encoder after warmup
            if has_graph_encoder and epoch == graph_warmup_epochs + 1:
                for param in self.model.graph_encoder.parameters():
                    param.requires_grad = True
                logger.info(f"Graph encoder UNFROZEN at epoch {epoch}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler_monitor and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_metrics.get(self.scheduler_monitor, val_metrics["val_fmax"]))
            else:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_fmax"].append(val_metrics["val_fmax"])
            self.history["val_auroc"].append(val_metrics["val_auroc"])
            self.history["lr"].append(current_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Metrics/val_fmax", val_metrics["val_fmax"], epoch)
                self.writer.add_scalar("Metrics/val_auroc", val_metrics["val_auroc"], epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # Logging
            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"F-max: {val_metrics['val_fmax']:.4f} | "
                f"AUROC: {val_metrics['val_auroc']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if self.save_best:
                monitor_val = val_metrics.get("val_fmax", 0)
                if self.best_metric is None or monitor_val > self.best_metric:
                    self.best_metric = monitor_val
                    self.save_checkpoint(epoch, val_metrics, f"{self.model_type}_best.pt")

            # Periodic checkpoint
            if self.save_every and epoch % self.save_every == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Early stopping
            if self.early_stopping.step(val_metrics.get("val_fmax", 0)):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best F-max: {self.best_metric:.4f}")

        # Save training history
        save_json(self.history, os.path.join(str(self.save_dir), f"{self.model_type}_history.json"))

        if self.writer:
            self.writer.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.history
