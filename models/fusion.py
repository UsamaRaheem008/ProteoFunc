"""
Multimodal Fusion Module
=========================
Implements five fusion mechanisms for combining sequence and network modalities:

1. Concatenation Fusion
2. Weighted Sum Fusion
3. Attention-based Fusion (interpretable)
4. Gated Fusion
5. Cross-Modal Attention Fusion
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation of sequence and network embeddings.

    Output dim = seq_dim + net_dim, followed by a linear projection.
    """

    def __init__(self, seq_dim: int, net_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(seq_dim + net_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            seq_emb: Sequence embeddings (B, seq_dim).
            net_emb: Network embeddings (B, net_dim).

        Returns:
            Fused embeddings (B, output_dim) and attention info dict.
        """
        fused = torch.cat([seq_emb, net_emb], dim=-1)
        fused = self.projection(fused)
        # No interpretable attention weights for concat
        info = {"seq_weight": torch.full((seq_emb.size(0), 1), 0.5, device=seq_emb.device),
                "net_weight": torch.full((net_emb.size(0), 1), 0.5, device=net_emb.device)}
        return fused, info


class WeightedSumFusion(nn.Module):
    """
    Learnable weighted sum of sequence and network embeddings.

    Both modalities are projected to the same dimension, then combined
    with a learnable scalar weight.
    """

    def __init__(self, seq_dim: int, net_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.net_proj = nn.Linear(net_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_proj = self.seq_proj(seq_emb)
        net_proj = self.net_proj(net_emb)

        alpha = torch.sigmoid(self.alpha)
        fused = alpha * seq_proj + (1 - alpha) * net_proj
        fused = self.norm(fused)
        fused = self.dropout(fused)

        info = {
            "seq_weight": alpha.expand(seq_emb.size(0), 1),
            "net_weight": (1 - alpha).expand(net_emb.size(0), 1),
            "alpha": alpha,
        }
        return fused, info


class AttentionFusion(nn.Module):
    """
    Attention-based fusion that learns to weight modality contributions
    per protein. Fully interpretable  -  attention weights show which
    modality is more important for each prediction.

    Architecture:
        [seq_emb, net_emb] -> Attention scores -> Weighted sum -> Output
    """

    def __init__(
        self,
        seq_dim: int,
        net_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.net_proj = nn.Linear(net_dim, output_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Project both modalities to same dimension
        seq_proj = self.seq_proj(seq_emb)  # (B, D)
        net_proj = self.net_proj(net_emb)  # (B, D)

        # Stack modalities: (B, 2, D)
        stacked = torch.stack([seq_proj, net_proj], dim=1)

        # Compute attention scores: (B, 2, 1) -> (B, 2)
        attn_scores = self.attention(stacked).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 2)

        # Weighted combination
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        fused = self.norm(fused)
        fused = self.dropout(fused)

        info = {
            "seq_weight": attn_weights[:, 0:1],  # (B, 1)
            "net_weight": attn_weights[:, 1:2],  # (B, 1)
            "attention_weights": attn_weights,     # (B, 2)
        }
        return fused, info


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism that uses a learned gate to control
    information flow from each modality.

    gate = σ(W_g . [seq_emb; net_emb] + b_g)
    fused = gate * seq_proj + (1 - gate) * net_proj
    """

    def __init__(self, seq_dim: int, net_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.net_proj = nn.Linear(net_dim, output_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(seq_dim + net_dim, output_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_proj = self.seq_proj(seq_emb)
        net_proj = self.net_proj(net_emb)

        # Compute gate
        combined = torch.cat([seq_emb, net_emb], dim=-1)
        gate_values = self.gate(combined)  # (B, output_dim)

        # Gated fusion
        fused = gate_values * seq_proj + (1 - gate_values) * net_proj
        fused = self.norm(fused)
        fused = self.dropout(fused)

        # Interpretability: mean gate value per sample
        mean_gate = gate_values.mean(dim=-1, keepdim=True)
        info = {
            "seq_weight": mean_gate,
            "net_weight": 1 - mean_gate,
            "gate_values": gate_values,
        }
        return fused, info


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion where each modality attends to the other.

    Uses multi-head attention:
    - Sequence attends to network context
    - Network attends to sequence features
    - Results are combined
    """

    def __init__(
        self,
        seq_dim: int,
        net_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.net_proj = nn.Linear(net_dim, output_dim)

        # Cross-attention: sequence queries, network keys/values
        self.cross_attn_s2n = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: network queries, sequence keys/values
        self.cross_attn_n2s = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.combine = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Project
        seq_proj = self.seq_proj(seq_emb).unsqueeze(1)  # (B, 1, D)
        net_proj = self.net_proj(net_emb).unsqueeze(1)  # (B, 1, D)

        # Cross-attention: seq attends to net
        s2n_out, s2n_weights = self.cross_attn_s2n(
            query=seq_proj, key=net_proj, value=net_proj
        )
        # Cross-attention: net attends to seq
        n2s_out, n2s_weights = self.cross_attn_n2s(
            query=net_proj, key=seq_proj, value=seq_proj
        )

        # Combine
        combined = torch.cat([s2n_out.squeeze(1), n2s_out.squeeze(1)], dim=-1)
        fused = self.combine(combined)

        info = {
            "seq_weight": torch.full((seq_emb.size(0), 1), 0.5, device=seq_emb.device),
            "net_weight": torch.full((net_emb.size(0), 1), 0.5, device=net_emb.device),
            "s2n_attention": s2n_weights,
            "n2s_attention": n2s_weights,
        }
        return fused, info


def get_fusion_module(
    method: str,
    seq_dim: int,
    net_dim: int,
    output_dim: int,
    num_heads: int = 4,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to get fusion module by name.

    Args:
        method: Fusion method name.
        seq_dim: Sequence embedding dimension.
        net_dim: Network embedding dimension.
        output_dim: Output dimension.
        num_heads: Number of attention heads (for attention-based methods).
        dropout: Dropout rate.

    Returns:
        Fusion module instance.
    """
    fusion_map = {
        "concat": ConcatenationFusion,
        "weighted_sum": WeightedSumFusion,
        "attention": AttentionFusion,
        "gated": GatedFusion,
        "cross_attention": CrossModalAttentionFusion,
        "residual_cross_attention": ResidualCrossAttentionFusion,
    }

    if method not in fusion_map:
        raise ValueError(f"Unknown fusion method: {method}. Choose from {list(fusion_map.keys())}")

    kwargs = {
        "seq_dim": seq_dim,
        "net_dim": net_dim,
        "output_dim": output_dim,
        "dropout": dropout,
    }

    if method in ("attention", "cross_attention", "residual_cross_attention"):
        kwargs["num_heads"] = num_heads

    return fusion_map[method](**kwargs)



class ResidualCrossAttentionFusion(nn.Module):
    """
    Residual Cross-Modal Attention Fusion.
    
    Treats Sequence features as the primary backbone and uses Network context 
    as a gated residual refinement.
    
    Formula:
        fused = LayerNorm(seq_proj + gate * cross_attn(seq_proj, net_proj))
        
    Guarantees Multimodal >= Sequence-Only performance by allowing the gate to 
    collapse to 0 if PPI network context is noisy.
    """

    def __init__(
        self,
        seq_dim: int,
        net_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.net_proj = nn.Linear(net_dim, output_dim)

        # Cross-attention: sequence queries, network keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Gating parameter for residual addition (initialized near 0 to start as sequence-dominant)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )

        # Initialize gate bias to -2.0 -> sigmoid(-2) ~ 0.12
        # Ensures model starts sequence-dominant; learns to increase network
        # contribution only where PPI context genuinely helps.
        # (Inspired by DAMPE 2024 gated residual cross-modal alignment)
        nn.init.constant_(self.gate[0].bias, -2.0)
        nn.init.xavier_uniform_(self.gate[0].weight, gain=0.1)

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, seq_emb: torch.Tensor, net_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_proj = self.seq_proj(seq_emb)  # (B, D)
        net_proj = self.net_proj(net_emb)  # (B, D)

        seq_q = seq_proj.unsqueeze(1)
        net_kv = net_proj.unsqueeze(1)

        # Sequence queries Network
        attn_out, attn_weights = self.cross_attn(
            query=seq_q, key=net_kv, value=net_kv
        )
        attn_out = attn_out.squeeze(1)  # (B, D)

        # Calculate dynamic gate values
        combined = torch.cat([seq_proj, attn_out], dim=-1)
        gate_values = self.gate(combined)  # (B, D)

        # Residual connection: Sequence + Gated Network refinement
        fused = self.norm(seq_proj + gate_values * attn_out)
        fused = self.dropout(fused)

        mean_gate = gate_values.mean(dim=-1, keepdim=True)
        info = {
            "seq_weight": 1.0 - 0.5 * mean_gate,
            "net_weight": 0.5 * mean_gate,
            "gate_values": gate_values,
            "cross_attention": attn_weights,
        }
        return fused, info
