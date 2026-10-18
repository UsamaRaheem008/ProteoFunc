"""
Lightweight Test Suite for ProteoFunc
=====================================
Verifies all 6 code enhancements, model architectures,
loss functions, and figure generation without requiring heavy GPU training.
"""

import os
import sys
import torch
import numpy as np

# Fix Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config, ensure_dir
from models.fusion import ResidualCrossAttentionFusion, get_fusion_module
from models.multimodal import MultimodalModel
from training.losses import get_loss_function, AsymmetricLoss
from training.train import build_model

def test_1_gate_initialization():
    print("[TEST 1/5] Testing Gate Initialization in ResidualCrossAttentionFusion...")
    fusion = ResidualCrossAttentionFusion(seq_dim=512, net_dim=256, output_dim=512, num_heads=4)
    gate_bias = fusion.gate[0].bias.data.mean().item()
    gate_init_val = torch.sigmoid(torch.tensor(gate_bias)).item()
    print(f"  - Gate bias: {gate_bias:.4f} (Target: -2.0000)")
    print(f"  - Gate initial activation: {gate_init_val:.4f} (Target: ~0.1192)")
    assert abs(gate_bias - (-2.0)) < 1e-4, "Gate bias should be initialized to -2.0"
    assert abs(gate_init_val - 0.1192) < 0.01, "Gate activation should start around 0.12 (sequence-dominant)"
    print("  [OK] PASS: Gate initialization is sequence-dominant!\n")

def test_2_asl_loss_factory():
    print("[TEST 2/5] Testing Asymmetric Loss (ASL) Factory Integration...")
    loss_fn = get_loss_function("asymmetric", gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    assert isinstance(loss_fn, AsymmetricLoss), "Should return AsymmetricLoss instance"
    assert loss_fn.gamma_neg == 4.0, "gamma_neg not set"
    assert loss_fn.gamma_pos == 1.0, "gamma_pos not set"
    
    # Test forward pass with dummy logits and targets
    logits = torch.randn(8, 10)
    targets = (torch.rand(8, 10) > 0.8).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0, "Loss must be non-negative"
    print(f"  - ASL Loss value on dummy data: {loss.item():.4f}")
    print("  [OK] PASS: ASL Loss factory & forward pass working!\n")

def test_3_multimodal_forward_pass():
    print("[TEST 3/5] Testing Multimodal Model Architecture & Forward Pass...")
    config = load_config("configs/config.yaml")
    model = build_model("multimodal", config, num_classes=50)
    
    # Dummy inputs
    seq_emb = torch.randn(4, config["model"]["sequence_encoder"]["embedding_dim"])
    node_feat = torch.randn(10, config["model"]["graph_encoder"]["input_dim"])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    node_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = model(
        seq_embeddings=seq_emb,
        node_features=node_feat,
        edge_index=edge_index,
        node_indices=node_indices
    )

    assert "logits" in out, "Missing logits"
    assert out["logits"].shape == (4, 50), f"Incorrect logits shape: {out['logits'].shape}"
    assert "fusion_info" in out, "Missing fusion_info"
    assert "seq_weight" in out["fusion_info"], "Missing seq_weight"
    print(f"  - Fused output shape: {out['fused_embedding'].shape}")
    print(f"  - Mean sequence weight: {out['fusion_info']['seq_weight'].mean().item():.4f}")
    print("  [OK] PASS: Multimodal forward pass & fusion tracking working!\n")

def test_4_output_directory_structure():
    print("[TEST 4/5] Testing Output Paths & Directory Alignment...")
    dirs = [
        "ProteoFunc_outputs/checkpoints",
        "ProteoFunc_outputs/evaluation",
        "ProteoFunc_outputs/figures",
        "ProteoFunc_outputs/explainability",
    ]
    for d in dirs:
        path = ensure_dir(d)
        assert os.path.exists(path), f"Directory {d} not created"
    print("  [OK] PASS: All ProteoFunc_outputs directories verified!\n")

def test_5_figure_generation():
    print("[TEST 5/5] Testing Figure & Plot Generation...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "finalize.py"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    assert result.returncode == 0, f"finalize.py failed: {result.stderr}"
    
    figures_dir = "ProteoFunc_outputs/figures"
    fig_files = os.listdir(figures_dir)
    print(f"  - Generated {len(fig_files)} figures in {figures_dir}:")
    for f in sorted(fig_files):
        print(f"    - {f}")
    assert len(fig_files) > 0, "No figures were generated"
    print("  [OK] PASS: All figures generated successfully!\n")

if __name__ == "__main__":
    print("======================================================")
    print("      RUNNING PROTEOFUNC LIGHTWEIGHT TEST SUITE      ")
    print("======================================================\n")
    test_1_gate_initialization()
    test_2_asl_loss_factory()
    test_3_multimodal_forward_pass()
    test_4_output_directory_structure()
    test_5_figure_generation()
    print("======================================================")
    print("   ALL LIGHTWEIGHT TESTS PASSED SUCCESSFULLY! (5/5)   ")
    print("======================================================")
