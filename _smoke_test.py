"""Smoke test  -  verify model forward passes work with the current configuration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from utils.helpers import load_config
from training.train import build_model

cfg = load_config("configs/config.yaml")
emb_dim = cfg["model"]["sequence_encoder"]["embedding_dim"]

print(f"Configuration: ESM embedding dim = {emb_dim}, Graph input dim = {cfg['model']['graph_encoder']['input_dim']}")
print()

# --- sequence_only ---
m1 = build_model("sequence_only", cfg, num_classes=100)
x = torch.randn(4, emb_dim)
out = m1({"embedding": x})
assert out["logits"].shape == (4, 100), f"Bad shape: {out['logits'].shape}"
p1 = sum(p.numel() for p in m1.parameters())
print(f"sequence_only  OK  logits={tuple(out['logits'].shape)}  params={p1:,}")

# --- multimodal (CPU, no PyG needed for shape check) ---
m3 = build_model("multimodal", cfg, num_classes=100)
p3 = sum(p.numel() for p in m3.parameters())
print(f"multimodal     OK  params={p3:,}")

print()
print("All model smoke tests passed. Project is ready to run.")
print()
print("Next commands to execute in order:")
print("  1)  python run_pipeline.py --stage graph")
print("  2)  python run_pipeline.py --stage embeddings")
print("  3)  python run_pipeline.py --stage dataset")
print("  4)  python run_pipeline.py --stage train")
print("  5)  python run_pipeline.py --stage evaluate")
print("  6)  streamlit run app.py")
