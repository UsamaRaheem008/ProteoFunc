"""Quick verification script  -  checks imports, config, and module loads."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  [OK]  {label}")
    except Exception as e:
        print(f"  [ERR] {label}: {e}")
        errors.append(label)


# ── utils ──────────────────────────────────────────────────────────────
check("utils.helpers", lambda: __import__("utils.helpers"))
check("utils.seed",    lambda: __import__("utils.seed"))
check("utils.logger",  lambda: __import__("utils.logger"))

# ── config ─────────────────────────────────────────────────────────────
def _check_config():
    from utils.helpers import load_config
    cfg = load_config("configs/config.yaml")
    esm = cfg["model"]["sequence_encoder"]["model_name"]
    dim = cfg["model"]["sequence_encoder"]["embedding_dim"]
    g_in = cfg["model"]["graph_encoder"]["input_dim"]
    nw   = cfg["project"]["num_workers"]
    ep   = cfg["training"]["epochs"]
    assert dim in (320, 640, 1280), f"Unexpected embedding_dim: {dim}"
    assert g_in >= dim, f"graph input_dim ({g_in}) should be >= embedding_dim ({dim})"
    assert nw == 0, f"num_workers not 0 on Windows: {nw}"
    assert ep > 0, f"epochs must be positive: {ep}"
    print(f"       ESM={esm}, dim={dim}, graph_in={g_in}, workers={nw}, epochs={ep}")

check("config.yaml values", _check_config)

# ── models ─────────────────────────────────────────────────────────────
check("models.sequence_only",   lambda: __import__("models.sequence_only"))
check("models.sequence_encoder",lambda: __import__("models.sequence_encoder"))
check("models.classifier",      lambda: __import__("models.classifier"))
check("models.fusion",          lambda: __import__("models.fusion"))

# ── training ───────────────────────────────────────────────────────────
check("training.losses",    lambda: __import__("training.losses"))
check("training.scheduler", lambda: __import__("training.scheduler"))

# ── trainer amp import ─────────────────────────────────────────────────
def _check_trainer_amp():
    import importlib.util, ast
    path = "training/trainer.py"
    src = open(path, encoding="utf-8").read()
    assert "from torch.amp import" in src or "torch.cuda.amp" in src, "AMP import missing"
    assert 'autocast("cuda")' in src or 'autocast()' in src, "autocast call missing"

check("trainer.py AMP imports",  _check_trainer_amp)

# ── build_dataset Windows fix ──────────────────────────────────────────
def _check_dataset_workers():
    src = open("preprocessing/build_dataset.py", encoding="utf-8").read()
    assert "_SAFE_NUM_WORKERS" in src, "_SAFE_NUM_WORKERS not found"
    assert 'sys.platform == "win32"' in src, "win32 check not found"

check("build_dataset Windows fix", _check_dataset_workers)

# ── build_graph vectorized ──────────────────────────────────────────────
def _check_graph_vectorized():
    src = open("preprocessing/build_graph.py", encoding="utf-8").read()
    assert "iterrows" not in src.split("def build_pyg_data")[1].split("def build_ppi_graph")[0], \
        "iterrows still in build_pyg_data"
    assert "k_sample" in src, "betweenness k_sample not found"

check("build_graph vectorized", _check_graph_vectorized)

# ── preprocessing download unicode ────────────────────────────────────
def _check_download_unicode():
    raw = open("preprocessing/download_data.py", "rb").read()
    assert b"\xe2\x80\xa6" not in raw, "Unicode ellipsis still present"  # UTF-8 for …

check("download_data no Unicode ellipsis", _check_download_unicode)

# ── Summary ────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED: {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
else:
    print(f"All {14 - len(errors)} checks passed. Project is ready.")
