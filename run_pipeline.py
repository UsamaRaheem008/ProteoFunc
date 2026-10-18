"""
Master Pipeline Script
=======================
Orchestrates the full Multimodal Protein Function Prediction pipeline:
  1. Data download
  2. Sequence processing & GO annotation
  3. Graph construction
  4. ESM-2 embedding extraction
  5. Dataset assembly
  6. Model training (sequence-only, network-only, multimodal)
  7. Evaluation & comparison
  8. Explainability analysis

Usage:
    python run_pipeline.py --config configs/config.yaml --stage all
    python run_pipeline.py --config configs/config.yaml --stage preprocess
    python run_pipeline.py --config configs/config.yaml --stage train
    python run_pipeline.py --config configs/config.yaml --stage evaluate
    python run_pipeline.py --config configs/config.yaml --stage explain
"""

import argparse
import io
import os
import sys
import time
from typing import List

# Fix Windows OpenMP duplicate library conflict (PyTorch + NumPy both load libiomp5md.dll)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Fix Windows terminal encoding (prevents UnicodeEncodeError on cp1252 consoles)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir
from utils.seed import set_seed

logger = get_logger(__name__)

BANNER = """
==================================================================
  Multimodal Protein Function Prediction Framework v1.0
  ESM-2 Sequences x GraphSAGE PPI Networks -> GO Terms
==================================================================
"""

STAGES = {
    "download":   "Data Download (UniProt + STRING + GO)",
    "process":    "Sequence & GO Processing",
    "embeddings": "ESM-2 Embedding Extraction",
    "graph":      "PPI Graph Construction",
    "dataset":    "Dataset Assembly & Splitting",
    "train":      "Model Training (all variants)",
    "evaluate":   "Model Evaluation & Comparison",
    "explain":    "Explainability Analysis",
}


def run_stage(stage: str, config_path: str) -> bool:
    """
    Run a single pipeline stage.

    Args:
        stage: Stage name.
        config_path: Path to config YAML.

    Returns:
        True if successful, False on error.
    """
    logger.info("-" * 68)
    logger.info(f"  STAGE: {STAGES.get(stage, stage)}")
    logger.info("-" * 68)
    t0 = time.time()

    try:
        config = load_config(config_path)

        if stage == "download":
            from preprocessing.download_data import main as fn
            sys.argv = ["download_data.py", "--config", config_path]
            fn()

        elif stage == "process":
            from preprocessing.process_sequences import main as seq_fn
            from preprocessing.process_go import main as go_fn
            sys.argv = ["process_sequences.py", "--config", config_path]
            seq_fn()
            sys.argv = ["process_go.py", "--config", config_path]
            go_fn()

        elif stage == "graph":
            from preprocessing.build_graph import main as fn
            sys.argv = ["build_graph.py", "--config", config_path]
            fn()

        elif stage == "embeddings":
            from preprocessing.extract_embeddings import main as fn
            sys.argv = ["extract_embeddings.py", "--config", config_path]
            fn()

        elif stage == "dataset":
            from preprocessing.build_dataset import main as fn
            sys.argv = ["build_dataset.py", "--config", config_path]
            fn()

        elif stage == "train":
            from training.train import main as fn
            sys.argv = ["train.py", "--config", config_path, "--model", "all"]
            fn()

        elif stage == "evaluate":
            from evaluation.evaluate import main as fn
            sys.argv = ["evaluate.py", "--config", config_path, "--model", "all", "--compare"]
            fn()

        elif stage == "explain":
            _run_explainability(config)

        # Force GPU and CPU memory cleanup between stages
        import gc
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        elapsed = time.time() - t0
        logger.info(f"  [OK] Stage '{stage}' completed in {elapsed:.1f}s")
        return True

    except Exception as exc:
        logger.error(f"  [FAIL] Stage '{stage}' failed: {exc}", exc_info=True)
        import gc
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return False


def _run_explainability(config: dict) -> None:
    """Run attention visualization and integrated gradients analysis."""
    import torch
    import numpy as np
    from utils.helpers import load_pickle, ensure_dir
    from utils.seed import get_device
    from training.train import build_model

    device = get_device(config["project"]["device"])
    processed_dir = config["data"]["processed_dir"]
    output_dir = ensure_dir("ProteoFunc_outputs/explainability")

    # Load multimodal model checkpoint
    ckpt_path = os.path.join(
        config["training"]["checkpoint"]["save_dir"], "multimodal_best.pt"
    )
    if not os.path.exists(ckpt_path):
        logger.warning("No multimodal checkpoint found - skipping explainability.")
        return

    # Load split data to get num_classes
    split_data = load_pickle(os.path.join(processed_dir, "split_data.pkl"))
    num_classes = split_data["num_classes"]

    model = build_model("multimodal", config, num_classes)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Load graph data
    graph_data = torch.load(os.path.join(processed_dir, "pyg_graph_data.pt"), map_location="cpu", weights_only=False)
    for key in ["edge_index", "edge_weight", "x"]:
        if key in graph_data and graph_data[key] is not None:
            graph_data[key] = graph_data[key].to(device)

    # Load embeddings sample (first 200 test proteins)
    embeddings = np.load(
        os.path.join(config["data"]["embeddings_dir"], "esm2_embeddings.npy")
    )
    test_idx = split_data["test_idx"][:200]
    sample_emb = torch.tensor(embeddings[test_idx], dtype=torch.float32).to(device)
    sample_acc = [split_data["test_accessions"][i] for i in range(min(200, len(split_data["test_accessions"])))]

    # Protein-to-graph-idx mapping
    p2g = graph_data.get("protein_to_idx", {})
    node_indices = torch.tensor(
        [p2g.get(acc, -1) for acc in sample_acc], dtype=torch.long
    ).to(device)

    # Run attention visualization
    from explainability.attention_viz import (
        extract_fusion_attention,
        plot_modality_contribution_distribution,
        plot_protein_attention_analysis,
    )

    logger.info("Extracting fusion attention weights...")
    attention_data = extract_fusion_attention(
        model, sample_emb, graph_data["x"],
        graph_data["edge_index"], graph_data.get("edge_weight"),
        node_indices, device,
    )

    if "seq_weight" in attention_data and "net_weight" in attention_data:
        plot_modality_contribution_distribution(
            attention_data["seq_weight"],
            attention_data["net_weight"],
            output_path=str(output_dir / "modality_distribution.png"),
        )
        plot_protein_attention_analysis(
            attention_data,
            sample_acc,
            output_path=str(output_dir / "protein_attention.png"),
        )
        logger.info(f"Attention plots saved to {output_dir}")
    else:
        logger.info("No attention weights available for this fusion method.")

    # Run integrated gradients on sequence-only model
    try:
        from explainability.integrated_gradients import (
            compute_integrated_gradients,
            plot_feature_importance,
        )
        seq_ckpt = os.path.join(
            config["training"]["checkpoint"]["save_dir"], "sequence_only_best.pt"
        )
        if os.path.exists(seq_ckpt):
            seq_model = build_model("sequence_only", config, num_classes)
            seq_ckpt_data = torch.load(seq_ckpt, map_location=device, weights_only=False)
            seq_model.load_state_dict(seq_ckpt_data["model_state_dict"])
            seq_model = seq_model.to(device).eval()

            logger.info("Computing integrated gradients for sequence model...")
            ig_results = compute_integrated_gradients(
                model=seq_model,
                embeddings=sample_emb[:50],
                target_class=0,
                n_steps=config["explainability"]["integrated_gradients"]["n_steps"],
            )
            plot_feature_importance(
                ig_results,
                output_path=str(output_dir / "integrated_gradients.png"),
            )
            logger.info("Integrated gradients plot saved.")
    except Exception as e:
        logger.warning(f"Integrated gradients failed: {e}")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Multimodal Protein Function Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:12s}  {v}" for k, v in STAGES.items()),
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=list(STAGES.keys()) + ["all", "preprocess", "train_eval"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Stages to skip (space-separated)",
    )
    args = parser.parse_args()

    # Load config once for seed setting
    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    # Ensure output directories exist
    for d in ["outputs/checkpoints", "outputs/logs", "outputs/evaluation",
              "outputs/explainability", "outputs/figures",
              "ProteoFunc_outputs/checkpoints", "ProteoFunc_outputs/logs",
              "ProteoFunc_outputs/evaluation", "ProteoFunc_outputs/explainability",
              "ProteoFunc_outputs/figures"]:
        ensure_dir(d)

    # Determine stages to run
    if args.stage == "all":
        stages_to_run = list(STAGES.keys())
    elif args.stage == "preprocess":
        stages_to_run = ["download", "process", "embeddings", "graph", "dataset"]
    elif args.stage == "train_eval":
        stages_to_run = ["train", "evaluate"]
    else:
        stages_to_run = [args.stage]

    # Remove skipped stages
    stages_to_run = [s for s in stages_to_run if s not in args.skip]

    logger.info(f"Running stages: {' -> '.join(stages_to_run)}")
    pipeline_start = time.time()

    results = {}
    for stage in stages_to_run:
        success = run_stage(stage, args.config)
        results[stage] = "OK" if success else "FAIL"
        if not success:
            logger.error(f"Pipeline aborted at stage '{stage}'.")
            break

    # Summary
    elapsed = time.time() - pipeline_start
    logger.info("\n" + "=" * 68)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 68)
    for stage, status in results.items():
        logger.info(f"  [{status}]  {stage:12s}  {STAGES.get(stage, '')}")
    logger.info(f"\n  Total time: {elapsed / 60:.1f} minutes")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
