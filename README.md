# 🧬 ProteoFunc  -  Multimodal Protein Function Prediction

> **MSc Bioinformatics Research Project**  
> Predicting Gene Ontology (GO) functional annotations using ESM-2 sequence embeddings + GraphSAGE PPI network context.

---

## Architecture

```
Protein Sequence ──► ESM-2 (650M) ──► SequenceEncoder ──┐
                                                          ├── AttentionFusion ──► GO Terms
PPI Graph ────────► GraphSAGE ───────► GraphEncoder ─────┘
```

| Component | Details |
|---|---|
| Sequence Encoder | ESM-2 t33 650M -> LayerNorm -> Linear(1280->512) -> GELU |
| Graph Encoder | GraphSAGE . 2 layers . hidden=256 . mean agg |
| Fusion | Multi-head cross-attention (4 heads) |
| Classifier | MLP [512->256->C] . GELU . LayerNorm . Dropout |
| Loss | Focal loss (γ=2, α=0.25) |
| Optimizer | AdamW + Cosine LR Warmup |

---

## Project Structure

```
ProteoFunc/
├── app.py                      # Streamlit dashboard
├── run_pipeline.py             # Master pipeline orchestrator
├── setup.py
├── requirements.txt
├── configs/
│   └── config.yaml             # Master configuration
├── preprocessing/
│   ├── download_data.py        # UniProt + STRING + GO download
│   ├── process_sequences.py    # Sequence filtering & annotation
│   ├── process_go.py           # GO ontology processing
│   ├── build_graph.py          # PPI graph -> PyG format
│   ├── extract_embeddings.py   # ESM-2 batch embedding
│   └── build_dataset.py        # Dataset assembly & splitting
├── models/
│   ├── sequence_encoder.py     # ESM-2 projection layer
│   ├── graph_encoder.py        # GraphSAGE encoder
│   ├── fusion.py               # Fusion mechanisms (5 types)
│   ├── classifier.py           # Classification head
│   ├── sequence_only.py        # Baseline: sequence only
│   ├── network_only.py         # Baseline: network only
│   └── multimodal.py           # Full multimodal model (SPFP)
├── training/
│   ├── losses.py               # Focal, BCE, weighted BCE
│   ├── scheduler.py            # Cosine, plateau, step LR
│   ├── trainer.py              # Unified trainer (AMP, early stop)
│   └── train.py                # Training entry point
├── evaluation/
│   ├── metrics.py              # F-max, S-min, AUPR, AUROC
│   ├── statistical_tests.py    # Bootstrap CI, Wilcoxon, McNemar
│   ├── visualizations.py       # PR curves, ROC, heatmaps
│   └── evaluate.py             # Evaluation pipeline
├── explainability/
│   ├── attention_viz.py        # Fusion attention visualization
│   ├── gnn_explainer.py        # GNNExplainer for graph edges
│   └── integrated_gradients.py # Captum IG for sequence features
├── inference/
│   └── predict.py              # Real-time prediction API
└── utils/
    ├── helpers.py              # IO, config, class weights
    ├── logger.py               # Structured logging
    └── seed.py                 # Reproducibility utilities
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For PyTorch Geometric (match your CUDA version):
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

### 2. Run full pipeline

```bash
python run_pipeline.py --stage all
```

Or run individual stages:

```bash
# Stage 1-5: data preparation
python run_pipeline.py --stage preprocess

# Stage 6: train all 3 model variants
python run_pipeline.py --stage train

# Stage 7: evaluate + compare
python run_pipeline.py --stage evaluate

# Stage 8: explainability analysis
python run_pipeline.py --stage explain
```

### 3. Launch dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Individual Scripts

```bash
# Download data
python preprocessing/download_data.py --config configs/config.yaml

# Process sequences & GO
python preprocessing/process_sequences.py --config configs/config.yaml
python preprocessing/process_go.py --config configs/config.yaml

# Build PPI graph
python preprocessing/build_graph.py --config configs/config.yaml

# Extract ESM-2 embeddings
python preprocessing/extract_embeddings.py --config configs/config.yaml

# Build dataset
python preprocessing/build_dataset.py --config configs/config.yaml

# Train specific model
python training/train.py --config configs/config.yaml --model sequence_only
python training/train.py --config configs/config.yaml --model network_only
python training/train.py --config configs/config.yaml --model multimodal

# Evaluate
python evaluation/evaluate.py --config configs/config.yaml --model all --compare
```

---

## Inference API

```python
from inference.predict import ProteinFunctionPredictor

predictor = ProteinFunctionPredictor(
    model_type="sequence_only",
    checkpoint_path="outputs/checkpoints/sequence_only_best.pt",
    config_path="configs/config.yaml",
    threshold=0.3,
    top_k=20,
)

result = predictor.predict("MEEPQSDPSVEPPLSQETFSDLWKLLPENN...")

for pred in result.top_predictions[:5]:
    print(f"{pred['go_id']}  {pred['go_name']:<40}  {pred['probability']:.3f}")
```

---

## Configuration

Edit `configs/config.yaml` to adjust:

| Section | Key settings |
|---|---|
| `data.uniprot` | organism, max_proteins, sequence length filters |
| `data.string` | confidence threshold, species |
| `data.go` | namespaces, min/max annotations |
| `model.sequence_encoder` | ESM-2 model name, embedding dim |
| `model.graph_encoder` | hidden/output dims, num layers |
| `model.fusion` | method: concat/weighted_sum/attention/gated/cross_attention |
| `training` | epochs, batch size, lr, loss type |
| `evaluation` | threshold range, top-k, stratified groups |

---

## Outputs

After a full pipeline run:

```
outputs/
├── checkpoints/
│   ├── sequence_only_best.pt
│   ├── network_only_best.pt
│   ├── multimodal_best.pt
│   └── *_history.json
├── evaluation/
│   ├── sequence_only/overall_metrics.json
│   ├── network_only/overall_metrics.json
│   ├── multimodal/overall_metrics.json
│   └── model_comparison.csv
├── explainability/
│   ├── modality_distribution.png
│   ├── protein_attention.png
│   └── integrated_gradients.png
└── logs/
    └── tensorboard/
```

---

## Metrics

| Metric | Description |
|---|---|
| **F-max** | Maximum F1 over all thresholds (CAFA standard) |
| **S-min** | Minimum semantic distance (ontology-aware) |
| **AUPR** | Area under precision-recall curve |
| **AUROC** | Area under ROC curve (macro-averaged) |
| **Micro/Macro F1** | Label-averaged F1 scores |
| **Hamming Loss** | Fraction of wrong label predictions |

---

## Citation

If you use this framework in your research:

```bibtex
@misc{proteofunc2024,
  title  = {ProteoFunc: Multimodal Protein Function Prediction via ESM-2 and GraphSAGE},
  author = {Usama Raheem},
  year   = {2024},
  note   = {GitHub repository: https://github.com/UsamaRaheem008/ProteoFunc}
}
```
