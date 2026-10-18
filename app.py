"""
ProteoFunc  -  Interactive Protein Function Prediction Tool
==========================================================
Professional Streamlit dashboard for the Multimodal Protein
Function Prediction project.

Tabs:
  1. Home            -  clean project intro + quick stats
  2. Predict         -  real-time GO prediction (all 3 model types)
  3. PPI Explorer    -  protein-protein interaction browser
  4. Training        -  training history plots
  5. Evaluation      -  test-set metrics & model comparison
  6. Explainability  -  attention maps & feature importance

Run:
    streamlit run app.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteoFunc  -  Protein Function Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  -  Dark professional theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: #0a0e1a; }
section[data-testid="stSidebar"] { background: #0f1526; border-right: 1px solid rgba(99,102,241,0.12); }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Scrollbar ──────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }

/* ── Hero Banner ────────────────────── */
.hero {
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(124,58,237,0.1));
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 20px;
  padding: 2.5rem 2rem;
  margin-bottom: 1.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #6366f1, #a78bfa, #6366f1, transparent);
}
.hero h1 {
  background: linear-gradient(135deg, #818cf8, #c084fc, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 2.8rem;
  font-weight: 800;
  margin: 0;
  letter-spacing: -0.02em;
}
.hero p {
  color: #94a3c6;
  font-size: 1.05rem;
  margin: 0.5rem 0 0;
  font-weight: 400;
}

/* ── Metric Cards ───────────────────── */
.metric-card {
  background: rgba(20,28,50,0.7);
  border: 1px solid rgba(99,102,241,0.12);
  border-radius: 16px;
  padding: 1.3rem 1.5rem;
  backdrop-filter: blur(10px);
  text-align: center;
  transition: all 0.3s ease;
}
.metric-card:hover {
  border-color: rgba(99,102,241,0.3);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.metric-card h2 {
  background: linear-gradient(135deg, #818cf8, #a5b4fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
}
.metric-card p {
  color: #5e6b8a;
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0.3rem 0 0;
}

/* ── Section Headers ────────────────── */
.section-header {
  color: #a5b4fc;
  font-size: 1.15rem;
  font-weight: 700;
  border-bottom: 1px solid rgba(99,102,241,0.15);
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
  letter-spacing: -0.01em;
}

/* ── GO Prediction Cards ────────────── */
.go-card {
  background: rgba(20,28,50,0.5);
  border-left: 4px solid #6366f1;
  border-radius: 10px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.5rem;
  transition: all 0.25s ease;
}
.go-card:hover {
  background: rgba(30,40,70,0.6);
  transform: translateX(3px);
}
.go-card.mf { border-left-color: #f59e0b; }
.go-card.bp { border-left-color: #10b981; }
.go-card.cc { border-left-color: #3b82f6; }

/* ── PPI Network Card ──────────────── */
.ppi-card {
  background: rgba(20,28,50,0.5);
  border: 1px solid rgba(99,102,241,0.1);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.4rem;
  transition: all 0.25s ease;
}
.ppi-card:hover {
  border-color: rgba(99,102,241,0.3);
  background: rgba(30,40,70,0.6);
}

/* ── Info Box ───────────────────────── */
.info-box {
  background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin: 0.8rem 0;
  font-size: 0.88rem;
  color: #94a3c6;
}
.info-box.warn {
  background: rgba(245,158,11,0.08);
  border-color: rgba(245,158,11,0.25);
}
.info-box.success {
  background: rgba(16,185,129,0.08);
  border-color: rgba(16,185,129,0.25);
}

/* ── Namespace badge ────────────────── */
.ns-badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}
.ns-mf { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.ns-bp { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.ns-cc { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }

/* ── Model type badge ──────────────── */
.model-badge {
  display: inline-block;
  padding: 0.2rem 0.7rem;
  border-radius: 50px;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.model-seq  { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.model-net  { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.model-mm   { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }

/* ── Graph Status ──────────────────── */
.graph-status {
  background: rgba(20,28,50,0.7);
  border: 1px solid rgba(99,102,241,0.12);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
}
.graph-status.found { border-color: rgba(34,197,94,0.3); }
.graph-status.missing { border-color: rgba(239,68,68,0.3); }

/* ── Tab styling ────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px;
  padding: 0.5rem 1.2rem;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_json_safe(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_config():
    try:
        from utils.helpers import load_config as _lc
        return _lc("configs/config.yaml")
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading predictor…")
def get_predictor(model_type: str, ckpt_path: str):
    from inference.predict import ProteinFunctionPredictor
    return ProteinFunctionPredictor(
        model_type=model_type,
        checkpoint_path=ckpt_path,
        config_path="configs/config.yaml",
        threshold=0.3,
        top_k=30,
    )


def ns_color(ns: str) -> str:
    return {"MF": "mf", "BP": "bp", "CC": "cc"}.get(ns, "")


def ns_full(ns: str) -> str:
    return {"MF": "Molecular Function", "BP": "Biological Process", "CC": "Cellular Component"}.get(ns, ns)


def fetch_uniprot_sequence(accession: str) -> Optional[str]:
    """Fetch protein sequence from UniProt by accession ID."""
    import requests
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            seq = "".join(l.strip() for l in lines if not l.startswith(">"))
            return seq
    except Exception:
        pass
    return None


def fetch_uniprot_info(accession: str) -> Optional[Dict]:
    """Fetch protein info from UniProt (name, gene, organism, function)."""
    import requests
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            info = {
                "accession": accession,
                "protein_name": "Unknown",
                "gene_name": " - ",
                "organism": " - ",
                "function": " - ",
                "subcellular_location": " - ",
                "sequence_length": 0,
            }
            # Protein name
            prot_desc = data.get("proteinDescription", {})
            rec_name = prot_desc.get("recommendedName", {})
            if rec_name:
                full_name = rec_name.get("fullName", {})
                info["protein_name"] = full_name.get("value", "Unknown")
            elif prot_desc.get("submissionNames"):
                info["protein_name"] = prot_desc["submissionNames"][0].get("fullName", {}).get("value", "Unknown")

            # Gene name
            genes = data.get("genes", [])
            if genes:
                info["gene_name"] = genes[0].get("geneName", {}).get("value", " - ")

            # Organism
            org = data.get("organism", {})
            info["organism"] = org.get("scientificName", " - ")

            # Function from comments
            for comment in data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        info["function"] = texts[0].get("value", " - ")
                if comment.get("commentType") == "SUBCELLULAR LOCATION":
                    locs = comment.get("subcellularLocations", [])
                    loc_names = []
                    for loc in locs:
                        loc_val = loc.get("location", {}).get("value", "")
                        if loc_val:
                            loc_names.append(loc_val)
                    if loc_names:
                        info["subcellular_location"] = ", ".join(loc_names[:5])

            # Sequence length
            seq_data = data.get("sequence", {})
            info["sequence_length"] = seq_data.get("length", 0)

            return info
    except Exception:
        pass
    return None


def load_ppi_data():
    """Load PPI edges and protein metadata."""
    ppi_path = os.path.join(config.get("data", {}).get("processed_dir", "data/processed"), "ppi_edges.tsv")
    proteins_path = os.path.join(config.get("data", {}).get("processed_dir", "data/processed"), "proteins_annotated.tsv")

    ppi_df = None
    proteins_df = None

    if os.path.exists(ppi_path):
        ppi_df = pd.read_csv(ppi_path, sep="\t")
    if os.path.exists(proteins_path):
        proteins_df = pd.read_csv(proteins_path, sep="\t")

    return ppi_df, proteins_df


# ── Sample proteins ──────────────────────────────────────────────────────────

SAMPLE_PROTEINS = {
    "P04637": {"name": "TP53", "desc": "Tumor protein p53", "seq": (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLFGRNSFEV"
        "PPERVEEGDRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    )},
    "P38398": {"name": "BRCA1", "desc": "DNA repair protein", "seq": (
        "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEV"
        "AGDNLANSPDKESIISVYVYKVLPQLKIEEQELPLSCEKAIDLEF"
    )},
    "P31749": {"name": "AKT1", "desc": "Protein kinase B", "seq": (
        "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTFIIRCLQWTTVIERTFHVETPEEREEWTTAIQTVADGLKKQEEEEMDFRSGSPSDNSGAEEMEVSLAKPKHRIVVITKDLMIQNLV"
    )},
    "P00533": {"name": "EGFR", "desc": "Epidermal growth factor receptor", "seq": (
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIV"
    )},
    "P42345": {"name": "MTOR", "desc": "Mechanistic target of rapamycin", "seq": (
        "MLGTGPAAATTAATTSSNVSVLQQFASGLKSRNEETRGKRDVSPKHCSLAFTFRQNLRGSGAYDNFKSGRVITLNKSRLEGDYSVMLERLNSDTNYALLQVIGDIFRGEYKDREQEFVQSCAAATSSSDNVPGEDQVATLRSISQLLFNLAKAKIDLSMQWDGLLS"
    )},
}

MODEL_LABELS = {
    "sequence_only": "Sequence Only",
    "network_only": "Network Only",
    "multimodal": "Multimodal (SPFP)",
}

MODEL_DESCRIPTIONS = {
    "sequence_only": "Uses ESM-2 protein language model embeddings only. Input: amino acid sequence.",
    "network_only": "Uses GraphSAGE on the PPI network only. Input: UniProt accession (must be in the PPI graph).",
    "multimodal": "Combines ESM-2 sequence embeddings + GraphSAGE PPI context via attention fusion. Input: UniProt accession + sequence.",
}

MODEL_BADGE_CLASS = {
    "sequence_only": "model-seq",
    "network_only": "model-net",
    "multimodal": "model-mm",
}

# ── Load config once ─────────────────────────────────────────────────────────
config = load_config()
processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
ckpt_dir = config.get("training", {}).get("checkpoint", {}).get("save_dir", "ProteoFunc_outputs/checkpoints")

# ── Check model status ───────────────────────────────────────────────────────
models_status = {}
for m in ["sequence_only", "network_only", "multimodal"]:
    ckpt = os.path.join(ckpt_dir, f"{m}_best.pt")
    models_status[m] = os.path.exists(ckpt)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ProteoFunc")
    st.markdown("*Protein Function Predictor*")
    st.divider()

    st.markdown("### ⚙️ Prediction Settings")
    inference_threshold = st.slider("Probability threshold", 0.05, 0.9, 0.25, 0.05,
                                    help="Minimum confidence for GO term predictions")
    top_k = st.slider("Max GO terms to show", 5, 50, 25, 5,
                       help="Maximum number of GO terms per prediction")

    st.divider()
    st.markdown("### 🏋️ Model Status")
    for m in ["sequence_only", "network_only", "multimodal"]:
        ready = models_status[m]
        icon = "✅" if ready else "⬜"
        st.caption(f"{icon} {MODEL_LABELS[m]}")

    st.divider()
    st.markdown("### 📊 Data Stats")
    proteins_path = os.path.join(processed_dir, "proteins_annotated.tsv")
    go_path = os.path.join(processed_dir, "go_term_metadata.tsv")
    if os.path.exists(proteins_path):
        try:
            n_p = len(pd.read_csv(proteins_path, sep="\t"))
            st.caption(f"🧬 {n_p:,} proteins")
        except Exception:
            pass
    if os.path.exists(go_path):
        try:
            n_g = len(pd.read_csv(go_path, sep="\t"))
            st.caption(f"🏷️ {n_g:,} GO terms")
        except Exception:
            pass
    ppi_path = os.path.join(processed_dir, "ppi_edges.tsv")
    if os.path.exists(ppi_path):
        try:
            n_e = len(pd.read_csv(ppi_path, sep="\t"))
            st.caption(f"🕸️ {n_e:,} PPI interactions")
        except Exception:
            pass

    st.divider()
    st.caption("ProteoFunc v1.0 . MSc Bioinformatics")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_infer, tab_ppi, tab_train, tab_eval, tab_explain = st.tabs(
    ["🏠 Home", "🔬 Predict", "🕸️ PPI Explorer", "📈 Training", "📊 Evaluation", "🔍 Explainability"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  -  HOME (Clean intro)
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
      <h1>🧬 ProteoFunc</h1>
      <p>Predict protein biological functions using deep learning on sequence & interaction networks</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)

    emb_path = os.path.join(config.get("data", {}).get("embeddings_dir", "data/embeddings"), "esm2_embeddings.npy")

    n_proteins = " - "
    n_go_terms = " - "
    emb_dim = " - "

    if os.path.exists(proteins_path):
        try:
            n_proteins = f"{len(pd.read_csv(proteins_path, sep=chr(9))):,}"
        except Exception:
            pass
    if os.path.exists(go_path):
        try:
            n_go_terms = f"{len(pd.read_csv(go_path, sep=chr(9))):,}"
        except Exception:
            pass
    if os.path.exists(emb_path):
        try:
            arr = np.load(emb_path, mmap_mode="r")
            emb_dim = str(arr.shape[1])
        except Exception:
            pass

    with col1:
        st.markdown(f'<div class="metric-card"><h2>{n_proteins}</h2><p>Proteins</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{n_go_terms}</h2><p>GO Terms</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{emb_dim}</h2><p>Embedding Dim</p></div>', unsafe_allow_html=True)
    with col4:
        models_ready = sum(1 for v in models_status.values() if v)
        st.markdown(f'<div class="metric-card"><h2>{models_ready}/3</h2><p>Models Trained</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # Brief description
    st.markdown("""
    <div class="info-box">
    <b>What is ProteoFunc?</b><br>
    ProteoFunc is a multimodal deep learning framework that predicts Gene Ontology (GO) functional
    annotations for proteins. It combines <b>ESM-2</b> protein language model embeddings with
    <b>GraphSAGE</b> protein-protein interaction (PPI) network context, fused via an
    attention mechanism, to predict what a protein does in the cell.<br><br>
    👉 Head to the <b>Predict</b> tab to try it on any protein.
    </div>
    """, unsafe_allow_html=True)

    # Show best results summary if available
    comp_path = "ProteoFunc_outputs/evaluation/model_comparison.csv"
    if os.path.exists(comp_path):
        st.markdown("")
        st.markdown('<div class="section-header">🏆 Model Performance Summary</div>', unsafe_allow_html=True)
        comp_df = pd.read_csv(comp_path, index_col=0)
        if "fmax" in comp_df.columns:
            show_cols = [c for c in ["fmax", "auroc", "aupr", "micro_f1"] if c in comp_df.columns]
            if show_cols:
                display_df = comp_df[show_cols].round(4)
                display_df.index = [MODEL_LABELS.get(m, m) for m in display_df.index]
                st.dataframe(display_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  -  PREDICT (Properly supports all 3 model types)
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    st.markdown('<div class="section-header">🔬 Protein Function Prediction</div>', unsafe_allow_html=True)

    # ── Model selection ──────────────────────────────────────────────────────
    available_models = [m for m, ready in models_status.items() if ready]
    if not available_models:
        st.error("No trained models found. Run the training pipeline first: `python run_pipeline.py --stage train`")
        st.stop()

    col_model, col_info = st.columns([1, 2])

    with col_model:
        model_choice = st.selectbox(
            "Select model",
            available_models,
            format_func=lambda x: MODEL_LABELS.get(x, x),
            help="Choose which trained model to use for prediction",
        )

    with col_info:
        badge_cls = MODEL_BADGE_CLASS.get(model_choice, "")
        st.markdown(
            f'<span class="model-badge {badge_cls}">{MODEL_LABELS.get(model_choice, model_choice)}</span> '
            f'<span style="color:#94a3c6;font-size:0.88rem;">{MODEL_DESCRIPTIONS.get(model_choice, "")}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Input area (different per model type) ────────────────────────────────
    needs_graph = model_choice in ["network_only", "multimodal"]

    if needs_graph:
        # Network & Multimodal: require UniProt accession for PPI graph lookup
        st.markdown("""
        <div class="info-box warn">
        <b>⚠️ This model uses PPI network context.</b> Enter a UniProt accession so the protein can be
        looked up in the interaction graph. You can also paste a sequence  -  if the protein is in our
        database, graph context will be used automatically.
        </div>
        """, unsafe_allow_html=True)

        col_acc, col_sample = st.columns([2, 1])

        with col_acc:
            accession_input = st.text_input(
                "UniProt Accession ID",
                value="P04637",
                placeholder="e.g. P04637, P38398, P31749",
                key="predict_accession",
            )

        with col_sample:
            st.markdown("**Quick select:**")
            sample_choice = st.selectbox(
                "Example proteins",
                [""] + list(SAMPLE_PROTEINS.keys()),
                format_func=lambda x: f"{SAMPLE_PROTEINS[x]['name']}  -  {SAMPLE_PROTEINS[x]['desc']}" if x else "Select a protein...",
                label_visibility="collapsed",
                key="predict_sample_acc",
            )
            if sample_choice:
                accession_input = sample_choice

        # Also allow optional sequence override
        with st.expander("📝 Optionally paste sequence (auto-fetched if left empty)"):
            sequence_override = st.text_area(
                "Amino acid sequence",
                value=SAMPLE_PROTEINS.get(accession_input, {}).get("seq", ""),
                height=120,
                placeholder="Leave empty to auto-fetch from UniProt...",
                key="predict_seq_override",
            )

        # Fetch protein info
        protein_info = None
        sequence_input = sequence_override.strip() if sequence_override.strip() else None

        if accession_input:
            with st.spinner(f"Looking up {accession_input}..."):
                protein_info = fetch_uniprot_info(accession_input.strip())
                if not sequence_input:
                    fetched = fetch_uniprot_sequence(accession_input.strip())
                    if fetched:
                        sequence_input = fetched

            if protein_info:
                st.markdown(
                    f'<div class="info-box success">'
                    f'<b>🧬 {protein_info.get("protein_name", "Unknown")}</b> '
                    f'({protein_info.get("gene_name", " - ")}) . '
                    f'{protein_info.get("organism", " - ")} . '
                    f'{protein_info.get("sequence_length", "?")} aa'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    else:
        # Sequence-only: simple sequence input
        st.markdown("""
        <div class="info-box">
        Enter an amino acid sequence or select an example protein. You can also enter a UniProt accession ID
        to auto-fetch the sequence.
        </div>
        """, unsafe_allow_html=True)

        input_method = st.radio(
            "Input method",
            ["🧬 Amino Acid Sequence", "🔑 UniProt Accession"],
            horizontal=True,
            label_visibility="collapsed",
        )

        protein_info = None
        sequence_input = ""
        accession_input = ""

        if input_method == "🧬 Amino Acid Sequence":
            col_left, col_right = st.columns([3, 1])
            with col_right:
                sample_choice = st.selectbox(
                    "Example",
                    ["Custom…"] + list(SAMPLE_PROTEINS.keys()),
                    format_func=lambda x: f"{SAMPLE_PROTEINS[x]['name']}" if x != "Custom…" else "Custom…",
                    key="predict_sample_seq",
                )
            with col_left:
                default_seq = SAMPLE_PROTEINS[sample_choice]["seq"] if sample_choice != "Custom…" else ""
                sequence_input = st.text_area(
                    "Paste amino acid sequence",
                    value=default_seq,
                    height=140,
                    placeholder="MKTAYIAKQRQISFVKSHFSRQ...",
                    key="predict_seq_input",
                )
        else:
            col_acc, col_sample = st.columns([2, 1])
            with col_acc:
                accession_input = st.text_input(
                    "UniProt Accession ID",
                    value="P04637",
                    placeholder="e.g. P04637",
                    key="predict_acc_seqonly",
                )
            with col_sample:
                st.markdown("**Quick select:**")
                sample_acc = st.selectbox(
                    "Known proteins",
                    [""] + list(SAMPLE_PROTEINS.keys()),
                    format_func=lambda x: f"{SAMPLE_PROTEINS[x]['name']}" if x else "Select...",
                    label_visibility="collapsed",
                    key="predict_quick_seqonly",
                )
                if sample_acc:
                    accession_input = sample_acc

            if accession_input:
                with st.spinner(f"Fetching {accession_input} from UniProt..."):
                    seq = fetch_uniprot_sequence(accession_input.strip())
                    protein_info = fetch_uniprot_info(accession_input.strip())
                if seq:
                    sequence_input = seq
                    st.success(f"✅ Fetched: **{len(seq)} amino acids**")
                else:
                    st.error(f"Could not fetch sequence for **{accession_input}**.")

    # ── Predict button ───────────────────────────────────────────────────────
    st.markdown("")
    predict_btn = st.button("🚀 Predict Functions", type="primary", use_container_width=True)

    if predict_btn:
        if not sequence_input or not sequence_input.strip():
            st.warning("Please provide a protein sequence or a valid UniProt accession ID.")
        else:
            seq = sequence_input.strip()
            # Clean FASTA header if present
            if seq.startswith(">"):
                lines = seq.split("\n")
                seq = "".join(l for l in lines if not l.startswith(">"))

            ckpt_path = os.path.join(ckpt_dir, f"{model_choice}_best.pt")
            if not os.path.exists(ckpt_path):
                st.error(f"Checkpoint not found for **{MODEL_LABELS[model_choice]}**. Run training first.")
            else:
                with st.spinner("Computing ESM-2 embedding & running prediction model…"):
                    try:
                        predictor = get_predictor(model_choice, ckpt_path)
                        predictor.threshold = inference_threshold
                        predictor.top_k = int(top_k)

                        # Use accession-based prediction for network/multimodal
                        if needs_graph and accession_input:
                            result = predictor.predict_by_accession(
                                accession=accession_input.strip(),
                                sequence=seq,
                                threshold=inference_threshold,
                                top_k=int(top_k),
                            )
                            # Show graph lookup info
                            graph_info = predictor.get_protein_graph_info(accession_input.strip())
                            if graph_info["in_graph"]:
                                st.markdown(
                                    f'<div class="graph-status found">'
                                    f'✅ <b>{accession_input}</b> found in PPI graph (node #{graph_info["node_index"]})  -  '
                                    f'full network context used for prediction'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="graph-status missing">'
                                    f'⚠️ <b>{accession_input}</b> not in PPI graph  -  '
                                    f'using fallback embedding (prediction may be less accurate)'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            result = predictor.predict(seq, threshold=inference_threshold, top_k=int(top_k))

                        st.success(
                            f"✅ Predicted **{len(result.predictions)}** GO terms in "
                            f"**{result.inference_time*1000:.0f} ms** using **{MODEL_LABELS[model_choice]}** model"
                        )

                        # Known function comparison
                        if protein_info and protein_info.get("function", " - ") != " - ":
                            st.markdown(
                                f'<div class="info-box">'
                                f'<b>Known Function (UniProt):</b> {protein_info["function"][:500]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        if result.predictions:
                            # ── Probability bar chart ──
                            df_pred = pd.DataFrame(result.top_predictions[:25])
                            fig = px.bar(
                                df_pred, x="probability", y="go_name",
                                orientation="h",
                                color="namespace",
                                color_discrete_map={"MF": "#f59e0b", "BP": "#10b981", "CC": "#3b82f6", "unknown": "#6366f1"},
                                title="Top GO Term Predictions",
                                template="plotly_dark",
                            )
                            fig.update_layout(
                                yaxis={"categoryorder": "total ascending"},
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=max(400, len(df_pred) * 28),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # ── Group by namespace ──
                            st.markdown('<div class="section-header">📋 Predictions by Namespace</div>', unsafe_allow_html=True)
                            by_ns = result.by_namespace()
                            ns_cols = st.columns(3)
                            for col, ns in zip(ns_cols, ["MF", "BP", "CC"]):
                                with col:
                                    items = sorted(by_ns.get(ns, []), key=lambda x: -x["probability"])
                                    st.markdown(
                                        f'<span class="ns-badge ns-{ns.lower()}">{ns}</span> '
                                        f'**{ns_full(ns)}** ({len(items)} terms)',
                                        unsafe_allow_html=True,
                                    )
                                    for p in items[:10]:
                                        css = ns_color(ns)
                                        st.markdown(
                                            f'<div class="go-card {css}">'
                                            f'<b>{p["go_id"]}</b>  -  {p["go_name"]}<br>'
                                            f'<small>Confidence: <b>{p["probability"]:.3f}</b> ({p["probability"]*100:.1f}%)</small>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )

                            # ── Summary stats ──
                            st.markdown("")
                            sum_cols = st.columns(4)
                            sum_cols[0].metric("Total Predictions", len(result.predictions))
                            sum_cols[1].metric("Avg Confidence", f"{np.mean([p['probability'] for p in result.predictions]):.3f}")
                            sum_cols[2].metric("Max Confidence", f"{max(p['probability'] for p in result.predictions):.3f}")
                            sum_cols[3].metric("Inference Time", f"{result.inference_time*1000:.0f} ms")

                            # ── Download ──
                            st.markdown("")
                            results_df = pd.DataFrame(result.top_predictions)
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download predictions as CSV",
                                data=csv_data,
                                file_name=f"proteofunc_{model_choice}_{accession_input or 'custom'}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No GO terms passed the confidence threshold. Try lowering the threshold in the sidebar.")

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.exception(exc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  -  PPI EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_ppi:
    st.markdown('<div class="section-header">🕸️ Protein-Protein Interaction Explorer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Explore the PPI network used for training. Enter a protein accession to see its interaction partners
    and STRING confidence scores.
    </div>
    """, unsafe_allow_html=True)

    ppi_df, proteins_df = load_ppi_data()

    if ppi_df is None:
        st.info("PPI data not found. Run the preprocessing pipeline to generate PPI edges.")
    else:
        col_ppi1, col_ppi2 = st.columns([1, 1])

        with col_ppi1:
            ppi_cols = ppi_df.columns.tolist()
            if len(ppi_cols) >= 2:
                src_col = ppi_cols[0]
                tgt_col = ppi_cols[1]
                score_col = ppi_cols[2] if len(ppi_cols) > 2 else None

                all_proteins = sorted(set(ppi_df[src_col].unique().tolist() + ppi_df[tgt_col].unique().tolist()))

                ppi_query = st.text_input(
                    "Protein accession",
                    value="",
                    placeholder="Enter a protein accession (e.g. P04637)",
                )

                st.caption(f"**{len(all_proteins):,}** proteins in PPI network . **{len(ppi_df):,}** interactions")

        with col_ppi2:
            min_confidence = st.slider("Minimum confidence score", 0.0, 1.0, 0.7, 0.05, key="ppi_conf")
            sort_by = st.selectbox("Sort by", ["Confidence (High -> Low)", "Confidence (Low -> High)", "Partner ID"])

        if ppi_query:
            ppi_query = ppi_query.strip()
            mask = (ppi_df[src_col] == ppi_query) | (ppi_df[tgt_col] == ppi_query)
            interactions = ppi_df[mask].copy()

            if score_col:
                interactions = interactions[interactions[score_col] >= min_confidence]

            if len(interactions) == 0:
                st.warning(f"No interactions found for **{ppi_query}** above confidence {min_confidence}.")
            else:
                partners = []
                for _, row in interactions.iterrows():
                    partner = row[tgt_col] if row[src_col] == ppi_query else row[src_col]
                    score = row[score_col] if score_col else 1.0
                    partners.append({"partner": partner, "confidence": score})

                partners_df = pd.DataFrame(partners)

                if sort_by == "Confidence (High -> Low)":
                    partners_df = partners_df.sort_values("confidence", ascending=False)
                elif sort_by == "Confidence (Low -> High)":
                    partners_df = partners_df.sort_values("confidence", ascending=True)
                else:
                    partners_df = partners_df.sort_values("partner")

                st.markdown(f'<div class="section-header">🔗 {len(partners_df)} Interaction Partners for {ppi_query}</div>', unsafe_allow_html=True)

                top_n = min(20, len(partners_df))
                top_partners = partners_df.head(top_n)
                fig = px.bar(
                    top_partners, x="confidence", y="partner",
                    orientation="h",
                    title=f"Top {top_n} PPI Partners by Confidence",
                    template="plotly_dark",
                    color="confidence",
                    color_continuous_scale=["#3b82f6", "#6366f1", "#a855f7"],
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=max(400, top_n * 25),
                )
                st.plotly_chart(fig, use_container_width=True)

                for _, row in partners_df.head(30).iterrows():
                    conf = row["confidence"]
                    st.markdown(
                        f'<div class="ppi-card">'
                        f'<b>{row["partner"]}</b> &nbsp;.&nbsp; '
                        f'Confidence: <b>{conf:.3f}</b> ({conf*100:.1f}%)'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                csv_ppi = partners_df.to_csv(index=False)
                st.download_button(
                    "📥 Download PPI partners as CSV",
                    data=csv_ppi,
                    file_name=f"ppi_partners_{ppi_query}.csv",
                    mime="text/csv",
                )

        # Network statistics
        st.divider()
        st.markdown('<div class="section-header">📊 Network Statistics</div>', unsafe_allow_html=True)
        net_stats_path = os.path.join(processed_dir, "network_statistics.json")
        if os.path.exists(net_stats_path):
            net_stats = load_json_safe(net_stats_path)
            if net_stats:
                stat_cols = st.columns(min(len(net_stats), 5))
                for col, (k, v) in zip(stat_cols, list(net_stats.items())[:5]):
                    col.metric(k.replace("_", " ").title(), f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  -  TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="section-header">📈 Training History</div>', unsafe_allow_html=True)

    model_selector = st.multiselect(
        "Select models to compare",
        ["sequence_only", "network_only", "multimodal"],
        default=[m for m in ["sequence_only", "network_only", "multimodal"] if models_status.get(m, False)],
        format_func=lambda x: MODEL_LABELS.get(x, x),
    )

    histories = {}
    for m in model_selector:
        hist_path = os.path.join(ckpt_dir, f"{m}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[m] = json.load(f)

    if not histories:
        st.info("No training history found. Train models first with `python run_pipeline.py --stage train`.")
    else:
        available_metrics = set()
        for hist in histories.values():
            available_metrics.update(hist.keys())
        metric_choices = sorted([m for m in available_metrics if isinstance(list(histories.values())[0].get(m, None), list)])

        if metric_choices:
            metric = st.selectbox("Metric", metric_choices,
                                  index=metric_choices.index("val_fmax") if "val_fmax" in metric_choices else 0)
        else:
            metric = "val_fmax"

        fig = go.Figure()
        colors = {"sequence_only": "#7c3aed", "network_only": "#3b82f6", "multimodal": "#10b981"}

        for m, hist in histories.items():
            if metric in hist:
                vals = hist[metric]
                epochs = list(range(1, len(vals) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs, y=vals,
                    mode="lines+markers",
                    name=MODEL_LABELS.get(m, m),
                    line=dict(color=colors.get(m, "#999"), width=2.5),
                    marker=dict(size=4),
                ))

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} over Epochs",
            xaxis_title="Epoch",
            yaxis_title=metric,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.15)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-header">📋 Best Values Per Model</div>', unsafe_allow_html=True)
        rows = []
        for m, hist in histories.items():
            row = {"Model": MODEL_LABELS.get(m, m)}
            for met in ["val_fmax", "val_auroc", "train_loss", "val_loss"]:
                if met in hist and hist[met]:
                    fn = max if "loss" not in met else min
                    row[met] = round(fn(hist[met]), 4)
            if "val_fmax" in hist and hist["val_fmax"]:
                row["best_epoch"] = int(np.argmax(hist["val_fmax"])) + 1
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  -  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-header">📊 Model Evaluation</div>', unsafe_allow_html=True)

    eval_model = st.selectbox(
        "Model", ["sequence_only", "network_only", "multimodal"], key="eval_m",
        format_func=lambda x: MODEL_LABELS.get(x, x),
    )
    eval_dir = f"ProteoFunc_outputs/evaluation/{eval_model}"
    overall = load_json_safe(f"{eval_dir}/overall_metrics.json")
    stratified = load_json_safe(f"{eval_dir}/stratified_metrics.json")

    if overall is None:
        st.info(f"No evaluation results for **{MODEL_LABELS[eval_model]}**. Run `python run_pipeline.py --stage evaluate`.")
    else:
        # Main metrics
        MAIN_METRICS = ["fmax", "smin", "aupr", "auroc", "micro_f1", "macro_f1", "hamming_loss"]
        present = {k: v for k, v in overall.items() if k in MAIN_METRICS}

        cols = st.columns(len(present))
        for col, (k, v) in zip(cols, present.items()):
            col.metric(k.replace("_", " ").upper(), f"{v:.4f}")

        # Stratified
        if stratified:
            st.divider()
            st.markdown('<div class="section-header">📈 Stratified Evaluation</div>', unsafe_allow_html=True)

            # By namespace
            ns_results = {k: v for k, v in stratified.items() if k.startswith("namespace_") and isinstance(v, dict)}
            if ns_results:
                st.markdown("**By GO Namespace:**")
                ns_rows = []
                for k, metrics in ns_results.items():
                    ns = k.replace("namespace_", "")
                    row = {"Namespace": ns}
                    for mk in ["fmax", "auroc", "aupr"]:
                        if mk in metrics:
                            row[mk] = round(metrics[mk], 4)
                    ns_rows.append(row)
                if ns_rows:
                    st.dataframe(pd.DataFrame(ns_rows).set_index("Namespace"), use_container_width=True)

            # By hub status
            hub_results = {k: v for k, v in stratified.items() if "hub" in k and isinstance(v, dict)}
            if hub_results:
                st.markdown("**By Hub Status:**")
                hub_rows = []
                for k, metrics in hub_results.items():
                    row = {"Group": k.replace("_", " ").title()}
                    for mk in ["fmax", "auroc", "aupr"]:
                        if mk in metrics:
                            row[mk] = round(metrics[mk], 4)
                    hub_rows.append(row)
                if hub_rows:
                    st.dataframe(pd.DataFrame(hub_rows).set_index("Group"), use_container_width=True)

            # By degree
            degree_results = {k: v for k, v in stratified.items() if "degree" in k and isinstance(v, dict)}
            if degree_results:
                st.markdown("**By Network Connectivity:**")
                deg_rows = []
                for k, metrics in degree_results.items():
                    row = {"Connectivity": k.replace("_", " ").title()}
                    for mk in ["fmax", "auroc", "aupr"]:
                        if mk in metrics:
                            row[mk] = round(metrics[mk], 4)
                    deg_rows.append(row)
                if deg_rows:
                    deg_df = pd.DataFrame(deg_rows).set_index("Connectivity")
                    st.dataframe(deg_df, use_container_width=True)

                    fig = px.bar(
                        deg_df.reset_index(), x="Connectivity", y="fmax",
                        color="Connectivity",
                        title="F-max by Network Connectivity",
                        template="plotly_dark",
                        color_discrete_sequence=["#6366f1", "#a855f7", "#ec4899"],
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.15)")
                    st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    comp_path = "ProteoFunc_outputs/evaluation/model_comparison.csv"
    if os.path.exists(comp_path):
        st.divider()
        st.markdown('<div class="section-header">🥊 Model Comparison</div>', unsafe_allow_html=True)
        comp_df = pd.read_csv(comp_path, index_col=0)
        comp_df.index = [MODEL_LABELS.get(m, m) for m in comp_df.index]
        comp_df.index.name = "Model"
        st.dataframe(comp_df.round(4), use_container_width=True)

        num_cols = [c for c in comp_df.columns if comp_df[c].dtype in [float, "float64"]]
        if num_cols:
            show_metrics = [c for c in ["fmax", "auroc", "aupr", "micro_f1", "macro_f1"] if c in num_cols]
            if show_metrics:
                comp_sub = comp_df[show_metrics].reset_index()
                first_col = comp_sub.columns[0]
                comp_long = comp_sub.melt(id_vars=first_col, var_name="Metric", value_name="Value")
                comp_long = comp_long.rename(columns={first_col: "Model"})
                fig = px.bar(
                    comp_long, x="Metric", y="Value", color="Model", barmode="group",
                    color_discrete_sequence=["#7c3aed", "#3b82f6", "#10b981"],
                    template="plotly_dark",
                    title="Model Comparison  -  Key Metrics",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.15)")
                st.plotly_chart(fig, use_container_width=True)

    # Statistical Tests
    stat_path = "ProteoFunc_outputs/evaluation/statistical_tests.json"
    if os.path.exists(stat_path):
        st.divider()
        st.markdown('<div class="section-header">🧪 Statistical Significance</div>', unsafe_allow_html=True)
        stat_data = load_json_safe(stat_path)
        if stat_data and "pairwise_comparisons" in stat_data:
            stat_rows = []
            for comp in stat_data["pairwise_comparisons"]:
                model_a = MODEL_LABELS.get(comp["model_a"], comp["model_a"])
                model_b = MODEL_LABELS.get(comp["model_b"], comp["model_b"])
                ttest = comp["ttest"]

                p_val = ttest["p_value"]
                sig = "✅ Yes" if ttest["significant"] else "❌ No"
                p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"

                stat_rows.append({
                    "Comparison": f"{model_a} vs {model_b}",
                    "P-Value": p_str,
                    "Cohen's d": round(ttest["cohens_d"], 4),
                    "Mean Diff": round(ttest["mean_diff"], 4),
                    "Significant": sig,
                })

            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows).set_index("Comparison"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6  -  EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown('<div class="section-header">🔍 Explainability & Interpretability</div>', unsafe_allow_html=True)

    explain_dir = Path("ProteoFunc_outputs/explainability")
    fig_dir = Path("ProteoFunc_outputs/figures")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Modality Contribution Distribution**")
        img_path = explain_dir / "modality_distribution.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Run: `python finalize.py` to generate explainability plots.")

        st.markdown("**Integrated Gradients  -  Feature Importance**")
        ig_path = explain_dir / "integrated_gradients.png"
        if ig_path.exists():
            st.image(str(ig_path), use_container_width=True)
        else:
            st.caption("Not yet generated.")

    with col2:
        st.markdown("**Per-Protein Attention Analysis**")
        attn_path = explain_dir / "protein_attention.png"
        if attn_path.exists():
            st.image(str(attn_path), use_container_width=True)
        else:
            st.info("Run: `python finalize.py` to generate attention plots.")

    # Additional figures
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        if figs:
            st.divider()
            st.markdown('<div class="section-header">📊 Generated Figures</div>', unsafe_allow_html=True)
            fig_cols = st.columns(min(3, len(figs)))
            for i, fp in enumerate(figs[:6]):
                fig_cols[i % 3].image(str(fp), caption=fp.stem.replace("_", " ").title(), use_container_width=True)

    # Interactive modality weight explorer
    st.divider()
    st.markdown('<div class="section-header">🎛️ Interactive Modality Weight Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        "Simulate how the fusion module weights sequence vs. network context "
        "under different protein properties."
    )

    c1, c2, c3 = st.columns(3)
    seq_quality = c1.slider("Sequence quality (↑ -> prefer sequence)", 0.0, 1.0, 0.6)
    network_deg = c2.slider("Network degree (↑ -> prefer network)", 0.0, 1.0, 0.4)
    go_coverage = c3.slider("GO coverage richness", 0.0, 1.0, 0.5)

    raw_seq = seq_quality * 1.5 + (1 - network_deg) * 0.5
    raw_net = network_deg * 1.5 + (1 - seq_quality) * 0.5
    total = raw_seq + raw_net
    w_seq = raw_seq / total
    w_net = raw_net / total

    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Bar(
        x=["Sequence", "Network"],
        y=[w_seq, w_net],
        marker_color=["#7c3aed", "#3b82f6"],
        text=[f"{w_seq:.2%}", f"{w_net:.2%}"],
        textposition="outside",
    ))
    fig_gauge.update_layout(
        title="Estimated Fusion Weights",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#5e6b8a;font-size:.78rem;'>"
    "ProteoFunc v1.0 . MSc Bioinformatics . ESM-2 x GraphSAGE . "
    "Multimodal Protein Function Prediction"
    "</div>",
    unsafe_allow_html=True,
)
