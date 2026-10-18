"""
Gene Ontology Processing Module
================================
Processes GO annotations, builds multi-label matrices, and categorizes GO terms.

Usage:
    python preprocessing/process_go.py --config configs/config.yaml
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir, save_json, save_pickle

logger = get_logger(__name__)

# GO namespace mapping
GO_NAMESPACE_MAP = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


def parse_go_obo(obo_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse GO OBO file to extract term information and hierarchy.

    Args:
        obo_path: Path to go-basic.obo file.

    Returns:
        Dictionary mapping GO ID -> {name, namespace, parents, is_obsolete}.
    """
    logger.info(f"Parsing GO OBO file: {obo_path}")

    try:
        import obonet
        graph = obonet.read_obo(obo_path)

        go_terms = {}
        for go_id, data in graph.nodes(data=True):
            if not go_id.startswith("GO:"):
                continue
            go_terms[go_id] = {
                "name": data.get("name", ""),
                "namespace": data.get("namespace", ""),
                "is_obsolete": data.get("is_obsolete", "false") == "true",
                "parents": [],
            }
            # Get parent terms (is_a relationships)
            if go_id in graph:
                for parent in graph.successors(go_id):
                    if parent.startswith("GO:"):
                        go_terms[go_id]["parents"].append(parent)

        logger.info(f"Parsed {len(go_terms)} GO terms from OBO file")
        return go_terms

    except ImportError:
        logger.warning("obonet not installed, using manual OBO parsing")
        return _parse_obo_manual(obo_path)


def _parse_obo_manual(obo_path: str) -> Dict[str, Dict[str, Any]]:
    """Manual OBO file parser without obonet dependency."""
    go_terms = {}
    current_term = None

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term = {}
            elif line == "" and current_term is not None:
                if "id" in current_term:
                    go_id = current_term["id"]
                    go_terms[go_id] = {
                        "name": current_term.get("name", ""),
                        "namespace": current_term.get("namespace", ""),
                        "is_obsolete": current_term.get("is_obsolete", False),
                        "parents": current_term.get("parents", []),
                    }
                current_term = None
            elif current_term is not None:
                if line.startswith("id: "):
                    current_term["id"] = line[4:]
                elif line.startswith("name: "):
                    current_term["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current_term["namespace"] = line[11:]
                elif line.startswith("is_obsolete: true"):
                    current_term["is_obsolete"] = True
                elif line.startswith("is_a: "):
                    parent = line[6:].split(" !")[0].strip()
                    current_term.setdefault("parents", []).append(parent)

    logger.info(f"Manual parse: {len(go_terms)} GO terms")
    return go_terms


def parse_uniprot_go_annotations(
    proteins_df: pd.DataFrame,
    go_col: str = "go_ids",
) -> Dict[str, List[str]]:
    """
    Extract GO annotations from UniProt protein data.

    Args:
        proteins_df: DataFrame with UniProt protein data.
        go_col: Column containing GO IDs (semicolon-separated).

    Returns:
        Dictionary mapping protein accession -> list of GO IDs.
    """
    protein_go = {}

    for _, row in proteins_df.iterrows():
        accession = row["accession"]
        go_str = row.get(go_col, "")

        if pd.isna(go_str) or not go_str:
            continue

        # Parse GO IDs (semicolon or comma separated)
        go_ids = []
        for term in str(go_str).replace(",", ";").split(";"):
            term = term.strip()
            if term.startswith("GO:"):
                go_ids.append(term)

        if go_ids:
            protein_go[accession] = go_ids

    logger.info(f"Extracted GO annotations for {len(protein_go)} proteins")
    return protein_go


def propagate_go_annotations(
    protein_go: Dict[str, List[str]],
    go_terms: Dict[str, Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """
    Propagate GO annotations using the true-path rule (bottom-up).
    Each protein inherits all ancestor annotations.

    Args:
        protein_go: Protein -> list of GO IDs.
        go_terms: GO term dictionary with parent information.

    Returns:
        Protein -> set of propagated GO IDs.
    """
    logger.info("Propagating GO annotations using true-path rule...")

    def get_ancestors(go_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Recursively get all ancestors of a GO term."""
        if visited is None:
            visited = set()
        if go_id in visited:
            return visited
        visited.add(go_id)
        if go_id in go_terms:
            for parent in go_terms[go_id].get("parents", []):
                get_ancestors(parent, visited)
        return visited

    propagated = {}
    for accession, go_ids in protein_go.items():
        all_terms = set()
        for go_id in go_ids:
            all_terms.update(get_ancestors(go_id))
        propagated[accession] = all_terms

    # Statistics
    orig_counts = [len(v) for v in protein_go.values()]
    prop_counts = [len(v) for v in propagated.values()]
    logger.info(
        f"GO propagation: avg {np.mean(orig_counts):.1f} -> {np.mean(prop_counts):.1f} terms/protein"
    )

    return propagated


def filter_go_terms(
    protein_go: Dict[str, Set[str]],
    go_terms: Dict[str, Dict[str, Any]],
    namespaces: List[str],
    min_annotations: int = 30,
    max_annotations: int = 5000,
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Filter GO terms by namespace, frequency, and obsolescence.

    Args:
        protein_go: Protein -> set of GO IDs.
        go_terms: GO term metadata.
        namespaces: List of GO namespaces to keep.
        min_annotations: Minimum proteins per GO term.
        max_annotations: Maximum proteins per GO term.

    Returns:
        Tuple of (filtered protein_go, list of selected GO terms).
    """
    logger.info(f"Filtering GO terms (namespaces={namespaces}, "
                f"min_ann={min_annotations}, max_ann={max_annotations})")

    # Count term frequencies
    term_counts = Counter()
    for terms in protein_go.values():
        term_counts.update(terms)

    # Filter terms
    selected_terms = []
    for go_id, count in term_counts.items():
        if go_id not in go_terms:
            continue
        term_info = go_terms[go_id]
        if term_info.get("is_obsolete", False):
            continue
        if term_info.get("namespace", "") not in namespaces:
            continue
        if count < min_annotations or count > max_annotations:
            continue
        selected_terms.append(go_id)

    selected_set = set(selected_terms)
    logger.info(f"Selected {len(selected_terms)} GO terms from {len(term_counts)} total")

    # Filter protein annotations
    filtered_go = {}
    for accession, terms in protein_go.items():
        filtered = terms & selected_set
        if filtered:
            filtered_go[accession] = filtered

    logger.info(f"Proteins with annotations: {len(filtered_go)}")

    # Log per-namespace counts
    for ns in namespaces:
        ns_terms = [t for t in selected_terms if go_terms.get(t, {}).get("namespace") == ns]
        logger.info(f"  {GO_NAMESPACE_MAP.get(ns, ns)}: {len(ns_terms)} terms")

    return filtered_go, sorted(selected_terms)


def categorize_go_terms(
    selected_terms: List[str],
    go_terms: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Categorize GO terms into functional groups:
    - Signaling / Complex / Regulatory
    - Metabolic / Housekeeping

    Args:
        selected_terms: List of selected GO term IDs.
        go_terms: GO term metadata.

    Returns:
        Dictionary mapping GO ID -> category.
    """
    # Keyword-based heuristic categorization
    signaling_keywords = [
        "signal", "receptor", "kinase", "phosphorylation", "transduction",
        "regulation", "regulatory", "transcription", "complex", "binding",
        "channel", "transport", "apoptosis", "cell cycle", "immune",
        "differentiation", "proliferation", "growth factor",
    ]
    metabolic_keywords = [
        "metabol", "biosynthe", "catabol", "oxid", "reduc", "hydrol",
        "transferase", "ligase", "lyase", "isomerase", "synthase",
        "dehydrogenase", "replication", "repair", "translation",
        "ribosom", "proteasom", "mitochond", "housekeeping",
    ]

    categories = {}
    for go_id in selected_terms:
        if go_id not in go_terms:
            categories[go_id] = "Other"
            continue

        name = go_terms[go_id].get("name", "").lower()

        is_signaling = any(kw in name for kw in signaling_keywords)
        is_metabolic = any(kw in name for kw in metabolic_keywords)

        if is_signaling and not is_metabolic:
            categories[go_id] = "Signaling/Regulatory"
        elif is_metabolic and not is_signaling:
            categories[go_id] = "Metabolic/Housekeeping"
        elif is_signaling and is_metabolic:
            categories[go_id] = "Signaling/Regulatory"  # default to signaling
        else:
            categories[go_id] = "Other"

    # Summary
    cat_counts = Counter(categories.values())
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  Category '{cat}': {count} terms")

    return categories


def build_label_matrix(
    protein_accessions: List[str],
    protein_go: Dict[str, Set[str]],
    selected_terms: List[str],
) -> np.ndarray:
    """
    Build binary multi-label matrix.

    Args:
        protein_accessions: Ordered list of protein accessions.
        protein_go: Protein -> set of GO IDs.
        selected_terms: Ordered list of selected GO terms.

    Returns:
        Binary numpy array of shape (num_proteins, num_terms).
    """
    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    n_proteins = len(protein_accessions)
    n_terms = len(selected_terms)

    label_matrix = np.zeros((n_proteins, n_terms), dtype=np.float32)

    for i, accession in enumerate(protein_accessions):
        if accession in protein_go:
            for term in protein_go[accession]:
                if term in term_to_idx:
                    label_matrix[i, term_to_idx[term]] = 1.0

    # Statistics
    annotations_per_protein = label_matrix.sum(axis=1)
    proteins_per_term = label_matrix.sum(axis=0)

    logger.info(f"Label matrix shape: {label_matrix.shape}")
    logger.info(f"  Annotations/protein: {annotations_per_protein.mean():.1f} +/- "
                f"{annotations_per_protein.std():.1f}")
    logger.info(f"  Proteins/term: {proteins_per_term.mean():.1f} +/- "
                f"{proteins_per_term.std():.1f}")
    logger.info(f"  Sparsity: {(1 - label_matrix.mean()) * 100:.2f}%")

    return label_matrix


def process_go_annotations(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main GO processing pipeline.

    Args:
        config: Project configuration.

    Returns:
        Dictionary with processed GO data.
    """
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    ensure_dir(processed_dir)

    # 1. Parse GO OBO
    obo_path = os.path.join(raw_dir, "go-basic.obo")
    go_terms = parse_go_obo(obo_path)

    # 2. Load cleaned proteins
    proteins_path = os.path.join(processed_dir, "proteins_clean.tsv")
    proteins_df = pd.read_csv(proteins_path, sep="\t")

    # 3. Parse UniProt GO annotations
    protein_go = parse_uniprot_go_annotations(proteins_df)

    # 4. Propagate annotations
    protein_go_propagated = propagate_go_annotations(protein_go, go_terms)

    # 5. Filter GO terms
    filtered_go, selected_terms = filter_go_terms(
        protein_go_propagated,
        go_terms,
        namespaces=data_cfg["go"]["namespaces"],
        min_annotations=data_cfg["go"]["min_annotations"],
        max_annotations=data_cfg["go"]["max_annotations"],
    )

    # 6. Filter proteins (keep only those with annotations)
    valid_accessions = [
        acc for acc in proteins_df["accession"].values
        if acc in filtered_go
    ]
    proteins_df = proteins_df[proteins_df["accession"].isin(valid_accessions)].reset_index(drop=True)
    logger.info(f"Proteins with GO annotations: {len(proteins_df)}")

    # 7. Build label matrix
    label_matrix = build_label_matrix(
        proteins_df["accession"].tolist(),
        filtered_go,
        selected_terms,
    )

    # 8. Categorize GO terms
    categories = categorize_go_terms(selected_terms, go_terms)

    # 9. Build term metadata
    term_metadata = []
    for go_id in selected_terms:
        info = go_terms.get(go_id, {})
        term_metadata.append({
            "go_id": go_id,
            "name": info.get("name", ""),
            "namespace": GO_NAMESPACE_MAP.get(info.get("namespace", ""), ""),
            "category": categories.get(go_id, "Other"),
        })
    term_metadata_df = pd.DataFrame(term_metadata)

    # 10. Save outputs
    proteins_df.to_csv(os.path.join(processed_dir, "proteins_annotated.tsv"), sep="\t", index=False)
    np.save(os.path.join(processed_dir, "label_matrix.npy"), label_matrix)
    term_metadata_df.to_csv(os.path.join(processed_dir, "go_term_metadata.tsv"), sep="\t", index=False)
    save_pickle(go_terms, os.path.join(processed_dir, "go_terms.pkl"))
    save_pickle(selected_terms, os.path.join(processed_dir, "selected_go_terms.pkl"))
    save_pickle(categories, os.path.join(processed_dir, "go_categories.pkl"))

    logger.info("GO processing complete!")

    return {
        "proteins_df": proteins_df,
        "label_matrix": label_matrix,
        "selected_terms": selected_terms,
        "go_terms": go_terms,
        "categories": categories,
        "term_metadata": term_metadata_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Process Gene Ontology annotations")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    process_go_annotations(config)


if __name__ == "__main__":
    main()
