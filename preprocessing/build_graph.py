"""
PPI Graph Construction Module
==============================
Builds protein-protein interaction graphs from STRING data.

Usage:
    python preprocessing/build_graph.py --config configs/config.yaml
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir, save_pickle, save_json

logger = get_logger(__name__)


def load_string_interactions(
    interactions_path: str,
    mapping_path: str,
    valid_proteins: List[str],
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Load STRING interactions and map to UniProt accessions.

    The STRING mapping file contains MANY aliases per ENSP ID (gene names,
    PDB IDs, RefSeq IDs, entry names, etc.). We filter to only keep aliases
    that are actual UniProt accessions present in our protein set.

    Args:
        interactions_path: Path to filtered STRING interactions TSV.
        mapping_path: Path to STRING-UniProt mapping TSV.
        valid_proteins: List of valid UniProt accessions to keep.
        confidence_threshold: Minimum edge weight/confidence threshold (0-1).

    Returns:
        DataFrame with columns: protein1, protein2, weight.
    """
    import re
    logger.info(f"Loading STRING interactions (confidence_threshold={confidence_threshold}) and mapping to UniProt IDs...")

    # Load mapping  -  the file has MANY rows per ENSP ID (one per alias)
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    logger.info(f"Loaded {len(mapping_df)} raw mapping rows")

    # Filter mapping by prioritizing aliases present in our valid protein set
    valid_set = set(valid_proteins)
    mapping_valid = mapping_df[mapping_df["uniprot_accession"].isin(valid_set)].copy()
    logger.info(f"Mapping rows matching dataset proteins: {len(mapping_valid)}")

    # Deduplicate: each ENSP maps to one canonical UniProt accession in our dataset
    mapping_dedup = mapping_valid.drop_duplicates(subset=["string_id"], keep="first")
    string_to_uniprot = dict(
        zip(mapping_dedup["string_id"], mapping_dedup["uniprot_accession"])
    )
    logger.info(f"Final ENSP->UniProt mappings: {len(string_to_uniprot)} unique ENSP IDs mapped to dataset proteins")

    # Load interactions
    interactions_df = pd.read_csv(interactions_path, sep="\t")
    logger.info(f"Loaded {len(interactions_df)} STRING interactions")

    # Add logging to show how many unique ENSPs are in the interaction file
    unique_ensp_in_interactions = set(interactions_df["protein1"]).union(set(interactions_df["protein2"]))
    logger.info(f"Unique ENSPs in interaction file: {len(unique_ensp_in_interactions)}")

    # Add logging to show how many of those ENSPs have ANY UniProt mapping
    ensp_with_mapping = sum(1 for ensp in unique_ensp_in_interactions if ensp in string_to_uniprot)
    logger.info(f"Unique ENSPs in interaction file with a UniProt mapping: {ensp_with_mapping}")

    # Map STRING IDs to UniProt
    interactions_df["uniprot1"] = interactions_df["protein1"].map(string_to_uniprot)
    interactions_df["uniprot2"] = interactions_df["protein2"].map(string_to_uniprot)
    
    # Add logging to show how many mapped UniProt IDs are in valid_set
    mapped_uniprot_ids = set(interactions_df["uniprot1"].dropna()).union(set(interactions_df["uniprot2"].dropna()))
    uniprot_in_valid_set = sum(1 for uid in mapped_uniprot_ids if uid in valid_set)
    logger.info(f"Mapped UniProt IDs that are in the valid_set: {uniprot_in_valid_set} / {len(mapped_uniprot_ids)}")

    # Diagnostic: how many mapped?
    mapped1 = interactions_df["uniprot1"].notna().sum()
    mapped2 = interactions_df["uniprot2"].notna().sum()
    logger.info(f"Mapping success: protein1={mapped1}/{len(interactions_df)}, "
                f"protein2={mapped2}/{len(interactions_df)}")

    # Drop unmapped
    interactions_df = interactions_df.dropna(subset=["uniprot1", "uniprot2"])
    logger.info(f"After mapping (both sides): {len(interactions_df)} interactions")

    # Filter to valid proteins
    interactions_df = interactions_df[
        interactions_df["uniprot1"].isin(valid_set) &
        interactions_df["uniprot2"].isin(valid_set)
    ].copy()
    logger.info(f"After filtering to valid proteins: {len(interactions_df)} interactions")

    # Remove self-loops
    interactions_df = interactions_df[
        interactions_df["uniprot1"] != interactions_df["uniprot2"]
    ].copy()

    # Ensure consistent order to drop bidirectional duplicates
    pairs = np.sort(interactions_df[["uniprot1", "uniprot2"]].values, axis=1)
    interactions_df["uniprot1"] = pairs[:, 0]
    interactions_df["uniprot2"] = pairs[:, 1]

    # Remove duplicate edges (same pair may appear multiple times)
    interactions_df = interactions_df.drop_duplicates(
        subset=["uniprot1", "uniprot2"], keep="first"
    )
    logger.info(f"After removing duplicates and self-loops: {len(interactions_df)} interactions")

    # Normalize edge weights to [0, 1]
    if "combined_score_norm" in interactions_df.columns:
        interactions_df["weight"] = interactions_df["combined_score_norm"]
    elif "combined_score" in interactions_df.columns:
        interactions_df["weight"] = interactions_df["combined_score"] / 1000.0
    else:
        interactions_df["weight"] = 1.0

    # Filter strictly by confidence threshold
    if confidence_threshold > 0.0:
        before_count = len(interactions_df)
        interactions_df = interactions_df[interactions_df["weight"] >= float(confidence_threshold)].copy()
        logger.info(f"Confidence threshold filter (>={confidence_threshold}): {before_count} -> {len(interactions_df)} edges")

    result = interactions_df[["uniprot1", "uniprot2", "weight"]].reset_index(drop=True)
    logger.info(f"Final PPI edges: {len(result)}")

    return result


def build_networkx_graph(
    edges_df: pd.DataFrame,
    protein_list: List[str],
) -> nx.Graph:
    """
    Build a NetworkX graph from edge DataFrame.

    Args:
        edges_df: DataFrame with columns protein1, protein2, weight (or uniprot1, uniprot2).
        protein_list: List of all protein accessions (nodes).

    Returns:
        NetworkX undirected graph.
    """
    G = nx.Graph()
    G.add_nodes_from(protein_list)

    col1 = "uniprot1" if "uniprot1" in edges_df.columns else "protein1"
    col2 = "uniprot2" if "uniprot2" in edges_df.columns else "protein2"

    if not edges_df.empty:
        has_weight = "weight" in edges_df.columns
        if has_weight:
            for row in edges_df[[col1, col2, "weight"]].values:
                G.add_edge(row[0], row[1], weight=float(row[2]))
        else:
            for row in edges_df[[col1, col2]].values:
                G.add_edge(row[0], row[1], weight=1.0)

    logger.info(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"  Density: {nx.density(G):.6f}")
    logger.info(f"  Connected components: {nx.number_connected_components(G)}")

    return G


def compute_network_statistics(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute network topology statistics.

    Args:
        G: NetworkX graph.

    Returns:
        Dictionary of network statistics.
    """
    logger.info("Computing network statistics...")

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "num_connected_components": nx.number_connected_components(G),
        "degree_mean": float(np.mean(degree_values)),
        "degree_median": float(np.median(degree_values)),
        "degree_std": float(np.std(degree_values)),
        "degree_max": int(np.max(degree_values)),
        "degree_min": int(np.min(degree_values)),
        "isolated_nodes": sum(1 for d in degree_values if d == 0),
    }

    # Hub proteins (top 1% by degree of connected nodes)
    nonzero_degrees = [d for d in degree_values if d > 0]
    if len(nonzero_degrees) > 0:
        threshold = max(np.percentile(nonzero_degrees, 99), 1)
        hub_proteins = [node for node, deg in degrees.items() if deg >= threshold]
    else:
        threshold = 1
        hub_proteins = []
    
    stats["num_hubs"] = len(hub_proteins)
    stats["hub_threshold_degree"] = int(threshold)

    logger.info(f"Network statistics:")
    for key, val in stats.items():
        logger.info(f"  {key}: {val}")

    return stats


def identify_hub_proteins(G: nx.Graph, percentile: float = 95) -> Dict[str, bool]:
    """
    Identify hub proteins based on degree centrality.

    Uses the percentile of NON-ZERO degrees to avoid marking all proteins
    as hubs when the graph is sparse (many isolated nodes).

    Args:
        G: NetworkX graph.
        percentile: Percentile threshold for hub classification (among connected nodes).

    Returns:
        Dictionary mapping protein -> is_hub boolean.
    """
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values()))

    # Only consider connected nodes (degree > 0) for hub threshold
    nonzero_degrees = degree_values[degree_values > 0]

    if len(nonzero_degrees) == 0:
        # No edges at all  -  no hubs
        logger.warning("No connected nodes in graph  -  no hub proteins identified.")
        hub_status = {node: False for node in degrees}
        return hub_status

    threshold = max(np.percentile(nonzero_degrees, percentile), 1)
    hub_status = {node: deg >= threshold for node, deg in degrees.items()}
    n_hubs = sum(hub_status.values())
    n_connected = len(nonzero_degrees)
    logger.info(f"Hub proteins (>= {percentile}th percentile of connected nodes, "
                f"degree >= {threshold}): {n_hubs} hubs out of {n_connected} connected nodes "
                f"({G.number_of_nodes() - n_connected} isolated)")

    return hub_status


def compute_centrality_metrics(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    Compute centrality metrics for all nodes.

    Args:
        G: NetworkX graph.

    Returns:
        Dictionary of centrality metric dictionaries.
    """
    logger.info("Computing centrality metrics...")

    centrality = {
        "degree": dict(G.degree()),
        "degree_centrality": nx.degree_centrality(G),
    }

    # Betweenness centrality  -  always use sampling (k=200) to prevent hanging
    # on large graphs (exact computation is O(V*E) = prohibitive for 10K+ nodes)
    k_sample = min(200, G.number_of_nodes())
    logger.info(f"  Computing approximate betweenness centrality (k={k_sample} samples)...")
    centrality["betweenness"] = nx.betweenness_centrality(G, k=k_sample, normalized=True, seed=42)

    # Closeness centrality  -  approximate via BFS from sample nodes (k=200) for instant computation
    logger.info(f"  Computing approximate closeness centrality (k={k_sample} samples)...")
    import random
    nodes_list = list(G.nodes())
    random.seed(42)
    sample_nodes = random.sample(nodes_list, k_sample)

    total_dist = {n: 0.0 for n in nodes_list}
    reachable_cnt = {n: 0 for n in nodes_list}

    for s in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G, s)
        for target, d in lengths.items():
            total_dist[target] += d
            reachable_cnt[target] += 1

    closeness = {}
    for n in nodes_list:
        cnt = reachable_cnt[n]
        if cnt > 0 and total_dist[n] > 0:
            closeness[n] = float((cnt / k_sample) * (cnt / total_dist[n]))
        else:
            closeness[n] = 0.0

    centrality["closeness"] = closeness

    return centrality


def build_pyg_data(
    edges_df: pd.DataFrame,
    protein_list: List[str],
    node_features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    centrality: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Build PyTorch Geometric compatible graph data.

    Args:
        edges_df: DataFrame with edge pairs.
        protein_list: Ordered list of protein accessions.
        node_features: Node feature matrix (num_nodes, feature_dim).
        labels: Label matrix (num_nodes, num_labels).

    Returns:
        Dictionary with edge_index, edge_weight, node_features, labels, mappings.
    """
    logger.info("Building PyTorch Geometric graph data...")

    protein_to_idx = {p: i for i, p in enumerate(protein_list)}
    n_nodes = len(protein_list)

    col1 = "uniprot1" if "uniprot1" in edges_df.columns else "protein1"
    col2 = "uniprot2" if "uniprot2" in edges_df.columns else "protein2"

    if edges_df.empty:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
    else:
        # Vectorized edge index construction
        p1_arr = edges_df[col1].values
        p2_arr = edges_df[col2].values
        w_arr  = edges_df["weight"].values if "weight" in edges_df.columns else np.ones(len(edges_df))

        valid = np.array(
            [(p in protein_to_idx and q in protein_to_idx) for p, q in zip(p1_arr, p2_arr)]
        )
        p1_arr, p2_arr, w_arr = p1_arr[valid], p2_arr[valid], w_arr[valid]

        i_idx = np.array([protein_to_idx[p] for p in p1_arr])
        j_idx = np.array([protein_to_idx[p] for p in p2_arr])

        # Undirected: both directions
        src = np.concatenate([i_idx, j_idx])
        dst = np.concatenate([j_idx, i_idx])
        wts = np.concatenate([w_arr, w_arr])
        
        # Add self-edges for isolated nodes
        connected_nodes = set(src)
        isolated_nodes = [i for i in range(n_nodes) if i not in connected_nodes]
        if isolated_nodes:
            iso_src = np.array(isolated_nodes)
            iso_dst = np.array(isolated_nodes)
            iso_wts = np.ones(len(isolated_nodes))
            src = np.concatenate([src, iso_src])
            dst = np.concatenate([dst, iso_dst])
            wts = np.concatenate([wts, iso_wts])

        edge_index  = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_weight = torch.tensor(wts, dtype=torch.float32)


    # Node features
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        # Fallback placeholder matching ESM-2 650M dimension (1280)
        x = torch.zeros((n_nodes, 1280), dtype=torch.float32)

    # Topological centrality features (3 dims: degree, betweenness, closeness)
    if centrality is not None and "degree_centrality" in centrality:
        logger.info("Injecting topological centrality features into GAT node embeddings...")
        deg_c = np.array([centrality["degree_centrality"].get(p, 0.0) for p in protein_list]).reshape(-1, 1)
        bet_c = np.array([centrality.get("betweenness", {}).get(p, 0.0) for p in protein_list]).reshape(-1, 1)
        clo_c = np.array([centrality.get("closeness", {}).get(p, 0.0) for p in protein_list]).reshape(-1, 1)
        cent_feats = torch.tensor(np.hstack([deg_c, bet_c, clo_c]), dtype=torch.float32)
        x = torch.cat([x, cent_feats], dim=-1)
    else:
        cent_zeros = torch.zeros((n_nodes, 3), dtype=torch.float32)
        x = torch.cat([x, cent_zeros], dim=-1)

    # Labels
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float32)
    else:
        y = None

    pyg_data = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "x": x,
        "y": y,
        "num_nodes": n_nodes,
        "num_edges": edge_index.shape[1],
        "protein_to_idx": protein_to_idx,
        "idx_to_protein": {i: p for p, i in protein_to_idx.items()},
        "protein_list": protein_list,
    }

    logger.info(f"PyG graph: {n_nodes} nodes, {edge_index.shape[1]} directed edges")

    return pyg_data


def build_ppi_graph(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main PPI graph construction pipeline.

    Args:
        config: Project configuration.

    Returns:
        Dictionary with graph data and statistics.
    """
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    ensure_dir(processed_dir)

    # Load annotated proteins
    proteins_df = pd.read_csv(os.path.join(processed_dir, "proteins_annotated.tsv"), sep="\t")
    protein_list = proteins_df["accession"].tolist()
    logger.info(f"Building PPI graph for {len(protein_list)} proteins")

    # Load STRING interactions
    species_id = data_cfg["string"]["species_id"]
    confidence_threshold = data_cfg["string"].get("confidence_threshold", 0.7)
    interactions_path = os.path.join(raw_dir, f"string_interactions_{species_id}.tsv")
    mapping_path = os.path.join(raw_dir, "string_uniprot_mapping.tsv")

    if not os.path.exists(interactions_path) or not os.path.exists(mapping_path):
        logger.warning("STRING data not found. Creating empty graph.")
        edges_df = pd.DataFrame(columns=["uniprot1", "uniprot2", "weight"])
    else:
        edges_df = load_string_interactions(
            interactions_path, mapping_path, protein_list, confidence_threshold=confidence_threshold
        )

    # Build NetworkX graph
    G = build_networkx_graph(edges_df, protein_list)

    # Compute statistics
    net_stats = compute_network_statistics(G)

    # Identify hubs
    hub_status = identify_hub_proteins(G)

    # Compute centrality
    centrality = compute_centrality_metrics(G)

    # Load labels
    label_matrix = np.load(os.path.join(processed_dir, "label_matrix.npy"))

    # Load ESM-2 embeddings as node features (instead of identity matrix placeholder)
    embeddings_dir = data_cfg["embeddings_dir"]
    embeddings_path = os.path.join(embeddings_dir, "esm2_embeddings.npy")
    node_features = None
    if os.path.exists(embeddings_path):
        node_features = np.load(embeddings_path)
        logger.info(f"Loaded ESM-2 embeddings as node features: {node_features.shape}")
    else:
        logger.warning("ESM-2 embeddings not found! Graph will use identity matrix placeholder. "
                       "Run preprocessing/extract_embeddings.py first for best results.")

    # Build PyG data
    pyg_data = build_pyg_data(edges_df, protein_list, node_features=node_features, labels=label_matrix, centrality=centrality)

    # Save outputs
    # NOTE: pyg_data contains PyTorch tensors  -  must use torch.save, NOT pickle
    import torch as _torch
    _torch.save(pyg_data, os.path.join(processed_dir, "pyg_graph_data.pt"))
    # Also save a pickle-compatible summary (no tensors)
    pyg_meta = {
        "num_nodes": pyg_data["num_nodes"],
        "num_edges": pyg_data["num_edges"],
        "protein_to_idx": pyg_data["protein_to_idx"],
    }
    save_pickle(pyg_meta, os.path.join(processed_dir, "pyg_graph_meta.pkl"))
    save_pickle(hub_status, os.path.join(processed_dir, "hub_status.pkl"))
    save_pickle(centrality, os.path.join(processed_dir, "centrality_metrics.pkl"))
    save_json(net_stats, os.path.join(processed_dir, "network_statistics.json"))
    edges_df.to_csv(os.path.join(processed_dir, "ppi_edges.tsv"), sep="\t", index=False)

    logger.info("PPI graph construction complete!")
    logger.info(f"  Saved graph to {processed_dir}/pyg_graph_data.pt")

    return {
        "graph": G,
        "pyg_data": pyg_data,
        "statistics": net_stats,
        "hub_status": hub_status,
        "centrality": centrality,
    }


def main():
    parser = argparse.ArgumentParser(description="Build PPI graph from STRING data")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    build_ppi_graph(config)


if __name__ == "__main__":
    main()
