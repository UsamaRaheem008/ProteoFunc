"""Patch build_graph.py  -  replace buggy build_networkx_graph body."""
import re

path = r"preprocessing/build_graph.py"
src = open(path, "r", encoding="utf-8").read()

# Find and replace the entire function body between the two markers
old = '''    G = nx.Graph()
    G.add_nodes_from(protein_list)

    col1 = "uniprot1" if "uniprot1" in edges_df.columns else "protein1"
    col2 = "uniprot2" if "uniprot2" in edges_df.columns else "protein2"

    if not edges_df.empty:
        # Vectorized edge addition (much faster than iterrows)
        edge_tuples = [
            (row[col1], row[col2], {"weight": row.get("weight", 1.0)})
            for _, row in edges_df[[col1, col2, "weight"]].itertuples(index=False, name=None)
        ] if "weight" in edges_df.columns else [
            (row[col1], row[col2], {"weight": 1.0})
            for _, row in edges_df[[col1, col2]].itertuples(index=False, name=None)
        ]
        G.add_edges_from(
            (r[0], r[1], {"weight": float(r[2]) if len(r) > 2 else 1.0})
            for r in edges_df[[col1, col2] + (["weight"] if "weight" in edges_df.columns else [])].itertuples(index=False, name=None)
        )

    logger.info(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"  Density: {nx.density(G):.6f}")
    logger.info(f"  Connected components: {nx.number_connected_components(G)}")

    return G'''

new = '''    G = nx.Graph()
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

    return G'''

if old in src:
    new_src = src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(new_src)
    print("PATCHED: build_networkx_graph body replaced successfully.")
else:
    # Print what we find around line 104 for debugging
    lines = src.splitlines()
    for i, l in enumerate(lines[100:130], start=101):
        print(f"{i}: {l}")
    print("\nOLD pattern not found  -  check the lines above.")
