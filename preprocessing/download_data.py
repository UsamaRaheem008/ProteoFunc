"""
Data Download Module
====================
Downloads protein data from UniProtKB/SwissProt, STRING, and Gene Ontology.

Usage:
    python preprocessing/download_data.py --config configs/config.yaml
"""

import argparse
import gzip
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir, save_json

logger = get_logger(__name__)


# ====================================================================
# UniProt Download
# ====================================================================

def download_uniprot_proteins(
    taxonomy_id: int = 9606,
    reviewed: bool = True,
    max_proteins: int = 15000,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Download reviewed human proteins from UniProtKB/SwissProt REST API.

    Args:
        taxonomy_id: NCBI taxonomy ID (9606 for Homo sapiens).
        reviewed: If True, download only reviewed (SwissProt) entries.
        max_proteins: Maximum number of proteins to download.
        output_dir: Directory to save downloaded data.

    Returns:
        DataFrame with protein accession, sequence, gene names, GO annotations.
    """
    logger.info(f"Downloading UniProt proteins (taxonomy={taxonomy_id}, reviewed={reviewed})")
    ensure_dir(output_dir)

    # UniProt REST API (2024+ format)
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    query = f"(organism_id:{taxonomy_id})"
    if reviewed:
        query += " AND (reviewed:true)"

    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,id,protein_name,gene_names,sequence,length,go_id,go,organism_name",
        "size": max_proteins,
    }

    logger.info(f"Querying UniProt API: {query}")

    try:
        response = requests.get(base_url, params=params, timeout=300)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"UniProt API request failed: {e}")
        logger.info("Attempting fallback download via TSV file...")
        return _download_uniprot_fallback(taxonomy_id, reviewed, max_proteins, output_dir)

    # Parse TSV response
    df = pd.read_csv(io.StringIO(response.text), sep="\t")

    # Rename columns for consistency
    column_mapping = {
        "Entry": "accession",
        "Entry Name": "entry_name",
        "Protein names": "protein_name",
        "Gene Names": "gene_names",
        "Sequence": "sequence",
        "Length": "length",
        "Gene Ontology IDs": "go_ids",
        "Gene Ontology (GO)": "go_terms",
        "Organism": "organism",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    logger.info(f"Downloaded {len(df)} proteins from UniProt")

    # Save raw data
    output_path = os.path.join(output_dir, "uniprot_proteins.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved raw UniProt data to {output_path}")

    return df


def _download_uniprot_fallback(
    taxonomy_id: int,
    reviewed: bool,
    max_proteins: int,
    output_dir: str,
) -> pd.DataFrame:
    """
    Fallback download using UniProt search API with pagination.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(organism_id:{taxonomy_id})"
    if reviewed:
        query += " AND (reviewed:true)"

    all_results = []
    page_size = 500
    cursor = None
    total_fetched = 0

    while total_fetched < max_proteins:
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,id,protein_name,gene_names,sequence,length,go_id,go",
            "size": min(page_size, max_proteins - total_fetched),
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Fallback download failed at page {total_fetched}: {e}")
            break

        lines = response.text.strip().split("\n")
        if total_fetched == 0:
            all_results.append(lines[0])  # header
        all_results.extend(lines[1:])
        total_fetched += len(lines) - 1

        # Extract cursor for next page from Link header
        link_header = response.headers.get("Link", "")
        if 'rel="next"' in link_header:
            cursor = link_header.split("cursor=")[1].split("&")[0].rstrip(">")
        else:
            break

        logger.info(f"  Fetched {total_fetched} proteins...")
        time.sleep(0.5)  # Rate limiting

    text = "\n".join(all_results)
    df = pd.read_csv(io.StringIO(text), sep="\t")

    column_mapping = {
        "Entry": "accession",
        "Entry Name": "entry_name",
        "Protein names": "protein_name",
        "Gene Names": "gene_names",
        "Sequence": "sequence",
        "Length": "length",
        "Gene Ontology IDs": "go_ids",
        "Gene Ontology (GO)": "go_terms",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    output_path = os.path.join(output_dir, "uniprot_proteins.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Fallback: Saved {len(df)} proteins to {output_path}")

    return df


# ====================================================================
# STRING PPI Download
# ====================================================================

def download_string_interactions(
    species_id: int = 9606,
    version: str = "12.0",
    confidence_threshold: float = 0.7,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Download protein-protein interactions from STRING database.

    Args:
        species_id: NCBI species ID (9606 for human).
        version: STRING database version.
        confidence_threshold: Minimum combined confidence score (0-1).
        output_dir: Output directory.

    Returns:
        DataFrame of filtered PPI interactions.
    """
    logger.info(f"Downloading STRING interactions (species={species_id}, v{version})")
    ensure_dir(output_dir)

    # STRING download URL for protein links
    url = (
        f"https://stringdb-downloads.org/download/"
        f"protein.links.v{version}/{species_id}.protein.links.v{version}.txt.gz"
    )

    output_gz = os.path.join(output_dir, f"string_interactions_{species_id}.txt.gz")
    output_tsv = os.path.join(output_dir, f"string_interactions_{species_id}.tsv")

    # Download compressed file
    logger.info(f"Downloading from {url}")
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_gz, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="STRING") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        logger.error(f"STRING download failed: {e}")
        logger.info("Attempting STRING API fallback...")
        return _download_string_api_fallback(species_id, confidence_threshold, output_dir)

    # Parse and filter in chunks to avoid OOM on the 13M-row file
    logger.info("Parsing STRING interactions in chunks (memory-safe)...")
    threshold_score = int(confidence_threshold * 1000)  # STRING scores are 0-1000
    CHUNK_SIZE = 500_000

    dtype_map = {
        "protein1":      str,
        "protein2":      str,
        "combined_score": "int32",
    }

    chunks_filtered = []
    total_rows = 0
    kept_rows  = 0

    try:
        reader = pd.read_csv(
            output_gz,
            sep=" ",
            compression="gzip",
            dtype=dtype_map,
            chunksize=CHUNK_SIZE,
        )
        for chunk in reader:
            total_rows += len(chunk)
            chunk_filt = chunk[chunk["combined_score"] >= threshold_score]
            kept_rows  += len(chunk_filt)
            chunks_filtered.append(chunk_filt)
            logger.info(f"  Parsed {total_rows:,} rows, kept {kept_rows:,} above threshold...")
    except Exception as e:
        logger.error(f"Chunked read failed: {e}. Trying plain read with explicit dtypes...")
        # Last-resort: read with explicit types only (no chunking)
        df_raw = pd.read_csv(
            output_gz, sep=" ", compression="gzip", dtype=dtype_map
        )
        chunks_filtered = [df_raw[df_raw["combined_score"] >= threshold_score]]
        total_rows = len(df_raw)
        kept_rows  = len(chunks_filtered[0])

    df = pd.concat(chunks_filtered, ignore_index=True) if chunks_filtered else \
         pd.DataFrame(columns=["protein1", "protein2", "combined_score"])

    logger.info(f"STRING: {total_rows:,} total rows -> {kept_rows:,} above threshold...")

    # Remove species prefix from protein IDs (e.g., "9606.ENSP00000..." -> "ENSP00000...")
    df["protein1"] = df["protein1"].str.replace(f"{species_id}.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace(f"{species_id}.", "", regex=False)

    df.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"Saved {len(df):,} filtered STRING interactions to {output_tsv}")

    return df


def _download_string_api_fallback(
    species_id: int,
    confidence_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    """Fallback: download STRING interactions via API for a subset of proteins."""
    logger.warning("STRING bulk download failed. Using API fallback (slower, limited).")

    # This creates a minimal placeholder  -  full bulk download is recommended
    url = f"https://string-db.org/api/tsv/network"

    # We'll create an empty DataFrame with the right columns as a placeholder
    df = pd.DataFrame(columns=["protein1", "protein2", "combined_score", "combined_score_norm"])
    output_path = os.path.join(output_dir, f"string_interactions_{species_id}.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.warning(f"Created empty STRING placeholder at {output_path}")
    logger.warning("Please manually download STRING data from https://string-db.org/")
    return df


# ====================================================================
# STRING ID Mapping
# ====================================================================

def download_string_id_mapping(
    species_id: int = 9606,
    version: str = "12.0",
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Download STRING-to-UniProt ID mapping.

    Args:
        species_id: NCBI species ID.
        version: STRING version.
        output_dir: Output directory.

    Returns:
        DataFrame mapping STRING IDs (ENSP) to UniProt accessions.
    """
    logger.info("Downloading STRING ID mapping...")
    ensure_dir(output_dir)

    # STRING alias/mapping file
    url = (
        f"https://stringdb-downloads.org/download/"
        f"protein.aliases.v{version}/{species_id}.protein.aliases.v{version}.txt.gz"
    )

    output_path = os.path.join(output_dir, f"string_aliases_{species_id}.txt.gz")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"STRING alias download failed: {e}")
        return pd.DataFrame(columns=["string_id", "uniprot_accession"])

    # Parse aliases file
    df = pd.read_csv(output_path, sep="\t", compression="gzip")

    # The aliases file has columns: #string_protein_id, alias, source
    # Filter for UniProt entries
    col_names = df.columns.tolist()
    if "#string_protein_id" in col_names:
        df = df.rename(columns={"#string_protein_id": "string_protein_id"})

    # Filter UniProt_AC source entries
    uniprot_mapping = df[df["source"].str.contains("UniProt", case=False, na=False)].copy()
    uniprot_mapping["string_protein_id"] = uniprot_mapping["string_protein_id"].str.replace(
        f"{species_id}.", "", regex=False
    )

    mapping = uniprot_mapping[["string_protein_id", "alias"]].rename(
        columns={"string_protein_id": "string_id", "alias": "uniprot_accession"}
    )
    mapping = mapping.drop_duplicates()

    output_map_path = os.path.join(output_dir, "string_uniprot_mapping.tsv")
    mapping.to_csv(output_map_path, sep="\t", index=False)
    logger.info(f"Saved {len(mapping)} STRING-UniProt mappings to {output_map_path}")

    return mapping


# ====================================================================
# Gene Ontology Download
# ====================================================================

def download_go_obo(output_dir: str = "data/raw") -> str:
    """
    Download Gene Ontology OBO file.

    Args:
        output_dir: Output directory.

    Returns:
        Path to the downloaded OBO file.
    """
    logger.info("Downloading Gene Ontology OBO file...")
    ensure_dir(output_dir)

    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    output_path = os.path.join(output_dir, "go-basic.obo")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info(f"Saved GO OBO file to {output_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"GO OBO download failed: {e}")

    return output_path


# ====================================================================
# Main Download Pipeline
# ====================================================================

def run_download_pipeline(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Execute the complete data download pipeline.

    Args:
        config: Project configuration dictionary.

    Returns:
        Dictionary of output file paths.
    """
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    ensure_dir(raw_dir)

    output_paths = {}

    # 1. Download UniProt proteins
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading UniProt proteins")
    logger.info("=" * 60)
    uniprot_df = download_uniprot_proteins(
        taxonomy_id=data_cfg["uniprot"]["taxonomy_id"],
        reviewed=data_cfg["uniprot"]["reviewed"],
        max_proteins=data_cfg["uniprot"]["max_proteins"],
        output_dir=raw_dir,
    )
    output_paths["uniprot"] = os.path.join(raw_dir, "uniprot_proteins.tsv")

    # 2. Download STRING interactions
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading STRING interactions")
    logger.info("=" * 60)
    string_df = download_string_interactions(
        species_id=data_cfg["string"]["species_id"],
        version=data_cfg["string"]["version"],
        confidence_threshold=data_cfg["string"]["confidence_threshold"],
        output_dir=raw_dir,
    )
    output_paths["string"] = os.path.join(raw_dir, f"string_interactions_{data_cfg['string']['species_id']}.tsv")

    # 3. Download STRING ID mapping
    logger.info("=" * 60)
    logger.info("STEP 3: Downloading STRING-UniProt mapping")
    logger.info("=" * 60)
    mapping_df = download_string_id_mapping(
        species_id=data_cfg["string"]["species_id"],
        version=data_cfg["string"]["version"],
        output_dir=raw_dir,
    )
    output_paths["mapping"] = os.path.join(raw_dir, "string_uniprot_mapping.tsv")

    # 4. Download GO OBO
    logger.info("=" * 60)
    logger.info("STEP 4: Downloading Gene Ontology")
    logger.info("=" * 60)
    go_path = download_go_obo(output_dir=raw_dir)
    output_paths["go_obo"] = go_path

    # Save manifest
    save_json(output_paths, os.path.join(raw_dir, "download_manifest.json"))
    logger.info("Download pipeline complete!")

    return output_paths


# ====================================================================
# CLI Entry Point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Download datasets for protein function prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_download_pipeline(config)


if __name__ == "__main__":
    main()
