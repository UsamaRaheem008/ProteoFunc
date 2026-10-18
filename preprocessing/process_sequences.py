"""
Sequence Processing Module
==========================
Parses, cleans, deduplicates, and filters protein sequences.

Usage:
    python preprocessing/process_sequences.py --config configs/config.yaml
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.helpers import load_config, ensure_dir

logger = get_logger(__name__)

# Standard amino acid alphabet
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def validate_sequence(sequence: str, allow_extended: bool = False) -> bool:
    """
    Validate that a string is a valid protein sequence.

    Args:
        sequence: Amino acid sequence string.
        allow_extended: If True, allow ambiguous amino acid codes.

    Returns:
        True if sequence is valid.
    """
    if not sequence or not isinstance(sequence, str):
        return False
    alphabet = EXTENDED_AA if allow_extended else AMINO_ACIDS
    return all(c in alphabet for c in sequence.upper())


def compute_sequence_hash(sequence: str) -> str:
    """Compute MD5 hash of a sequence for deduplication."""
    return hashlib.md5(sequence.upper().encode()).hexdigest()


def parse_fasta_file(fasta_path: str) -> pd.DataFrame:
    """
    Parse a FASTA file into a DataFrame.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        DataFrame with columns: accession, description, sequence, length.
    """
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract UniProt accession from header
        # Format: >sp|P04637|P53_HUMAN Cellular tumor antigen p53 ...
        parts = record.id.split("|")
        accession = parts[1] if len(parts) >= 2 else record.id
        records.append({
            "accession": accession,
            "description": record.description,
            "sequence": str(record.seq),
            "length": len(record.seq),
        })
    return pd.DataFrame(records)


def clean_sequences(
    df: pd.DataFrame,
    min_length: int = 50,
    max_length: int = 1200,
    remove_duplicates: bool = True,
    sequence_col: str = "sequence",
    id_col: str = "accession",
) -> pd.DataFrame:
    """
    Clean and filter protein sequences.

    Args:
        df: DataFrame with protein sequences.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.
        remove_duplicates: Whether to remove duplicate sequences.
        sequence_col: Name of the sequence column.
        id_col: Name of the identifier column.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    logger.info(f"Starting sequence cleaning with {initial_count} proteins")

    # 1. Remove rows with missing sequences
    df = df.dropna(subset=[sequence_col]).copy()
    logger.info(f"  After removing missing sequences: {len(df)}")

    # 2. Uppercase and strip whitespace
    df[sequence_col] = df[sequence_col].str.upper().str.strip()

    # 3. Remove non-standard characters (keep standard AAs only)
    def clean_seq(seq: str) -> str:
        return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)

    df[sequence_col] = df[sequence_col].apply(clean_seq)

    # 4. Add length column
    df["length"] = df[sequence_col].str.len()

    # 5. Filter by length
    df = df[(df["length"] >= min_length) & (df["length"] <= max_length)].copy()
    logger.info(f"  After length filter ({min_length}-{max_length}): {len(df)}")

    # 6. Remove duplicate sequences (keep first occurrence)
    if remove_duplicates:
        df["seq_hash"] = df[sequence_col].apply(compute_sequence_hash)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["seq_hash"], keep="first")
        df = df.drop(columns=["seq_hash"])
        logger.info(f"  After deduplication: {len(df)} (removed {before_dedup - len(df)} duplicates)")

    # 7. Validate sequences
    valid_mask = df[sequence_col].apply(validate_sequence)
    df = df[valid_mask].copy()
    logger.info(f"  After validation: {len(df)}")

    # 8. Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Sequence cleaning: {initial_count} -> {len(df)} proteins")
    return df


def compute_sequence_statistics(df: pd.DataFrame, sequence_col: str = "sequence") -> Dict[str, Any]:
    """
    Compute descriptive statistics for the protein sequences.

    Args:
        df: DataFrame with sequences.
        sequence_col: Sequence column name.

    Returns:
        Dictionary of statistics.
    """
    lengths = df[sequence_col].str.len()

    # Amino acid composition
    all_seqs = "".join(df[sequence_col].values)
    aa_counts = Counter(all_seqs)
    total_aa = sum(aa_counts.values())
    aa_freq = {aa: count / total_aa for aa, count in sorted(aa_counts.items())}

    stats = {
        "num_proteins": len(df),
        "length_mean": float(lengths.mean()),
        "length_median": float(lengths.median()),
        "length_std": float(lengths.std()),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_q25": float(lengths.quantile(0.25)),
        "length_q75": float(lengths.quantile(0.75)),
        "aa_frequencies": aa_freq,
        "total_residues": total_aa,
    }

    logger.info(f"Sequence statistics:")
    logger.info(f"  Proteins: {stats['num_proteins']}")
    logger.info(f"  Length: {stats['length_mean']:.0f} +/- {stats['length_std']:.0f} "
                f"(range: {stats['length_min']}-{stats['length_max']})")

    return stats


def write_fasta(
    df: pd.DataFrame,
    output_path: str,
    id_col: str = "accession",
    sequence_col: str = "sequence",
) -> None:
    """
    Write sequences to FASTA format.

    Args:
        df: DataFrame with protein data.
        output_path: Output FASTA file path.
        id_col: Column with sequence identifiers.
        sequence_col: Column with sequences.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:  # only call ensure_dir if there is a directory component
        ensure_dir(out_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row[id_col]}\n")
            # Wrap at 80 characters
            seq = row[sequence_col]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

    logger.info(f"Wrote {len(df)} sequences to {output_path}")


def process_sequences(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main sequence processing pipeline.

    Args:
        config: Project configuration.

    Returns:
        Cleaned protein DataFrame.
    """
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    ensure_dir(processed_dir)

    # Load raw UniProt data
    raw_path = os.path.join(raw_dir, "uniprot_proteins.tsv")
    if not os.path.exists(raw_path):
        logger.error(f"UniProt data not found at {raw_path}. Run download_data.py first.")
        return pd.DataFrame()

    logger.info(f"Loading raw UniProt data from {raw_path}")
    df = pd.read_csv(raw_path, sep="\t")

    # Clean sequences
    df = clean_sequences(
        df,
        min_length=data_cfg["uniprot"]["min_sequence_length"],
        max_length=data_cfg["uniprot"]["max_sequence_length"],
        remove_duplicates=True,
    )

    # Compute statistics
    stats = compute_sequence_statistics(df)

    # Save processed sequences
    output_path = os.path.join(processed_dir, "proteins_clean.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved cleaned proteins to {output_path}")

    # Save FASTA
    fasta_path = os.path.join(processed_dir, "proteins_clean.fasta")
    write_fasta(df, fasta_path)

    return df


def main():
    parser = argparse.ArgumentParser(description="Process protein sequences")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    process_sequences(config)


if __name__ == "__main__":
    main()
