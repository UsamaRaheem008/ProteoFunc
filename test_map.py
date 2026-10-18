import pandas as pd
import re

mapping_path = 'data/raw/string_uniprot_mapping.tsv'
interactions_path = 'data/raw/string_interactions_9606.tsv'
proteins_path = 'data/processed/proteins_annotated.tsv'

proteins_df = pd.read_csv(proteins_path, sep='\t')
valid_proteins = proteins_df['accession'].tolist()
valid_set = set(valid_proteins)
print(f'Valid proteins: {len(valid_set)}')

mapping_df = pd.read_csv(mapping_path, sep='\t')
uniprot_pattern = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|'
    r'^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]$|'
    r'^[A-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z0-9]{2,4}$'
)

is_valid_uniprot = mapping_df['uniprot_accession'].apply(
    lambda x: bool(uniprot_pattern.match(str(x)))
)
mapping_uniprot = mapping_df[is_valid_uniprot].copy()
print(f'Valid UniProt regex matches: {len(mapping_uniprot)}')
print(f'Unique UniProt regex matches: {mapping_uniprot["uniprot_accession"].nunique()}')
print(f'Regex matches IN dataset: {mapping_uniprot["uniprot_accession"].isin(valid_set).sum()}')

mapping_in_dataset = mapping_uniprot[
    mapping_uniprot['uniprot_accession'].isin(valid_set)
].copy()
mapping_dedup = mapping_in_dataset.drop_duplicates(subset=['string_id'], keep='first')
string_to_uniprot = dict(
    zip(mapping_dedup['string_id'], mapping_dedup['uniprot_accession'])
)
print(f'Final dictionary size (ENSPs mapped): {len(string_to_uniprot)}')

interactions_df = pd.read_csv(interactions_path, sep='\t')
print(f'Raw interactions: {len(interactions_df)}')

interactions_df['uniprot1'] = interactions_df['protein1'].map(string_to_uniprot)
interactions_df['uniprot2'] = interactions_df['protein2'].map(string_to_uniprot)

print(f'Mapped p1: {interactions_df["uniprot1"].notna().sum()}')
print(f'Mapped p2: {interactions_df["uniprot2"].notna().sum()}')
