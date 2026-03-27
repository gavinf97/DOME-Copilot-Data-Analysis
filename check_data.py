import pandas as pd

file_path = 'Human_30_Copilot_vs_Human_Evaluations_Interface/evaluation_results.tsv'
df = pd.read_csv(file_path, sep='\t')

# Count entries per PMCID
counts = df['PMCID'].value_counts()
print("=== All PMCID Counts ===")
print(counts)

# Let's find the ones that don't match the max count (27 fields generally)
print("\n=== Incomplete PMCIDs ===")
max_count = counts.max()
incomplete = counts[counts < max_count]
print(incomplete)

for pmcid in incomplete.index:
    # 1-based index including header = df index + 2
    rows = df[df['PMCID'] == pmcid].index + 2
    print(f"\nRow numbers for {pmcid} in the TSV (including header):")
    print(list(rows))
