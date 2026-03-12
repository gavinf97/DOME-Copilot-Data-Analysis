import os
import json
import pandas as pd

def main():
    metadata_files = [
        "Datasets_Metadata/DOME_Registry_222_Metadata.tsv",
        "Datasets_Metadata/Negative_1012_Metadata.tsv",
        "Datasets_Metadata/Positive_1012_Metadata.tsv"
    ]
    
    metadata_dict = {}
    for mf in metadata_files:
        if not os.path.exists(mf):
            print(f"File not found: {mf}")
            continue
        df = pd.read_csv(mf, sep="\t")
        for _, row in df.iterrows():
            pmcid = str(row["pmcid"]).strip()
            if pd.isna(row["pmcid"]) or pmcid == "nan":
                continue
            
            metadata_dict[pmcid] = {
                "title": str(row["title"]) if pd.notna(row["title"]) else "",
                "authors": str(row["authors"]) if pd.notna(row["authors"]) else "",
                "journal": str(row["journal"]) if pd.notna(row["journal"]) else "",
                "year": str(int(row["year"])) if pd.notna(row["year"]) else "",
                "pmid": str(int(row["pmid"])) if pd.notna(row["pmid"]) else "",
                "pmcid": pmcid,
                "doi": str(row["doi"]) if pd.notna(row["doi"]) else ""
            }

    print(f"Loaded metadata for {len(metadata_dict)} PMCIDs.")

    base_dir = "Copilot_Processed_Datasets_JSON"
    count_updated = 0
    count_not_found = 0

    for folder in os.listdir(base_dir):
        if not folder.endswith("_Updated_Metadata"):
            continue
        
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                pmcid = filename.replace(".json", "")
                
                if pmcid in metadata_dict:
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Error reading JSON: {file_path}")
                            continue

                    info = metadata_dict[pmcid]
                    data["publication/title"] = info["title"]
                    data["publication/authors"] = info["authors"]
                    data["publication/journal"] = info["journal"]
                    if info["year"]: data["publication/year"] = info["year"]
                    if info["pmid"]: data["publication/pmid"] = info["pmid"]
                    data["publication/pmcid"] = info["pmcid"]
                    if info["doi"]: data["publication/doi"] = info["doi"]

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                    
                    count_updated += 1
                else:
                    count_not_found += 1
                    # print(f"Metadata not found for PMCID: {pmcid}")

    print(f"Updated {count_updated} files.")
    print(f"Could not find metadata for {count_not_found} files.")

if __name__ == "__main__":
    main()
