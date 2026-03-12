import os
import json
import csv

def main():
    base_dir = "Copilot_Processed_Datasets_JSON"
    
    # Identify the base folders and their corresponding updated folders
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.endswith("_Updated_Metadata")]
    
    pub_fields_to_check = [
        "publication/title", "publication/authors", "publication/journal", 
        "publication/year", "publication/pmid", "publication/pmcid", "publication/doi"
    ]
    
    total_files = 0
    pub_diff_count = 0
    unexpected_diff_files = []
    
    field_diff_stats = {field: 0 for field in pub_fields_to_check}

    for base_folder in folders:
        updated_folder = base_folder + "_Updated_Metadata"
        base_path = os.path.join(base_dir, base_folder)
        updated_path = os.path.join(base_dir, updated_folder)
        
        if not os.path.exists(updated_path):
            print(f"Skipping {base_folder}: No updated folder found.")
            continue
            
        json_files = [f for f in os.listdir(base_path) if f.endswith(".json")]
        
        for j_file in json_files:
            b_file_path = os.path.join(base_path, j_file)
            u_file_path = os.path.join(updated_path, j_file)
            
            if not os.path.exists(u_file_path):
                continue
                
            total_files += 1
            
            try:
                with open(b_file_path, "r", encoding="utf-8") as bf:
                    b_data = json.load(bf)
                with open(u_file_path, "r", encoding="utf-8") as uf:
                    u_data = json.load(uf)
            except Exception as e:
                print(f"Error reading {j_file}: {e}")
                continue
            
            # 1. Check publication metadata fields
            has_pub_diff = False
            for field in pub_fields_to_check:
                b_val = str(b_data.get(field, "")).strip()
                u_val = str(u_data.get(field, "")).strip()
                if b_val != u_val:
                    has_pub_diff = True
                    field_diff_stats[field] += 1
            
            if has_pub_diff:
                pub_diff_count += 1
                
            # 2. Check all other fields (non-metadata fields + tags)
            all_keys = set(list(b_data.keys()) + list(u_data.keys()))
            unexpected_diffs = {}
            for key in all_keys:
                if key not in pub_fields_to_check:
                    b_val = b_data.get(key)
                    u_val = u_data.get(key)
                    if b_val != u_val:
                        unexpected_diffs[key] = {
                            "base": str(b_val)[:100] + "..." if b_val else "None",
                            "updated": str(u_val)[:100] + "..." if u_val else "None"
                        }
            
            if unexpected_diffs:
                pmcid = j_file.replace(".json", "")
                unexpected_diff_files.append({
                    "pmcid": pmcid,
                    "folder": base_folder,
                    "diffs": json.dumps(unexpected_diffs)
                })

    print("=================== QC REPORT ===================")
    print(f"Total files compared: {total_files}")
    if total_files > 0:
        print(f"Files with publication metadata differences: {pub_diff_count} ({pub_diff_count/total_files*100:.2f}%)")
        print("\nBreakdown of publication field differences:")
        for field, count in field_diff_stats.items():
            print(f"  - {field}: {count} files ({count/total_files*100:.2f}%)")
            
        print(f"\nFiles with UNEXPECTED differences (other fields): {len(unexpected_diff_files)}")
    
    if unexpected_diff_files:
        csv_file = "unexpected_metadata_diffs_qc.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pmcid", "folder", "diffs"])
            writer.writeheader()
            writer.writerows(unexpected_diff_files)
        print(f"\nSaved unexpected differences to {csv_file}")
        
        print("\nExamples of unexpected differences (up to 3):")
        for ex in unexpected_diff_files[:3]:
            print(f"PMCID: {ex['pmcid']} | Keys: {list(json.loads(ex['diffs']).keys())}")
    else:
        print("\nSUCCESS: No unexpected differences found in non-publication fields! All other fields are perfectly identical.")

if __name__ == "__main__":
    main()
