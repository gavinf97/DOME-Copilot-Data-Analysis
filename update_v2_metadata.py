import os
import json
import csv
import shutil
import math

def clean_float_str(val):
    if not val or val != val:
        return ""
    try:
        fval = float(val)
        if math.isnan(fval):
            return ""
        if fval.is_integer():
            return str(int(fval))
        return str(fval)
    except ValueError:
        return str(val).strip()

def update_from_csv():
    # Source CSV
    csv_file = 'negative_entries.csv'
    
    # Paths for Negative
    source_folder_neg = 'Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Neg_Processed_2026-03-02'
    dest_folder_neg = 'Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Neg_Processed_2026-03-02_Updated_Metadata'
    
    # Read the CSV to get mappings
    pmcid_to_meta = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = row.get('PMCID', '').strip()
            if not pmcid:
                continue
            
            pmcid_to_meta[pmcid] = {
                "publication/title": row.get('Title', '').strip(),
                "publication/authors": row.get('Authors', '').strip(),
                "publication/journal": row.get('Journal', '').strip(),
                "publication/year": clean_float_str(row.get('Year', '')),
                "publication/pmid": clean_float_str(row.get('PMID', '')),
                "publication/pmcid": pmcid
            }
            
    # Process negative folder
    os.makedirs(dest_folder_neg, exist_ok=True)
    if os.path.exists(source_folder_neg):
        for filename in os.listdir(source_folder_neg):
            if filename.endswith('.json'):
                source_path = os.path.join(source_folder_neg, filename)
                dest_path = os.path.join(dest_folder_neg, filename)
                
                pmcid = filename.replace('.json', '')
                
                with open(source_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON from {source_path}")
                        continue
                
                # Update from csv mappings
                if pmcid in pmcid_to_meta:
                    for k, v in pmcid_to_meta[pmcid].items():
                        if v:
                            data[k] = v
                else:
                    print(f"Warning: {pmcid} not found in negative_entries.csv")
                
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed Negative Dataset -> {dest_folder_neg}")
    else:
        print(f"Source folder not found: {source_folder_neg}")

def update_from_json(target_source, target_dest, metadata_source_folder):
    os.makedirs(target_dest, exist_ok=True)
    if not os.path.exists(target_source):
        print(f"Source folder not found: {target_source}")
        return
        
    for filename in os.listdir(target_source):
        if filename.endswith('.json'):
            source_path = os.path.join(target_source, filename)
            dest_path = os.path.join(target_dest, filename)
            metadata_path = os.path.join(metadata_source_folder, filename)
            
            with open(source_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {source_path}")
                    continue
                    
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    try:
                        meta_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading {metadata_path}")
                        meta_data = {}
                        
                # Update specific fields
                keys_to_update = [
                    "publication/title", "publication/authors", 
                    "publication/journal", "publication/year", 
                    "publication/pmid", "publication/pmcid"
                ]
                for k in keys_to_update:
                    if k in meta_data and meta_data[k]:
                        data[k] = meta_data[k]
            else:
                print(f"Warning: Metadata source not found for {filename}")
                
            with open(dest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
    print(f"Processed {target_source} -> {target_dest}")

if __name__ == '__main__':
    # 1. Update Negative from CSV
    update_from_csv()
    
    # 2. Update 222
    target_source_222 = 'Copilot_Processed_Datasets_JSON/Copilot_222_v2_Processed_2026-03-02'
    target_dest_222 = 'Copilot_Processed_Datasets_JSON/Copilot_222_v2_Processed_2026-03-02_Updated_Metadata'
    metadata_source_222 = 'Copilot_Processed_Datasets_JSON/Copilot_222_v0_Processed_2025-12-04_Updated_Metadata'
    update_from_json(target_source_222, target_dest_222, metadata_source_222)
    
    # 3. Update 1012 Positive
    target_source_1012_pos = 'Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Pos_Processed_2026-03-02'
    target_dest_1012_pos = 'Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Pos_Processed_2026-03-02_Updated_Metadata'
    metadata_source_1012_pos = 'Copilot_Processed_Datasets_JSON/Copilot_1012_v0_Processed_2026-01-15_Updated_Metadata'
    update_from_json(target_source_1012_pos, target_dest_1012_pos, metadata_source_1012_pos)
