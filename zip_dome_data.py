import os
import json
import zipfile
import re
import csv
import sys

# Configuration
ZIP_NAME = "DOME_Copilot_Data_Package.zip"
ROOT_DIR = "DOME_Copilot_Data_Package"

# Map source file system path -> Destination folder name in zip
SOURCES = {
    "Copilot_Processed_Datasets_JSON": "Copilot_Processed_Datasets_JSON",
    "Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary/DOME_Registry_PMC_Supplementary": "DOME_Registry_PMC_Supplementary",
    "Download_1012_Negative_PMC_Full_Text_and_Supplementary/Negative_PMC_Supplementary": "Negative_PMC_Supplementary",
    "Download_1012_Positive_PMC_Full_Text_and_Supplementary/Positive_PMC_Supplementary": "Positive_PMC_Supplementary",
    "Human_30_Copilot_vs_Human_Evaluations_Interface/30_Evaluation_Source_JSONs_Human_and_Copilot_Including_PDFs": "30_Evaluation_Source_JSONs_Human_and_Copilot_Including_PDFs",
    "DOME_Registry_Human_Reviews_258_20260205.json": "DOME_Registry_Human_Reviews_258_20260205.json"
}

# Metadata tracking: PMCID -> {source_name: boolean}
metadata = {}

def extract_pmcid_from_string(text):
    match = re.search(r'(PMC\d+)', text)
    return match.group(1) if match else None

def scan_and_zip():
    print(f"Creating {ZIP_NAME}...")
    
    try:
        # Use ZIP_DEFLATED for compression
        with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            
            for source_path, dest_name in SOURCES.items():
                if not os.path.exists(source_path):
                    print(f"Warning: Source not found: {source_path}")
                    continue
                    
                print(f"Processing {source_path} -> {dest_name}...")
                
                # CASE 1: Single File
                if os.path.isfile(source_path):
                    # Add file to zip
                    arcname = os.path.join(ROOT_DIR, dest_name)
                    zf.write(source_path, arcname)
                    
                    # Metadata extraction for the specific registry JSON file
                    if "DOME_Registry_Human_Reviews" in source_path and source_path.endswith('.json'):
                        try:
                            print("Extracting metadata from Registry Review JSON...")
                            with open(source_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Handle both list of objects and potentially other structures if changed, but assuming list based on read_file
                                if isinstance(data, list):
                                    for entry in data:
                                        # Safe extraction
                                        pub = entry.get('publication', {})
                                        pmcid = pub.get('pmcid')
                                        
                                        # Sometimes PMCID might be just a number or empty
                                        if pmcid:
                                            # Normalize PMCID (ensure it starts with PMC if it is just numbers, though the context showed "PMC10087011")
                                            if not str(pmcid).startswith('PMC'):
                                                pmcid = f"PMC{pmcid}"
                                                
                                            if pmcid not in metadata: metadata[pmcid] = {}
                                            metadata[pmcid][dest_name] = True
                        except Exception as e:
                            print(f"Error reading registry file metadata: {e}")
                
                # CASE 2: Directory
                elif os.path.isdir(source_path):
                    for root, dirs, files in os.walk(source_path):
                        # Relative path from the source folder
                        rel_path = os.path.relpath(root, source_path)
                        
                        # 1. Check directories for metadata (e.g. .../PMC12345/)
                        for d in dirs:
                            pmcid = extract_pmcid_from_string(d)
                            if pmcid:
                                if pmcid not in metadata: metadata[pmcid] = {}
                                metadata[pmcid][dest_name] = True
                                
                        # 2. Check files for metadata (e.g. PMC12345.json)
                        for f in files:
                            pmcid = extract_pmcid_from_string(f)
                            if pmcid:
                                if pmcid not in metadata: metadata[pmcid] = {}
                                metadata[pmcid][dest_name] = True
                            
                            # Add file to zip
                            abs_file_path = os.path.join(root, f)
                            
                            # Construct archive name
                            if rel_path == ".":
                                arc_path = os.path.join(ROOT_DIR, dest_name, f)
                            else:
                                arc_path = os.path.join(ROOT_DIR, dest_name, rel_path, f)
                                
                            try:
                                zf.write(abs_file_path, arc_path)
                            except Exception as e:
                                print(f"Error zipping {abs_file_path}: {e}")

            # Generate and Write Metadata CSV
            print("Generating overarching metadata file...")
            meta_filename = "DOME_Copilot_Data_Package_Metadata.csv"
            
            # Columns: PMCID, Source1, Source2, ...
            source_cols = list(SOURCES.values())
            fieldnames = ['PMCID'] + source_cols
            
            # We need to write this to a temp file first or write string to zip
            # Using temp file
            with open(meta_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                
                # Sort PMCIDs numerically if possible, otherwise string sort
                sorted_pmcids = sorted(metadata.keys())
                
                for pmcid in sorted_pmcids:
                    row = [pmcid]
                    for source in source_cols:
                        if metadata[pmcid].get(source):
                            row.append('Yes')
                        else:
                            row.append('No')
                    writer.writerow(row)
            
            # Add metadata file to zip
            zf.write(meta_filename, os.path.join(ROOT_DIR, meta_filename))
            
            # Cleanup temp file
            os.remove(meta_filename)
        
        print(f"Zip created successfully: {ZIP_NAME}")

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    scan_and_zip()
