import json
import os

def main():
    # -------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------
    source_json_path = 'DOME_Registry_Human_Reviews_258_20260205.json'
    
    # Target 1: Copilot Processed Dataset JSONs
    processed_json_dir = 'Copilot_Processed_Datasets_JSON/Copilot_222_v0_Processed_2025-12-04_Updated_Metadata'

    # Target 2: Folder PMC Titles
    target_folders_dir = 'Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary/DOME_Registry_PMC_Supplementary'


    # -------------------------------------------------------------
    # 1. READ SOURCE OF TRUTH (JSON)
    # -------------------------------------------------------------
    print(f"Reading Source JSON: {source_json_path}")
    source_pmcids = set()
    
    if os.path.exists(source_json_path):
        try:
            with open(source_json_path, 'r') as f:
                data = json.load(f)
                count = 0
                for entry in data:
                    if isinstance(entry, dict):
                        pub = entry.get('publication')
                        if isinstance(pub, dict):
                            pmcid = pub.get('pmcid')
                            # Ensure PMCID is a valid string
                            if isinstance(pmcid, str) and pmcid.strip():
                                source_pmcids.add(pmcid.strip())
                                count += 1
            print(f"-> Found {len(source_pmcids)} unique PMCIDs in Source JSON.")
        except Exception as e:
            print(f"ERROR reading Source JSON: {e}")
            return
    else:
        print(f"ERROR: Source JSON file not found at {source_json_path}")
        return

    # -------------------------------------------------------------
    # 2. READ PROCESSED JSON DIRECTORY
    # -------------------------------------------------------------
    print(f"\nReading Processed JSON Directory: {processed_json_dir}")
    processed_pmcids = set()
    
    if os.path.exists(processed_json_dir):
        files = os.listdir(processed_json_dir)
        for f in files:
            if f.lower().endswith('.json'):
                # Start with filename, strip extension
                pmcid = os.path.splitext(f)[0]
                if pmcid.strip():
                    processed_pmcids.add(pmcid.strip())
        print(f"-> Found {len(processed_pmcids)} unique JSON files (PMCIDs) in Processed Directory.")
    else:
        print(f"ERROR: Processed Directory not found at {processed_json_dir}")

    # -------------------------------------------------------------
    # 3. READ TARGET FOLDERS DIRECTORY
    # -------------------------------------------------------------
    print(f"\nReading Target Folders Directory: {target_folders_dir}")
    folder_pmcids = set()
    
    if os.path.exists(target_folders_dir):
        items = os.listdir(target_folders_dir)
        for item in items:
            item_path = os.path.join(target_folders_dir, item)
            # Only count directories
            if os.path.isdir(item_path):
                # Folder name corresponds to PMCID
                pmcid = item
                if pmcid.strip():
                    folder_pmcids.add(pmcid.strip())
        print(f"-> Found {len(folder_pmcids)} unique folders (PMCIDs) in Target Directory.")
    else:
        print(f"ERROR: Target Directory not found at {target_folders_dir}")


    # -------------------------------------------------------------
    # 4. PERFORM COMPARISONS & PRINT REPORT
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    # --- Comparison A: Source vs Processed JSONs ---
    print("\n--- A. Source JSON PMCIDs vs Copilot Processed JSON Content ---")
    
    # Intersection
    matches_A = source_pmcids.intersection(processed_pmcids)
    # In Source but Missing in Processed
    missing_A = source_pmcids - processed_pmcids
    # In Processed but not in Source (Unexpected?)
    extra_A = processed_pmcids - source_pmcids
    
    coverage_A = (len(matches_A) / len(source_pmcids) * 100) if len(source_pmcids) > 0 else 0
    
    print(f"Total Source PMCIDs: {len(source_pmcids)}")
    print(f"Total Processed JSONs: {len(processed_pmcids)}")
    print(f"Matches: {len(matches_A)}")
    print(f"Coverage: {coverage_A:.2f}%")
    
    print(f"\n[MISSING] PMCIDs present in Source but NOT in Processed JSONs ({len(missing_A)}):")
    if missing_A:
        print(", ".join(sorted(list(missing_A))))
    else:
        print("None")
        
    print(f"\n[EXTRA] PMCIDs present in Processed JSONs but NOT in Source ({len(extra_A)}):")
    if extra_A:
        print(", ".join(sorted(list(extra_A))))
    else:
        print("None")


    # --- Comparison B: Source vs Target Folders ---
    print("\n" + "-"*60)
    print("\n--- B. Source JSON PMCIDs vs Target Folders ---")
    
    # Intersection
    matches_B = source_pmcids.intersection(folder_pmcids)
    # In Source but Missing in Folders
    missing_B = source_pmcids - folder_pmcids
    # In Folders but not in Source
    extra_B = folder_pmcids - source_pmcids
    
    coverage_B = (len(matches_B) / len(source_pmcids) * 100) if len(source_pmcids) > 0 else 0
    
    print(f"Total Source PMCIDs: {len(source_pmcids)}")
    print(f"Total Target Folders: {len(folder_pmcids)}")
    print(f"Matches: {len(matches_B)}")
    print(f"Coverage: {coverage_B:.2f}%")
    
    print(f"\n[MISSING] PMCIDs present in Source but NOT in Target Folders ({len(missing_B)}):")
    if missing_B:
        print(", ".join(sorted(list(missing_B))))
    else:
        print("None")
        
    print(f"\n[EXTRA] PMCIDs present in Target Folders but NOT in Source ({len(extra_B)}):")
    if extra_B:
         print(", ".join(sorted(list(extra_B))))
    else:
        print("None")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
