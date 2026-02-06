import os

def check_missing_folders():
    # Define paths
    json_dir = 'Copilot_Processed_Datasets_JSON/Copilot_222_v0_Processed_2025-12-04_Updated_Metadata'
    supp_dir = 'Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary/DOME_Registry_PMC_Supplementary'

    # 1. Get PMCIDs from JSON files
    if not os.path.exists(json_dir):
        print(f"JSON directory not found: {json_dir}")
        return

    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
    json_pmcids = set([os.path.splitext(f)[0] for f in json_files])
    
    print(f"Total JSON files in source: {len(json_pmcids)}")

    # 2. Get PMCIDs from Supplementary directory folders
    if not os.path.exists(supp_dir):
        print(f"Supplementary directory not found: {supp_dir}")
        return

    supp_folders = [f for f in os.listdir(supp_dir) if os.path.isdir(os.path.join(supp_dir, f))]
    supp_pmcids = set(supp_folders)
    
    print(f"Total folders in target: {len(supp_pmcids)}")

    # 3. Find JSONs that are not folders
    missing_folders = json_pmcids - supp_pmcids
    
    print(f"\nFound {len(missing_folders)} PMCIDs that are in the JSON folder but NOT in the Supplementary folder:")
    for pmcid in sorted(list(missing_folders)):
        print(pmcid)

if __name__ == "__main__":
    check_missing_folders()
