import os

def create_missing_folders():
    target_base_dir = 'Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary/DOME_Registry_PMC_Supplementary'
    
    missing_pmcids = [
        "PMC10316696",
        "PMC11223784",
        "PMC11512451",
        "PMC12077394"
    ]

    if not os.path.exists(target_base_dir):
        print(f"Target directory does not exist: {target_base_dir}")
        return

    print(f"Creating missing folders in: {target_base_dir}")

    for pmcid in missing_pmcids:
        # User requested prefix "empty_"
        folder_name = f"empty_{pmcid}"
        folder_path = os.path.join(target_base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"Created folder: {folder_name}")
            except OSError as e:
                print(f"Error creating folder {folder_name}: {e}")
        else:
            print(f"Folder already exists: {folder_name}")

if __name__ == "__main__":
    create_missing_folders()
