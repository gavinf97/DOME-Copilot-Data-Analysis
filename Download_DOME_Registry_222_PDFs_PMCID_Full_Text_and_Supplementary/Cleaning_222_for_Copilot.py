import os
import shutil

def main():
    # Define paths relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir_name = 'DOME_Registry_PMC_PDFs'
    dest_dir_name = 'DOME_Registry_PMC_Supplementary'
    
    source_path = os.path.join(base_dir, source_dir_name)
    dest_path = os.path.join(base_dir, dest_dir_name)

    # Check if source directory exists
    if not os.path.exists(source_path):
        print(f"Source directory not found: {source_path}")
        return

    # Check/Create destination base directory
    if not os.path.exists(dest_path):
        print(f"Destination base directory not found: {dest_path}. Creating it.")
        os.makedirs(dest_path, exist_ok=True)

    # List PDF files in source directory
    pdf_files = [f for f in os.listdir(source_path) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in source directory.")

    for filename in pdf_files:
        source_file_path = os.path.join(source_path, filename)
        
        # Extract PMCID
        # Helper to extract PMCID from filename like 'PMC12345_main.pdf' or 'PMC12345.pdf'
        if '_main' in filename:
            pmcid = filename.split('_main')[0]
        else:
            pmcid = os.path.splitext(filename)[0]
            
        target_folder = os.path.join(dest_path, pmcid)
        target_file_name = f"{pmcid}_main.pdf"
        target_file_path = os.path.join(target_folder, target_file_name)
        
        # Case 1: No matching folder
        if not os.path.exists(target_folder):
            print(f"Folder for {pmcid} does not exist. Creating and copying file.")
            os.makedirs(target_folder)
            shutil.copy2(source_file_path, target_file_path)
        
        # Case 2: Matching folder exists
        else:
            match_found = False
            source_size = os.path.getsize(source_file_path)
            
            # Check existing PDF files in target folder
            existing_pdfs = [f for f in os.listdir(target_folder) if f.lower().endswith('.pdf')]
            
            for exist_pdf in existing_pdfs:
                exist_pdf_path = os.path.join(target_folder, exist_pdf)
                exist_size = os.path.getsize(exist_pdf_path)
                
                # Check for exact size match
                if exist_size == source_size:
                    print(f"Exact size match found in {pmcid} for {exist_pdf}. Removing existing file.")
                    try:
                        os.remove(exist_pdf_path)
                        match_found = True
                    except OSError as e:
                        print(f"Error removing file {exist_pdf_path}: {e}")
            
            # Copy in the matching one from source
            # "copy in the matching one from the DOME Registry PMC PDFs folder"
            # This happens regardless of whether there was a size match or not, to ensure the correct file is present.
            try:
                shutil.copy2(source_file_path, target_file_path)
                status = "Copied (replaced)" if match_found else "Copied"
                print(f"{status} {filename} to {target_folder} as {target_file_name}")
            except OSError as e:
                 print(f"Error copying file to {target_file_path}: {e}")

if __name__ == "__main__":
    main()
