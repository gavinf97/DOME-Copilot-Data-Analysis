Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary/DOME_Registry_PMC_PDFs#!/usr/bin/env python3
"""
Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary.py

Converted from Jupyter Notebook.
Downloads DOME Registry contents, retrieves full text papers & supplementary files,
and provides metadata analysis.

Usage:
    python Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary.py [options]

Options:
    --automated         Run all steps automatically without user prompts.
    --skip-manual       Skip steps that might strictly require manual intervention (mostly remediation checks).
    --force-download    Force re-download of files even if they already exist.
    --help              Show this help message.
"""

import os
import sys
import json
import csv
import re
import time
import glob
import shutil
import argparse
import tarfile
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_DIR = os.getcwd() # Script runs in its own folder, so cwd is fine
JSON_FOLDER = "DOME_Registry_JSON_Files"
TSV_FOLDER = "DOME_Registry_TSV_Files"
PDF_FOLDER = "DOME_Registry_PMC_PDFs"
SUPP_FOLDER = "DOME_Registry_PMC_Supplementary"
REMEDIATION_FOLDER = "DOME_Registry_Remediation"

# Ensure directories exist
for folder in [JSON_FOLDER, TSV_FOLDER, PDF_FOLDER, SUPP_FOLDER, REMEDIATION_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def setup_args():
    parser = argparse.ArgumentParser(description="DOME Registry Downloader and Analyzer")
    parser.add_argument("--automated", action="store_true", help="Run in fully automated mode")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of existing files")
    parser.add_argument("--skip-manual", action="store_true", help="Skip manual remediation checks")
    return parser.parse_args()

# ==============================================================================
# 1. Download Registry Data
# ==============================================================================
def step_1_download_registry(args):
    print("\n" + "="*60)
    print("STEP 1: Downloading DOME Registry Contents")
    print("="*60)
    
    url = "https://registry.dome-ml.org/api/review?skip=0&limit=1000&text=%20&public=true&sort=publication.year&asc=true"
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_name = f"DOME_Registry_Contents_{current_date}.json"
    json_file_path = os.path.join(JSON_FOLDER, file_name)

    if os.path.exists(json_file_path) and not args.force_download:
        print(f"File already exists: {json_file_path}")
        print("Skipping download. Use --force-download to overwrite.")
        return json_file_path

    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, headers={'accept': '*/*'})
        if response.status_code == 200:
            entries_count = len(response.json())
            print(f"Entries returned: {entries_count}")
            with open(json_file_path, 'w', encoding='utf-8') as file:
                file.write(response.text)
            print(f"Saved to: {json_file_path}")
            return json_file_path
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading registry: {e}")
        return None

# ==============================================================================
# 2. Flatten JSON and Convert to TSV
# ==============================================================================
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def step_2_flatten_and_convert(json_file_path):
    print("\n" + "="*60)
    print("STEP 2: Flattening JSON and Converting to TSV")
    print("="*60)
    
    if not json_file_path or not os.path.exists(json_file_path):
        print("Input JSON file invalid or missing.")
        return None

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

    if not data:
        print("No data to process.")
        return None

    # Flatten
    flattened_data = [flatten_json(entry) for entry in data]
    file_name = os.path.basename(json_file_path)
    flattened_file_name = "flattened_" + file_name
    flattened_json_path = os.path.join(JSON_FOLDER, flattened_file_name)
    
    with open(flattened_json_path, 'w', encoding='utf-8') as file:
        json.dump(flattened_data, file, indent=4)
    print(f"Flattened JSON saved to: {flattened_json_path}")

    # Convert to TSV
    tsv_file_name = flattened_file_name.replace('.json', '.tsv')
    tsv_file_path = os.path.join(TSV_FOLDER, tsv_file_name)
    
    try:
        headers = set()
        for entry in flattened_data:
            headers.update(entry.keys())
        headers = list(headers)
        
        with open(tsv_file_path, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=headers, delimiter='\t')
            writer.writeheader()
            for entry in flattened_data:
                writer.writerow(entry)
        print(f"TSV saved to: {tsv_file_path}")
        return tsv_file_path
    except Exception as e:
        print(f"Error writing TSV: {e}")
        return None

# ==============================================================================
# 3. Format/Reorder TSV
# ==============================================================================
def step_3_format_tsv(tsv_file_path):
    print("\n" + "="*60)
    print("STEP 3: Formatting TSV Columns")
    print("="*60)
    
    if not tsv_file_path or not os.path.exists(tsv_file_path):
        return None

    try:
        df = pd.read_csv(tsv_file_path, sep='\t')
        
        # Column prefixes
        prefixes = {
            'publication_': [],
            'publication_tags_': [],
            'matches_data': [],
            'matches_optimization': [],
            'matches_model': [],
            'matches_evaluation': []
        }
        
        # Categorize columns
        other_columns = []
        for col in df.columns:
            matched = False
            if col.startswith('matches_data'):
                prefixes['matches_data'].append(col)
                matched = True
            elif col.startswith('matches_optimization'):
                prefixes['matches_optimization'].append(col)
                matched = True
            elif col.startswith('matches_model'):
                prefixes['matches_model'].append(col)
                matched = True
            elif col.startswith('matches_evaluation'):
                prefixes['matches_evaluation'].append(col)
                matched = True
            elif col.startswith('publication_tags_'):
                prefixes['publication_tags_'].append(col)
                matched = True
            elif col.startswith('publication_'):
                prefixes['publication_'].append(col)
                matched = True
            
            if not matched:
                other_columns.append(col)

        # Sort tags numerically
        prefixes['publication_tags_'] = sorted(
            prefixes['publication_tags_'], 
            key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0
        )

        reordered = (other_columns + 
                     prefixes['publication_'] + 
                     prefixes['publication_tags_'] + 
                     prefixes['matches_data'] + 
                     prefixes['matches_optimization'] + 
                     prefixes['matches_model'] + 
                     prefixes['matches_evaluation'])
        
        # Filter reordered to include only columns existing in df
        reordered = [col for col in reordered if col in df.columns]

        df = df[reordered]
        
        # Set index to shortid if exists
        if 'shortid' in df.columns:
            df = df.set_index('shortid')
            df.to_csv(tsv_file_path, sep='\t', index=True, encoding='utf-8')
        else:
            df.to_csv(tsv_file_path, sep='\t', index=False, encoding='utf-8')
            
        print(f"Reordered TSV saved to: {tsv_file_path}")
        return tsv_file_path
    
    except Exception as e:
        print(f"Error reordering TSV: {e}")
        return tsv_file_path # Return original if fail

# ==============================================================================
# Helper
# ==============================================================================
def clean_doi(doi_string):
    if pd.isna(doi_string): return None
    s = str(doi_string).strip()
    # Regex cleanup
    s = re.sub(r'^https?://(dx\.|www\.)?doi\.org/', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^doi:\s*', '', s, flags=re.IGNORECASE)
    s = s.strip()
    if not s.startswith('10.'): return None
    return s

# ==============================================================================
# 4. Map DOIs to ID
# ==============================================================================
def step_4_map_dois(tsv_file_path, args):
    print("\n" + "="*60)
    print("STEP 4: Mapping DOIs to PMCIDs")
    print("="*60)

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_tsv = os.path.join(TSV_FOLDER, f'PMCIDs_DOME_Registry_Contents_{current_date}.tsv')

    if os.path.exists(output_tsv) and not args.force_download:
        print(f"Output file exists: {output_tsv}")
        df = pd.read_csv(output_tsv, sep='\t')
        if all(col in df.columns for col in ['mapped_pmcid', 'mapped_europepmc_id', 'mapped_pmid']):
            print("Mapping columns exist. Skipping API calls.")
            return output_tsv

    print("Reading input TSV...")
    df = pd.read_csv(tsv_file_path, sep='\t')
    raw_dois = df['publication_doi'].dropna().unique()
    
    cleaned_dois = []
    for d in raw_dois:
        c = clean_doi(d)
        if c: cleaned_dois.append(c)
    
    print(f"Found {len(cleaned_dois)} valid DOIs.")
    
    # Mapping Logic
    id_mapping = {}
    batch_size = 20 # Can increase batch size for efficiency
    
    print("Querying NCBI service...")
    for i in range(0, len(cleaned_dois), batch_size):
        batch = cleaned_dois[i:i + batch_size]
        doi_str = ','.join(batch)
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=dome_copilot&email=dome@example.com&ids={doi_str}&format=json"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for rec in data.get('records', []):
                    doi = rec.get('doi')
                    id_mapping[doi] = {
                        'pmcid': rec.get('pmcid'),
                        'pmid': rec.get('pmid'),
                        'europepmc_id': rec.get('pmcid') if rec.get('pmcid') else (f"MED/{rec.get('pmid')}" if rec.get('pmid') else None)
                    }
            time.sleep(0.34) # Rate limit inside loop
        except Exception as e:
            print(f"Batch error: {e}")
        
    

    # Apply mapping
    print("Applying mappings to DataFrame...")
    df['mapped_pmcid'] = df['publication_doi'].apply(lambda x: id_mapping.get(clean_doi(x), {}).get('pmcid'))
    df['mapped_europepmc_id'] = df['publication_doi'].apply(lambda x: id_mapping.get(clean_doi(x), {}).get('europepmc_id'))
    df['mapped_pmid'] = df['publication_doi'].apply(lambda x: id_mapping.get(clean_doi(x), {}).get('pmid'))

    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Mapped TSV saved to: {output_tsv}")
    return output_tsv

# ==============================================================================
# 5. Remediation & Simplification
# ==============================================================================
def step_4_5_remediation(output_tsv, args):
    if args.skip_manual:
        print("Skipping remediation check (flag --skip-manual used).")
        return output_tsv

    print("\n" + "="*60)
    print("STEP 4.5: Checking for Manual Remediation Files")
    print("="*60)
    
    pattern = os.path.join(TSV_FOLDER, 'remediated_Failed_DOI_Mappings_*.tsv')
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    if not files:
        print("No remediation files found.")
        return output_tsv
        
    df = pd.read_csv(output_tsv, sep='\t')
    
    updates = 0
    for raf in files:
        print(f"Processing remediation file: {os.path.basename(raf)}")
        rdf = pd.read_csv(raf, sep='\t')
        
        # Filter RESOLVED
        valid = rdf[(rdf['Remediation_Status'] == 'RESOLVED')]
        
        id_col = 'shortid' if 'shortid' in df.columns else '_id'
        r_id_col = 'shortid' if 'shortid' in valid.columns else ('_id' if '_id' in valid.columns else None)
        
        if not r_id_col: continue

        for _, row in valid.iterrows():
            eid = row[r_id_col]
            man_pmcid = row.get('Manual_PMCID')
            man_pmid = row.get('Manual_PMID')
            
            mask = df[id_col] == eid
            if mask.any():
                if pd.notna(man_pmcid):
                    df.loc[mask, 'mapped_pmcid'] = str(man_pmcid).strip()
                    df.loc[mask, 'mapped_europepmc_id'] = str(man_pmcid).strip()
                    updates += 1
                if pd.notna(man_pmid):
                    df.loc[mask, 'mapped_pmid'] = str(man_pmid).strip()
    
    if updates > 0:
        df.to_csv(output_tsv, sep='\t', index=False)
        print(f"Applied {updates} remediation updates.")
    
    return output_tsv

def step_4_6_simplify(output_tsv):
    print("\n" + "="*60)
    print("STEP 4.6: Creating Simplified Mapping File")
    print("="*60)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    simple_file = os.path.join(TSV_FOLDER, f'DOME_ID_PMID_PMCID_Mapping_{current_date}.tsv')
    
    df = pd.read_csv(output_tsv, sep='\t')
    id_col = 'shortid' if 'shortid' in df.columns else '_id'
    
    if id_col in df.columns:
        simple = df[[id_col, 'mapped_pmid', 'mapped_pmcid']].copy()
        simple.columns = ['domeshort_id', 'PMID', 'PMCID']
        simple.to_csv(simple_file, sep='\t', index=False)
        print(f"Simplified mapping saved to: {simple_file}")

# ==============================================================================
# 6. PDF Download
# ==============================================================================
def step_5_download_pdfs(output_tsv, args):
    """
    Download PDFs from Europe PMC OA service.
    Checks if PDF already exists before downloading.
    """
    print("\n" + "="*60)
    print("STEP 5: Downloading PDFs")
    print("="*60)
    
    df = pd.read_csv(output_tsv, sep='\t')
    pmcids = df['mapped_pmcid'].dropna().unique()
    
    to_download = []
    for pmcid in pmcids:
        # Check if already exists in target folder
        if not os.path.exists(os.path.join(PDF_FOLDER, f"{pmcid}_main.pdf")):
            to_download.append(pmcid)
            
    print(f"Need to download {len(to_download)} PDFs out of {len(pmcids)} mapped PMCIDs.")
    
    success_count = 0
    
    for idx, pmcid in enumerate(to_download):
        print(f"[{idx+1}/{len(to_download)}] Processing: {pmcid}")
        # Simplified Logic for brevity - EuroPMC OA
        url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        try:
            r = requests.get(url, timeout=30, allow_redirects=True)
            if r.status_code == 200 and r.content[:4] == b'%PDF':
                with open(os.path.join(PDF_FOLDER, f"{pmcid}_main.pdf"), 'wb') as f:
                    f.write(r.content)
                print("  [SUCCESS] Downloaded via OA service")
                success_count += 1
            else:
                print("  [FAIL] Not available via OA service or redirect fail")
        except:
            print("  [ERROR] Request failed")
        time.sleep(0.5)

    # Update TSV with status based on file existence
    df['pdf_downloadable'] = df['mapped_pmcid'].apply(
        lambda x: 'yes' if pd.notna(x) and os.path.exists(os.path.join(PDF_FOLDER, f"{x}_main.pdf")) else 'no'
    )
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Updated TSV with PDF status (Total new downloads: {success_count}).")
    return output_tsv

# ==============================================================================
# 7. Supplementary Files
# ==============================================================================
def step_6_download_supp(output_tsv, args):
    """
    Download Supplementary Files using NCBI OA API.
    Identifies FTP link via XML API, downloads tar.gz, extracts PDFs.
    """
    print("\n" + "="*60)
    print("STEP 6: Downloading Supplementary Files")
    print("="*60)
    
    df = pd.read_csv(output_tsv, sep='\t')
    pmcids = df['mapped_pmcid'].dropna().unique()
    
    # Pre-check how many need download
    to_download = []
    for pmcid in pmcids:
        pmcid_dir = os.path.join(SUPP_FOLDER, pmcid)
        # Condition to skip: Folder exists AND has contents AND (not force download)
        if os.path.exists(pmcid_dir) and os.listdir(pmcid_dir) and not args.force_download:
            continue
        to_download.append(pmcid)

    print(f"Need to process supplementary files for {len(to_download)} entries.")

    for idx, pmcid in enumerate(to_download):
        pmcid_dir = os.path.join(SUPP_FOLDER, pmcid)
        print(f"[{idx+1}/{len(to_download)}] Processing Supplementary for {pmcid}...")
        
        try:
            # 1. Get OA details from NCBI API
            api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            
            # Use urllib for XML fetching
            with urllib.request.urlopen(api_url, timeout=30) as response:
                xml_data = response.read()

            # 2. Parse XML to find FTP link
            try:
                root = ET.fromstring(xml_data)
                link = root.find(".//link[@format='tgz']")
            except ET.ParseError:
                print("  [ERROR] Could not parse XML response from NCBI.")
                link = None
            
            if link is not None:
                href = link.get("href")
                # NCBI moved to HTTPS recently, ensure protocol
                if href.startswith("ftp://"):
                    href = href.replace("ftp://", "https://")
                
                print(f"  Found package URL: {href}")
                
                # 3. Download tar.gz
                tar_path = os.path.join(SUPP_FOLDER, f"{pmcid}.tar.gz")
                
                # Create parent dir if needed (SUPP_FOLDER exists, but just in case)
                os.makedirs(SUPP_FOLDER, exist_ok=True)
                
                print("  Downloading package...")
                urllib.request.urlretrieve(href, tar_path)
                
                # 4. Extract Files
                os.makedirs(pmcid_dir, exist_ok=True)
                print("  Extracting PDFs...")
                
                extracted_count = 0
                try:
                    with tarfile.open(tar_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile():
                                # Check extension
                                if member.name.lower().endswith('.pdf'):
                                    # Strip folder structure from tar member name
                                    member.name = os.path.basename(member.name)
                                    tar.extract(member, path=pmcid_dir)
                                    extracted_count += 1
                except tarfile.ReadError:
                     print("  [ERROR] Tarfile corrupted or invalid.")

                print(f"  Extracted {extracted_count} PDF(s).")
                
                # 5. Cleanup
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                
                # Remove empty folders if no PDFs found
                if not os.listdir(pmcid_dir):
                    os.rmdir(pmcid_dir)
                    print("  No PDFs found in package (folder removed).")
            else:
                print("  No OA package available (link not found in XML).")

        except urllib.error.HTTPError as e:
            print(f"  HTTP Error: {e.code}")
        except Exception as e:
            print(f"  Error: {e}")
            
        # Rate limit
        time.sleep(0.5)

    # Update TSV
    print("Updating supplementary status in TSV...")
    status = []
    counts = []
    
    for pmcid in df['mapped_pmcid']:
        if pd.notna(pmcid):
            p_dir = os.path.join(SUPP_FOLDER, pmcid)
            if os.path.exists(p_dir):
                files = os.listdir(p_dir)
                status.append('yes' if files else 'no')
                counts.append(len(files))
            else:
                status.append('no')
                counts.append(0)
        else:
            status.append('no')
            counts.append(0)
            
    df['supplementary_downloadable'] = status
    df['supplementary_file_count'] = counts
    df.to_csv(output_tsv, sep='\t', index=False)
    print("Updated TSV with supplementary file status.")
    return output_tsv

# ==============================================================================
# 8. Enrich TSV (Title/Abstract)
# ==============================================================================
def step_7_enrich_tsv(output_tsv):
    print("\n" + "="*60)
    print("STEP 7: Enriching TSV with Title/Abstract")
    print("="*60)
    
    df = pd.read_csv(output_tsv, sep='\t')
    if 'article_title' not in df.columns:
        df['article_title'] = None
        df['article_abstract'] = None
    
    # Check how many need enrichment
    # We only care about mapped_pmcid rows that lack title/abstract
    needs_enrichment = df[df['mapped_pmcid'].notna() & (df['article_title'].isna())]
    print(f"Found {len(needs_enrichment)} entries needing title/abstract enrichment.")

    for idx in needs_enrichment.index:
        pmcid = df.at[idx, 'mapped_pmcid']
        print(f"Enriching: {pmcid}")
        try:
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:{pmcid}&resultType=core&format=json"
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get('hitCount', 0) > 0:
                    res = data['resultList']['result'][0]
                    df.at[idx, 'article_title'] = res.get('title')
                    df.at[idx, 'article_abstract'] = res.get('abstractText')
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.3)
            
    df.to_csv(output_tsv, sep='\t', index=False)
    print("Enrichment complete.")
    return output_tsv

# ==============================================================================
# 9. Metadata Reporting & Visualization (Step 8/9 in Notebook)
# ==============================================================================
def step_8_visualize(output_tsv):
    print("\n" + "="*60)
    print("STEP 8: Generating Metadata Report and Visualizations")
    print("="*60)
    
    df = pd.read_csv(output_tsv, sep='\t')
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_png = os.path.join(TSV_FOLDER, f'DOME_Metadata_Complete_Analysis_{current_date}.png')
    
    # Calculate Metrics
    total_entries = len(df)
    # Check if publication_doi exists
    if 'publication_doi' in df:
        entries_with_doi = df['publication_doi'].notna().sum()
    else:
        entries_with_doi = 0
        
    entries_with_pmcid = df['mapped_pmcid'].notna().sum()
    
    # pdf_downloadable might be missing if no PDFs downloaded
    if 'pdf_downloadable' in df.columns:
        pdf_success = (df['pdf_downloadable'] == 'yes').sum()
    else: 
        pdf_success = 0
    
    # Full Success (PDF + Supp + Title/Abs) - assuming Supp count > 0 is needed
    if 'supplementary_file_count' in df.columns and 'article_title' in df.columns and 'pdf_downloadable' in df.columns:
        full_success = ((df['pdf_downloadable'] == 'yes') & 
                        (df['supplementary_file_count'] > 0) & 
                        (df['article_title'].notna())).sum()
    else:
        full_success = 0

    print(f"Total Entries: {total_entries}")
    print(f"With PMCID: {entries_with_pmcid}")
    print(f"PDFs Downloaded: {pdf_success}")
    print(f"Full Retrieval: {full_success}")

    # Plotting
    try:
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Pipeline Funnel
        ax1 = plt.subplot(2, 2, 1)
        stages = ['Total', 'DOI', 'PMCID', 'PDF', 'Full (w/Supp)']
        counts = [total_entries, entries_with_doi, entries_with_pmcid, pdf_success, full_success]
        ax1.bar(stages, counts, color=['gray', 'blue', 'orange', 'green', 'purple'])
        ax1.set_title('Retrieval Pipeline Funnel')
        ax1.set_ylabel('Count')
        for i, v in enumerate(counts):
            ax1.text(i, v + 5, str(v), ha='center')

        # 2. PDF Availability Pie
        ax2 = plt.subplot(2, 2, 2)
        if entries_with_pmcid > 0:
            pdf_fail = entries_with_pmcid - pdf_success
            ax2.pie([pdf_success, pdf_fail], labels=['Downloadable', 'Failed/Unavailable'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
            ax2.set_title('PDF Availability (for mapped PMCIDs)')
        
        # 3. Supp Files
        ax3 = plt.subplot(2, 2, 3)
        if 'supplementary_file_count' in df.columns:
            has_supp = (df['supplementary_file_count'] > 0).sum()
        else:
            has_supp = 0
            
        no_supp = entries_with_pmcid - has_supp
        ax3.pie([has_supp, no_supp], labels=['With Supp Files', 'No Supp Files'], autopct='%1.1f%%', colors=['#ffcc99','#99ff99'])
        ax3.set_title('Supplementary Files (for mapped PMCIDs)')

        plt.tight_layout()
        plt.savefig(output_png)
        print(f"Saved visualization to: {output_png}")
        # plt.show() # Disabled for script mode
        
    except Exception as e:
        print(f"Plotting error: {e}")

def step_9_identify_failed_mappings(output_tsv):
    print("\n" + "="*60)
    print("STEP 9: Identifying Failed Mappings for Remediation")
    print("="*60)
    
    df = pd.read_csv(output_tsv, sep='\t')
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Failed = Has DOI but no PMCID
    if 'publication_doi' in df.columns and 'mapped_pmcid' in df.columns:
        failed = df[df['publication_doi'].notna() & df['mapped_pmcid'].isna()].copy()
    else:
        print("Required columns missing for checking failed mappings.")
        return

    if len(failed) > 0:
        print(f"Found {len(failed)} failed mappings.")
        
        # Export for manual check
        failed_file = os.path.join(REMEDIATION_FOLDER, f'Failed_DOI_Mappings_{current_date}.tsv')
        
        # Add remediation columns
        failed['Remediation_Status'] = 'NEEDS_REVIEW'
        failed['Manual_PMCID'] = ''
        failed['Manual_PMID'] = ''
        
        # Select useful columns
        cols = ['shortid' if 'shortid' in failed.columns else '_id', 
                'publication_doi', 'publication_title', 
                'Remediation_Status', 'Manual_PMCID', 'Manual_PMID']
        
        # Ensure cols exist
        cols = [c for c in cols if c in failed.columns]
        
        failed[cols].to_csv(failed_file, sep='\t', index=False)
        print(f"Saved remediation file: {failed_file}")
        
    else:
        print("No failed mappings found! All DOIs mapped successfully.")

# ==============================================================================
# Main
# ==============================================================================
def main():
    args = setup_args()
    print("Starting DOME Registry Automator...")
    
    # Sequence matching Notebook
    # 1. Download Registry
    json_path = step_1_download_registry(args)
    if not json_path: return

    # 2. Flatten/Convert
    tsv_path = step_2_flatten_and_convert(json_path)
    if not tsv_path: return

    # 3. Format
    formatted_tsv = step_3_format_tsv(tsv_path)
    
    # 4. Map DOIs
    mapped_tsv = step_4_map_dois(formatted_tsv, args)
    
    # 4.5 Remediation Integration
    remediated_tsv = step_4_5_remediation(mapped_tsv, args)
    
    # 4.6 Simplify ID TSV
    step_4_6_simplify(remediated_tsv)
    
    # 5. Download PDFs
    with_pdfs_tsv = step_5_download_pdfs(remediated_tsv, args)
    
    # 6. Download Supp Files (FIXED logic)
    with_supp_tsv = step_6_download_supp(with_pdfs_tsv, args)
    
    # 7. Enrich Title/Abs
    final_tsv = step_7_enrich_tsv(with_supp_tsv)
    
    # 8. Visualize
    step_8_visualize(final_tsv)
    
    # 9. Failed Report
    step_9_identify_failed_mappings(final_tsv)
    
    print("\nDone! Final output at:", final_tsv)

if __name__ == "__main__":
    main()
