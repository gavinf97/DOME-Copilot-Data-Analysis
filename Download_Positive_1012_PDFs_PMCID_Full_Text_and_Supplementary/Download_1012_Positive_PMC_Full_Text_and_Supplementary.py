#!/usr/bin/env python3
"""
Download_1012_Positive_PMC_Full_Text_and_Supplementary.py

Converted from: Download_1012_Positive_PMC_Full_Text_and_Supplementary.ipynb
Downloads PDF and Supplementary files for a positive set of entries (1012).
Includes robustness checks, automated remediation, and visualization.

Usage:
    python Download_1012_Positive_PMC_Full_Text_and_Supplementary.py [options]

Options:
    --automated         Run non-interactively.
    --force-download    Force re-download of existing files.
    --input-file        Path to input TSV (default: positive_entries_pmid_pmcid_filtered.tsv)
"""

import os
import sys
import argparse
import pandas as pd
import requests
import time
import glob
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import tarfile
import shutil
import re
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.getcwd() # Script assumes it runs inside its specific folder
PDF_OUTPUT_FOLDER = 'Positive_PMC_PDFs'
SUPP_OUTPUT_FOLDER = 'Positive_PMC_Supplementary'
TSV_OUTPUT_FOLDER = 'Positive_PMC_TSV_Files'

def setup_dirs():
    for f in [PDF_OUTPUT_FOLDER, SUPP_OUTPUT_FOLDER, TSV_OUTPUT_FOLDER]:
        os.makedirs(f, exist_ok=True)
    print(f"Directories checked: {PDF_OUTPUT_FOLDER}, {SUPP_OUTPUT_FOLDER}, {TSV_OUTPUT_FOLDER}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download Positive PMC Full Text and Supplementary")
    parser.add_argument("--automated", action="store_true", help="Run without user prompts")
    parser.add_argument("--force-download", action="store_true", help="Re-download all files")
    parser.add_argument("--input-file", type=str, default=None, help="Input TSV file path") 
    return parser.parse_args()

def find_input_file(user_path):
    candidates = [
        user_path,
        "positive_entries_pmid_pmcid_filtered.tsv",
        "../positive_entries_pmid_pmcid_filtered.tsv",
        os.path.join(BASE_DIR, "positive_entries_pmid_pmcid_filtered.tsv")
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None

# ==============================================================================
# FUNCTIONS (Adapted from Notebook)
# ==============================================================================

def download_pdfs(df, output_folder, force=False):
    """
    Download PDFs from Europe PMC.
    """
    pmcids = df['mapped_pmcid'].dropna().unique()
    
    to_download = []
    for pmcid in pmcids:
        path = os.path.join(output_folder, f"{pmcid}_main.pdf")
        if not os.path.exists(path) or force:
            to_download.append(pmcid)

    print(f"\nNeed to download {len(to_download)} PDFs out of {len(pmcids)} entries.")
    
    success = 0
    fail = 0
    
    for idx, pmcid in enumerate(to_download, 1):
        print(f"[{idx}/{len(to_download)}] Processing {pmcid}...")
        
        # Method 1: Metadata
        downloaded = False
        try:
            meta_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:{pmcid}&resultType=core&format=json"
            r = requests.get(meta_url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get('hitCount', 0) > 0:
                    res = data['resultList']['result'][0]
                    if 'fullTextUrlList' in res:
                        for u in res['fullTextUrlList']['fullTextUrl']:
                            if u.get('documentStyle') == 'pdf' or u.get('availabilityCode') == 'OA':
                                pdf_url = u.get('url')
                                if pdf_url and '.pdf' in pdf_url.lower():
                                    pr = requests.get(pdf_url, timeout=30, allow_redirects=True)
                                    if pr.status_code == 200 and pr.headers.get('Content-Type','').startswith('application/pdf'):
                                        with open(os.path.join(output_folder, f"{pmcid}_main.pdf"), 'wb') as f:
                                            f.write(pr.content)
                                        print("  ✓ Downloaded via metadata link")
                                        success += 1
                                        downloaded = True
                                        break
        except Exception as e:
            pass

        # Method 2: OA Render
        if not downloaded:
            url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            try:
                r = requests.get(url, timeout=30, allow_redirects=True)
                if r.status_code == 200 and r.content[:4] == b'%PDF':
                    with open(os.path.join(output_folder, f"{pmcid}_main.pdf"), 'wb') as f:
                        f.write(r.content)
                    print("  ✓ Downloaded via OA service")
                    success += 1
                else:
                    print("  ✗ Not available via OA")
                    fail += 1
            except:
                print("  ✗ Request failed")
                fail += 1
        
        time.sleep(0.5)
        
    print(f"\nPDF Download Summary: Success={success}, Fail={fail}")

def download_supplementary_files(df, output_folder, force=False):
    """
    Download supplementary files using NCBI OA API.
    """
    pmcids = df['mapped_pmcid'].dropna().unique()
    
    to_download = []
    for pmcid in pmcids:
        p_dir = os.path.join(output_folder, pmcid)
        if (not os.path.exists(p_dir) or not os.listdir(p_dir)) or force:
            to_download.append(pmcid)
            
    print(f"\nNeed to download supplementary for {len(to_download)} entries.")
    
    success = 0
    fail = 0
    
    KEEP_EXTENSIONS = {'.pdf'}
    
    for idx, pmcid in enumerate(to_download, 1):
        print(f"[{idx}/{len(to_download)}] Processing {pmcid}...")
        p_dir = os.path.join(output_folder, pmcid)
        os.makedirs(p_dir, exist_ok=True)
        
        try:
            api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            with urllib.request.urlopen(api_url, timeout=30) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            link = root.find(".//link[@format='tgz']")
            
            if link is not None:
                href = link.get("href")
                if href.startswith("ftp://"): href = href.replace("ftp://", "https://")
                
                print(f"  Found package: {href}")
                tar_path = os.path.join(output_folder, f"{pmcid}.tar.gz")
                urllib.request.urlretrieve(href, tar_path)
                
                print("  Extracting PDFs...")
                extracted = 0
                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            _, ext = os.path.splitext(member.name.lower())
                            if ext in KEEP_EXTENSIONS:
                                member.name = os.path.basename(member.name)
                                tar.extract(member, path=p_dir)
                                extracted += 1
                                print(f"    Extracted: {member.name}")
                
                os.remove(tar_path)
                
                if extracted > 0:
                    success += 1
                else:
                    print("  No PDFs in package.")
                    if not os.listdir(p_dir): os.rmdir(p_dir)
                    fail += 1
            else:
                print("  No OA package available.")
                if not os.listdir(p_dir): os.rmdir(p_dir)
                fail += 1
                
        except Exception as e:
            print(f"  Error: {e}")
            fail += 1
        
        time.sleep(0.5)

    print(f"\nSupplementary Download Summary: Success={success}, Fail={fail}")

def clean_and_organize(pdf_folder, supp_folder):
    print("\n" + "="*60)
    print("CLEANING AND ORGANIZING")
    print("="*60)
    
    stats = {'prefixes':0, 'main_moved':0, 'exact':0, 'irrelevant':0}
    
    REMOVE_KWS = [r'review', r'comment', r'response', r'revision', r'original', r'author', r'letter', r'correction']
    
    # 0. Clean Prefixes
    if os.path.exists(supp_folder):
        for root, _, files in os.walk(supp_folder):
            for file in files:
                if 'potential_duplicate_' in file:
                    path = os.path.join(root, file)
                    new_n = file.replace('potential_duplicate_', '')
                    new_p = os.path.join(root, new_n)
                    if os.path.exists(new_p):
                        os.remove(path)
                    else:
                        os.rename(path, new_p)
                    stats['prefixes'] += 1

    # 1. Integrate Main PDFs
    if os.path.exists(pdf_folder):
        for f in os.listdir(pdf_folder):
            if f.endswith('_main.pdf'):
                pmcid = f.replace('_main.pdf', '')
                src = os.path.join(pdf_folder, f)
                dst_dir = os.path.join(supp_folder, pmcid)
                dst = os.path.join(dst_dir, f)
                
                os.makedirs(dst_dir, exist_ok=True)
                
                # Check exact sizedupes
                src_sz = os.path.getsize(src)
                existing = [x for x in os.listdir(dst_dir) if x.endswith('.pdf') and x != f]
                for e in existing:
                    ep = os.path.join(dst_dir, e)
                    if os.path.getsize(ep) == src_sz:
                        os.remove(ep)
                        stats['exact'] += 1
                
                shutil.copy2(src, dst)
                stats['main_moved'] += 1
                
    # 2. Clean Irrelevant
    if os.path.exists(supp_folder):
        for root, _, files in os.walk(supp_folder):
            for file in files:
                if file.endswith('_main.pdf'): continue
                name_l = file.lower()
                for kw in REMOVE_KWS:
                    if re.search(kw, name_l):
                        try:
                            os.remove(os.path.join(root, file))
                            stats['irrelevant'] += 1
                        except: pass
                        break
                        
    print(f"Cleanup Stats: {stats}")

def analyze_and_visualize(df, tsv_file):
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    # Reload TSV to get latest
    df = pd.read_csv(tsv_file, sep='\t')
    
    total = len(df)
    with_pmcid = df['mapped_pmcid'].notna().sum()
    pdfs = (df['pdf_downloadable'] == 'yes').sum()
    with_supp = (df['supplementary_file_count'] > 0).sum()
    
    # Calculate Success
    # Note: df may not have 'article_title' in this simplified input unless we fetch it.
    # The original notebook fetched title/abstract in Step 2.
    # Our input file only has PMID/PMCID. We will skip title enrichment for now as it wasn't explicitly Step 7 in previous turn.
    # Actually, notebook Step 2 loads CSVs with Title/Abstract.
    # Since we are using a simplified TSV input (PMID/PMCID), we lose title/abstract unless valid.
    # We will assume partial success based on what we have.
    
    print(f"Total: {total}")
    print(f"With PMCID: {with_pmcid}")
    print(f"PDFs: {pdfs}")
    print(f"With Supp: {with_supp}")
    
    # Plotting
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        plot_path = os.path.join(TSV_OUTPUT_FOLDER, f'Positive_Entries_Analysis_{current_date}.png')
        
        plt.figure(figsize=(12, 6))
        
        # Funnel
        plt.subplot(1, 2, 1)
        stages = ['Total', 'With PMCID', 'PDF Downloaded', 'With Supplementary']
        vals = [total, with_pmcid, pdfs, with_supp]
        bars = plt.bar(stages, vals, color=['gray', 'blue', 'green', 'orange'])
        plt.title('Retrieval Pipeline')
        plt.xticks(rotation=15)
        for b in bars:
            plt.text(b.get_x() + b.get_width()/2, b.get_height(), str(int(b.get_height())), ha='center', va='bottom')
            
        # Pie
        plt.subplot(1, 2, 2)
        plt.pie([pdfs, with_pmcid-pdfs], labels=['PDF Available', 'Missing'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        plt.title('PDF Success Rate (of mapped PMCIDs)')
        
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved chart: {plot_path}")
        
    except Exception as e:
        print(f"Plotting Error: {e}")

def verify_main_pdf_integration():
    print("\n" + "="*60)
    print("VERIFYING MAIN PDF INTEGRATION")
    print("="*60)
    
    if not os.path.exists(SUPP_OUTPUT_FOLDER): return
    
    missing = 0
    total = 0
    
    for pmcid in os.listdir(SUPP_OUTPUT_FOLDER):
        d = os.path.join(SUPP_OUTPUT_FOLDER, pmcid)
        if os.path.isdir(d):
            total += 1
            has_main = any(f.endswith('_main.pdf') for f in os.listdir(d))
            if not has_main:
                missing += 1
                
    print(f"Total Supp Folders: {total}")
    print(f"Folders Missing Main PDF: {missing}")
    
    if missing > 0:
        # Cross check
        restored = 0
        for pmcid in os.listdir(SUPP_OUTPUT_FOLDER):
             d = os.path.join(SUPP_OUTPUT_FOLDER, pmcid)
             if os.path.isdir(d):
                 has_main = any(f.endswith('_main.pdf') for f in os.listdir(d))
                 if not has_main:
                     src = os.path.join(PDF_OUTPUT_FOLDER, f"{pmcid}_main.pdf")
                     if os.path.exists(src):
                         shutil.copy2(src, os.path.join(d, f"{pmcid}_main.pdf"))
                         restored += 1
        if restored > 0:
            print(f"Restored {restored} missing Main PDFs from source folder.")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    args = parse_arguments()
    setup_dirs()
    
    # 1. Load Data
    infile = find_input_file(args.input_file)
    if not infile:
        print("Error: Input file 'positive_entries_pmid_pmcid_filtered.tsv' not found.")
        sys.exit(1)
        
    print(f"Loading {infile}...")
    df = pd.read_csv(infile, sep='\t')
    
    # Normalize Columns
    if 'PMCID' in df.columns and 'mapped_pmcid' not in df.columns:
        df['mapped_pmcid'] = df['PMCID']
    if 'PMID' in df.columns and 'mapped_pmid' not in df.columns:
        df['mapped_pmid'] = df['PMID']
        
    # Mock other cols for strict compatibility if needed, though simpler is better
    if 'publication_doi' not in df.columns: df['publication_doi'] = None
    if 'article_title' not in df.columns: df['article_title'] = None
    if 'article_abstract' not in df.columns: df['article_abstract'] = None

    status_file = os.path.join(TSV_OUTPUT_FOLDER, 'positive_entries_status.tsv')
    df.to_csv(status_file, sep='\t', index=False)
    
    # 2. Downloads
    download_pdfs(df, PDF_OUTPUT_FOLDER, args.force_download)
    download_supplementary_files(df, SUPP_OUTPUT_FOLDER, args.force_download)
    
    # Update TSV Status (Pre-cleanup)
    # Actually logic in clean_and_organize relies on file existence, so we can update status after cleanup often
    
    # 3. Clean
    clean_and_organize(PDF_OUTPUT_FOLDER, SUPP_OUTPUT_FOLDER)
    
    # 4. Verification
    verify_main_pdf_integration()
    
    # 5. Final Status Update
    print("Updating Final Status TSV...")
    # PDF Status
    df['pdf_downloadable'] = df['mapped_pmcid'].apply(
        lambda x: 'yes' if pd.notna(x) and os.path.exists(os.path.join(PDF_OUTPUT_FOLDER, f"{x}_main.pdf")) else 'no'
    )
    # Supp Status
    supp_status = []
    supp_counts = []
    
    for pmcid in df['mapped_pmcid']:
        if pd.notna(pmcid):
            d = os.path.join(SUPP_OUTPUT_FOLDER, pmcid)
            if os.path.exists(d):
                fls = [f for f in os.listdir(d) if not f.startswith('.')]
                supp_status.append('yes' if fls else 'no')
                supp_counts.append(len(fls))
            else:
                supp_status.append('no')
                supp_counts.append(0)
        else:
            supp_status.append('no')
            supp_counts.append(0)
            
    df['supplementary_downloadable'] = supp_status
    df['supplementary_file_count'] = supp_counts
    
    df.to_csv(status_file, sep='\t', index=False)
    
    # 6. Report
    analyze_and_visualize(df, status_file)
    
    print("\nDone! Check Positive_PMC_TSV_Files for reports.")

if __name__ == "__main__":
    main()
