import os
import requests
import csv
import time
from collections import defaultdict

# We are no longer using JSON directories to find PMCIDs.
# Instead, we read from dataset_pmcids.csv.
CSV_SOURCE = "dataset_pmcids.csv"

# The output TSV filename per dataset
OUTPUT_TSV_MAP = {
    "DOME_Registry_222": "DOME_Registry_222_Metadata.tsv",
    "Positive_1012": "Positive_1012_Metadata.tsv",
    "Negative_1012": "Negative_1012_Metadata.tsv"
}

ALL_FIELDS = ["title", "authors", "journal", "year", "pmid", "pmcid", "doi"]

def load_pmcids_from_csv(csv_path):
    """
    Returns a dict mapping dataset_name to a list of PMCIDs.
    E.g., { 'Positive_1012': ['PMC123', 'PMC456'], ... }
    """
    dataset_to_pmcids = defaultdict(list)
    if not os.path.exists(csv_path):
        print(f"Source CSV {csv_path} not found!")
        return dataset_to_pmcids
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = row.get("pmcid", "").strip()
            ds = row.get("dataset", "").strip()
            if pmcid and ds:
                dataset_to_pmcids[ds].append(pmcid)
    return dataset_to_pmcids

def fetch_europe_pmc(pmcids):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
    results = {}
    batch_size = 50
    for i in range(0, len(pmcids), batch_size):
        batch = pmcids[i:i+batch_size]
        query = " OR ".join([f"PMCID:{pmcid}" for pmcid in batch])
        data = {"query": query, "format": "json", "resultType": "core", "pageSize": len(batch)}
        try:
            res = requests.post(url, data=data)
            res.raise_for_status()
            records = res.json().get("resultList", {}).get("result", [])
            for r in records:
                pmcid = r.get("pmcid", "")
                if pmcid:
                    results[pmcid] = {
                        "title": r.get("title", ""),
                        "authors": r.get("authorString", ""),
                        "journal": r.get("journalInfo", {}).get("journal", {}).get("title", ""),
                        "year": str(r.get("pubYear", "")),
                        "pmid": r.get("pmid", ""),
                        "pmcid": pmcid,
                        "doi": r.get("doi", "")
                    }
        except Exception as e:
            print(f"Europe PMC API error on batch: {e}")
        time.sleep(0.5)
    return results

def fetch_ncbi_pmc(pmcids):
    results = {}
    batch_size = 50
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    for i in range(0, len(pmcids), batch_size):
        batch = pmcids[i:i+batch_size]
        # Remove 'PMC' prefix for NCBI API
        ids_only = [p.replace("PMC", "") for p in batch if p.startswith("PMC")]
        if not ids_only:
            continue
        params = {"db": "pmc", "id": ",".join(ids_only), "retmode": "json"}
        try:
            res = requests.get(base_url, params=params)
            res.raise_for_status()
            data = res.json().get("result", {})
            for uid in data.get("uids", []):
                record = data.get(uid, {})
                pmcid = f"PMC{uid}"
                authors = ", ".join([a.get("name", "") for a in record.get("authors", [])])
                doi = ""
                pmid = ""
                for aid in record.get("articleids", []):
                    if aid.get("idtype") == "doi":
                        doi = aid.get("value")
                    elif aid.get("idtype") == "pmid":
                        pmid = aid.get("value")
                results[pmcid] = {
                    "title": record.get("title", ""),
                    "authors": authors,
                    "journal": record.get("fulljournalname", ""),
                    "year": record.get("pubdate", "")[:4] if record.get("pubdate") else "",
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "doi": doi
                }
        except Exception as e:
            print(f"NCBI API error on batch: {e}")
        time.sleep(0.5)
    return results

def load_existing_tsv(filepath):
    data = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                pmcid = row.get("pmcid")
                if pmcid:
                    data[pmcid] = row
    return data

def write_tsv(filepath, data_dict):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, delimiter='\t')
        writer.writeheader()
        for pmcid, record in data_dict.items():
            out = {field: record.get(field, "") for field in ALL_FIELDS}
            writer.writerow(out)

def analyze_missing(data_dict):
    missing_by_field = {f: [] for f in ALL_FIELDS}
    for pmcid, record in data_dict.items():
        for f in ALL_FIELDS:
            if not record.get(f, "").strip():
                missing_by_field[f].append(pmcid)
    return missing_by_field

def main():
    metadata_dir = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(metadata_dir, "coverage_report.txt")
    csv_path = os.path.join(metadata_dir, CSV_SOURCE)
    
    dataset_to_pmcids = load_pmcids_from_csv(csv_path)
    
    if not dataset_to_pmcids:
        print(f"No data found in {csv_path}. Please make sure it exists and has 'pmcid' and 'dataset' columns.")
        return

    with open(report_file, 'w', encoding='utf-8') as rf:
        rf.write("Metadata Coverage Report\n")
        rf.write("========================\n\n")

        for ds_name, target_pmcids in dataset_to_pmcids.items():
            print(f"Processing dataset {ds_name}...")
            output_filename = OUTPUT_TSV_MAP.get(ds_name, f"{ds_name}_Metadata.tsv")
            output_file = os.path.join(metadata_dir, output_filename)
            
            existing_data = load_existing_tsv(output_file)
            final_data = {}
            pmcids_to_fetch = []
            
            for pmcid in target_pmcids:
                if pmcid not in existing_data:
                    pmcids_to_fetch.append(pmcid)
                else:
                    record = existing_data[pmcid]
                    is_missing_fields = any(not record.get(f, "").strip() for f in ALL_FIELDS)
                    if is_missing_fields:
                        pmcids_to_fetch.append(pmcid)
                    final_data[pmcid] = record
            
            if pmcids_to_fetch:
                print(f"Fetching {len(pmcids_to_fetch)} records for {ds_name} via Europe PMC...")
                epmc_data = fetch_europe_pmc(pmcids_to_fetch)
                
                for pmcid, new_record in epmc_data.items():
                    if pmcid not in final_data:
                        final_data[pmcid] = {"pmcid": pmcid}
                    for f in ALL_FIELDS:
                        if not final_data[pmcid].get(f, "").strip() and new_record.get(f, "").strip():
                            final_data[pmcid][f] = new_record[f]

                still_missing_pmcids = set()
                for pmcid in pmcids_to_fetch:
                    record = final_data.get(pmcid, {})
                    if any(not record.get(f, "").strip() for f in ALL_FIELDS):
                        still_missing_pmcids.add(pmcid)
                
                if still_missing_pmcids:
                    print(f"Fallback to NCBI PMC API for {len(still_missing_pmcids)} items...")
                    ncbi_data = fetch_ncbi_pmc(list(still_missing_pmcids))
                    for pmcid, new_record in ncbi_data.items():
                        for f in ALL_FIELDS:
                            if not final_data.get(pmcid, {}).get(f, "").strip() and new_record.get(f, "").strip():
                                final_data[pmcid][f] = new_record[f]
            
            write_tsv(output_file, final_data)
            print(f"Updated {output_file}")
            
            rf.write(f"--- Dataset: {ds_name} ---\n")
            rf.write(f"Total entries: {len(final_data)}\n")
            missing_info = analyze_missing(final_data)
            has_missing = False
            for f, missing_pmcids in missing_info.items():
                if missing_pmcids:
                    has_missing = True
                    rf.write(f"  Missing '{f}': {len(missing_pmcids)} ({(len(missing_pmcids)/len(final_data))*100:.2f}%)\n")
                    rf.write(f"    Affected PMCIDs: {', '.join(missing_pmcids)}\n")
            if not has_missing:
                rf.write("  No missing output values for any column!\n")
            rf.write("\n")
            
    print(f"Done. Check {report_file} for details.")

if __name__ == "__main__":
    main()
