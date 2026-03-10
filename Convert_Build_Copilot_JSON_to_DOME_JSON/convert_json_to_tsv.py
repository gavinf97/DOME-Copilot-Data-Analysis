import json
import os
import pandas as pd
import glob
from pathlib import Path

def process_jsons_to_tsv(input_dir, template_tsv_path, output_tsv_path, qc_failed_tsv_path):
    # Read the template to get the exact columns
    template_df = pd.read_csv(template_tsv_path, sep="\t")
    columns = template_df.columns.tolist()
    if 'isAiGenerated' not in columns:
        columns.append('isAiGenerated')

    all_data = []
    qc_failed_data = []

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON from {file_path}")
                continue
        
        row_data = {col: "" for col in columns}
        row_data['isAiGenerated'] = 'TRUE'
        row_data['user'] = '665a01aa7089c469b4646267'
        
        is_qc_failed = False
        
        for key, value in data.items():
            # Check if this field indicates it's not a relevant publication
            # Ignore publication fields for the QC check
            if not key.startswith("publication/") and isinstance(value, str):
                if "Not a relevant AI or machine learning publication" in value:
                    is_qc_failed = True

            # Map JSON key format (e.g. dataset/availability) to TSV column format (e.g. dataset.availability)
            tsv_col = key.replace("/", ".")
            
            if tsv_col == "publication.tags":
                # Handle tags up to 7
                tags = []
                if isinstance(value, list):
                    tags = value
                elif isinstance(value, str):
                    try:
                        # Sometimes tags are provided as a JSON string list
                        import ast
                        parsed_tags = ast.literal_eval(value)
                        if isinstance(parsed_tags, list):
                            tags = parsed_tags
                        else:
                            # It might be a Markdown list like "* Tag 1\n* Tag 2"
                            if '\n' in value and '*' in value:
                                tags = [t.strip().lstrip("*").strip() for t in value.split("\n") if t.strip()]
                            else:
                                tags = [t.strip() for t in value.split(",")]
                    except Exception:
                        if '\n' in value and '*' in value:
                            tags = [t.strip().lstrip("*").strip() for t in value.split("\n") if t.strip()]
                        else:
                            tags = [t.strip() for t in value.split(",")]
                    
                for i in range(min(7, len(tags))):
                    tag_col = f"publication.tags[{i}]"
                    if tag_col in columns:
                        row_data[tag_col] = tags[i]
            elif tsv_col in columns:
                if isinstance(value, (list, dict)):
                    row_data[tsv_col] = json.dumps(value)
                else:
                    row_data[tsv_col] = str(value)
        
        if is_qc_failed:
            qc_failed_data.append(row_data)
        else:
            all_data.append(row_data)

    df_valid = pd.DataFrame(all_data, columns=columns)
    df_qc_failed = pd.DataFrame(qc_failed_data, columns=columns)

    df_valid.to_csv(output_tsv_path, sep="\t", index=False)
    df_qc_failed.to_csv(qc_failed_tsv_path, sep="\t", index=False)

    print(f"Total files processed: {len(json_files)}")
    print(f"Successfully mapped entries (Valid): {len(all_data)}")
    print(f"QC Failed entries (Not rel. AI/ML pub): {len(qc_failed_data)}")
    print(f"Saved Valid TSV to: {output_tsv_path}")
    print(f"Saved QC Failed TSV to: {qc_failed_tsv_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    
    input_dir = base_dir / "Copilot_Processed_Datasets_JSON" / "Copilot_1012_v2_Pos_Processed_2026-03-02_Updated_Metadata"
    template_tsv_path = base_dir / "Convert_Build_Copilot_JSON_to_DOME_JSON" / "TSV_Fields_DOME_MongoDB_20260310 .tsv"
    
    output_dir = base_dir / "Convert_Build_Copilot_JSON_to_DOME_JSON"
    output_tsv_path = output_dir / "Pos_1012_Mapped_Output.tsv"
    qc_failed_tsv_path = output_dir / "Pos_1012_QC_Failed_Output.tsv"
    
    process_jsons_to_tsv(input_dir, template_tsv_path, output_tsv_path, qc_failed_tsv_path)
