import pandas as pd
import json
import glob
import os
from datetime import datetime
import sys
import numpy as np # Import numpy for handling potential NaN values

# --- Configuration ---
# Define the paths for your data
METADATA_CSV_PATH = 'corpus/processed/metadata.csv'
PASSAGES_DIR = 'corpus/processed/passages/'
MASTER_METADATA_CSV_PATH = 'corpus/processed/master_metadata.csv'
# MASTER_METADATA_PARQUET_PATH = 'corpus/processed/master_metadata.parquet' # Optional Parquet output
# MASTER_METADATA_SQLITE_PATH = 'corpus/processed/master_metadata.sqlite' # Optional SQLite output

# Define token count thresholds for quality check
MIN_TOKENS = 10
MAX_TOKENS = 500

# Define critical document metadata fields expected after join for validation
# These should be the *new* names after renaming
CRITICAL_DOC_FIELDS = ['doc_type', 'doc_source']

# Specify the actual key name used for the unique passage ID in your JSONL files.
# Based on your example, this is 'id'.
ACTUAL_PASSAGE_ID_JSON_KEY = 'id'

# Define default value for empty 'section' fields
SECTION_DEFAULT_VALUE = 'unknown' # Or 'root'

# Define columns to drop if not needed (set to empty list [] to keep them)
COLUMNS_TO_DROP = ['char_start', 'char_end'] # Add or remove column names as needed


# Define column renaming map
# This maps the column names *after* the pandas merge/aggregation
# to the desired final column names in the master catalog.
# Note: pandas adds _x (left df) and _y (right df) suffixes on merge conflicts.
COLUMN_RENAME_MAP = {
    ACTUAL_PASSAGE_ID_JSON_KEY: 'passage_id',  # Your original JSON ID becomes passage_id
    'source_x': 'passage_source',          # 'source' from passage JSONL becomes passage_source (assuming it exists and conflicts)
    'filename': 'doc_filename',            # 'filename' from metadata.csv
    'type': 'doc_type',                    # 'type' from metadata.csv
    'source_y': 'doc_source',              # 'source' from metadata.csv becomes doc_source (assuming it exists and conflicts)
    'date': 'doc_date',                    # 'date' from metadata.csv
    'notes': 'doc_notes',                  # 'notes' from metadata.csv
    # Add other renames if necessary, e.g., 'content' -> 'passage_content'
}


# --- Implementation ---

def consolidate_metadata(metadata_csv_path, passages_dir, master_metadata_csv_path, min_tokens, max_tokens, critical_doc_fields, actual_passage_id_key, rename_map, section_default, cols_to_drop):
    """
    Consolidates document-level and passage-level metadata into a single catalog.

    Args:
        metadata_csv_path (str): Path to the document-level metadata CSV.
        passages_dir (str): Directory containing passage JSONL files (recursive scan).
        master_metadata_csv_path (str): Output path for the master metadata CSV.
        min_tokens (int): Minimum token count for quality check.
        max_tokens (int): Maximum token count for quality check (base value).
        critical_doc_fields (list): List of *final* column names for critical document metadata fields to check for missing values.
        actual_passage_id_key (str): The actual key name used for the passage ID in the JSONL files (e.g., 'id').
        rename_map (dict): Dictionary mapping columns *after* merge to desired final names.
        section_default (str): Default string value for empty/null sections.
        cols_to_drop (list): List of column names to drop from the final DataFrame.
    """
    print("--- Phase E: Metadata Consolidation Started ---")

    # 2. Extraction & Join Logic

    # Load document metadata
    print(f"Loading document metadata from {metadata_csv_path}...")
    try:
        doc_meta_df = pd.read_csv(metadata_csv_path)
        if 'doc_id' not in doc_meta_df.columns:
             print(f"Error: '{metadata_csv_path}' must contain a 'doc_id' column for joining.")
             sys.exit(1)

        # Warn about missing expected document columns if any
        expected_doc_cols_for_rename = [col for col in rename_map if col in ['filename', 'type', 'source', 'date', 'notes']]
        missing_expected_doc_cols = [col for col in expected_doc_cols_for_rename if col not in doc_meta_df.columns]
        if missing_expected_doc_cols:
            print(f"Warning: Document metadata CSV '{metadata_csv_path}' is missing expected columns for renaming: {missing_expected_doc_cols}")


        print(f"Loaded {len(doc_meta_df)} document records.")
    except FileNotFoundError:
        print(f"Error: Document metadata file not found at {metadata_csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading document metadata from {metadata_csv_path}: {e}")
        sys.exit(1)


    # Aggregate passage metadata
    print(f"\nAggregating passage metadata from {passages_dir}/**/*.jsonl...")
    passage_data_list = []
    jsonl_files = glob.glob(os.path.join(passages_dir, '**/*.jsonl'), recursive=True)
    total_jsonl_files = len(jsonl_files)
    print(f"Found {total_jsonl_files} JSONL files.")

    if total_jsonl_files == 0:
        print("Warning: No JSONL files found in the passages directory. Cannot create master catalog.")
        return

    processed_files_count = 0
    for jsonl_file in jsonl_files:
        processed_files_count += 1
        if processed_files_count % 100 == 0 or processed_files_count == total_jsonl_files:
             print(f"  Processing file {processed_files_count}/{total_jsonl_files}")

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        passage_data = json.loads(line)
                        # Add passage_file path for provenance
                        passage_data['passage_file'] = os.path.join('corpus', 'processed', os.path.relpath(jsonl_file, 'corpus/processed/'))
                        passage_data_list.append(passage_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {line_num} in {jsonl_file} from file {jsonl_file}")
                    except Exception as e:
                         print(f"Warning: Error processing line {line_num} in {jsonl_file} from file {jsonl_file}: {e}")

        except FileNotFoundError:
             print(f"Error: File not found (should not happen with glob): {jsonl_file}")
        except Exception as e:
             print(f"Error reading file {jsonl_file}: {e}")


    if not passage_data_list:
        print("No valid passage data could be aggregated from JSONL files. Cannot create master catalog.")
        return

    passage_meta_df = pd.DataFrame(passage_data_list)
    print(f"Aggregated {len(passage_meta_df)} raw passage records.")

    # --- Tweak 1: Handle Section Defaults ---
    if 'section' in passage_meta_df.columns:
        print(f"Filling empty/null 'section' values with '{section_default}'...")
        # Fill NaN values (from None in JSON or missing key)
        passage_meta_df['section'].fillna(section_default, inplace=True)
        # Replace empty strings if sections were represented as "" in JSON
        passage_meta_df['section'].replace('', section_default, inplace=True, regex=False)
    else:
        print("Info: 'section' column not found in passage data. Skipping section defaulting.")


    # Derive doc_id for passages using the actual key name from the JSONL data
    if actual_passage_id_key not in passage_meta_df.columns:
        print(f"Error: Aggregated passage data must contain a '{actual_passage_id_key}' column (as specified in configuration 'ACTUAL_PASSAGE_ID_JSON_KEY').")
        print(f"Please check your JSONL files and the script's configuration.")
        sys.exit(1)

    try:
        passage_meta_df['doc_id'] = passage_meta_df[actual_passage_id_key].apply(
            lambda x: str(x).split('-', 1)[0] if isinstance(x, (str, bytes)) and '-' in str(x) else None
        )

        if passage_meta_df['doc_id'].isnull().any():
             num_missing_doc_ids = passage_meta_df['doc_id'].isnull().sum()
             print(f"Warning: {num_missing_doc_ids} passages ({num_missing_doc_ids/len(passage_meta_df):.2%}) did not yield a valid doc_id after splitting '{actual_passage_id_key}'.")
    except Exception as e:
        print(f"Error deriving doc_id from '{actual_passage_id_key}': {e}")
        sys.exit(1)


    # Join on doc_id
    print("\nJoining passage metadata with document metadata on 'doc_id'...")
    # Use an 'inner' join to only include passages that successfully match a document
    master_meta_df = pd.merge(passage_meta_df, doc_meta_df, on='doc_id', how='inner')
    print(f"Resulting master catalog has {len(master_meta_df)} records after inner join.")

    # Add provenance columns
    master_meta_df['processed_date'] = datetime.now().strftime('%Y-%m-%d')

    # --- Column Renaming ---
    print("\nRenaming columns...")
    existing_rename_map = {old_name: new_name for old_name, new_name in rename_map.items() if old_name in master_meta_df.columns}

    if existing_rename_map:
        master_meta_df = master_meta_df.rename(columns=existing_rename_map)
        print(f"Applied renames: {existing_rename_map}")
    else:
        print("No columns matched the rename map.")

    # --- Tweak 3: Drop Unnecessary Columns ---
    if cols_to_drop:
        existing_cols_to_drop = [col for col in cols_to_drop if col in master_meta_df.columns]
        if existing_cols_to_drop:
            print(f"\nDropping columns: {existing_cols_to_drop}...")
            master_meta_df = master_meta_df.drop(columns=existing_cols_to_drop)
            print("Columns dropped.")
        else:
             print(f"\nNone of the columns specified to drop {cols_to_drop} were found.")
    else:
         print("\nNo columns specified to drop.")


    # --- Tweak 2: CSV Quoting Note ---
    # Pandas to_csv uses csv.QUOTE_MINIMAL by default, which quotes fields
    # containing special characters like commas or newlines. This should
    # handle multi-line content cells correctly for standard CSV readers.
    # If you still encounter issues with specific readers or data, consider
    # using quoting=csv.QUOTE_ALL or exporting to Parquet/JSON Lines instead.


    # Export master catalog
    print(f"\nExporting master metadata to {master_metadata_csv_path}...")
    output_dir = os.path.dirname(master_metadata_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        master_meta_df.to_csv(master_metadata_csv_path, index=False)
        print("Export complete.")
    except Exception as e:
        print(f"Error exporting master metadata to {master_metadata_csv_path}: {e}")
        sys.exit(1)

    # 3. Quality Checks
    print("\n--- Quality Checks ---")

    # Row counts
    print(f"Total passages aggregated from JSONL files: {len(passage_meta_df)}")
    print(f"Total records in master catalog (after inner join): {len(master_meta_df)}")
    if len(master_meta_df) != len(passage_meta_df):
        unjoined_passages_count = len(passage_meta_df) - len(master_meta_df)
        print(f"Warning: {unjoined_passages_count} passages ({unjoined_passages_count/len(passage_meta_df):.2%}) from JSONL files could not be joined with document metadata (likely missing matching 'doc_id' in metadata.csv).")

    # Missing critical document metadata fields check (uses renamed columns)
    print(f"Checking for missing values in critical document metadata fields: {critical_doc_fields}...")
    existing_critical_fields_after_rename = [field for field in critical_doc_fields if field in master_meta_df.columns]

    if existing_critical_fields_after_rename:
        missing_critical_data = master_meta_df[existing_critical_fields_after_rename].isnull().sum()
        if missing_critical_data.sum() > 0:
            print("Warning: Found missing values in the following critical document metadata fields:")
            print(missing_critical_data[missing_critical_data > 0])
            print("This could indicate blank entries in your original metadata.csv for these documents or issues during join.")
        else:
            print(f"Check: No missing values found in critical document metadata fields ({', '.join(existing_critical_fields_after_rename)}).")
    else:
        print(f"Info: None of the specified critical document metadata fields {critical_doc_fields} found in the master catalog columns after renaming.")


    # Token-count sanity (uses assumed final column name 'token_count')
    print(f"Checking token counts (expected range: {min_tokens} - {max_tokens * 1.1:.0f})...")
    if 'token_count' in master_meta_df.columns:
        master_meta_df['token_count'] = pd.to_numeric(master_meta_df['token_count'], errors='coerce')

        invalid_token_count_mask = (master_meta_df['token_count'].isnull()) | \
                                 (master_meta_df['token_count'] < min_tokens) | \
                                 (master_meta_df['token_count'] > max_tokens * 1.1)

        invalid_token_passages = master_meta_df[invalid_token_count_mask]

        if not invalid_token_passages.empty:
            print(f"Warning: Found {len(invalid_token_passages)} passages ({len(invalid_token_passages)/len(master_meta_df):.2%}) with suspicious or missing token counts.")
            # Use the renamed 'passage_id' column for reporting if it exists
            id_col_for_report = 'passage_id' if 'passage_id' in master_meta_df.columns else actual_passage_id_key
            # Use renamed 'doc_filename' column for reporting if it exists
            filename_col_for_report = 'doc_filename' if 'doc_filename' in master_meta_df.columns else 'filename' # fallback to original if not renamed

            report_cols = [id_col_for_report, 'token_count', filename_col_for_report]
            report_cols = [col for col in report_cols if col in master_meta_df.columns]
            if report_cols:
                 print("Suspicious passages sample:")
                 print(invalid_token_passages[report_cols].head())
            else:
                 print("Cannot display sample as required columns for report are missing.")
        else:
            print("Check: All passage token counts seem within the expected range and are numeric.")
    else:
         print("Warning: 'token_count' column not found in master metadata. Cannot perform token count sanity check.")


    print("\n--- Phase E: Metadata Consolidation Finished ---")

# --- How to run the script ---
if __name__ == "__main__":
    # Ensure the processed directory structure exists if you haven't already
    os.makedirs('corpus/processed', exist_ok=True)
    os.makedirs('corpus/processed/passages', exist_ok=True)

    # Run the consolidation process
    consolidate_metadata(
        metadata_csv_path=METADATA_CSV_PATH,
        passages_dir=PASSAGES_DIR,
        master_metadata_csv_path=MASTER_METADATA_CSV_PATH,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        critical_doc_fields=CRITICAL_DOC_FIELDS,
        actual_passage_id_key=ACTUAL_PASSAGE_ID_JSON_KEY,
        rename_map=COLUMN_RENAME_MAP,
        section_default=SECTION_DEFAULT_VALUE, # Pass the section default
        cols_to_drop=COLUMNS_TO_DROP           # Pass columns to drop
    )

    # Optional: Export to other formats like SQLite or Parquet for faster querying
    print("\nAttempting optional export to SQLite and Parquet...")
    try:
        # Reload the saved CSV to ensure we're working with the final data
        master_meta_df = pd.read_csv(MASTER_METADATA_CSV_PATH)

        # SQLite Export
        sqlite_db_path = 'corpus/processed/master_metadata.sqlite'
        try:
            from sqlalchemy import create_engine
            engine = create_engine(f'sqlite:///{sqlite_db_path}')
            master_meta_df.to_sql('master_metadata', engine, if_exists='replace', index=False)
            print(f"Exported to SQLite: {sqlite_db_path}")
        except ImportError:
            print("Skipping SQLite export: Please install sqlalchemy (`pip install sqlalchemy`)")
        except Exception as e:
            print(f"Error during SQLite export: {e}")

        # Parquet Export
        parquet_path = 'corpus/processed/master_metadata.parquet'
        try:
            master_meta_df.to_parquet(parquet_path, index=False)
            print(f"Exported to Parquet: {parquet_path}")
        except ImportError:
            print("Skipping Parquet export: Please install pyarrow or fastparquet (`pip install pyarrow`)")
        except Exception as e:
            print(f"Error during Parquet export: {e}")

    except FileNotFoundError:
        print(f"Cannot perform optional export: Master metadata CSV not found at {MASTER_METADATA_CSV_PATH}")
    except Exception as e:
         print(f"Error during optional export setup: {e}")