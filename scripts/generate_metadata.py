#!/usr/bin/env python3
import os
import csv
import datetime
import re # Import regex module for case-insensitive matching (alternative to .lower().startswith())

RAW_DIR       = "corpus/raw"
PROCESSED_DIR = "corpus/processed"
METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.csv")

# --- Configuration for automatic filling ---
# Define regex patterns for source
TD_PATTERN = re.compile(r"^td", re.IGNORECASE) # Case-insensitive match for start with "td"
TP_PATTERN = re.compile(r"^tp", re.IGNORECASE) # Case-insensitive match for start with "tp"

# --- Script ---

# 1. Ensure processed folder exists
print(f"Ensuring directory exists: {PROCESSED_DIR}")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Scan raw directory and create records
print(f"Scanning raw directory: {RAW_DIR}")
records = []
skipped_files = []

if not os.path.exists(RAW_DIR):
    print(f"Error: Raw directory not found at {RAW_DIR}")
    sys.exit(1) # Exit if the raw directory doesn't exist

for root, _, files in os.walk(RAW_DIR):
    for fname in files:
        full_path = os.path.join(root, fname)
        # Skip hidden files (optional, adjust as needed)
        if fname.startswith('.'):
            print(f"Skipping hidden file: {full_path}")
            continue
        # Skip directories encountered as files (e.g., on some systems or unusual names)
        if os.path.isdir(full_path):
             print(f"Skipping directory found in file list: {full_path}")
             continue

        try:
            # relative path from RAW_DIR, using forward slashes
            rel_path = os.path.relpath(full_path, RAW_DIR).replace(os.sep, "/")

            # doc_id: replace slashes with underscores, strip extension
            # doc_id = os.path.splitext(rel_path)[0].replace("/", "_")
            doc_id = os.path.splitext(fname)[0] # Use the base filename without extension

            # --- Automatically determine 'type', 'source', 'date' ---

            # Determine 'type' from file extension
            # Get extension including the dot, then strip dot and convert to lowercase
            file_ext = os.path.splitext(fname)[1].lstrip('.').lower()
            file_type = file_ext if file_ext else "unknown" # Use extension as type, or 'unknown' if no extension

            # Determine 'source' based on file name prefix (case-insensitive)
            base_name_without_ext = os.path.splitext(fname)[0]
            file_source = "lecture" # Default source

            if TD_PATTERN.match(base_name_without_ext):
                 file_source = "tutorial"
            elif TP_PATTERN.match(base_name_without_ext):
                 file_source = "assignment"
            # else remains "lecture"

            # Determine 'date' from file modification time
            try:
                mtime = os.path.getmtime(full_path)
                file_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Warning: Could not get modification date for {full_path}: {e}")
                file_date = "" # Leave blank if date cannot be retrieved


            records.append({
                "doc_id":   doc_id,
                "filename": rel_path,
                "type":     file_type,
                "source":   file_source,
                "date":     file_date,
                "notes":    ""    # notes field remains empty for manual filling later
            })
        except Exception as e:
            print(f"Error processing file {full_path}: {e}")
            skipped_files.append(full_path)


# 3. Write CSV
print(f"Writing metadata for {len(records)} files to {METADATA_PATH}")
try:
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id","filename","type","source","date","notes"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Successfully wrote metadata for {len(records)} files.")
    if skipped_files:
        print(f"Warning: Skipped {len(skipped_files)} files due to errors.")
        # print("Skipped files:", skipped_files) # Uncomment to see the list
except Exception as e:
    print(f"Error writing CSV file {METADATA_PATH}: {e}")
    sys.exit(1)


print("\nMetadata creation script finished.")
print(f"Remember to manually review and edit '{METADATA_PATH}' to correct any automatically assigned values and fill in 'notes'.")