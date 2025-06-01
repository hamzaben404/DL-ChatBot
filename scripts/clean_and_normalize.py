#!/usr/bin/env python3
from pathlib import Path
import ftfy
import unicodedata
import re
from datetime import datetime

# Directory configuration
TEXT_IN_DIR = Path("corpus/processed/texts")
TEXT_OUT_DIR = Path("corpus/processed/texts_clean")
TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_line(line):
    """Clean and normalize text lines while preserving code blocks"""
    line = ftfy.fix_text(line)
    line = unicodedata.normalize('NFKC', line)
    
    replacements = {
        r"D´eveloppement": "Développement",
        r"Probl`emes": "Problèmes",
        r"f´evrier": "février",
        r"R´egression": "Régression",
        r"´e": "é",
        r"`e": "è",
        r"ˆe": "ê",
        r"¨ı": "ï",
        r"´E": "É",
        r"‘": "'",
        r"’": "'",
        r"\b(\d{1,2})\s+([a-zA-Zéû]+)\s+(\d{4})\b": lambda m: format_french_date(m),
        r"(\d),(\d{3})": r"\1\2"
    }

    for pattern, replacement in replacements.items():
        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)

    return line.rstrip()

def format_french_date(match):
    months = {
        'janvier': '01', 'février': '02', 'mars': '03',
        'avril': '04', 'mai': '05', 'juin': '06',
        'juillet': '07', 'août': '08', 'septembre': '09',
        'octobre': '10', 'novembre': '11', 'décembre': '12'
    }
    day = match.group(1).zfill(2)
    month = months.get(match.group(2).lower(), '00')
    return f"{match.group(3)}-{month}-{day}"

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    cleaned = []
    in_code_block = False
    
    for line in content:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            cleaned.append(line)
            continue
        
        if in_code_block:
            cleaned.append(ftfy.fix_text(line))
        else:
            cleaned_line = clean_line(line)
            cleaned.append(cleaned_line + '\n')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned)

def process_corpus():
    """Process all files recursively while preserving directory structure"""
    for input_file in TEXT_IN_DIR.glob('**/*.txt'):
        if input_file.is_file():
            # Get relative path from input directory
            rel_path = input_file.relative_to(TEXT_IN_DIR)
            output_file = TEXT_OUT_DIR / rel_path
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            clean_file(input_file, output_file)
            print(f"Processed: {rel_path}")

if __name__ == "__main__":
    process_corpus()
    print(f"Cleaned files saved to: {TEXT_OUT_DIR}")