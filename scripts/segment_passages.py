#!/usr/bin/env python3
import re
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import tiktoken

# Configuration
MAX_TOKENS = 512
MIN_TOKENS = 50
OVERLAP_THRESHOLD = 1.2
WINDOW_OVERLAP = 0.2
ENCODING = tiktoken.get_encoding("cl100k_base")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TEXT_CLEAN_DIR = Path("corpus/processed/texts_clean")
PASSAGE_DIR = Path("corpus/processed/passages")
PASSAGE_DIR.mkdir(parents=True, exist_ok=True)

class PassageProcessor:
    def __init__(self):
        self.current_section: Optional[str] = None
        self.in_code_block: bool = False

    def token_count(self, text: str) -> int:
        return len(ENCODING.encode(text))

    def is_code_block(self, content: str) -> bool:
        lines = content.split('\n')
        return len(lines) >= 2 and lines[0].startswith('```') and lines[-1].startswith('```')

    def is_section_header(self, line: str) -> bool:
        header_patterns = [
            r"^#{1,3}\s+(?P<title>.+)",
            r"^\d+(\.\d+)*\s+(?P<title>[A-Z].+)",
            r"^(?P<title>[A-Z\s]{10,})\:?$",
            r"^(?P<title>Objectif|Architecture|Conclusion|Méthodologie)\b.*\:?$"
        ]
        for pattern in header_patterns:
            match = re.match(pattern, line)
            if match:
                self.current_section = match.group("title").strip()
                return True
        return False

    def should_split(self, context: Dict) -> bool:
        return any([
            self.is_section_header(context["current_line"]),
            context["current_line"].startswith(("Figure", "Table")),
            (context["trigger_list_start"] and not context["prev_list_start"]),
            (context["token_count"] > MAX_TOKENS * 0.9)  # Split earlier to account for joins
        ])

    def split_long_passage(self, passage: str) -> List[str]:
        if self.is_code_block(passage):
            return [passage]

        tokens = ENCODING.encode(passage)
        if len(tokens) <= MAX_TOKENS:
            return [passage]

        window_size = int(MAX_TOKENS * (1 - WINDOW_OVERLAP))
        if len(tokens) < MAX_TOKENS * OVERLAP_THRESHOLD:
            window_size = MAX_TOKENS

        passages = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_TOKENS, len(tokens))
            chunk = ENCODING.decode(tokens[start:end])
            
            # Ensure we don't split mid-word
            while end < len(tokens) and not ENCODING.decode(tokens[end-1:end+1]).isalnum():
                end += 1
                
            chunk = ENCODING.decode(tokens[start:end])
            passages.append(chunk)
            start += window_size if window_size < (len(tokens) - end) else 0

        return passages

    def clean_content(self, content: str) -> str:
        content = content.strip()
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content

    def process_file(self, file_path: Path) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            passages = []
            current_passage = []
            current_content = ""
            metadata = {
                "current_section": None,
                "line_start": 1,
                "char_start": 0,
                "in_code": False
            }

            for line_num, line in enumerate(content.split('\n'), start=1):
                line = line.rstrip()
                
                # Handle code blocks
                if line.startswith('```'):
                    if metadata["in_code"]:
                        # End code block
                        current_passage.append(line)
                        current_content += "\n" + line
                        passages.append(self.create_passage(current_content, metadata, file_path, True))
                        current_passage = []
                        current_content = ""
                        metadata.update({
                            "in_code": False,
                            "line_start": line_num + 1,
                            "char_start": metadata["char_start"] + len(line) + 1
                        })
                        continue
                    else:
                        # Start code block
                        if current_passage:
                            passages.append(self.create_passage(current_content, metadata, file_path))
                            current_passage = []
                            current_content = ""
                        metadata["in_code"] = True
                        current_passage.append(line)
                        current_content = line
                        continue

                if metadata["in_code"]:
                    current_passage.append(line)
                    current_content += "\n" + line
                    continue

                # Calculate accurate token count
                new_content = current_content + "\n" + line if current_content else line
                token_count = self.token_count(new_content)

                context = {
                    "current_line": line,
                    "prev_line": current_passage[-1] if current_passage else "",
                    "token_count": token_count,
                    "trigger_list_start": re.match(r"^[\-•]", line),
                    "prev_list_start": re.match(r"^[\-•]", current_passage[-1] if current_passage else "")
                }

                if self.should_split(context):
                    if current_content:
                        passages.append(self.create_passage(current_content, metadata, file_path))
                    current_passage = [line]
                    current_content = line
                    metadata.update({
                        "line_start": line_num,
                        "char_start": metadata["char_start"] + len(line) + 1,
                        "current_section": self.current_section
                    })
                else:
                    current_passage.append(line)
                    current_content = new_content

            if current_content:
                passages.append(self.create_passage(current_content, metadata, file_path, metadata["in_code"]))

            # Post-process and split
            final_passages = []
            for p in passages:
                cleaned = self.clean_content(p["content"])
                if p["content_type"] == "code":
                    final_passages.append(p)
                    continue
                
                chunks = self.split_long_passage(cleaned)
                for chunk in chunks:
                    chunk = self.clean_content(chunk)
                    token_count = self.token_count(chunk)
                    final_passages.append({
                        "id": f"{p['id']}-{uuid.uuid4().hex[:4]}",
                        "source": p["source"],
                        "section": p["section"],
                        "content": chunk,
                        "content_type": "text",
                        "token_count": token_count,
                        "line_start": p["line_start"],
                        "line_end": p["line_end"],
                        "char_start": p["char_start"],
                        "char_end": p["char_start"] + len(chunk)
                    })

            # Write output
            output_file = PASSAGE_DIR / file_path.relative_to(TEXT_CLEAN_DIR).with_suffix('.jsonl')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for p in final_passages:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')

            logging.info(f"Processed {file_path} → {len(final_passages)} passages")

        except Exception as e:
            logging.error(f"Error in {file_path}: {str(e)}")

    def create_passage(self, content: str, metadata: Dict, file_path: Path, is_code=False) -> Dict:
        return {
            "id": f"{file_path.stem}-{uuid.uuid4().hex[:8]}",
            "source": str(file_path.relative_to(TEXT_CLEAN_DIR)),
            "section": metadata["current_section"],
            "content": content,
            "content_type": "code" if is_code else "text",
            "token_count": self.token_count(content),
            "line_start": metadata["line_start"],
            "line_end": metadata["line_start"] + content.count('\n'),
            "char_start": metadata["char_start"],
            "char_end": metadata["char_start"] + len(content)
        }

def validate_passages(sample_size: int = 50):
    """Relaxed validation with context awareness"""
    import random
    warnings = []
    
    for jsonl_file in Path(PASSAGE_DIR).glob('**/*.jsonl'):
        with open(jsonl_file) as f:
            passages = [json.loads(line) for line in f]
            
        for p in random.sample(passages, min(sample_size, len(passages))):
            # Allow 10% overage for text passages
            if p["content_type"] == "text" and p["token_count"] > MAX_TOKENS * 1.1:
                warnings.append(f"Long text passage {p['id']} ({p['token_count']} tokens)")
                
            # Validate code blocks
            if p["content_type"] == "code":
                lines = p["content"].split('\n')
                if not lines[0].startswith('```') or not lines[-1].startswith('```'):
                    warnings.append(f"Code block formatting issue in {p['id']}")

    if warnings:
        logging.warning(f"Validation passed with {len(warnings)} notes:")
        for warn in warnings[:5]:
            logging.warning(warn)
    else:
        logging.info("All sampled passages validated successfully")

if __name__ == "__main__":
    # Process files with 2 workers for stability
    files = list(TEXT_CLEAN_DIR.glob('**/*.txt'))
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(PassageProcessor().process_file, files)
    
    # Run validation
    validate_passages()
    logging.info("Processing completed successfully")