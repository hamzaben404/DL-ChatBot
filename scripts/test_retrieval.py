#!/usr/bin/env python3
import sqlite3
import numpy as np
import yaml
import json
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CFG_PATH = Path("configs/sample_queries.yaml")
EMB_DIR  = Path("corpus/processed/embeddings")
IDX_DIR  = Path("corpus/processed/indexes")
DB_PATH  = Path("corpus/processed/master_metadata.sqlite")
TOP_K    = 5

# -----------------------------------------------------------------------------
# Load sample queries
# -----------------------------------------------------------------------------
with open(CFG_PATH, 'r', encoding='utf-8') as f:
    samples = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Load embedding model & FAISS index
# -----------------------------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(str(IDX_DIR / "text_index.faiss"))
id_map = np.load(EMB_DIR / "text_ids.npy", allow_pickle=True)

# -----------------------------------------------------------------------------
# Connect to metadata DB
# -----------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

results = {}

for entry in samples:
    qid  = entry["id"]
    qtxt = entry["question"]
    qvec = model.encode(qtxt, convert_to_numpy=True)

    # retrieve top-K
    D, I = index.search(np.expand_dims(qvec, axis=0), TOP_K)
    retrieved = []
    for dist, idx in zip(D[0], I[0]):
        pid = id_map[idx]
        # fetch passage metadata
        cursor.execute(
            "SELECT section, substr(content,1,200) as excerpt "
            "FROM master_metadata WHERE passage_id = ?", (pid,)
        )
        row = cursor.fetchone()
        retrieved.append({
            "passage_id": pid,
            "score": float(dist),
            "section": row[0],
            "excerpt": row[1].replace('\n',' ')
        })

    results[qid] = {
        "question": qtxt,
        "retrieved": retrieved
    }

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
OUT_PATH = Path("validation/retrieval_results.json")
OUT_PATH.parent.mkdir(exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Retrieval results written to {OUT_PATH.resolve()}")
