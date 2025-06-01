# scripts/embed_passages.py

import sqlite3
import numpy as np
import yaml
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def main():
    # Load config
    cfg = yaml.safe_load(Path("configs/embedding_config.yaml").read_text())

    # Prepare output dirs
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    text_model = SentenceTransformer(cfg["model"]["text_encoder"], device=cfg["device"])
    code_model = SentenceTransformer(cfg["model"]["code_encoder"], device=cfg["device"])

    # Connect to metadata DB
    conn = sqlite3.connect(cfg["paths"]["master_sqlite"])
    cursor = conn.cursor()

    for content_type, model in [("text", text_model), ("code", code_model)]:
        # Fetch all passages of this type from the master_metadata table
        cursor.execute(
            "SELECT passage_id, content FROM master_metadata WHERE content_type = ?", (content_type,)
        )
        records = cursor.fetchall()
        if not records:
            print(f"No passages of type '{content_type}' found, skipping.")
            continue

        ids = []
        embs = []

        for batch in chunks(records, cfg["batch_size"]):
            batch_ids, texts = zip(*batch)
            vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embs.append(vectors)
            ids.extend(batch_ids)

        # Stack and save
        all_embs = np.vstack(embs)
        np.save(emb_dir / cfg["paths"][f"{content_type}_embeddings_file"], all_embs)
        np.save(emb_dir / f"{content_type}_ids.npy", np.array(ids, dtype=object))

        print(f"Saved {content_type} embeddings: {all_embs.shape}, ids: {len(ids)}")

    conn.close()

if __name__ == "__main__":
    main()
