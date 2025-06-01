# scripts/build_index.py

import faiss
import numpy as np
import json
from pathlib import Path
import yaml

def main():
    # Load config
    cfg = yaml.safe_load(Path("configs/embedding_config.yaml").read_text())

    idx_dir = Path(cfg["paths"]["indexes_dir"])
    idx_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(cfg["paths"]["embeddings_dir"])

    for content_type in ["text", "code"]:
        emb_file = emb_dir / cfg["paths"][f"{content_type}_embeddings_file"]
        id_file = emb_dir / f"{content_type}_ids.npy"
        if not emb_file.exists() or not id_file.exists():
            print(f"Missing embeddings or ids for '{content_type}', skipping.")
            continue

        embeddings = np.load(emb_file)
        ids = np.load(id_file, allow_pickle=True)

        # Build FAISS index (inner product)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        idx_path = idx_dir / cfg["paths"][f"{content_type}_index_file"]
        faiss.write_index(index, str(idx_path))

        # Save ID mapping
        with open(idx_dir / f"{content_type}_id_map.json", "w", encoding="utf-8") as f:
            json.dump(ids.tolist(), f, ensure_ascii=False)

        print(f"Built {content_type} index with {index.ntotal} vectors → {idx_path}")

if __name__ == "__main__":
    main()
