# rag_retriever.py (FIXED)

"""
rag_retriever.py

Module implementing the RAGRetriever for embedding queries,
searching FAISS index, and fetching passage metadata.
"""
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import yaml
import os

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class RAGRetriever:
    def __init__(self, config_path: str = "configs/embedding_config.yaml"):
        # Load configuration and override device to CPU
        cfg = yaml.safe_load(Path(config_path).read_text())
        cfg["device"] = "cpu"
        paths = cfg["paths"]
        model_cfg = cfg["model"]

        # Load embedding model on CPU
        self.model = SentenceTransformer(model_cfg["text_encoder"], device="cpu")

        # Load FAISS index
        index_path = Path(paths["indexes_dir"]) / cfg["paths"]["text_index_file"]
        self.index = faiss.read_index(str(index_path))

        # Load ID map
        ids_path = Path(paths["embeddings_dir"]) / "text_ids.npy"
        self.id_map = np.load(ids_path, allow_pickle=True)

        # Connect to metadata database
        self.conn = sqlite3.connect(paths["master_sqlite"])
        self.cursor = self.conn.cursor()
        
        # self.conn_pool = []

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top_k passages for the given query.
        Returns list of dicts with keys: passage_id, content, section, doc_source, score.
        """
        # conn = self.get_connection()
        # cursor = conn.cursor()
        try:
            # Encode query
            qvec = self.model.encode(query, convert_to_numpy=True)
            qvec = np.expand_dims(qvec, axis=0)

            # Search index
            distances, indices = self.index.search(qvec, top_k)
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {e}")

        results = []
        for score, idx in zip(distances[0], indices[0]):
            try:
                pid = self.id_map[idx]
            except IndexError:
                continue
            # Fetch metadata - REMOVED PAGE_NUMBER
            self.cursor.execute(
                "SELECT passage_id, content, section, doc_source "
                "FROM master_metadata "
                "WHERE passage_id = ?", (pid,)
            )
            row = self.cursor.fetchone()
            if row:
                results.append({
                    "passage_id": row[0],
                    "content": row[1],
                    "section": row[2] or "",
                    "doc_source": row[3],
                    "score": float(score)
                })
                
        # self.return_connection(conn)
        return results

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

## new  
    def get_connection(self):
        if not self.conn_pool:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        return self.conn_pool.pop()
    
    def return_connection(self, conn):
        self.conn_pool.append(conn)

# Example usage
if __name__ == "__main__":
    retriever = RAGRetriever()
    query = input("Enter your question: ")
    out = retriever.retrieve(query, top_k=5)
    for i, r in enumerate(out, 1):
        print(f"{i}. [{r['score']:.3f}] ({r['section']}) {r['content'][:200]}...")