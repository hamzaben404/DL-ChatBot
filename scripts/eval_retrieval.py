#!/usr/bin/env python3
# scripts/eval_retrieval.py
# Basic retrieval evaluation: Recall@5 on a small gold set

import sys
from pathlib import Path
import yaml

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from rag_retriever import RAGRetriever

def main():
    # Load gold retrieval set
    gold = yaml.safe_load(open("evaluation/gold_retrieval.yaml", 'r', encoding='utf-8'))
    retriever = RAGRetriever(config_path="configs/embedding_config.yaml")

    correct_at_5 = 0
    total = len(gold)

    for entry in gold:
        q = entry["question"]
        gold_pid = entry["gold_passage_id"]
        results = retriever.retrieve(q, top_k=5)
        retrieved_ids = [r['passage_id'] for r in results]
        if gold_pid in retrieved_ids:
            correct_at_5 += 1
        print(f"Question: {q}")
        print(f"Retrieved top-5: {retrieved_ids}")
        print(f"Gold passage found: {gold_pid in retrieved_ids}\n")

    recall_at_5 = correct_at_5 / total
    print(f"Recall@5: {recall_at_5:.2f} ({correct_at_5}/{total})")

if __name__ == "__main__":
    main()
