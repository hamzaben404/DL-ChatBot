#!/usr/bin/env python3
"""
scripts/demo_retrieval.py

Simple CLI demo for the RAGRetriever.
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so we can import rag_retriever
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import argparse
from rag_retriever import RAGRetriever


def main():
    parser = argparse.ArgumentParser(description="RAG Retriever CLI Demo")
    parser.add_argument(
        "--question", type=str, required=True,
        help="The question to retrieve passages for"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top passages to retrieve"
    )
    args = parser.parse_args()

    retriever = RAGRetriever(config_path="configs/embedding_config.yaml")
    results = retriever.retrieve(args.question, top_k=args.top_k)

    print(f"\nTop {args.top_k} passages for question: '{args.question}'\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [Score: {r['score']:.3f}] (Section: {r['section']})")
        excerpt = r['content'].replace('\n', ' ')
        print(f"Excerpt: {excerpt[:200]}...\n")

if __name__ == "__main__":
    main()
