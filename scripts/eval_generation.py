#!/usr/bin/env python3
# scripts/eval_generation.py

import sys
from pathlib import Path
import yaml

# Add ROUGE
from rouge_score import rouge_scorer

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from scripts.generate_answer import generate_answer

def main():
    # Load gold generation set
    gold = yaml.safe_load(open("evaluation/gold_generation.yaml", 'r', encoding='utf-8'))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total = len(gold)
    rouge_l_sum = 0.0

    for entry in gold:
        q = entry["question"]
        ref = entry["reference"].strip().lower()
        gen, _ = generate_answer(q, top_k=5)
        gen_text = gen.strip().lower()

        scores = rouge.score(ref, gen_text)
        rouge_l_f = scores["rougeL"].fmeasure
        rouge_l_sum += rouge_l_f

        print(f"Question: {q}")
        print(f"Reference: {entry['reference']}")
        print(f"Generated: {gen}")
        print(f"ROUGE-L F1: {rouge_l_f:.3f}\n")

    avg_rouge_l = rouge_l_sum / total
    print(f"Avg ROUGE-L: {avg_rouge_l:.3f}")

if __name__ == "__main__":
    main()
