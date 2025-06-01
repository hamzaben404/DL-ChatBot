#!/usr/bin/env python3
"""
scripts/generate_answer.py
"""
import sys
from pathlib import Path
import argparse
import yaml
from rag_retriever import RAGRetriever
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

EMB_CFG = ROOT / "configs/embedding_config.yaml"
LLM_CFG = ROOT / "configs/llm_config.yaml"

# Improved prompt template
PROMPT_TEMPLATE = """
Context:
{context}

Instructions:
1. Answer using ONLY these sources
2. Cite sources like [Document:Page#]
3. Be concise and technical

Question: {question}
Answer:"""

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())

def generate_answer(query, top_k=5):
    try:
        # 1. Retrieve passages
        retriever = RAGRetriever(config_path=str(EMB_CFG))
        passages = retriever.retrieve(query, top_k=top_k)
        
        # Format context with references
        context = ""
        for i, p in enumerate(passages, 1):
            context += f"[Source {i}] {p['section']}:\n{p['content']}\n\n---\n"
        
        # 2. Load LLM config
        llm_cfg = load_yaml(LLM_CFG)
        hf_token = llm_cfg.get("api_key", None)
        model_id = llm_cfg.get("model", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        system_prompt = llm_cfg.get("system_prompt", "")

        if not hf_token:
            raise ValueError("Hugging Face API token is missing in configs/llm_config.yaml")

        # 3. Initialize client
        client = InferenceClient(model=model_id, token=hf_token, timeout=120)

        # 4. Build prompt
        full_prompt = (
            f"<s>[INST] {system_prompt}\n\n"
            f"{PROMPT_TEMPLATE.format(context=context, question=query)}"
            " [/INST]"
        )

        # 5. Generate response
        response = client.text_generation(
            full_prompt,
            max_new_tokens=llm_cfg.get("max_tokens", 512),
            temperature=llm_cfg.get("temperature", 0.3),
            top_p=llm_cfg.get("top_p", 0.9),
            repetition_penalty=llm_cfg.get("repetition_penalty", 1.1),
            stop=["</s>"]
        )

        return response.strip(), passages
    
    except HfHubHTTPError as e:
        print(f"\nERROR: Hugging Face API error ({e.response.status_code})")
        print(f"URL: {e.response.url}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="RAG end-to-end Q&A generator")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to retrieve")
    args = parser.parse_args()

    answer, passages = generate_answer(args.question, args.top_k)
    print(f"\nAnswer:\n{answer}\n")
    print("Retrieved Passages:")
    for i, p in enumerate(passages, 1):
        excerpt = p['content'].replace('\n', ' ')[:100]
        print(f"{i}. [Score: {p['score']:.3f}] [{p['section']}] {excerpt}...")

if __name__ == "__main__":
    main()