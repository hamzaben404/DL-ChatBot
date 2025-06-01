# test_rag.py
from scripts.generate_answer import generate_answer

def test():
    answer, passages = generate_answer("What is backpropagation?", 3)
    print("TEST SUCCESS!")
    print(f"Answer: {answer[:50]}...")
    print(f"Passages: {len(passages)} retrieved")

if __name__ == "__main__":
    test()