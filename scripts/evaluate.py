import yaml
from generate_answer import generate_answer

queries = yaml.safe_load(Path("configs/sample_queries.yaml").read_text())
for i, q in enumerate(queries):
    answer, _ = generate_answer(q['question'])
    print(f"Q{i+1}: {q['question']}")
    print(f"A: {answer[:200]}...\n")