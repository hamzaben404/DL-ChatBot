# -----------------------------------------------------------------------------
# tests/test_retriever.py
# -----------------------------------------------------------------------------
import unittest
from rag_retriever import RAGRetriever

class TestRAGRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = RAGRetriever(config_path="configs/embedding_config.yaml")

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("What is backpropagation?", top_k=3)
        self.assertIsInstance(results, list)

    def test_retrieve_structure(self):
        results = self.retriever.retrieve("What is backpropagation?", top_k=2)
        for r in results:
            self.assertIn("passage_id", r)
            self.assertIn("content", r)
            self.assertIn("section", r)
            self.assertIn("doc_source", r)
            self.assertIn("score", r)

    def test_retrieve_non_empty(self):
        results = self.retriever.retrieve("neural network", top_k=3)
        self.assertGreaterEqual(len(results), 1)

    def test_score_range(self):
        results = self.retriever.retrieve("activation function", top_k=3)
        for r in results:
            self.assertGreaterEqual(r['score'], 0.0)
            self.assertLessEqual(r['score'], 1.0)

    def test_known_backprop_passage(self):
        results = self.retriever.retrieve("backpropagation", top_k=1)
        # Expect the top passage to come from the DeepLearning1 document
        pid = results[0]['passage_id'] if results else ''
        self.assertTrue(pid.startswith("DeepLearning1"))

if __name__ == '__main__':
    unittest.main()