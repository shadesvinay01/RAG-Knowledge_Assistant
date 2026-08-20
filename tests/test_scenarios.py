import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.chunking import HierarchicalChunker
from src.embeddings import EmbeddingEngine
from src.azure_search import HybridSearchEngine
from src.rag_engine import ImprovedEnterpriseRAGEngine
from eval.evaluate import load_knowledge_base

class TestRAGScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = "data/knowledge_base"
        chunks = load_knowledge_base(data_dir)
        embedding_engine = EmbeddingEngine()
        vectors = embedding_engine.embed_texts([c.content for c in chunks])
        
        search_engine = HybridSearchEngine()
        search_engine.index_chunks(chunks, vectors)

        cls.engine = ImprovedEnterpriseRAGEngine(search_engine, embedding_engine)

    def test_scenario_1_wrong_chunk(self):
        query = "How many days of paid sick leave do employees get under the 2026 policy and when is a doctor certificate needed?"
        res = self.engine.query(query, bypass_cache=True)
        self.assertIn("12 days", res["answer"])
        self.assertIn("4 consecutive", res["answer"])

    def test_scenario_2_multi_document(self):
        query = "Compare the refund policy for Enterprise and Standard customers."
        res = self.engine.query(query, bypass_cache=True)
        self.assertIn("Enterprise", res["answer"])
        self.assertIn("Standard", res["answer"])
        self.assertGreaterEqual(len(res["citations"]), 1)

    def test_scenario_3_version_conflict(self):
        query = "How many days of annual paid leave are employees entitled to?"
        res = self.engine.query(query, bypass_cache=True)
        self.assertIn("25 days", res["answer"])

    def test_scenario_4_hallucination_guardrail(self):
        query = "What is the company policy on pet insurance reimbursement for remote employees?"
        res = self.engine.query(query, bypass_cache=True)
        self.assertTrue(res.get("insufficient_evidence"))

    def test_scenario_5_ambiguous_query(self):
        query = "What is the limit?"
        res = self.engine.query(query, bypass_cache=True)
        self.assertTrue(res.get("is_ambiguous"))

    def test_scenario_6_conversational_context(self):
        history = [{"user": "What is the Enterprise plan cancellation policy?", "assistant": "Enterprise plans require 30 days notice."}]
        query = "What about Standard?"
        res = self.engine.query(query, chat_history=history, bypass_cache=True)
        self.assertIn("Standard", res["answer"])

    def test_rbac_department_isolation(self):
        # Querying HR department restricted leave document as an Engineering user
        query = "How many days of paid sick leave do employees get under the 2026 HR policy?"
        res = self.engine.query(query, user_department="Engineering", bypass_cache=True)
        # HR documents restricted for Engineering user -> Insufficient evidence or unretrieved
        self.assertTrue(res.get("insufficient_evidence") or "Leave_Policy_2026.md" not in str(res["citations"]))

if __name__ == "__main__":
    unittest.main()
