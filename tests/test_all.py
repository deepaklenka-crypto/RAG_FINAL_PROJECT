"""
Comprehensive Test Suite for Multi-Format 3-Type RAG System:
Validates Ingestion, Chunkers, Vector DB, 3 RAG Types, Evaluation, Corruption, and Cache.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import init_db, SessionLocal, DocumentModel, ChunkModel
from index import DocumentParser, index_document
from rag.chunking import get_chunker
from rag.embeddings import get_embedding_provider, LocalFallbackEmbeddingProvider
from rag.vector_store import QdrantVectorStore
from rag.scoring import compute_cosine_similarity, BM25Scorer, reciprocal_rank_fusion
from rag.simple_rag import SimpleRAG
from rag.hybrid_rag import HybridRAG
from rag.graph_rag import GraphRAG
from evaluation.prompt_eval import PromptEvaluator
from evaluation.response_eval import ResponseEvaluator
from evaluation.corruption import CorruptionAnalyzer
from optimization.cache import global_semantic_cache
from main import app


class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.sample_txt = "./data/samples/ai_rag_overview.txt"
        cls.sample_csv = "./data/samples/products_catalog.csv"
        cls.sample_xlsx = "./data/samples/financial_report.xlsx"
        cls.sample_docx = "./data/samples/system_specifications.docx"
        cls.sample_pdf = "./data/samples/executive_summary.pdf"

    def test_01_parsers_all_formats(self):
        """Test document parsers for all 5 formats: TXT, CSV, XLSX, DOCX, PDF."""
        for path in [self.sample_txt, self.sample_csv, self.sample_xlsx, self.sample_docx, self.sample_pdf]:
            self.assertTrue(os.path.exists(path), f"Sample file missing: {path}")
            text, meta = DocumentParser.parse(path)
            self.assertGreater(len(text), 10, f"Failed parsing non-empty text from {path}")
            self.assertIn("filename", meta)

    def test_02_chunking_strategies(self):
        """Test recursive, semantic, and structured chunkers."""
        sample_text = "RAG stands for Retrieval-Augmented Generation. It combines search with generation. " * 20

        # Recursive chunker
        rec_chunker = get_chunker("recursive", chunk_size=150, chunk_overlap=30)
        rec_chunks = rec_chunker.chunk(sample_text)
        self.assertGreater(len(rec_chunks), 1)

        # Semantic chunker
        sem_chunker = get_chunker("semantic", chunk_size=200, chunk_overlap=40)
        sem_chunks = sem_chunker.chunk(sample_text)
        self.assertGreater(len(sem_chunks), 1)

        # Structured chunker
        struct_chunker = get_chunker("structured")
        struct_chunks = struct_chunker.chunk("Col1: Val1, Col2: Val2\nCol1: Val3, Col2: Val4", metadata={"file_type": "csv"})
        self.assertGreater(len(struct_chunks), 0)

    def test_03_embeddings_and_vector_store(self):
        """Test vector generation and Qdrant storage."""
        embedder = get_embedding_provider("local")
        v1 = embedder.embed_text("Deep Learning and Neural Networks")
        v2 = embedder.embed_text("Deep Learning and Artificial Intelligence")
        self.assertEqual(len(v1), 768)
        sim = compute_cosine_similarity(v1, v2)
        self.assertGreater(sim, 0.3)

        store = QdrantVectorStore(dimension=768)
        ids = store.upsert_chunks(
            chunks=["Test Chunk Content"],
            embeddings=[v1],
            metadatas=[{"doc_test": True}]
        )
        self.assertEqual(len(ids), 1)
        search_res = store.search(v1, top_k=1)
        self.assertGreaterEqual(len(search_res), 1)

    def test_04_scoring_and_rrf(self):
        """Test BM25 scoring and Reciprocal Rank Fusion."""
        corpus = [
            "Qdrant is a high performance vector database.",
            "PostgreSQL handles relational data and knowledge graphs.",
            "Gemini provides advanced multimodal and coding intelligence."
        ]
        bm25 = BM25Scorer(corpus)
        hits = bm25.score("Qdrant database", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 0)  # first document is top hit

        dense_res = [{"text": corpus[0], "score": 0.95}, {"text": corpus[1], "score": 0.70}]
        sparse_res = [{"text": corpus[1], "score": 2.5}, {"text": corpus[0], "score": 1.8}]
        fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60)
        self.assertEqual(len(fused), 2)
        self.assertIn("rrf_score", fused[0])

    def test_05_three_rag_types(self):
        """Test Simple RAG, Hybrid RAG, and Graph RAG pipelines."""
        q = "What are the benefits of Hybrid RAG and PagedAttention?"

        # 1. Simple RAG
        sim_rag = SimpleRAG()
        res_sim = sim_rag.query(q, top_k=2, language="en")
        self.assertIn("answer", res_sim)
        self.assertEqual(res_sim["rag_type"], "simple_rag")

        # 2. Hybrid RAG
        hyb_rag = HybridRAG()
        res_hyb = hyb_rag.query(q, top_k=2, language="en")
        self.assertIn("answer", res_hyb)
        self.assertEqual(res_hyb["rag_type"], "hybrid_rag")

        # 3. Graph RAG
        g_rag = GraphRAG()
        res_graph = g_rag.query(q, top_k=2, language="en")
        self.assertIn("answer", res_graph)
        self.assertEqual(res_graph["rag_type"], "graph_rag")
        self.assertIn("graph_context", res_graph)

    def test_06_multilingual_and_code_mode(self):
        """Test Hindi language mode and Python code generation mode."""
        hyb_rag = HybridRAG()

        # Hindi query
        res_hi = hyb_rag.query("RAG आर्किटेक्चर क्या है?", language="hi")
        self.assertIn("answer", res_hi)

        # Code query
        res_code = hyb_rag.query("Write a Python function for cosine similarity", mode="code")
        self.assertIn("answer", res_code)

    def test_07_evaluations_and_corruption(self):
        """Test Prompt evaluation, Response evaluation, and Corruption stress test."""
        # Prompt eval
        p_eval = PromptEvaluator.evaluate("Explain the mechanism of Hybrid RAG with Qdrant and BM25")
        self.assertGreater(p_eval["clarity_score"], 0.5)
        self.assertTrue(p_eval["passed_safety"])

        # Response eval
        r_eval = ResponseEvaluator().evaluate(
            query="What is Qdrant?",
            response="Qdrant is a high performance vector database built for similarity search.",
            context_passages=["Qdrant is a high performance vector database."]
        )
        self.assertGreaterEqual(r_eval["faithfulness_score"], 0.5)

        # Corruption test
        analyzer = CorruptionAnalyzer()
        report = analyzer.run_stress_test(
            query="What is RAG?",
            clean_context="Retrieval-Augmented Generation enhances language models with external vector search."
        )
        self.assertIn("robustness_score", report)
        self.assertIn("degradation_curve", report)

    def test_08_semantic_cache(self):
        """Test that repeated or similar query hits the cache."""
        global_semantic_cache.clear()
        q = "How does Reciprocal Rank Fusion work in Hybrid RAG?"
        hyb_rag = HybridRAG()

        # First query (populates cache)
        res1 = hyb_rag.query(q, use_cache=True)
        self.assertFalse(res1["telemetry"]["cache_hit"])

        # Second identical query (cache hit)
        res2 = hyb_rag.query(q, use_cache=True)
        self.assertTrue(res2["telemetry"]["cache_hit"])

    def test_09_api_endpoints(self):
        """Test FastAPI endpoints via TestClient."""
        # Health
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "online")

        # Documents list
        res_docs = self.client.get("/api/documents")
        self.assertEqual(res_docs.status_code, 200)
        self.assertIsInstance(res_docs.json(), list)

        # Query endpoint
        res_q = self.client.post("/api/query", json={
            "query": "What is Qdrant and how does it store vectors?",
            "rag_type": "hybrid",
            "language": "en",
            "mode": "general"
        })
        self.assertEqual(res_q.status_code, 200)
        self.assertIn("answer", res_q.json())

        # Knowledge graph endpoint
        res_g = self.client.get("/api/graph")
        self.assertEqual(res_g.status_code, 200)
        self.assertIn("nodes", res_g.json())

        # Benchmark stats endpoint
        res_b = self.client.get("/api/benchmark")
        self.assertEqual(res_b.status_code, 200)

        # Conversational memory session endpoints
        res_sess = self.client.get("/api/sessions")
        self.assertEqual(res_sess.status_code, 200)
        self.assertIsInstance(res_sess.json(), list)

    def test_10_conversational_rag_with_memory(self):
        """Test multi-turn conversational session memory and coreference resolution."""
        from rag.memory import global_conversation_memory
        sess_id = "test_turn_suite_session"
        global_conversation_memory.clear_session(sess_id)

        hyb_rag = HybridRAG()
        # Turn 1
        r1 = hyb_rag.query("What is Qdrant?", session_id=sess_id, top_k=2)
        self.assertIn("answer", r1)
        self.assertEqual(r1["session_id"], sess_id)

        # Turn 2: uses pronoun 'it'
        r2 = hyb_rag.query("Does it support fast filtering?", session_id=sess_id, top_k=2)
        self.assertIn("answer", r2)

        # Verify history length
        history = global_conversation_memory.get_history(sess_id)
        self.assertGreaterEqual(len(history), 4)  # 2 user turns + 2 assistant turns


if __name__ == "__main__":
    unittest.main()
