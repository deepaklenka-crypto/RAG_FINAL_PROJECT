"""
Hybrid RAG Implementation:
Fuses Dense Semantic Vectors (Qdrant) and Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF) and Contextual Cross-Reranking.
"""

import time
from typing import Dict, Any, Optional, List
from .vector_store import QdrantVectorStore
from .embeddings import get_embedding_provider
from .scoring import BM25Scorer, reciprocal_rank_fusion, FastReRanker
from llm.engine_factory import get_llm_engine
from llm.prompts import get_rag_prompt
from database import SessionLocal, ChunkModel, DocumentModel
from optimization.cache import global_semantic_cache
from optimization.compression import ContextCompressor
from rag.memory import global_conversation_memory


class HybridRAG:
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        llm_backend: Optional[str] = None
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.embed_provider = get_embedding_provider()
        self.llm_engine = get_llm_engine(llm_backend)
        self.compressor = ContextCompressor()
        self._bm25_scorer: Optional[BM25Scorer] = None
        self._corpus_chunks: List[Dict[str, Any]] = []
        self._load_corpus()

    def _load_corpus(self):
        """Loads corpus from database to power the BM25 sparse index."""
        try:
            with SessionLocal() as db:
                chunks = db.query(ChunkModel).all()
                self._corpus_chunks = [
                    {
                        "id": c.id,
                        "text": c.content,
                        "document_id": c.document_id,
                        "metadata": {"chunk_index": c.chunk_index}
                    }
                    for c in chunks
                ]
                texts = [c["text"] for c in self._corpus_chunks]
                if texts:
                    self._bm25_scorer = BM25Scorer(texts)
        except Exception as e:
            print(f"[HybridRAG] Warning loading corpus: {e}")

    def refresh_index(self):
        """Refreshes the BM25 index after new documents are ingested."""
        self._load_corpus()

    def query(
        self,
        question: str,
        top_k: int = 4,
        language: str = "en",
        mode: str = "general",
        rrf_k: int = 60,
        compress_context: bool = False,
        use_cache: bool = True,
        backend: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Hybrid RAG query:
        Dense Search + BM25 Search -> Reciprocal Rank Fusion -> Contextual Rerank -> LLM.
        Supports multi-turn conversational session memory.
        """
        start_total = time.perf_counter()

        # 1. Semantic Cache check
        if use_cache:
            cached_result = global_semantic_cache.get(question, rag_type="hybrid_rag")
            if cached_result:
                if session_id:
                    global_conversation_memory.add_turn(
                        session_id, question, cached_result["response"], sources=cached_result.get("context", [])
                    )
                total_latency = (time.perf_counter() - start_total) * 1000
                return {
                    "answer": cached_result["response"],
                    "rag_type": "hybrid_rag",
                    "sources": cached_result.get("context", []),
                    "session_id": session_id,
                    "telemetry": {
                        "total_latency_ms": round(total_latency, 2),
                        "cache_hit": True,
                        "matched_query": cached_result.get("matched_query"),
                        "similarity": cached_result.get("similarity")
                    }
                }

        # 2. Contextual Query Reformulation for multi-turn sessions
        effective_query = global_conversation_memory.contextualize_query(question, session_id)

        # 3. Dense Vector Retrieval (Qdrant)
        start_dense = time.perf_counter()
        q_vector = self.embed_provider.embed_text(effective_query)
        dense_results = self.vector_store.search(
            query_vector=q_vector,
            top_k=top_k * 2
        )
        dense_ms = (time.perf_counter() - start_dense) * 1000

        # 4. Sparse BM25 Retrieval
        start_sparse = time.perf_counter()
        sparse_results = []
        if self._bm25_scorer and self._corpus_chunks:
            bm25_hits = self._bm25_scorer.score(effective_query, top_k=top_k * 2)
            for idx, score in bm25_hits:
                chunk_data = self._corpus_chunks[idx]
                sparse_results.append({
                    "id": str(chunk_data["id"]),
                    "text": chunk_data["text"],
                    "score": score,
                    "metadata": chunk_data.get("metadata", {})
                })
        sparse_ms = (time.perf_counter() - start_sparse) * 1000

        # 5. Reciprocal Rank Fusion (RRF)
        start_fusion = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)

        # 6. Cross-Reranker
        reranked_docs = FastReRanker.rerank(effective_query, fused_candidates, top_k=top_k)
        fusion_ms = (time.perf_counter() - start_fusion) * 1000

        # 7. Context assembly & optional compression
        if compress_context:
            context_str = self.compressor.compress_chunks(effective_query, reranked_docs)
        else:
            context_parts = []
            for i, doc in enumerate(reranked_docs, start=1):
                context_parts.append(f"[{i}] {doc['text']}")
            context_str = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # Inject conversation history into context
        history_str = global_conversation_memory.format_history_for_prompt(session_id)
        if history_str:
            context_str = f"{history_str}\n\nRetrieved Passages:\n{context_str}"

        sources = [
            {
                "source_id": i + 1,
                "text": d["text"][:200] + "..." if len(d["text"]) > 200 else d["text"],
                "rrf_score": round(d.get("rrf_score", 0.0), 4),
                "rerank_score": round(d.get("rerank_score", 0.0), 4),
                "metadata": d.get("metadata", {})
            }
            for i, d in enumerate(reranked_docs)
        ]

        # 8. LLM Generation
        system_prompt, user_prompt = get_rag_prompt(question, context_str, language=language, mode=mode)
        start_gen = time.perf_counter()
        engine = get_llm_engine(backend) if backend else self.llm_engine
        answer, gen_telemetry = engine.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.2
        )
        gen_ms = (time.perf_counter() - start_gen) * 1000
        total_ms = (time.perf_counter() - start_total) * 1000

        # Record conversation turn in memory
        if session_id and answer:
            global_conversation_memory.add_turn(session_id, question, answer, sources=sources)

        # Update cache
        if use_cache and answer:
            global_semantic_cache.set(
                query=question,
                response=answer,
                query_embedding=q_vector,
                rag_type="hybrid_rag",
                context=sources
            )

        return {
            "answer": answer,
            "rag_type": "hybrid_rag",
            "sources": sources,
            "session_id": session_id,
            "telemetry": {
                "total_latency_ms": round(total_ms, 2),
                "dense_retrieval_ms": round(dense_ms, 2),
                "sparse_bm25_ms": round(sparse_ms, 2),
                "fusion_rerank_ms": round(fusion_ms, 2),
                "generation_latency_ms": round(gen_ms, 2),
                "ttft_ms": round(gen_telemetry.get("ttft_ms", 0.0), 2),
                "tokens_generated": gen_telemetry.get("tokens_generated", 0),
                "tokens_per_second": gen_telemetry.get("tokens_per_second", 0.0),
                "cache_hit": False,
                "model_used": gen_telemetry.get("model", "unknown")
            }
        }
