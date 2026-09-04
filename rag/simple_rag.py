"""
Simple RAG Implementation:
Dense vector similarity search via Qdrant followed by LLM synthesis.
Supports English, Hindi, and Python Code Generation.
"""

import time
from typing import Dict, Any, Optional, List
from .vector_store import QdrantVectorStore
from .embeddings import get_embedding_provider
from llm.engine_factory import get_llm_engine
from llm.prompts import get_rag_prompt
from rag.scoring import compute_cosine_similarity
from optimization.cache import global_semantic_cache
from rag.memory import global_conversation_memory


class SimpleRAG:
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        llm_backend: Optional[str] = None
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.embed_provider = get_embedding_provider()
        self.llm_engine = get_llm_engine(llm_backend)

    def query(
        self,
        question: str,
        top_k: int = 4,
        language: str = "en",
        mode: str = "general",
        filter_doc_id: Optional[int] = None,
        use_cache: bool = True,
        backend: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Simple RAG query:
        Embed Query -> Qdrant Search -> Format Context -> Generate Answer.
        """
        start_total = time.perf_counter()

        # 1. Semantic Cache check
        if use_cache:
            cached_result = global_semantic_cache.get(question, rag_type="simple_rag")
            if cached_result:
                if session_id:
                    global_conversation_memory.add_turn(
                        session_id, question, cached_result["response"], sources=cached_result.get("context", [])
                    )
                total_latency = (time.perf_counter() - start_total) * 1000
                return {
                    "answer": cached_result["response"],
                    "rag_type": "simple_rag",
                    "sources": cached_result.get("context", []),
                    "session_id": session_id,
                    "telemetry": {
                        "total_latency_ms": round(total_latency, 2),
                        "cache_hit": True,
                        "matched_query": cached_result.get("matched_query"),
                        "similarity": cached_result.get("similarity")
                    }
                }

        # 2. Vector Retrieval (with contextual query reformulation if session history exists)
        effective_query = global_conversation_memory.contextualize_query(question, session_id)
        start_retrieval = time.perf_counter()
        q_vector = self.embed_provider.embed_text(effective_query)
        retrieved_docs = self.vector_store.search(
            query_vector=q_vector,
            top_k=top_k,
            filter_doc_id=filter_doc_id
        )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

        # 3. Context Construction with Conversation Memory
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc.get("metadata", {})
            src_name = meta.get("filename", "Document")
            context_parts.append(f"[{i}] Source: {src_name}\n{doc['text']}")
            sources.append({
                "source_id": i,
                "score": round(doc["score"], 4),
                "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "metadata": meta
            })

        history_str = global_conversation_memory.format_history_for_prompt(session_id)
        docs_str = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        context_str = f"{history_str}\n\nDocument Context:\n{docs_str}" if history_str else docs_str

        # 4. LLM Generation
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

        # Cache valid response
        if use_cache and answer:
            global_semantic_cache.set(
                query=question,
                response=answer,
                query_embedding=q_vector,
                rag_type="simple_rag",
                context=sources
            )

        return {
            "answer": answer,
            "rag_type": "simple_rag",
            "sources": sources,
            "telemetry": {
                "total_latency_ms": round(total_ms, 2),
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "generation_latency_ms": round(gen_ms, 2),
                "ttft_ms": round(gen_telemetry.get("ttft_ms", 0.0), 2),
                "tokens_generated": gen_telemetry.get("tokens_generated", 0),
                "tokens_per_second": gen_telemetry.get("tokens_per_second", 0.0),
                "cache_hit": False,
                "model_used": gen_telemetry.get("model", "unknown")
            }
        }
