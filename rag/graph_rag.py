"""
Graph RAG Implementation:
Constructs and queries Knowledge Graphs stored in PostgreSQL and NetworkX.
Performs 1-hop and 2-hop relational subgraph expansion, community context aggregation,
and fused vector-graph multi-hop synthesis.
"""

import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
import networkx as nx

from .vector_store import QdrantVectorStore
from .embeddings import get_embedding_provider
from llm.engine_factory import get_llm_engine
from llm.prompts import get_rag_prompt, GRAPH_EXTRACTION_PROMPT
from database import SessionLocal, GraphEntityModel, GraphRelationModel, ChunkModel
from optimization.cache import global_semantic_cache
from rag.memory import global_conversation_memory


class GraphRAG:
    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        llm_backend: Optional[str] = None
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.embed_provider = get_embedding_provider()
        self.llm_engine = get_llm_engine(llm_backend)
        self.graph = nx.DiGraph()
        self._load_graph_from_db()

    def _load_graph_from_db(self):
        """Loads all nodes and edges from PostgreSQL/SQLite into NetworkX."""
        self.graph.clear()
        try:
            with SessionLocal() as db:
                entities = db.query(GraphEntityModel).all()
                for ent in entities:
                    self.graph.add_node(
                        ent.name,
                        entity_type=ent.entity_type,
                        description=ent.description
                    )

                relations = db.query(GraphRelationModel).all()
                for rel in relations:
                    self.graph.add_edge(
                        rel.source_entity,
                        rel.target_entity,
                        relation=rel.relation_type,
                        description=rel.description,
                        weight=rel.weight
                    )
        except Exception as e:
            print(f"[GraphRAG] Notice loading graph: {e}")

    def extract_and_index_graph(self, text: str, doc_id: Optional[int] = None) -> Dict[str, int]:
        """
        Uses LLM (or heuristic fallback) to extract knowledge graph entities and relations from text chunk,
        persists to database tables and NetworkX graph.
        """
        entities_added = 0
        relations_added = 0

        # Heuristic entity extraction if offline or quick indexing
        extracted = self._extract_triples(text)

        with SessionLocal() as db:
            for ent in extracted.get("entities", []):
                name = ent.get("name", "").strip()
                if not name:
                    continue
                ent_type = ent.get("type", "CONCEPT")
                desc = ent.get("description", "")
                
                # Check existing
                existing = db.query(GraphEntityModel).filter_by(name=name).first()
                if not existing:
                    new_ent = GraphEntityModel(
                        name=name,
                        entity_type=ent_type,
                        description=desc,
                        source_doc_id=doc_id
                    )
                    db.add(new_ent)
                    entities_added += 1

                self.graph.add_node(name, entity_type=ent_type, description=desc)

            for rel in extracted.get("relations", []):
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                if not src or not tgt or src == tgt:
                    continue
                rel_type = rel.get("relation", "RELATED_TO")
                desc = rel.get("description", "")

                new_rel = GraphRelationModel(
                    source_entity=src,
                    target_entity=tgt,
                    relation_type=rel_type,
                    description=desc,
                    weight=1.0,
                    source_doc_id=doc_id
                )
                db.add(new_rel)
                relations_added += 1

                self.graph.add_edge(src, tgt, relation=rel_type, description=desc, weight=1.0)

            db.commit()

        return {"entities": entities_added, "relations": relations_added}

    def _extract_triples(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """Performs entity and relation extraction via LLM or regex fallback."""
        prompt = GRAPH_EXTRACTION_PROMPT.format(text=text[:1200])
        try:
            raw_res, _ = self.llm_engine.generate(
                prompt=prompt,
                system_prompt="You are a Knowledge Graph extraction system. Output strictly valid JSON.",
                max_tokens=512,
                temperature=0.1
            )
            clean_json = raw_res.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r'^```(json)?\s*|\s*```$', '', clean_json, flags=re.MULTILINE)
            return json.loads(clean_json)
        except Exception:
            # Deterministic domain entity extraction fallback
            return self._heuristic_extractor(text)

    def _heuristic_extractor(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """Fast regex extractor for technical concepts and relations."""
        known_concepts = [
            "Simple RAG", "Hybrid RAG", "Graph RAG", "Qdrant", "PostgreSQL",
            "BM25", "Reciprocal Rank Fusion", "Gemini", "vLLM", "SGLang",
            "PagedAttention", "RadixAttention", "Embedding", "Chunking"
        ]
        found_entities = []
        for c in known_concepts:
            if c.lower() in text.lower():
                found_entities.append({"name": c, "type": "TECHNOLOGY", "description": f"Core component: {c}"})

        found_relations = []
        if len(found_entities) >= 2:
            for i in range(len(found_entities) - 1):
                found_relations.append({
                    "source": found_entities[i]["name"],
                    "target": found_entities[i+1]["name"],
                    "relation": "INTERACTS_WITH",
                    "description": "Found together in document context"
                })

        return {"entities": found_entities, "relations": found_relations}

    def query(
        self,
        question: str,
        top_k: int = 4,
        language: str = "en",
        mode: str = "general",
        hop_depth: int = 2,
        use_cache: bool = True,
        backend: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Graph RAG:
        1. Contextualizes query with multi-turn session memory if active.
        2. Identifies entities in query.
        3. Traverses knowledge graph (1-hop / 2-hop neighborhood expansion).
        4. Retrieves vector chunks from Qdrant.
        5. Fuses relational knowledge graph facts with vector context for multi-hop reasoning.
        """
        start_total = time.perf_counter()

        # Cache check
        if use_cache:
            cached_result = global_semantic_cache.get(question, rag_type="graph_rag")
            if cached_result:
                if session_id:
                    global_conversation_memory.add_turn(
                        session_id, question, cached_result["response"], sources=cached_result.get("context", [])
                    )
                total_latency = (time.perf_counter() - start_total) * 1000
                return {
                    "answer": cached_result["response"],
                    "rag_type": "graph_rag",
                    "sources": cached_result.get("context", []),
                    "graph_context": cached_result.get("extra", {}).get("graph_context", {"matched_entities": [], "traversed_triples": []}),
                    "session_id": session_id,
                    "telemetry": {
                        "total_latency_ms": round(total_latency, 2),
                        "cache_hit": True
                    }
                }

        # 1. Contextual Query Reformulation
        effective_query = global_conversation_memory.contextualize_query(question, session_id)

        # 2. Graph Entity Identification
        start_graph = time.perf_counter()
        matched_nodes = []
        for node in self.graph.nodes:
            if str(node).lower() in effective_query.lower():
                matched_nodes.append(node)

        # 3. Multi-hop subgraph expansion
        subgraph_triples = []
        subgraph_nodes = set(matched_nodes)

        for seed in matched_nodes:
            # 1-hop outgoing
            for _, target, data in self.graph.out_edges(seed, data=True):
                subgraph_nodes.add(target)
                subgraph_triples.append(f"({seed}) -[{data.get('relation', 'RELATED_TO')}]-> ({target})")
                if hop_depth > 1:
                    # 2-hop outgoing
                    for _, t2, d2 in self.graph.out_edges(target, data=True):
                        subgraph_nodes.add(t2)
                        subgraph_triples.append(f"({target}) -[{d2.get('relation', 'RELATED_TO')}]-> ({t2})")

            # 1-hop incoming
            for source, _, data in self.graph.in_edges(seed, data=True):
                subgraph_nodes.add(source)
                subgraph_triples.append(f"({source}) -[{data.get('relation', 'RELATED_TO')}]-> ({seed})")

        graph_ms = (time.perf_counter() - start_graph) * 1000

        # 4. Dense Vector Retrieval
        start_dense = time.perf_counter()
        q_vector = self.embed_provider.embed_text(effective_query)
        retrieved_chunks = self.vector_store.search(
            query_vector=q_vector,
            top_k=top_k
        )
        dense_ms = (time.perf_counter() - start_dense) * 1000

        # Format graph context
        if subgraph_triples:
            triples_text = "\n".join([f"- {t}" for t in list(set(subgraph_triples))[:15]])
            graph_section = f"Knowledge Graph Relations:\n{triples_text}"
        else:
            graph_section = "Knowledge Graph Relations: None directly matched in 1/2-hop neighborhood."

        # Format vector context
        text_parts = []
        sources = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            text_parts.append(f"[{i}] {chunk['text']}")
            sources.append({
                "source_id": i,
                "text": chunk["text"][:200] + "...",
                "score": round(chunk["score"], 4),
                "metadata": chunk.get("metadata", {})
            })

        text_section = "Document Passages:\n" + ("\n\n".join(text_parts) if text_parts else "No relevant passages found.")
        
        # Inject conversation history into context
        history_str = global_conversation_memory.format_history_for_prompt(session_id)
        if history_str:
            fused_context = f"{history_str}\n\n{graph_section}\n\n{text_section}"
        else:
            fused_context = f"{graph_section}\n\n{text_section}"

        # 5. LLM Synthesis
        system_prompt, user_prompt = get_rag_prompt(question, fused_context, language=language, mode=mode)
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

        graph_context_data = {
            "matched_entities": matched_nodes,
            "traversed_triples": list(set(subgraph_triples))[:15],
            "subgraph_node_count": len(subgraph_nodes)
        }

        # Record conversation turn in memory
        if session_id and answer:
            global_conversation_memory.add_turn(session_id, question, answer, sources=sources)

        # Cache answer
        if use_cache and answer:
            global_semantic_cache.set(
                query=question,
                response=answer,
                query_embedding=q_vector,
                rag_type="graph_rag",
                context=sources,
                extra={"graph_context": graph_context_data}
            )

        return {
            "answer": answer,
            "rag_type": "graph_rag",
            "sources": sources,
            "graph_context": graph_context_data,
            "session_id": session_id,
            "telemetry": {
                "total_latency_ms": round(total_ms, 2),
                "graph_traversal_ms": round(graph_ms, 2),
                "vector_search_ms": round(dense_ms, 2),
                "generation_latency_ms": round(gen_ms, 2),
                "ttft_ms": round(gen_telemetry.get("ttft_ms", 0.0), 2),
                "tokens_generated": gen_telemetry.get("tokens_generated", 0),
                "tokens_per_second": gen_telemetry.get("tokens_per_second", 0.0),
                "cache_hit": False,
                "model_used": gen_telemetry.get("model", "unknown")
            }
        }

    def get_graph_data(self) -> Dict[str, Any]:
        """Returns nodes and edges formatted for visualizers (D3, Cytoscape, Vis.js)."""
        nodes = []
        for n, d in self.graph.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": str(n),
                "type": d.get("entity_type", "CONCEPT"),
                "description": d.get("description", "")
            })

        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "relation": d.get("relation", "RELATED_TO"),
                "weight": d.get("weight", 1.0)
            })

        return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}
