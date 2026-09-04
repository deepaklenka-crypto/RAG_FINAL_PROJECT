# Multi-Format 3-Type RAG System Implementation Plan (Updated)

This project implements an end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) platform supporting:
1. Multi-format ingestion (**PDF, CSV, XLSX, DOCX, TXT**).
2. PostgreSQL database with automatic SQLite fallback + **Qdrant Vector DB** (local & remote).
3. **3 RAG Paradigms**: Simple RAG, Hybrid RAG (Dense + BM25 with RRF scoring), and Graph RAG (Entity-Relation knowledge graph).
4. **Multi-Level Evaluation**: Prompt-level, Response-level, and Model-level evaluations.
5. **Corruption & Robustness Analysis**: Testing resilience against typos, adversarial distractors, context truncation, and OCR noise.
6. **Latency & Speed Profiling**: Benchmarking TTFT (Time to First Token), retrieval latency, rerank time, total E2E latency, and token throughput (tokens/sec).
7. **Production Optimizations**: Semantic response caching, asynchronous parallel dual-stream retrieval, context compression/pruning, and HyDE.
8. **Multi-Engine LLM Support**: **Google Gemini API** (`gemini-2.5-flash` / `gemini-1.5-flash`), with built-in adapters for high-throughput local engines: **vLLM** (PagedAttention) and **SGLang** (RadixAttention with KV cache reuse) via OpenAI-compatible endpoints.
9. **Multilingual & Coding**: English, Hindi (`हिन्दी`), and PEP 8 Python code generation.

---

## User Review Required

> [!IMPORTANT]
> - **Inference Engine Flexibility**: The system will feature an `LLMFactory` in `llm/` that seamlessly supports:
>   - `gemini`: Google GenAI API (cloud default for reasoning, Hindi & Python coding).
>   - `vllm`: Local vLLM server via OpenAI-compatible API (`http://localhost:8000/v1`) with PagedAttention and continuous batching.
>   - `sglang`: Local SGLang server (`http://localhost:30000/v1`) with RadixAttention for ultra-fast KV cache reuse across RAG retrieved contexts.
> - **Multi-Level Evaluation & Corruption Suite**:
>   - **Prompt-level**: Token density, prompt injection guard, ambiguity score.
>   - **Response-level**: Faithfulness (hallucination detection), answer relevance, semantic similarity, format adherence.
>   - **Model-level**: Cross-model benchmark (e.g. Gemini Flash vs Pro vs local vLLM/SGLang model), output consistency, latency comparisons.
>   - **Corruption Analysis**: Noise injection test (character typos, token masking, distractor insertion) to measure RAG degradation.
> - **Latency & Profiling**: Built-in timers measuring retrieval latency, rerank latency, LLM generation time, and tokens per second.

---

## Proposed System Architecture

```
                                  +---------------------------------------+
                                  |           FastAPI Web API             |
                                  |  (Upload, Index, Query, Evaluate,     |
                                  |   Corruption Suite, Benchmarking, UI) |
                                  +-------------------+-------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
            +-------v-------+                 +-------v-------+                 +-------v-------+
            |  Simple RAG   |                 |  Hybrid RAG   |                 |   Graph RAG   |
            | (Dense Vector |                 | (Dense Qdrant |                 | (PostgreSQL   |
            |  in Qdrant)   |                 | + Sparse BM25 |                 |  + NetworkX   |
            |               |                 | + RRF Score)  |                 |  Multi-hop)   |
            +-------+-------+                 +-------+-------+                 +-------+-------+
                    |                                 |                                 |
                    +---------------------------------+---------------------------------+
                                                      |
                                     +----------------v---------------+
                                     |       Optimization Layer       |
                                     |  - Semantic Vector Cache       |
                                     |  - Async Dual-Stream Retrieval |
                                     |  - Context Pruning & HyDE      |
                                     +----------------+---------------+
                                                      |
                                     +----------------v---------------+
                                     |    Multi-Engine LLM Factory    |
                                     |  - Google Gemini API           |
                                     |  - vLLM (PagedAttention)       |
                                     |  - SGLang (RadixAttention)     |
                                     |  - English / Hindi / Python    |
                                     +----------------+---------------+
                                                      |
                                     +----------------v---------------+
                                     |   Evaluation & Benchmark Lab   |
                                     |  - Prompt / Response / Model   |
                                     |  - Corruption Stress Testing   |
                                     |  - Latency & TTFT Profiler     |
                                     +--------------------------------+
```

---

## Detailed File Structure & Implementation

### 1. Root Configuration & Dependencies
- `requirements.txt`: FastAPI, Uvicorn, Qdrant-Client, Google-GenAI, PyPDF, Python-Docx, OpenPyXL, Pandas, Rank-BM25, SQLAlchemy, Psycopg2-Binary, NetworkX, Scikit-Learn, Python-Dotenv, Httpx, Pydantic.
- `.env.example`: Configuration parameters for Gemini API, PostgreSQL, Qdrant, vLLM, and SGLang endpoints.
- `database.py`: SQLAlchemy schema for PostgreSQL / SQLite:
  - `documents`, `document_chunks`, `graph_entities`, `graph_relations`, `query_logs`, `eval_logs`, `benchmark_logs`.

### 2. Ingestion & Preprocessing
- `index.py`: CLI and automated ingestion pipeline for PDF, CSV, XLSX, DOCX, TXT.
- `data/`:
  - `data/uploads/`: Raw uploaded documents.
  - `data/qdrant_storage/`: Local Qdrant persistent database.
  - `data/samples/`: Sample multi-format files for immediate test verification.

### 3. RAG Core Modules (`rag/`)
- `rag/chunking.py`:
  - `RecursiveCharacterChunker` (sliding window with configurable overlap).
  - `SemanticSentenceChunker` (boundary & semantic distance breakpoint detection).
  - `StructuredDocumentChunker` (table row serialization for CSV/XLSX; heading/paragraph structure for DOCX/PDF).
- `rag/embeddings.py`:
  - `GeminiEmbeddingProvider` (`text-embedding-004`).
  - `LocalFallbackEmbeddingProvider` (deterministic feature hashing vector provider).
- `rag/vector_store.py`:
  - `QdrantVectorStore` (collection lifecycle, payload filtering, cosine & dot-product search).
- `rag/scoring.py`:
  - Dense cosine similarity, BM25 score normalization, Reciprocal Rank Fusion (RRF: $\sum \frac{1}{60 + r}$), and LLM cross-reranking scoring.
- `rag/simple_rag.py`: Dense single-stream vector retrieval + prompt augmentation.
- `rag/hybrid_rag.py`: Asynchronous parallel retrieval (Qdrant dense + BM25 sparse) fused via RRF.
- `rag/graph_rag.py`: Entity-relationship extraction, subgraph expansion (1-hop / 2-hop traversal via NetworkX + PostgreSQL), combining relational triples with semantic vectors for multi-hop questions.

### 4. LLM Layer & vLLM / SGLang Integration (`llm/`)
- `llm/engine_factory.py`:
  - `GeminiEngine`: Google GenAI integration with retry logic and streaming.
  - `VLLMEngine`: OpenAI-compatible client configured for vLLM endpoints (`http://localhost:8000/v1`), utilizing PagedAttention.
  - `SGLangEngine`: OpenAI-compatible client optimized for SGLang (`http://localhost:30000/v1`), taking advantage of RadixAttention prefix caching.
- `llm/prompts.py`:
  - Multilingual templates: English, Hindi (`हिन्दी`), and bilingual instructions.
  - Python coding prompt template: Clean PEP 8 code, docstrings, type hints, and runnable verification tests.
  - Entity-relation extraction prompts for Graph RAG.

### 5. Optimization Layer (`optimization/`)
- `optimization/cache.py`: **Semantic Response Cache** using vector similarity. Similar queries get served in <10ms without hitting the LLM.
- `optimization/query_rewriter.py`: Query expansion, sub-query decomposition, and keyword enhancement.
- `optimization/hyde.py`: Hypothetical Document Embeddings for zero-shot domain adaptation.
- `optimization/compression.py`: Context pruning and extractive sentence compression to minimize token overhead and latency.

### 6. Evaluation, Corruption & Latency Suite (`evaluation/`)
- `evaluation/prompt_eval.py`: **Prompt-level evaluation** (clarity, token length efficiency, injection resistance).
- `evaluation/response_eval.py`: **Response-level evaluation** (faithfulness, hallucination rate, answer relevancy, completeness, format adherence).
- `evaluation/model_eval.py`: **Model-level evaluation** (comparative benchmarking between Gemini Flash, Gemini Pro, and local vLLM/SGLang).
- `evaluation/corruption.py`: **Corruption Analysis**:
  - Context corruption injection (typos, word omission, sentence shuffling, distractor chunk injection).
  - Robustness score: measures how well the RAG pipeline answers when given 0%, 25%, 50%, and 75% corrupted context.
- `evaluation/latency_profiler.py`:
  - Fine-grained timing breakdown: retrieval time, reranking time, TTFT (Time to First Token), generation time, tokens per second (TPS).

### 7. FastAPI Endpoints & UI (`main.py`)
- REST API:
  - `POST /api/upload`: Upload PDF, CSV, XLSX, DOCX, TXT.
  - `POST /api/index`: Trigger indexing with chunking & embedding selections.
  - `POST /api/query`: Execute query with `rag_type` (`simple`, `hybrid`, `graph`), `engine` (`gemini`, `vllm`, `sglang`), `language` (`en`, `hi`), and `mode` (`general`, `code`).
  - `POST /api/evaluate`: Comprehensive multi-level evaluation.
  - `POST /api/corruption-test`: Run corruption stress test and get resilience report.
  - `GET /api/benchmark`: Retrieve latency and throughput profiles.
  - `GET /api/graph`: Graph nodes and edges for Knowledge Graph visualizer.
  - `GET /`: Modern Glassmorphic dashboard UI to test all features visually.

---

## Verification Plan

### Automated Verification
1. Test document parsers with all 5 formats (PDF, CSV, XLSX, DOCX, TXT).
2. Test chunking strategies (`recursive`, `semantic`, `structured`).
3. Test Qdrant vector database storage and query retrieval.
4. Test Hybrid RAG RRF fusion and BM25 scoring.
5. Test Graph RAG entity extraction, network expansion, and multi-hop reasoning.
6. Test Prompt-level, Response-level, and Model-level evaluation functions.
7. Test Corruption Analysis suite with simulated noisy contexts.
8. Test Latency Profiler and semantic cache hit rate.
9. Verify FastAPI endpoints with an automated test suite (`tests/test_all.py`).

### Manual Verification
1. Run `python main.py` or `uvicorn main:app --reload`.
2. Open `http://127.0.0.1:8000` to interact with the web dashboard.
3. Test querying in English and Hindi, requesting Python code generation, and running corruption tests.
