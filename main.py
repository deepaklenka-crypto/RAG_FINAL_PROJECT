"""
FastAPI Application Entrypoint:
Exposes RESTful API endpoints for Multi-Format 3-Type RAG, Document Upload,
Multi-Level Evaluation, Corruption Analysis, Latency Profiling, and Knowledge Graph Visualization.
Includes an embedded sleek Glassmorphism Web Dashboard.
"""

import os
import shutil
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from database import (
    init_db, SessionLocal, get_db,
    DocumentModel, ChunkModel, QueryLogModel, EvaluationLogModel, BenchmarkLogModel
)
from index import index_document, DocumentParser
from rag.simple_rag import SimpleRAG
from rag.hybrid_rag import HybridRAG
from rag.graph_rag import GraphRAG
from rag.vector_store import QdrantVectorStore
from evaluation.evaluator import RAGEvaluator
from evaluation.corruption import CorruptionAnalyzer
from evaluation.latency_profiler import LatencyProfiler
from optimization.cache import global_semantic_cache
from rag.memory import global_conversation_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables and storage directories exist
    init_db()
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data/qdrant_storage", exist_ok=True)
    yield
    # Shutdown


app = FastAPI(
    title="Multi-Format 3-Type RAG Platform",
    description="Enterprise RAG supporting Simple, Hybrid, and Graph RAG with PostgreSQL, Qdrant, Gemini, vLLM, and SGLang.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global pipeline lazy singletons
_vector_store = None
_simple_rag = None
_hybrid_rag = None
_graph_rag = None
evaluator = RAGEvaluator()
corruption_analyzer = CorruptionAnalyzer()

def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store

def get_simple_rag() -> SimpleRAG:
    global _simple_rag
    if _simple_rag is None:
        _simple_rag = SimpleRAG(vector_store=get_vector_store())
    return _simple_rag

def get_hybrid_rag() -> HybridRAG:
    global _hybrid_rag
    if _hybrid_rag is None:
        _hybrid_rag = HybridRAG(vector_store=get_vector_store())
    return _hybrid_rag

def get_graph_rag() -> GraphRAG:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = GraphRAG(vector_store=get_vector_store())
    return _graph_rag


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the three types of RAG and their differences?"})
    rag_type: str = Field("hybrid", description="'simple', 'hybrid', or 'graph'")
    backend: Optional[str] = Field("gemini", description="'gemini', 'vllm', or 'sglang'")
    language: str = Field("en", description="'en' for English or 'hi' for Hindi")
    mode: str = Field("general", description="'general' or 'code'")
    top_k: int = Field(4, ge=1, le=20)
    use_cache: bool = Field(True)
    compress_context: bool = Field(False)
    hop_depth: int = Field(2, ge=1, le=3)
    session_id: Optional[str] = Field(None, description="Conversational session ID for multi-turn chat memory")


class EvaluationRequest(BaseModel):
    query: str
    response: str
    context_passages: List[str]
    rag_type: str = "hybrid"
    language: str = "en"
    mode: str = "general"
    run_corruption_test: bool = False


class CorruptionTestRequest(BaseModel):
    query: str
    clean_context: str
    noise_levels: List[float] = [0.0, 0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    vs = get_vector_store()
    gr = get_graph_rag()
    return {
        "status": "online",
        "service": "Multi-Format 3-Type RAG API",
        "qdrant_points": vs.count(),
        "semantic_cache_size": global_semantic_cache.size(),
        "graph_nodes": gr.graph.number_of_nodes(),
        "graph_edges": gr.graph.number_of_edges()
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    embedding_strategy: str = Form("gemini"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    extract_graph: bool = Form(True)
):
    """
    Upload and index PDF, CSV, XLSX, DOCX, or TXT documents.
    """
    allowed_extensions = ["pdf", "csv", "xlsx", "xls", "docx", "doc", "txt"]
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{ext}. Supported formats: {', '.join(allowed_extensions)}"
        )

    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = index_document(
            file_path=file_path,
            chunking_strategy=chunking_strategy,
            embedding_strategy=embedding_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_graph=extract_graph
        )
        # Refresh hybrid BM25 and graph
        get_hybrid_rag().refresh_index()
        get_graph_rag()._load_graph_from_db()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")


@app.post("/api/query")
def execute_query(req: QueryRequest):
    """
    Executes a query against the selected RAG pipeline (Simple, Hybrid, Graph).
    Supports English, Hindi, and Python Code Mode.
    """
    rag_choice = req.rag_type.lower()
    start_t = time.perf_counter()

    if rag_choice == "simple":
        res = get_simple_rag().query(
            question=req.query,
            top_k=req.top_k,
            language=req.language,
            mode=req.mode,
            use_cache=req.use_cache,
            backend=req.backend,
            session_id=req.session_id
        )
    elif rag_choice == "graph":
        res = get_graph_rag().query(
            question=req.query,
            top_k=req.top_k,
            language=req.language,
            mode=req.mode,
            hop_depth=req.hop_depth,
            use_cache=req.use_cache,
            backend=req.backend,
            session_id=req.session_id
        )
    else:  # default to hybrid
        res = get_hybrid_rag().query(
            question=req.query,
            top_k=req.top_k,
            language=req.language,
            mode=req.mode,
            compress_context=req.compress_context,
            use_cache=req.use_cache,
            backend=req.backend,
            session_id=req.session_id
        )

    # Telemetry logging to database
    telemetry = res.get("telemetry", {})
    try:
        with SessionLocal() as db:
            log_entry = QueryLogModel(
                query_text=req.query,
                response_text=res.get("answer", ""),
                rag_type=req.rag_type,
                llm_backend=req.backend or "gemini",
                language=req.language,
                mode=req.mode,
                total_latency_ms=telemetry.get("total_latency_ms", 0.0),
                retrieval_latency_ms=telemetry.get("retrieval_latency_ms", telemetry.get("dense_retrieval_ms", 0.0)),
                generation_latency_ms=telemetry.get("generation_latency_ms", 0.0),
                chunks_retrieved=len(res.get("sources", [])),
                tokens_generated=telemetry.get("tokens_generated", 0),
                tokens_per_second=telemetry.get("tokens_per_second", 0.0),
                cache_hit=telemetry.get("cache_hit", False)
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            res["query_id"] = log_entry.id
    except Exception as e:
        print(f"[API Query] Notice logging query: {e}")

    return res


@app.post("/api/evaluate")
def run_evaluation(req: EvaluationRequest):
    """
    Executes Prompt-Level, Response-Level, and optional Corruption evaluation.
    """
    results = evaluator.evaluate_turn(
        query=req.query,
        response=req.response,
        context_passages=req.context_passages,
        rag_type=req.rag_type,
        language=req.language,
        mode=req.mode,
        include_corruption_test=req.run_corruption_test
    )
    return results


@app.post("/api/corruption-test")
def run_corruption_analysis(req: CorruptionTestRequest):
    """
    Stress-tests context degradation under 0%, 25%, 50%, 75% noise.
    """
    report = corruption_analyzer.run_stress_test(
        query=req.query,
        clean_context=req.clean_context,
        noise_levels=req.noise_levels
    )
    return report


@app.get("/api/documents")
def get_documents():
    """Returns list of indexed documents."""
    with SessionLocal() as db:
        docs = db.query(DocumentModel).order_by(DocumentModel.id.desc()).all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "chunk_count": d.chunk_count,
                "chunking_strategy": d.chunking_strategy,
                "embedding_strategy": d.embedding_strategy,
                "status": d.status,
                "created_at": d.created_at.strftime("%Y-%m-%d %H:%M:%S") if d.created_at else ""
            }
            for d in docs
        ]


@app.get("/api/graph")
def get_knowledge_graph():
    """Returns nodes and edges for Knowledge Graph visualizer."""
    return get_graph_rag().get_graph_data()


@app.get("/api/benchmark")
def get_benchmark_statistics():
    """Returns latency, TTFT, and tokens/sec telemetry statistics."""
    return LatencyProfiler.get_summary_statistics()


@app.post("/api/cache/clear")
def clear_cache():
    """Clears the semantic vector cache."""
    global_semantic_cache.clear()
    return {"status": "cleared", "cache_size": 0}


@app.get("/api/sessions")
def list_sessions():
    """Lists all active and stored conversational chat sessions."""
    return global_conversation_memory.list_sessions()


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    """Retrieves conversation turn history for a session."""
    return global_conversation_memory.get_history(session_id, window_size=50)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Clears conversation memory for a session."""
    global_conversation_memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Interactive Web Dashboard UI (Glassmorphic)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OmniRAG Studio - Enterprise 3-Type RAG Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-primary: #0a0d14;
      --bg-secondary: #101622;
      --card-bg: rgba(22, 30, 46, 0.65);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #00f2fe;
      --accent-blue: #4facfe;
      --accent-purple: #7928ca;
      --accent-emerald: #10b981;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --font-sans: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08), transparent 35%),
                  radial-gradient(circle at 85% 85%, rgba(121, 40, 202, 0.12), transparent 45%),
                  var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Header */
    header {
      border-bottom: 1px solid var(--card-border);
      background: rgba(10, 13, 20, 0.75);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo-box {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #fff;
      font-size: 1.1rem;
      box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    }
    .logo-title {
      font-weight: 700;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #fff, var(--accent-cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .nav-tabs {
      display: flex;
      gap: 0.5rem;
      background: rgba(255, 255, 255, 0.04);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .tab-btn.active, .tab-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
    }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(121, 40, 202, 0.2));
      border: 1px solid rgba(79, 172, 254, 0.3);
      color: var(--accent-cyan);
    }

    /* Main Container */
    main {
      flex: 1;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease-out; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Cards */
    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      position: relative;
    }

    /* Grid layout */
    .grid-2 {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 1.5rem;
    }
    @media (max-width: 1024px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    /* Form controls */
    .form-group {
      margin-bottom: 1.25rem;
    }
    label {
      display: block;
      font-size: 0.825rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    input[type="text"], textarea, select {
      width: 100%;
      background: rgba(10, 13, 20, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 16px;
      color: var(--text-main);
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type="text"]:focus, textarea:focus, select:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 12px rgba(0, 242, 254, 0.2);
    }
    textarea { resize: vertical; min-height: 100px; }

    /* Button */
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      color: #000;
      font-weight: 700;
      border: none;
      border-radius: 10px;
      padding: 12px 24px;
      font-size: 0.95rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(0, 242, 254, 0.45);
    }

    /* Badge & Tag */
    .tag {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .tag-cyan { background: rgba(0, 242, 254, 0.15); color: var(--accent-cyan); border: 1px solid rgba(0, 242, 254, 0.3); }
    .tag-purple { background: rgba(121, 40, 202, 0.15); color: #c084fc; border: 1px solid rgba(121, 40, 202, 0.3); }
    .tag-emerald { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Telemetry grid */
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-top: 1rem;
    }
    .metric-card {
      background: rgba(10, 13, 20, 0.6);
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
      text-align: center;
    }
    .metric-label { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 2px; }
    .metric-val { font-size: 1.1rem; font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono); }

    /* Answer display */
    .answer-box {
      margin-top: 1.5rem;
      background: rgba(10, 13, 20, 0.9);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.5rem;
      line-height: 1.65;
      font-size: 0.98rem;
    }
    .code-block {
      background: #0d1117;
      border-radius: 8px;
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.875rem;
      overflow-x: auto;
      border: 1px solid rgba(255, 255, 255, 0.1);
      margin-top: 1rem;
    }

    /* Table styles */
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 0.9rem; }
    th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }

    /* Conversational Chat styling */
    .chat-container {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 0.5rem;
      margin-top: 1rem;
    }
    .chat-bubble-user {
      align-self: flex-end;
      background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(121, 40, 202, 0.25));
      border: 1px solid rgba(79, 172, 254, 0.4);
      border-radius: 14px 14px 2px 14px;
      padding: 0.75rem 1.1rem;
      max-width: 82%;
      font-size: 0.95rem;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .chat-bubble-assistant {
      align-self: flex-start;
      background: rgba(10, 13, 20, 0.9);
      border: 1px solid var(--card-border);
      border-radius: 14px 14px 14px 2px;
      padding: 1.1rem 1.25rem;
      max-width: 92%;
      font-size: 0.95rem;
      line-height: 1.65;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
    .chat-meta {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .session-bar {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 1.25rem;
      background: rgba(0, 0, 0, 0.35);
      padding: 0.75rem 1rem;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      flex-wrap: wrap;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="logo-box">
      <div class="logo-badge">R</div>
      <div>
        <div class="logo-title">OmniRAG Studio</div>
        <div style="font-size: 0.7rem; color: var(--text-muted);">Simple &bull; Hybrid &bull; Graph &bull; Gemini / vLLM / SGLang</div>
      </div>
    </div>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('query-tab')">Query Studio</button>
      <button class="tab-btn" onclick="switchTab('upload-tab')">Document Ingestion</button>
      <button class="tab-btn" onclick="switchTab('graph-tab')">Knowledge Graph</button>
      <button class="tab-btn" onclick="switchTab('eval-tab')">Evaluation & Corruption</button>
      <button class="tab-btn" onclick="switchTab('telemetry-tab')">Latency & Profiling</button>
    </div>
  </header>

  <!-- Main Container -->
  <main>

    <!-- 1. Query Studio -->
    <section id="query-tab" class="tab-content active">
      <!-- Session & Cache Management Bar -->
      <div class="session-bar">
        <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan); display: flex; align-items: center; gap: 6px;">
          💬 Conversational Memory:
        </span>
        <select id="session-select" onchange="onSessionChange()" style="flex: 1; min-width: 220px; padding: 6px 12px; font-size: 0.85rem;">
          <option value="session_default">Default Session (Persistent Memory)</option>
        </select>
        <button type="button" class="btn-primary" style="padding: 6px 14px; font-size: 0.8rem;" onclick="createNewSession()">
          + New Chat
        </button>
        <button type="button" class="btn-primary" style="padding: 6px 14px; font-size: 0.8rem; background: rgba(244,63,94,0.15); border: 1px solid rgba(244,63,94,0.35); color: #f43f5e;" onclick="deleteCurrentSession()">
          Clear Memory
        </button>
        <button type="button" class="btn-primary" style="padding: 6px 14px; font-size: 0.8rem; background: rgba(121,40,202,0.15); border: 1px solid rgba(121,40,202,0.35); color: #c084fc;" onclick="clearSemanticCache()">
          Flush Cache
        </button>
      </div>

      <div class="grid-2">
        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Ask Your Knowledge Base</h2>
          
          <div class="form-group">
            <label>Question / Follow-up Query</label>
            <textarea id="query-input" placeholder="e.g. What is Hybrid RAG? (Follow-up: How does its latency compare to Simple RAG?)"></textarea>
          </div>

          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.25rem;">
            <div class="form-group">
              <label>RAG Architecture</label>
              <select id="rag-type-select">
                <option value="hybrid" selected>Hybrid RAG (Dense + BM25 + RRF)</option>
                <option value="simple">Simple RAG (Dense Vector)</option>
                <option value="graph">Graph RAG (Entity-Relation KG)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Language Output</label>
              <select id="language-select">
                <option value="en" selected>English (EN)</option>
                <option value="hi">हिन्दी (Hindi)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Output Mode</label>
              <select id="mode-select">
                <option value="general" selected>General Analysis</option>
                <option value="code">Python Code Generation</option>
              </select>
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.25rem;">
            <div class="form-group">
              <label>Inference Engine</label>
              <select id="backend-select">
                <option value="gemini" selected>Google Gemini (Cloud)</option>
                <option value="ollama:qwen2.5:3b">Ollama: Qwen 2.5 3B (Local Laptop)</option>
                <option value="ollama:qwen2.5vl:7b">Ollama: Qwen 2.5 6GB/7B (Local Laptop)</option>
                <option value="ollama:gemma2:2b">Ollama: Gemma 2 2B (Local Laptop)</option>
                <option value="vllm">vLLM (PagedAttention - Remote GPU)</option>
                <option value="sglang">SGLang (RadixAttention - Remote GPU)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Semantic Cache</label>
              <select id="cache-select">
                <option value="true" selected>Enabled (<10ms)</option>
                <option value="false">Bypass Cache</option>
              </select>
            </div>
            <div class="form-group">
              <label>Context Pruning</label>
              <select id="compression-select">
                <option value="false" selected>Disabled</option>
                <option value="true">Enable Compression</option>
              </select>
            </div>
          </div>

          <button class="btn-primary" id="query-submit-btn" onclick="executeRAGQuery()">
            <span>Execute Retrieval</span> &rarr;
          </button>
        </div>

        <!-- Telemetry & Graph preview -->
        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 0.75rem;">Real-Time Retrieval Telemetry</h2>
          
          <div class="telemetry-grid">
            <div class="metric-card">
              <div class="metric-label">Total Latency</div>
              <div class="metric-val" id="metric-latency">0 ms</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">TTFT (First Token)</div>
              <div class="metric-val" id="metric-ttft">0 ms</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Tokens / Sec</div>
              <div class="metric-val" id="metric-tps">0</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Cache Status</div>
              <div class="metric-val" id="metric-cache" style="font-size: 0.9rem;">None</div>
            </div>
          </div>

          <h3 style="font-size: 0.95rem; margin-top: 1.5rem; margin-bottom: 0.5rem;">Retrieved Sources & Scoring</h3>
          <div id="sources-container" style="max-height: 260px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="color: var(--text-muted); font-size: 0.85rem;">Execute a query to inspect citations and scores.</div>
          </div>
        </div>
      </div>

      <!-- Multi-Turn Conversational Stream -->
      <div class="glass-card" style="margin-top: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 0.75rem;">
          <div style="display: flex; align-items: center; gap: 0.75rem;">
            <h3 style="font-size: 1.1rem;">Conversation Thread</h3>
            <span id="session-badge" class="tag tag-purple">Session: session_default</span>
          </div>
          <span id="active-rag-tag" class="tag tag-cyan">Hybrid RAG</span>
        </div>
        <div id="chat-history-container" class="chat-container">
          <div class="chat-bubble-assistant">
            <div class="chat-meta">
              <span class="tag tag-cyan">System</span>
              <span>Memory & Cache Ready</span>
            </div>
            Ask any question above! Multi-turn conversational memory is enabled, so you can ask follow-ups referencing earlier turns (e.g. "tell me more about it", "compare its speed").
          </div>
        </div>
      </div>
    </section>

    <!-- 2. Document Ingestion -->
    <section id="upload-tab" class="tab-content">
      <div class="grid-2">
        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Upload & Index Documents</h2>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.25rem;">
            Supports <b>PDF, CSV, XLSX, DOCX, and TXT</b>. Documents are parsed, split using selected chunking, embedded, and stored in Qdrant Vector DB & PostgreSQL.
          </p>

          <form id="upload-form" onsubmit="handleUpload(event)">
            <div class="form-group">
              <label>Select File</label>
              <input type="file" id="file-upload-input" required style="padding: 10px; background: rgba(0,0,0,0.5); border: 1px dashed var(--card-border); border-radius: 8px;" />
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
              <div class="form-group">
                <label>Chunking Technique</label>
                <select id="chunk-tech-select">
                  <option value="recursive" selected>Recursive Character</option>
                  <option value="semantic">Semantic Sentence</option>
                  <option value="structured">Structured (Row / Section)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Embedding Engine</label>
                <select id="embed-tech-select">
                  <option value="gemini" selected>Gemini text-embedding-004</option>
                  <option value="local">Deterministic Local Fallback</option>
                </select>
              </div>
            </div>

            <button type="submit" class="btn-primary" id="upload-btn">
              <span>Upload & Index Document</span>
            </button>
            <span id="upload-status" style="margin-left: 1rem; font-size: 0.875rem;"></span>
          </form>
        </div>

        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Indexed Knowledge Base Documents</h2>
          <div style="max-height: 400px; overflow-y: auto;">
            <table id="doc-table">
              <thead>
                <tr>
                  <th>Doc ID</th>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Chunks</th>
                  <th>Strategy</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody id="doc-tbody">
                <tr><td colspan="6" style="text-align: center;">Loading documents...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- 3. Knowledge Graph -->
    <section id="graph-tab" class="tab-content">
      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
          <div>
            <h2 style="font-size: 1.2rem;">Relational Knowledge Graph</h2>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Extracted entities and multi-hop relationships stored in PostgreSQL & NetworkX.</div>
          </div>
          <button class="btn-primary" onclick="loadKnowledgeGraph()">Refresh Graph</button>
        </div>

        <div id="graph-summary" style="display: flex; gap: 1.5rem; margin-bottom: 1.5rem;">
          <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Graph Nodes (Entities)</div>
            <div class="metric-val" id="graph-node-count">0</div>
          </div>
          <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Graph Edges (Relations)</div>
            <div class="metric-val" id="graph-edge-count">0</div>
          </div>
        </div>

        <div style="max-height: 480px; overflow-y: auto;">
          <table>
            <thead>
              <tr>
                <th>Source Entity</th>
                <th>Relationship</th>
                <th>Target Entity</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody id="graph-edges-tbody">
              <tr><td colspan="4" style="text-align: center;">Click Refresh to load knowledge graph relationships.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- 4. Evaluation & Corruption -->
    <section id="eval-tab" class="tab-content">
      <div class="grid-2">
        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Multi-Level Evaluation & Stress Testing</h2>
          
          <div class="form-group">
            <label>Evaluation Target Query</label>
            <input type="text" id="eval-query" value="What are the key benefits of Hybrid RAG over Simple RAG?" />
          </div>

          <div class="form-group">
            <label>Context Passages (Sample)</label>
            <textarea id="eval-context">Hybrid RAG combines dense semantic vector search via Qdrant with sparse BM25 lexical token matching. The outputs are fused using Reciprocal Rank Fusion (RRF) with constant k=60 to normalize rank discrepancies.</textarea>
          </div>

          <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <button class="btn-primary" onclick="runComprehensiveEvaluation()">
              Run Full Evaluation
            </button>
            <button class="btn-primary" style="background: linear-gradient(135deg, #f43f5e, #be123c); color: #fff;" onclick="runCorruptionStressTest()">
              Run Corruption Test
            </button>
          </div>
        </div>

        <div class="glass-card">
          <h2 style="font-size: 1.2rem; margin-bottom: 1rem;">Evaluation Diagnostic Metrics</h2>
          <div id="eval-results-container" style="line-height: 1.6; font-size: 0.9rem;">
            <p style="color: var(--text-muted);">Run evaluation to view Prompt-level, Response-level, and Corruption Robustness metrics.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 5. Latency & Profiling -->
    <section id="telemetry-tab" class="tab-content">
      <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
          <div>
            <h2 style="font-size: 1.2rem;">Average Latency & Speed Telemetry</h2>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Aggregated latency percentiles and token throughput across all queries.</div>
          </div>
          <button class="btn-primary" onclick="loadTelemetryStats()">Refresh Stats</button>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
          <div class="metric-card">
            <div class="metric-label">Average Total Latency</div>
            <div class="metric-val" id="stat-avg-lat">0 ms</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">P95 Latency</div>
            <div class="metric-val" id="stat-p95-lat">0 ms</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Avg Retrieval Time</div>
            <div class="metric-val" id="stat-retrieval-lat">0 ms</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Avg Tokens / Sec</div>
            <div class="metric-val" id="stat-tps">0</div>
          </div>
        </div>

        <div style="margin-top: 2rem;">
          <h3 style="font-size: 1rem; margin-bottom: 1rem;">Optimization Highlights</h3>
          <ul style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.8; margin-left: 1.5rem;">
            <li><b>Semantic Vector Cache</b>: Sub-10ms response time for semantically identical queries.</li>
            <li><b>Dual-Stream Hybrid Retrieval</b>: Parallel dense vector and BM25 token matching with Reciprocal Rank Fusion.</li>
            <li><b>Context Compression & Pruning</b>: Drops redundant sentences to minimize prompt token count and latency.</li>
            <li><b>RadixAttention & PagedAttention Support</b>: Optimized for vLLM and SGLang high-throughput local engines.</li>
          </ul>
        </div>
      </div>
    </section>

  </main>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (targetBtn) targetBtn.classList.add('active');
      const targetContent = document.getElementById(tabId);
      if (targetContent) targetContent.classList.add('active');

      if (tabId === 'upload-tab') loadDocuments();
      if (tabId === 'graph-tab') loadKnowledgeGraph();
      if (tabId === 'telemetry-tab') loadTelemetryStats();
    }

    let activeSessionId = "session_default";

    async function loadSessions() {
      try {
        const res = await fetch('/api/sessions');
        const sessions = await res.json();
        const sel = document.getElementById('session-select');
        sel.innerHTML = "";
        
        let hasActive = false;
        sessions.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.session_id;
          opt.textContent = `${s.session_id} (${s.message_count} msgs)`;
          if (s.session_id === activeSessionId) {
            opt.selected = true;
            hasActive = true;
          }
          sel.appendChild(opt);
        });

        if (!hasActive) {
          const defaultOpt = document.createElement('option');
          defaultOpt.value = activeSessionId;
          defaultOpt.textContent = `${activeSessionId} (Active)`;
          defaultOpt.selected = true;
          sel.insertBefore(defaultOpt, sel.firstChild);
        }

        document.getElementById('session-badge').innerText = "Session: " + activeSessionId;
      } catch (err) {
        console.error("Error loading sessions:", err);
      }
    }

    async function onSessionChange() {
      const sel = document.getElementById('session-select');
      activeSessionId = sel.value;
      document.getElementById('session-badge').innerText = "Session: " + activeSessionId;
      loadSessionHistory(activeSessionId);
    }

    async function createNewSession() {
      const newId = "session_" + Math.random().toString(36).substring(2, 9);
      activeSessionId = newId;
      const sel = document.getElementById('session-select');
      const opt = document.createElement('option');
      opt.value = newId;
      opt.textContent = `${newId} (New)`;
      opt.selected = true;
      sel.appendChild(opt);
      document.getElementById('session-badge').innerText = "Session: " + newId;

      const container = document.getElementById('chat-history-container');
      container.innerHTML = `
        <div class="chat-bubble-assistant">
          <div class="chat-meta">
            <span class="tag tag-cyan">System</span>
            <span>New Session Initialized</span>
          </div>
          Started new chat session <b>${newId}</b>. Ask questions, and conversational memory will track follow-up context.
        </div>
      `;
    }

    async function deleteCurrentSession() {
      if (!confirm(`Clear memory for session "${activeSessionId}"?`)) return;
      try {
        await fetch(`/api/sessions/${activeSessionId}`, { method: 'DELETE' });
        createNewSession();
        loadSessions();
      } catch (err) {
        alert("Error clearing session: " + err);
      }
    }

    async function clearSemanticCache() {
      try {
        const res = await fetch('/api/cache/clear', { method: 'POST' });
        const d = await res.json();
        document.getElementById('metric-cache').innerText = "Flushed";
        alert("Semantic vector cache flushed successfully!");
      } catch (err) {
        alert("Error flushing cache: " + err);
      }
    }

    async function loadSessionHistory(sessionId) {
      const container = document.getElementById('chat-history-container');
      try {
        const res = await fetch(`/api/sessions/${sessionId}/history`);
        const history = await res.json();
        if (!history || history.length === 0) {
          container.innerHTML = `
            <div class="chat-bubble-assistant">
              <div class="chat-meta">
                <span class="tag tag-cyan">System</span>
                <span>Session Active</span>
              </div>
              No messages in this session yet. Ask a question to begin!
            </div>
          `;
          return;
        }

        container.innerHTML = "";
        history.forEach(msg => {
          if (msg.role === 'user') {
            container.innerHTML += `
              <div class="chat-bubble-user">
                <div class="chat-meta" style="justify-content: flex-end;">
                  <span style="color: var(--accent-cyan);">You</span>
                  <span>${msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}</span>
                </div>
                ${msg.content.replace(/\\n/g, '<br/>')}
              </div>
            `;
          } else {
            container.innerHTML += `
              <div class="chat-bubble-assistant">
                <div class="chat-meta">
                  <span class="tag tag-purple">Assistant</span>
                  <span>${msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}</span>
                </div>
                ${formatResponseText(msg.content)}
              </div>
            `;
          }
        });
        container.scrollTop = container.scrollHeight;
      } catch (err) {
        console.error("Error loading session history:", err);
      }
    }

    function formatResponseText(rawText) {
      if (!rawText) return "No response received.";
      if (rawText.includes("```")) {
        return rawText.replace(/```python([\\s\\S]*?)```/g, '<div class="code-block"><code>$1</code></div>')
                      .replace(/```([\\s\\S]*?)```/g, '<div class="code-block"><code>$1</code></div>');
      }
      return rawText.replace(/\\n/g, '<br/>');
    }

    async function executeRAGQuery() {
      const qInput = document.getElementById('query-input');
      const q = qInput.value.trim();
      if (!q) return alert("Please enter a question.");

      const ragType = document.getElementById('rag-type-select').value;
      const lang = document.getElementById('language-select').value;
      const mode = document.getElementById('mode-select').value;
      const backend = document.getElementById('backend-select').value;
      const useCache = document.getElementById('cache-select').value === 'true';
      const compress = document.getElementById('compression-select').value === 'true';
      const submitBtn = document.getElementById('query-submit-btn');

      document.getElementById('active-rag-tag').innerText = ragType.toUpperCase() + " RAG";
      submitBtn.disabled = true;
      submitBtn.innerText = "Retrieving & Synthesizing...";

      const container = document.getElementById('chat-history-container');
      const timeStr = new Date().toLocaleTimeString();

      // Append user bubble
      container.innerHTML += `
        <div class="chat-bubble-user">
          <div class="chat-meta" style="justify-content: flex-end;">
            <span style="color: var(--accent-cyan);">You</span>
            <span>${timeStr}</span>
          </div>
          ${q.replace(/\\n/g, '<br/>')}
        </div>
      `;

      // Temporary assistant pending bubble
      const pendingBubbleId = "pending-" + Date.now();
      container.innerHTML += `
        <div id="${pendingBubbleId}" class="chat-bubble-assistant">
          <div class="chat-meta">
            <span class="tag tag-cyan">${ragType.toUpperCase()} RAG</span>
            <span>Thinking & synthesizing...</span>
          </div>
          <em style="color: var(--text-muted);">Querying vector store / graph and generating grounded answer...</em>
        </div>
      `;
      container.scrollTop = container.scrollHeight;

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            rag_type: ragType,
            backend: backend,
            language: lang,
            mode: mode,
            top_k: 4,
            use_cache: useCache,
            compress_context: compress,
            session_id: activeSessionId
          })
        });
        const data = await res.json();
        
        // Render answer into bubble
        const pendingEl = document.getElementById(pendingBubbleId);
        if (pendingEl) {
          const tel = data.telemetry || {};
          const cacheBadge = tel.cache_hit ? '<span class="tag tag-emerald">CACHE HIT</span>' : '<span class="tag tag-purple">LIVE RAG</span>';
          pendingEl.innerHTML = `
            <div class="chat-meta">
              <span class="tag tag-cyan">${ragType.toUpperCase()} RAG</span>
              ${cacheBadge}
              <span style="margin-left: auto;">${tel.total_latency_ms || 0} ms</span>
            </div>
            ${formatResponseText(data.answer)}
          `;
        }
        container.scrollTop = container.scrollHeight;
        qInput.value = "";

        // Render telemetry
        const tel = data.telemetry || {};
        document.getElementById('metric-latency').innerText = (tel.total_latency_ms || 0) + " ms";
        document.getElementById('metric-ttft').innerText = (tel.ttft_ms || 0) + " ms";
        document.getElementById('metric-tps').innerText = tel.tokens_per_second || 0;
        document.getElementById('metric-cache').innerText = tel.cache_hit ? "HIT (Cache)" : "MISS";

        // Render sources
        const sourcesDiv = document.getElementById('sources-container');
        sourcesDiv.innerHTML = "";
        const sources = data.sources || [];
        if (sources.length === 0) {
          sourcesDiv.innerHTML = "<div style='color: var(--text-muted);'>No source chunks retrieved.</div>";
        } else {
          sources.forEach((s, idx) => {
            const scoreLabel = s.rrf_score ? `RRF: ${s.rrf_score}` : `Score: ${s.score || 0}`;
            sourcesDiv.innerHTML += `
              <div style="background: rgba(10, 13, 20, 0.8); padding: 8px 12px; border-radius: 8px; border: 1px solid var(--card-border); font-size: 0.85rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span style="font-weight: 700; color: var(--accent-cyan);">Source #${idx+1}</span>
                  <span class="tag tag-purple">${scoreLabel}</span>
                </div>
                <div style="color: var(--text-muted);">${s.text}</div>
              </div>
            `;
          });
        }

        // Refresh sessions list
        loadSessions();
      } catch (err) {
        const pendingEl = document.getElementById(pendingBubbleId);
        if (pendingEl) {
          pendingEl.innerHTML = `<span style="color: #f43f5e;">Error executing query: ${err}</span>`;
        }
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = "<span>Execute Retrieval</span> &rarr;";
      }
    }

    async function handleUpload(e) {
      e.preventDefault();
      const fileInput = document.getElementById('file-upload-input');
      if (!fileInput.files.length) return alert("Select a file first.");

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('chunking_strategy', document.getElementById('chunk-tech-select').value);
      formData.append('embedding_strategy', document.getElementById('embed-tech-select').value);

      document.getElementById('upload-status').innerText = "Indexing document...";

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          document.getElementById('upload-status').innerHTML = "<span style='color: var(--accent-emerald);'>Success! Indexed " + data.chunks_indexed + " chunks.</span>";
          loadDocuments();
        } else {
          document.getElementById('upload-status').innerHTML = "<span style='color: #f43f5e;'>Failed: " + (data.detail || "Error") + "</span>";
        }
      } catch (err) {
        document.getElementById('upload-status').innerHTML = "<span style='color: #f43f5e;'>Upload error: " + err + "</span>";
      }
    }

    async function loadDocuments() {
      try {
        const res = await fetch('/api/documents');
        const docs = await res.json();
        const tbody = document.getElementById('doc-tbody');
        tbody.innerHTML = "";
        if (!docs.length) {
          tbody.innerHTML = "<tr><td colspan='6' style='text-align:center;'>No documents indexed yet.</td></tr>";
          return;
        }
        docs.forEach(d => {
          tbody.innerHTML += `
            <tr>
              <td>#${d.id}</td>
              <td style="font-weight: 600;">${d.filename}</td>
              <td><span class="tag tag-cyan">${d.file_type}</span></td>
              <td>${d.chunk_count}</td>
              <td>${d.chunking_strategy}</td>
              <td><span class="tag tag-emerald">${d.status}</span></td>
            </tr>
          `;
        });
      } catch (err) {}
    }

    async function loadKnowledgeGraph() {
      try {
        const res = await fetch('/api/graph');
        const data = await res.json();
        document.getElementById('graph-node-count').innerText = data.node_count || 0;
        document.getElementById('graph-edge-count').innerText = data.edge_count || 0;

        const tbody = document.getElementById('graph-edges-tbody');
        tbody.innerHTML = "";
        const edges = data.edges || [];
        if (!edges.length) {
          tbody.innerHTML = "<tr><td colspan='4' style='text-align: center;'>No relationships discovered yet.</td></tr>";
          return;
        }
        edges.slice(0, 30).forEach(e => {
          tbody.innerHTML += `
            <tr>
              <td style="font-weight: 600; color: var(--accent-cyan);">${e.source}</td>
              <td><span class="tag tag-purple">${e.relation}</span></td>
              <td style="font-weight: 600; color: #fff;">${e.target}</td>
              <td>${e.weight}</td>
            </tr>
          `;
        });
      } catch (err) {}
    }

    async function runComprehensiveEvaluation() {
      const q = document.getElementById('eval-query').value;
      const ctx = document.getElementById('eval-context').value;
      const container = document.getElementById('eval-results-container');
      container.innerHTML = "<em>Running multi-level evaluation (Prompt clarity, faithfulness, answer relevancy)...</em>";

      try {
        const res = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            response: "Hybrid RAG fuses dense vector retrieval with BM25 keyword matching via Reciprocal Rank Fusion.",
            context_passages: [ctx],
            rag_type: "hybrid",
            language: "en",
            mode: "general",
            run_corruption_test: false
          })
        });
        const data = await res.json();
        const p = data.prompt_level || {};
        const r = data.response_level || {};

        container.innerHTML = `
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
            <div class="metric-card">
              <div class="metric-label">Prompt Clarity Score</div>
              <div class="metric-val">${p.clarity_score || 1.0} / 1.0</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Injection Risk</div>
              <div class="metric-val" style="color: ${p.injection_risk_score > 0.3 ? '#f43f5e' : 'var(--accent-emerald)'};">${p.injection_risk_score || 0.0}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Faithfulness (Grounded)</div>
              <div class="metric-val">${r.faithfulness_score || 1.0} / 1.0</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Answer Relevancy</div>
              <div class="metric-val">${r.answer_relevancy_score || 1.0} / 1.0</div>
            </div>
          </div>
          <p><b>Overall Health Score:</b> <span class="tag tag-emerald" style="font-size: 0.9rem;">${data.overall_health_score || 0.9} / 1.0</span></p>
          <p style="margin-top: 0.5rem; color: var(--text-muted);">Total claims analyzed: ${r.total_claims_analyzed || 1} &bull; Grounded claims: ${r.grounded_claims || 1}</p>
        `;
      } catch (err) {
        container.innerText = "Evaluation error: " + err;
      }
    }

    async function runCorruptionStressTest() {
      const q = document.getElementById('eval-query').value;
      const ctx = document.getElementById('eval-context').value;
      const container = document.getElementById('eval-results-container');
      container.innerHTML = "<em>Injecting OCR typos, missing tokens, and distractor noise (0%, 25%, 50%, 75%)...</em>";

      try {
        const res = await fetch('/api/corruption-test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            clean_context: ctx,
            noise_levels: [0.0, 0.25, 0.50, 0.75]
          })
        });
        const data = await res.json();
        let rows = "";
        (data.degradation_curve || []).forEach(d => {
          rows += `<tr><td>${d.noise_level}</td><td><b>${d.similarity_to_baseline}</b></td><td>${d.latency_ms} ms</td></tr>`;
        });

        container.innerHTML = `
          <div style="margin-bottom: 1rem;">
            <span class="tag tag-purple" style="font-size: 0.9rem;">Robustness Score: ${data.robustness_score} / 1.0</span>
            <span class="tag tag-cyan" style="font-size: 0.9rem; margin-left: 0.5rem;">Resilience: ${data.resilience_rating}</span>
          </div>
          <table>
            <thead><tr><th>Noise Level</th><th>Semantic Retention</th><th>Latency</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      } catch (err) {
        container.innerText = "Corruption test error: " + err;
      }
    }

    async function loadTelemetryStats() {
      try {
        const res = await fetch('/api/benchmark');
        const d = await res.json();
        document.getElementById('stat-avg-lat').innerText = (d.avg_total_latency_ms || 0) + " ms";
        document.getElementById('stat-p95-lat').innerText = (d.p95_latency_ms || 0) + " ms";
        document.getElementById('stat-retrieval-lat').innerText = (d.avg_retrieval_latency_ms || 0) + " ms";
        document.getElementById('stat-tps').innerText = d.avg_tokens_per_second || 0;
      } catch (err) {}
    }

    // Initial load
    window.onload = () => {
      loadDocuments();
      loadSessions();
    };
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Multi-Format 3-Type RAG API on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)
