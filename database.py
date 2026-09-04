"""
Database Layer: PostgreSQL & SQLite Dual Support.
Stores document records, chunk metadata, Knowledge Graph entities/relations,
and query/evaluation/latency benchmark telemetry.
"""

import os
import json
from datetime import datetime
from typing import Generator, Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float,
    DateTime, ForeignKey, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from dotenv import load_dotenv

load_dotenv()

# Determine Database URL (PostgreSQL with individual parameters or direct URL, fallback to SQLite)
postgres_user = os.getenv("POSTGRES_USER", "postgres")
postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
postgres_host = os.getenv("POSTGRES_HOST", "localhost")
postgres_port = os.getenv("POSTGRES_PORT", "5432")
postgres_db = os.getenv("POSTGRES_DB", "rag_project")

constructed_pg_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
DATABASE_URL = os.getenv("DATABASE_URL", constructed_pg_url)

# Test or fallback connection
if DATABASE_URL.startswith("sqlite"):
    os.makedirs("./data", exist_ok=True)
    connect_args = {"check_same_thread": False}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
else:
    try:
        # Test PostgreSQL connection with connection pooling
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        with engine.connect() as conn:
            pass
    except Exception as e:
        print(f"[Database Warning] PostgreSQL connection failed ({e}). Falling back to local SQLite at ./data/rag_app.db")
        os.makedirs("./data", exist_ok=True)
        DATABASE_URL = "sqlite:///./data/rag_app.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentModel(Base):
    """Tracks uploaded and indexed files."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, csv, xlsx, docx, txt
    file_size = Column(Integer, default=0)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default="uploaded")  # uploaded, indexed, failed
    chunk_count = Column(Integer, default=0)
    chunking_strategy = Column(String(100), default="recursive")
    embedding_strategy = Column(String(100), default="gemini")
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    """Stores text chunks and metadata."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    embedding_id = Column(String(100), nullable=True, index=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("DocumentModel", back_populates="chunks")


class GraphEntityModel(Base):
    """Knowledge Graph Entity node for Graph RAG."""
    __tablename__ = "graph_entities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(100), default="CONCEPT")
    description = Column(Text, default="")
    source_doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class GraphRelationModel(Base):
    """Knowledge Graph Relationship edge for Graph RAG."""
    __tablename__ = "graph_relations"

    id = Column(Integer, primary_key=True, index=True)
    source_entity = Column(String(255), nullable=False, index=True)
    target_entity = Column(String(255), nullable=False, index=True)
    relation_type = Column(String(100), default="RELATED_TO")
    description = Column(Text, default="")
    weight = Column(Float, default=1.0)
    source_doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLogModel(Base):
    """Telemetry log for queries, latency, and tokens."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    rag_type = Column(String(50), nullable=False)  # simple, hybrid, graph
    llm_backend = Column(String(50), nullable=False)  # gemini, vllm, sglang
    language = Column(String(20), default="en")
    mode = Column(String(30), default="general")  # general, code
    total_latency_ms = Column(Float, default=0.0)
    retrieval_latency_ms = Column(Float, default=0.0)
    generation_latency_ms = Column(Float, default=0.0)
    chunks_retrieved = Column(Integer, default=0)
    tokens_generated = Column(Integer, default=0)
    tokens_per_second = Column(Float, default=0.0)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationLogModel(Base):
    """Multi-level evaluation results: prompt, response, model, corruption."""
    __tablename__ = "evaluation_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_logs.id", ondelete="SET NULL"), nullable=True)
    query_text = Column(Text, nullable=False)
    rag_type = Column(String(50), default="hybrid")
    
    # Prompt-level metrics
    prompt_clarity_score = Column(Float, default=1.0)
    prompt_token_efficiency = Column(Float, default=1.0)
    prompt_injection_risk = Column(Float, default=0.0)
    
    # Response-level metrics
    faithfulness_score = Column(Float, default=1.0)
    answer_relevancy_score = Column(Float, default=1.0)
    hallucination_rate = Column(Float, default=0.0)
    format_adherence_score = Column(Float, default=1.0)
    
    # Model-level metrics
    model_consistency_score = Column(Float, default=1.0)
    
    # Corruption robustness
    corruption_noise_level = Column(Float, default=0.0)
    corruption_robustness_score = Column(Float, default=1.0)

    feedback_notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class BenchmarkLogModel(Base):
    """Latency and throughput benchmark results."""
    __tablename__ = "benchmark_logs"

    id = Column(Integer, primary_key=True, index=True)
    engine_name = Column(String(50), nullable=False)
    test_type = Column(String(50), nullable=False)  # concurrency, single, corruption
    concurrent_requests = Column(Integer, default=1)
    avg_latency_ms = Column(Float, default=0.0)
    p95_latency_ms = Column(Float, default=0.0)
    avg_ttft_ms = Column(Float, default=0.0)
    avg_tokens_per_sec = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSessionModel(Base):
    """Stores conversational RAG chat sessions."""
    __tablename__ = "chat_sessions"

    id = Column(String(100), primary_key=True, index=True)
    title = Column(String(255), default="New Conversation")
    rag_type = Column(String(50), default="hybrid")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatMessageModel", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessageModel.created_at")


class ChatMessageModel(Base):
    """Stores individual dialogue turns in a conversational RAG session."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    sources_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSessionModel", back_populates="messages")


def init_db():
    """Initializes the database schema."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
