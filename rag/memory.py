"""
Conversational RAG Memory Module:
Provides multi-turn conversational session memory:
1. In-Memory Session Cache + Persistent Database Sync (PostgreSQL/SQLite)
2. Sliding-window conversation buffer (retaining last K turns)
3. Conversational Contextualizer: Reformulates follow-up queries using chat history
4. History injection into RAG generation prompts
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from database import SessionLocal, ChatSessionModel, ChatMessageModel


class ConversationalMemoryManager:
    """
    Manages conversational memory across multiple turns for RAG.
    Maintains fast in-memory buffers with persistent database backing.
    """
    def __init__(self, default_window_size: int = 6):
        self.default_window_size = default_window_size
        # In-memory cache: session_id -> list of message dicts
        self._memory_cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Returns existing session_id or creates a new one."""
        sid = session_id or str(uuid.uuid4())
        
        if sid not in self._memory_cache:
            self._memory_cache[sid] = []
            # Check or persist to DB
            try:
                with SessionLocal() as db:
                    existing = db.query(ChatSessionModel).filter(ChatSessionModel.id == sid).first()
                    if not existing:
                        new_sess = ChatSessionModel(
                            id=sid,
                            title=title or f"Conversation {sid[:8]}",
                            rag_type="hybrid"
                        )
                        db.add(new_sess)
                        db.commit()
                    else:
                        # Load existing messages from DB
                        for msg in existing.messages:
                            self._memory_cache[sid].append({
                                "role": msg.role,
                                "content": msg.content,
                                "sources": json.loads(msg.sources_json or "[]"),
                                "created_at": msg.created_at.isoformat() if msg.created_at else None
                            })
            except Exception as e:
                print(f"[MemoryManager] DB sync notice: {e}")

        return sid

    def add_message(self, session_id: str, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
        """Appends a turn to in-memory history and commits to persistent database."""
        sid = self.get_or_create_session(session_id)
        
        msg_obj = {
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": datetime.utcnow().isoformat()
        }
        self._memory_cache[sid].append(msg_obj)

        # Persist to database
        try:
            with SessionLocal() as db:
                db_msg = ChatMessageModel(
                    session_id=sid,
                    role=role,
                    content=content,
                    sources_json=json.dumps(sources or [])
                )
                db.add(db_msg)
                db.commit()
        except Exception as e:
            print(f"[MemoryManager] Notice adding message to DB: {e}")

    def add_turn(self, session_id: str, user_query: str, assistant_response: str, sources: Optional[List[Dict[str, Any]]] = None):
        """Convenience method to record a complete user-assistant turn."""
        self.add_message(session_id, role="user", content=user_query)
        self.add_message(session_id, role="assistant", content=assistant_response, sources=sources)

    def get_history(self, session_id: str, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns the conversation history up to window_size messages."""
        if not session_id or session_id not in self._memory_cache:
            self.get_or_create_session(session_id)
        
        limit = window_size or self.default_window_size
        return self._memory_cache.get(session_id, [])[-limit:]

    def format_history_for_prompt(self, session_id: Optional[str] = None, window_size: int = 4) -> str:
        """
        Formats recent chat history into a string to insert into LLM synthesis prompts.
        """
        if not session_id:
            return ""

        history = self.get_history(session_id, window_size=window_size)
        if not history:
            return ""

        lines = ["--- Conversation History (Previous Context) ---"]
        for msg in history:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            lines.append(f"{prefix}{msg['content']}")
        lines.append("--- End of Previous Context ---\n")
        return "\n".join(lines)

    def contextualize_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Resolves coreferences or follow-up pronouns in the query using previous turns.
        e.g., Turn 1: 'What is Qdrant?'
              Turn 2: 'Can it run in memory?' -> 'Can Qdrant run in memory?'
        """
        if not session_id:
            return query

        history = self.get_history(session_id, window_size=4)
        if not history:
            return query

        # Fast heuristic check for pronouns or dependent query patterns
        lower_q = query.lower()
        referential_triggers = [
            " it ", " it?", " its ", " that ", " this ", " they ", " them ",
            "these ", "those ", "what about", "how about", "tell me more",
            "can you explain more", "does it", "is it", "can it"
        ]
        
        needs_context = any(t in f" {lower_q} " for t in referential_triggers)
        if not needs_context:
            return query

        # Find the last user or assistant topic
        last_turns = [m["content"] for m in history if m["role"] == "user"]
        if last_turns:
            prior_topic = last_turns[-1]
            return f"[Context: {prior_topic}] {query}"

        return query

    def clear_session(self, session_id: str):
        """Clears memory for a session."""
        if session_id in self._memory_cache:
            self._memory_cache[session_id] = []
        try:
            with SessionLocal() as db:
                db.query(ChatMessageModel).filter(ChatMessageModel.session_id == session_id).delete()
                db.commit()
        except Exception as e:
            print(f"[MemoryManager] Notice clearing session: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all active and stored sessions."""
        try:
            with SessionLocal() as db:
                sessions = db.query(ChatSessionModel).order_by(ChatSessionModel.updated_at.desc()).all()
                return [
                    {
                        "session_id": s.id,
                        "title": s.title,
                        "rag_type": s.rag_type,
                        "message_count": len(s.messages),
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None
                    }
                    for s in sessions
                ]
        except Exception as e:
            print(f"[MemoryManager] Notice listing sessions: {e}")
            return []


# Global conversation memory manager singleton
global_conversation_memory = ConversationalMemoryManager()
