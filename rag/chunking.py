"""
Advanced Chunking Strategies:
1. Recursive Character Chunking (sliding window with overlap)
2. Semantic Sentence Chunking (boundary-preserving semantic flow)
3. Structured Document Chunking (row-aware for CSV/XLSX, section/heading-aware for DOCX/PDF)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


class BaseChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        raise NotImplementedError


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text using hierarchical separators to maintain semantic coherence.
    Separators: ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return list(text)
        
        separator = separators[0]
        splits = []
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        good_splits = []
        for s in splits:
            if separator and s:
                # Retain separator for readability except space
                piece = s if separator == " " else s + (separator if not s.endswith(separator) else "")
            else:
                piece = s
            
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if len(separators) > 1:
                    sub_splits = self._split_text(piece, separators[1:])
                    good_splits.extend(sub_splits)
                else:
                    # Hard truncate if no more separators
                    for i in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                        good_splits.append(piece[i : i + self.chunk_size])
        return good_splits

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        base_meta = metadata or {}
        raw_splits = self._split_text(text.strip(), self.separators)
        
        chunks: List[TextChunk] = []
        current_chunk = ""
        chunk_idx = 0

        for split in raw_splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                if current_chunk.strip():
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        chunk_index=chunk_idx,
                        metadata={**base_meta, "chunk_strategy": "recursive"},
                        token_count=len(current_chunk.split())
                    ))
                    chunk_idx += 1
                
                # Overlap logic
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + split

        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                chunk_index=chunk_idx,
                metadata={**base_meta, "chunk_strategy": "recursive"},
                token_count=len(current_chunk.split())
            ))

        return chunks


class SemanticSentenceChunker(BaseChunker):
    """
    Groups complete sentences until the target token size is reached,
    ensuring sentences are never split in half and contextual flow is preserved.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self.sentence_regex = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        base_meta = metadata or {}
        # Split into sentences
        sentences = [s.strip() for s in self.sentence_regex.split(text.strip()) if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len <= self.chunk_size or not current_sentences:
                current_sentences.append(sent)
                current_len += sent_len + 1
            else:
                chunk_text = " ".join(current_sentences)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    metadata={**base_meta, "chunk_strategy": "semantic_sentence"},
                    token_count=len(chunk_text.split())
                ))
                chunk_idx += 1

                # Keep overlap sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                current_sentences = overlap_sentences + [sent]
                current_len = sum(len(s) + 1 for s in current_sentences)

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_idx,
                metadata={**base_meta, "chunk_strategy": "semantic_sentence"},
                token_count=len(chunk_text.split())
            ))

        return chunks


class StructuredDocumentChunker(BaseChunker):
    """
    Specialized chunking for:
    - Tabular data (CSV, XLSX): serializes rows into semantic key-value records.
    - Structured documents (DOCX, PDF): preserves headings and hierarchical section flow.
    """
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        base_meta = metadata or {}
        file_type = base_meta.get("file_type", "").lower()

        # Tabular line-by-line or section-by-section
        if file_type in ["csv", "xlsx"]:
            return self._chunk_tabular(text, base_meta)
        else:
            return self._chunk_structured_text(text, base_meta)

    def _chunk_tabular(self, text: str, base_meta: Dict[str, Any]) -> List[TextChunk]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return []

        chunks: List[TextChunk] = []
        batch_size = 5  # Group 5 table rows per chunk for dense semantic representation
        chunk_idx = 0

        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            chunk_content = "\n".join(batch)
            chunks.append(TextChunk(
                text=chunk_content,
                chunk_index=chunk_idx,
                metadata={
                    **base_meta,
                    "chunk_strategy": "structured_tabular",
                    "row_range": f"{i+1}-{min(i+batch_size, len(lines))}"
                },
                token_count=len(chunk_content.split())
            ))
            chunk_idx += 1

        return chunks

    def _chunk_structured_text(self, text: str, base_meta: Dict[str, Any]) -> List[TextChunk]:
        # Split by section headings or markdown headers
        sections = re.split(r'(?=\n#{1,3}\s+|\nSection\s+\d+:)', text)
        chunks: List[TextChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            
            # If section is within bounds, keep whole
            if len(sec) <= self.chunk_size:
                chunks.append(TextChunk(
                    text=sec,
                    chunk_index=chunk_idx,
                    metadata={**base_meta, "chunk_strategy": "structured_section"},
                    token_count=len(sec.split())
                ))
                chunk_idx += 1
            else:
                # Sub-chunk large section with recursive fallback
                sub_chunker = RecursiveCharacterChunker(self.chunk_size, self.chunk_overlap)
                sub_chunks = sub_chunker.chunk(sec, {**base_meta, "chunk_strategy": "structured_section_split"})
                for sc in sub_chunks:
                    sc.chunk_index = chunk_idx
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks


def get_chunker(strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 100) -> BaseChunker:
    """Factory function for chunker selection."""
    strategy = strategy.lower()
    if strategy in ["semantic", "semantic_sentence"]:
        return SemanticSentenceChunker(chunk_size, chunk_overlap)
    elif strategy in ["structured", "table", "tabular"]:
        return StructuredDocumentChunker(chunk_size, chunk_overlap)
    else:
        return RecursiveCharacterChunker(chunk_size, chunk_overlap)
