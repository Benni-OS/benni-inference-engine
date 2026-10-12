"""Context Window Rolling Chunker & Semantic Document Compressor for BIE.

Splits large documents, codebases, or conversation histories into semantic chunks,
ranks them by relevance against the user prompt using vector embeddings,
and compiles an optimal high-density context within a target token budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .embeddings import EmbeddingsEngine


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    token_count: int
    relevance_score: float = 0.0


class SemanticChunker:
    """Splits large text into semantic blocks and ranks them by query relevance."""

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        chunk_overlap_chars: int = 200,
        embeddings_engine: EmbeddingsEngine | None = None,
    ) -> None:
        self.max_chunk_chars = max(100, max_chunk_chars)
        self.chunk_overlap_chars = max(0, chunk_overlap_chars)
        self.embeddings = embeddings_engine or EmbeddingsEngine()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def split_into_chunks(self, document: str) -> list[TextChunk]:
        """Split document on paragraph or heading boundaries with sliding overlap."""
        if not document.strip():
            return []

        # Split on markdown headings or double newlines
        raw_sections = re.split(r"(?:\n\n+|(?=^#{1,4}\s+))", document, flags=re.MULTILINE)
        chunks: list[TextChunk] = []
        current_text = ""
        chunk_idx = 0

        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            if len(current_text) + len(sec_clean) <= self.max_chunk_chars:
                current_text = f"{current_text}\n\n{sec_clean}".strip()
            else:
                if current_text:
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_idx,
                            text=current_text,
                            token_count=self.estimate_tokens(current_text),
                        )
                    )
                    chunk_idx += 1
                    # Carry over overlap
                    overlap = current_text[-self.chunk_overlap_chars :] if self.chunk_overlap_chars else ""
                    current_text = f"{overlap}\n\n{sec_clean}".strip()
                else:
                    # Section is larger than max_chunk_chars, force split
                    for i in range(0, len(sec_clean), self.max_chunk_chars - self.chunk_overlap_chars):
                        sub_text = sec_clean[i : i + self.max_chunk_chars]
                        chunks.append(
                            TextChunk(
                                chunk_id=chunk_idx,
                                text=sub_text,
                                token_count=self.estimate_tokens(sub_text),
                            )
                        )
                        chunk_idx += 1
                    current_text = ""

        if current_text:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_idx,
                    text=current_text,
                    token_count=self.estimate_tokens(current_text),
                )
            )

        return chunks

    def rank_and_compress(
        self,
        document: str,
        query: str,
        budget_tokens: int = 4096,
        top_k: int = 5,
    ) -> tuple[str, list[TextChunk]]:
        """Rank document chunks by similarity to query and assemble context within token budget."""
        chunks = self.split_into_chunks(document)
        if not chunks:
            return "", []

        # If document already fits inside budget, return it directly
        total_tokens = sum(c.token_count for c in chunks)
        if total_tokens <= budget_tokens:
            return document, chunks

        # Compute query vector and chunk vectors
        q_vec = self.embeddings.embed(query)[0]
        chunk_texts = [c.text for c in chunks]
        chunk_vecs = self.embeddings.embed(chunk_texts)

        # Score relevance
        for i, c in enumerate(chunks):
            c.relevance_score = self.embeddings.cosine_similarity(q_vec, chunk_vecs[i])

        # Sort chunks by relevance descending
        ranked = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)[:top_k]

        # Sort selected chunks back into original chronological document order
        chronological = sorted(ranked, key=lambda c: c.chunk_id)

        # Assemble into budget
        selected_text = []
        accumulated_tokens = 0
        final_chunks: list[TextChunk] = []

        for c in chronological:
            if accumulated_tokens + c.token_count <= budget_tokens:
                selected_text.append(c.text)
                accumulated_tokens += c.token_count
                final_chunks.append(c)
            else:
                break

        compressed_doc = "\n\n---\n\n".join(selected_text)
        return compressed_doc, final_chunks
