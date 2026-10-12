"""Vector Embeddings Pipeline & Similarity Engine for BIE.

Generates normalized vector embeddings for local RAG, memory recall (Skill 60),
and semantic search. Compatible with OpenAI `/v1/embeddings` endpoint.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


class EmbeddingsEngine:
    """Computes normalized vector embeddings and similarity metrics."""

    def __init__(self, default_dimension: int = 1536) -> None:
        self.default_dimension = default_dimension

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [round(x / norm, 6) for x in vec]

    def _fallback_embedding(self, text: str, dim: int) -> list[float]:
        """Deterministic feature hashing embedding when backend is text-only."""
        vec = [0.0] * dim
        tokens = text.lower().split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return self._l2_normalize(vec)

    def embed(
        self,
        texts: str | list[str],
        model: Any | None = None,
        dimension: int | None = None,
    ) -> list[list[float]]:
        """Generate normalized vector embeddings for one or more strings."""
        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = list(texts)

        dim = dimension or self.default_dimension
        results: list[list[float]] = []

        for text in input_list:
            if not text.strip():
                results.append([0.0] * dim)
                continue

            # Try llama_cpp backend embedding if supported
            if model is not None and hasattr(model, "create_embedding"):
                try:
                    res = model.create_embedding(text)
                    raw_vec = res["data"][0]["embedding"]
                    results.append(self._l2_normalize(raw_vec))
                    continue
                except Exception:
                    pass

            # Fallback embedding
            results.append(self._fallback_embedding(text, dim))

        return results

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two unit or raw vectors."""
        if len(v1) != len(v2) or not v1:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return round(dot / (norm1 * norm2), 4)

    def find_nearest(
        self,
        query_vec: list[float],
        candidates: list[list[float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return indices and similarity scores of top-K closest vectors."""
        scores = [
            (idx, self.cosine_similarity(query_vec, cand))
            for idx, cand in enumerate(candidates)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
