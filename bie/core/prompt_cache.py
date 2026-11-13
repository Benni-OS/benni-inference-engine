"""Deterministic LRU + TTL Prompt & Response Cache for BIE.

Dramatically lowers latency for repeated deterministic prompts (temperature <= 0.2),
structured extraction queries, and recurring agent system+Skill 81 turns.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    key: str
    payload: dict[str, Any]
    created_at: float
    tokens_saved: int
    hits: int = 0


class PromptResponseCache:
    """Thread-safe bounded LRU+TTL cache for inference responses."""

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: float = 3600.0,
        max_cacheable_temp: float = 0.25,
    ) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(1.0, ttl_seconds)
        self.max_cacheable_temp = max_cacheable_temp
        self._lock = threading.RLock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @staticmethod
    def build_key(namespace: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"ns": namespace, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_cacheable(
        self,
        *,
        temperature: float,
        stream: bool = False,
        cache_bypass: bool = False,
    ) -> bool:
        if stream or cache_bypass:
            return False
        return float(temperature) <= self.max_cacheable_temp

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if (now - entry.created_at) > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            entry.hits += 1
            self.hits += 1
            self.tokens_saved += entry.tokens_saved
            return json.loads(json.dumps(entry.payload))

    def put(self, key: str, payload: dict[str, Any], *, tokens_saved: int = 0) -> None:
        now = time.time()
        stored_copy = json.loads(json.dumps(payload))
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(
                key=key,
                payload=stored_copy,
                created_at=now,
                tokens_saved=max(0, int(tokens_saved)),
            )
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = round((self.hits / total) * 100.0, 1) if total > 0 else 0.0
            return {
                "entries": len(self._store),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": hit_rate,
                "tokens_saved": self.tokens_saved,
            }
