"""
Milestone 6: Redis Caching Layer with TTL, Telemetry & In-Memory LRU Fallback.

Caches ranked recommendations keyed by `(user_id, model_variant, top_k)` so repeat
requests bypass Stage-1 FAISS + Stage-2 deep ranking in < 1ms. Automatically falls back
to a bounded in-memory TTL cache when a Redis server is not running locally.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

try:
    import redis
    REDIS_LIB_AVAILABLE = True
except ImportError:
    REDIS_LIB_AVAILABLE = False


class RecommendationCache:
    """Production Redis cache with seamless local LRU+TTL fallback."""

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 300, max_local_items: int = 2000):
        self.ttl_seconds = ttl_seconds
        self.max_local_items = max_local_items
        self._local_store: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._client = None
        self.backend = "in_memory_lru_fallback"
        self.hits = 0
        self.misses = 0

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if REDIS_LIB_AVAILABLE and os.getenv("DISABLE_REDIS", "0") != "1":
            try:
                client = redis.Redis.from_url(url, socket_connect_timeout=0.5, decode_responses=True)
                client.ping()
                self._client = client
                self.backend = "redis"
            except Exception:
                self._client = None

    @staticmethod
    def make_key(user_id: int, variant: str, top_k: int) -> str:
        return f"recs:v1:u:{user_id}:m:{variant}:k:{top_k}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self._client is not None:
            try:
                raw = self._client.get(key)
                if raw is not None:
                    self.hits += 1
                    return json.loads(raw)
                self.misses += 1
                return None
            except Exception:
                pass

        now = time.time()
        if key in self._local_store:
            expires_at, raw = self._local_store[key]
            if expires_at > now:
                self._local_store.move_to_end(key)
                self.hits += 1
                return json.loads(raw)
            del self._local_store[key]

        self.misses += 1
        return None

    def set(self, key: str, payload: Dict[str, Any], ttl: Optional[int] = None) -> None:
        ttl_val = ttl or self.ttl_seconds
        serialized = json.dumps(payload)
        if self._client is not None:
            try:
                self._client.setex(key, ttl_val, serialized)
                return
            except Exception:
                pass

        self._local_store[key] = (time.time() + ttl_val, serialized)
        self._local_store.move_to_end(key)
        while len(self._local_store) > self.max_local_items:
            self._local_store.popitem(last=False)

    def invalidate_user(self, user_id: int) -> int:
        prefix = f"recs:v1:u:{user_id}:"
        removed = 0
        if self._client is not None:
            try:
                keys = list(self._client.scan_iter(match=f"{prefix}*"))
                if keys:
                    removed = int(self._client.delete(*keys))
                return removed
            except Exception:
                pass

        to_del = [k for k in self._local_store if k.startswith(prefix)]
        for k in to_del:
            del self._local_store[k]
            removed += 1
        return removed

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flushdb()
            except Exception:
                pass
        self._local_store.clear()

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "cached_keys": len(self._local_store) if self._client is None else -1,
            "ttl_seconds": self.ttl_seconds,
        }
