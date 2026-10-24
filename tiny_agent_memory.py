"""
tiny_agent_memory.py
Zero-dependency memory system for AI agents.

Provides three memory types:
  - SemanticMemory  : Long-term storage with TF-IDF / hash-based retrieval
  - WorkingMemory   : Short-term key-value store for session context
  - EpisodicMemory  : Append-only interaction log with timestamps

All three are wrapped in a unified MemoryStore interface.
If numpy is available, VectorMemory provides cosine-similarity retrieval.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional numpy availability check
# ---------------------------------------------------------------------------
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "SemanticMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "VectorMemory",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single entry stored in memory."""
    id: str
    content: str
    metadata: dict[str, Any]
    timestamp: str
    importance: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


@dataclass
class InteractionEntry:
    """A single interaction in episodic memory."""
    id: str
    role: str           # "user" | "agent" | "system"
    content: str
    metadata: dict[str, Any]
    timestamp: str
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# Tokeniser (stdlib only)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"(?u)\w+")


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, stripped of punctuation."""
    return [t.lower() for t in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# TF-IDF Similarity  (stdlib only)
# ---------------------------------------------------------------------------

class _TFIDFIndex:
    """
    Simple in-memory TF-IDF index using pure Python / stdlib.
    No numpy required. Cosine similarity for retrieval.
    """

    def __init__(self) -> None:
        self.documents: dict[str, list[str]] = {}   # id → token list
        self.doc_count: int = 0
        self.term_df: dict[str, int] = defaultdict(int)  # document frequency
        self.doc_norms: dict[str, float] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenise(text)
        self.documents[doc_id] = tokens

        for term in set(tokens):
            self.term_df[term] += 1
        self.doc_count += 1

        # Pre-compute L2 norm for cosine similarity
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        norm_sq = sum((tf[t] * self._idf(t)) ** 2 for t in tf)
        self.doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def _idf(self, term: str) -> float:
        df = self.term_df.get(term, 0)
        return math.log((self.doc_count + 1) / (df + 1)) + 1.0

    def _tf(self, term: str, tokens: list[str]) -> float:
        return tokens.count(term) / max(len(tokens), 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return list of (doc_id, score) sorted descending by cosine similarity.
        """
        if not self.documents:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        # Vectorise query
        q_tf = defaultdict(int)
        for t in query_tokens:
            q_tf[t] += 1

        scores: dict[str, float] = {}
        for doc_id, tokens in self.documents.items():
            doc_norm = self.doc_norms.get(doc_id, 1.0)
            if doc_norm == 0:
                doc_norm = 1.0

            dot = sum(
                self._tf(t, tokens) * self._idf(t) * (q_tf[t] / len(query_tokens))
                for t in q_tf
                if t in tokens
            )
            q_norm = math.sqrt(sum((q_tf[t] / len(query_tokens)) ** 2 for t in q_tf))
            if q_norm > 0:
                scores[doc_id] = dot / (doc_norm * q_norm)
            else:
                scores[doc_id] = 0.0

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Hash-based similarity  (stdlib fallback when numpy unavailable)
# ---------------------------------------------------------------------------

class _HashIndex:
    """
    Fast hash-based similarity using minhash-style shingling.
    No external dependencies.
    """

    def __init__(self, ngram_size: int = 3, num_hashes: int = 32) -> None:
        self.ngram_size = ngram_size
        self.num_hashes = num_hashes
        self.documents: dict[str, str] = {}  # id → original text
        self.minhashes: dict[str, list[int]] = {}

    def add(self, doc_id: str, text: str) -> None:
        self.documents[doc_id] = text
        self.minhashes[doc_id] = self._minhash(text)

    def _shingles(self, text: str) -> set[str]:
        tokens = _tokenise(text)
        return {
            " ".join(tokens[i:i + self.ngram_size])
            for i in range(len(tokens) - self.ngram_size + 1)
        }

    def _minhash(self, text: str) -> list[int]:
        shingles = self._shingles(text)
        if not shingles:
            return [0] * self.num_hashes
        hashes: list[int] = []
        for i in range(self.num_hashes):
            mn = min(
                int(hashlib.sha256(f"{s}_{i}".encode()).hexdigest(), 16)
                % (2 ** 30)
                for s in shingles
            )
            hashes.append(mn)
        return hashes

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        query_mh = self._minhash(query)
        scores = {}
        for doc_id, mh in self.minhashes.items():
            # Jaccard approximation via minhash intersection
            intersection = sum(a == b for a, b in zip(query_mh, mh))
            jaccard = intersection / self.num_hashes
            scores[doc_id] = jaccard
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Long-term semantic memory using TF-IDF or hash-based retrieval.

    No external dependencies. If numpy is available, cosine similarity
    via VectorMemory is used automatically.

    Args:
        persist_dir: Directory for optional disk persistence (JSON).
        top_k: Default number of results to return on recall.
    """

    def __init__(
        self,
        name: str = "semantic",
        persist_dir: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        self.name = name
        self.top_k = top_k
        self._entries: dict[str, MemoryEntry] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._index: _TFIDFIndex = _TFIDFIndex()

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # --- public API ---

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        """Add a memory entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())[:12]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            timestamp=_now(),
            importance=importance,
        )
        self._entries[entry_id] = entry
        self._index.add(entry_id, content)
        self._maybe_save()
        return entry_id

    def recall(self, query: str, top_k: Optional[int] = None) -> list[MemoryEntry]:
        """Retrieve the top-K most relevant memories for a query."""
        k = top_k if top_k is not None else self.top_k
        ranked = self._index.search(query, top_k=k)
        return [self._entries[oid] for oid, score in ranked if oid in self._entries and score > 0]

    def forget(self, pattern: Optional[str] = None, entry_id: Optional[str] = None) -> int:
        """
        Remove entries matching a content pattern (substring match, case-insensitive)
        or by exact ID. Returns number of entries removed.
        """
        removed = 0
        if entry_id:
            if entry_id in self._entries:
                del self._entries[entry_id]
                removed += 1
        elif pattern:
            p = pattern.lower()
            to_delete = [eid for eid, e in self._entries.items() if p in e.content.lower()]
            for eid in to_delete:
                del self._entries[eid]
                removed += 1

        if removed:
            # Rebuild index from remaining entries
            self._index = _TFIDFIndex()
            for e in self._entries.values():
                self._index.add(e.id, e.content)
            self._maybe_save()
        return removed

    def count(self) -> int:
        return len(self._entries)

    def snapshot(self) -> dict[str, Any]:
        """Return full semantic memory as a dict."""
        return {
            "type": "semantic",
            "name": self.name,
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    # --- persistence ---

    def _persist_path(self) -> Path:
        return (self._persist_dir or Path("/tmp")) / f"{self.name}.json"

    def _save(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"entries": [e.to_dict() for e in self._entries.values()]},
                fh,
                indent=2,
            )

    def _load(self) -> None:
        path = self._persist_path()
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for ed in data.get("entries", []):
            entry = MemoryEntry.from_dict(ed)
            self._entries[entry.id] = entry
            self._index.add(entry.id, entry.content)

    def _maybe_save(self) -> None:
        if self._persist_dir:
            self._save()

    def __repr__(self) -> str:
        return f"SemanticMemory(name={self.name!r}, entries={len(self._entries)})"


# ---------------------------------------------------------------------------
# VectorMemory (numpy cosine similarity)
# ---------------------------------------------------------------------------

class VectorMemory:
    """
    Optional numpy-backed vector memory using TF-IDF + cosine similarity.

    Automatically used by MemoryStore when numpy is available.
    Falls back to SemanticMemory (TF-IDF stdlib) when it is not.
    """

    def __init__(
        self,
        name: str = "vector",
        persist_dir: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for VectorMemory. "
                "Install it with: pip install numpy"
            )
        self.name = name
        self.top_k = top_k
        self._entries: dict[str, MemoryEntry] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._entry_ids: list[str] = []

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        entry_id = str(uuid.uuid4())[:12]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            timestamp=_now(),
            importance=importance,
        )
        self._entries[entry_id] = entry
        self._rebuild_index()
        self._maybe_save()
        return entry_id

    def recall(self, query: str, top_k: Optional[int] = None) -> list[MemoryEntry]:
        k = top_k if top_k is not None else self.top_k
        if not self._vectors is not None and len(self._entry_ids) == 0:
            return []
        q_vec = self._query_vector(query)
        scores = self._vectors.dot(q_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._entries[self._entry_ids[i]] for i in top_indices if scores[i] > 0]

    def forget(self, pattern: Optional[str] = None, entry_id: Optional[str] = None) -> int:
        removed = 0
        if entry_id and entry_id in self._entries:
            del self._entries[entry_id]
            removed += 1
        elif pattern:
            p = pattern.lower()
            to_delete = [eid for eid, e in self._entries.items() if p in e.content.lower()]
            for eid in to_delete:
                del self._entries[eid]
                removed += 1
        if removed:
            self._rebuild_index()
            self._maybe_save()
        return removed

    def count(self) -> int:
        return len(self._entries)

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "vector",
            "name": self.name,
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    def _rebuild_index(self) -> None:
        if not self._entries:
            self._vectors = np.zeros((0, 0))
            self._entry_ids = []
            self._vocab = {}
            self._idf = None
            return

        texts = list(self._entries.values())
        self._entry_ids = [e.id for e in texts]
        corpus = [e.content.lower() for e in texts]

        # Build vocab
        vocab_set: set[str] = set()
        for doc in corpus:
            vocab_set.update(_tokenise(doc))
        self._vocab = {t: i for i, t in enumerate(sorted(vocab_set))}
        V = len(self._vocab)
        if V == 0:
            self._vectors = np.zeros((len(corpus), 1))
            return

        # TF-IDF matrix
        N = len(corpus)
        tf = np.zeros((N, V))
        df = np.zeros(V)
        for d, doc in enumerate(corpus):
            tokens = _tokenise(doc)
            for t in tokens:
                if t in self._vocab:
                    tf[d, self._vocab[t]] += 1
                    df[self._vocab[t]] += 1

        idf = np.log((N + 1) / (df + 1)) + 1
        self._idf = idf
        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._vectors = tfidf / norms

    def _query_vector(self, query: str) -> np.ndarray:
        V = len(self._vocab)
        q_vec = np.zeros(V)
        tokens = _tokenise(query.lower())
        for t in tokens:
            if t in self._vocab:
                q_vec[self._vocab[t]] += 1
        if self._idf is not None and V == len(self._idf):
            q_vec = q_vec * self._idf
        norm = np.linalg.norm(q_vec)
        return q_vec / (norm if norm > 0 else 1)

    def _persist_path(self) -> Path:
        return (self._persist_dir or Path("/tmp")) / f"{self.name}_vector.json"

    def _save(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"entries": [e.to_dict() for e in self._entries.values()]},
                fh,
                indent=2,
            )

    def _load(self) -> None:
        path = self._persist_path()
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for ed in data.get("entries", []):
            entry = MemoryEntry.from_dict(ed)
            self._entries[entry.id] = entry
        self._rebuild_index()

    def _maybe_save(self) -> None:
        if self._persist_dir:
            self._save()


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Short-term key-value store for session context.
    Resets on agent restart. Optional disk persistence.
    """

    def __init__(
        self,
        name: str = "working",
        persist_dir: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self.name = name
        self._store: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

        atexit.register(self._save_sync)

    # --- public API ---

    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair. Overwrites existing."""
        self._store[key] = value
        self._timestamps[key] = time.monotonic()
        self._maybe_save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key. Returns default if missing or expired."""
        if key not in self._store:
            return default
        if self._expired(key):
            self.delete(key)
            return default
        return self._store[key]

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if the key existed."""
        if key in self._store:
            del self._store[key]
            self._timestamps.pop(key, None)
            self._maybe_save()
            return True
        return False

    def keys(self) -> list[str]:
        """Return active (non-expired) keys."""
        self._purge_expired()
        return list(self._store.keys())

    def clear(self) -> None:
        """Clear all working memory."""
        self._store.clear()
        self._timestamps.clear()
        self._maybe_save()

    def update(self, data: dict[str, Any]) -> None:
        """Bulk update from a dict."""
        for k, v in data.items():
            self.set(k, v)

    def count(self) -> int:
        self._purge_expired()
        return len(self._store)

    def snapshot(self) -> dict[str, Any]:
        """Return full working memory state."""
        self._purge_expired()
        return {
            "type": "working",
            "name": self.name,
            "count": len(self._store),
            "store": dict(self._store),
        }

    # --- internals ---

    def _expired(self, key: str) -> bool:
        if self._ttl is None:
            return False
        if key not in self._timestamps:
            return True
        return (time.monotonic() - self._timestamps[key]) > self._ttl

    def _purge_expired(self) -> None:
        if self._ttl is None:
            return
        now = time.monotonic()
        expired = [k for k, ts in self._timestamps.items() if (now - ts) > self._ttl]
        for k in expired:
            self._store.pop(k, None)
            self._timestamps.pop(k, None)

    def _persist_path(self) -> Path:
        return (self._persist_dir or Path("/tmp")) / f"{self.name}_working.json"

    def _save(self) -> None:
        if not self._persist_dir:
            return
        self._save_sync()

    def _save_sync(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_path()
        data = {"store": self._store, "timestamps": self._timestamps}
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
        except (IOError, OSError):
            pass

    def _load(self) -> None:
        path = self._persist_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._store = data.get("store", {})
            self._timestamps = {
                k: float(v) for k, v in data.get("timestamps", {}).items()
            }
            self._purge_expired()
        except (IOError, OSError, json.JSONDecodeError):
            pass

    def _maybe_save(self) -> None:
        if self._persist_dir:
            self._save()

    def __repr__(self) -> str:
        return f"WorkingMemory(name={self.name!r}, keys={len(self._store)})"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Append-only log of agent interactions with timestamps.
    Every role ("user", "agent", "system") gets a tagged entry.
    """

    MAX_ENTRIES = 100_000  # guardrail for unbounded growth

    def __init__(
        self,
        name: str = "episodic",
        persist_dir: Optional[str] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.name = name
        self.max_entries = max_entries or self.MAX_ENTRIES
        self._entries: list[InteractionEntry] = []
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._dirty = False

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

        atexit.register(self._save_sync)

    # --- public API ---

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> str:
        """Append an interaction. Returns the entry ID."""
        if role not in ("user", "agent", "system", "tool"):
            raise ValueError(f"Unknown role: {role!r}")

        entry_id = str(uuid.uuid4())[:12]
        entry = InteractionEntry(
            id=entry_id,
            role=role,
            content=content,
            metadata=metadata or {},
            timestamp=_now(),
            duration_ms=duration_ms,
        )
        self._entries.append(entry)
        self._dirty = True

        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
            self._dirty = True

        self._maybe_save()
        return entry_id

    def user(self, content: str, **kwargs: Any) -> str:
        return self.add("user", content, **kwargs)

    def agent(self, content: str, **kwargs: Any) -> str:
        return self.add("agent", content, **kwargs)

    def system(self, content: str, **kwargs: Any) -> str:
        return self.add("system", content, **kwargs)

    def recent(self, n: int = 10, role: Optional[str] = None) -> list[InteractionEntry]:
        """Return the N most recent entries, optionally filtered by role."""
        entries = self._entries if role is None else [e for e in self._entries if e.role == role]
        return entries[-n:]

    def search(self, query: str, top_k: int = 5) -> list[InteractionEntry]:
        """Text search over interaction content."""
        q = query.lower()
        scored = [
            (i, e) for i, e in enumerate(self._entries)
            if q in e.content.lower()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Clear all episodic memory."""
        self._entries.clear()
        self._dirty = True
        self._maybe_save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "episodic",
            "name": self.name,
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    # --- persistence ---

    def _persist_path(self) -> Path:
        return (self._persist_dir or Path("/tmp")) / f"{self.name}_episodic.json"

    def _save(self) -> None:
        if not self._persist_dir:
            return
        self._save_sync()

    def _save_sync(self) -> None:
        if not self._persist_dir or not self._dirty:
            return
        path = self._persist_path()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"entries": [e.to_dict() for e in self._entries]},
                    fh,
                    indent=2,
                    default=str,
                )
            self._dirty = False
        except (IOError, OSError):
            pass

    def _load(self) -> None:
        path = self._persist_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = [
                InteractionEntry.from_dict(ed)
                for ed in data.get("entries", [])
            ]
        except (IOError, OSError, json.JSONDecodeError):
            pass

    def _maybe_save(self) -> None:
        if self._persist_dir and self._dirty:
            self._save()

    def __repr__(self) -> str:
        return f"EpisodicMemory(name={self.name!r}, entries={len(self._entries)})"


# ---------------------------------------------------------------------------
# MemoryStore — unified interface
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Unified memory interface wrapping Semantic, Working, and Episodic memory.

    Provides a single entry point for agent memory operations with optional
    persistence to disk.

    Args:
        name: Agent name, used as prefix for persistence files.
        persist_dir: Directory for JSON persistence files. Created if needed.
        use_vector: Use numpy-backed VectorMemory when available (default: auto).
        top_k: Default retrieval count.

    Example:
        >>> store = MemoryStore(name="my-agent", persist_dir="/tmp/agent-memory")
        >>> store.add("user asked about the deploy pipeline", metadata={"type": "user_message"})
        >>> store.add("found the bug in config_loader.py line 47", metadata={"type": "agent_action"})
        >>> results = store.remember("deploy pipeline")
        >>> store.snapshot()
    """

    def __init__(
        self,
        name: str = "agent",
        persist_dir: Optional[str] = None,
        use_vector: bool = False,
        top_k: int = 5,
    ) -> None:
        self.name = name
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if use_vector and HAS_NUMPY:
            self._semantic: Any = VectorMemory(
                name=f"{name}_semantic", persist_dir=str(persist_dir) if persist_dir else None, top_k=top_k
            )
        else:
            self._semantic = SemanticMemory(
                name=f"{name}_semantic", persist_dir=str(persist_dir) if persist_dir else None, top_k=top_k
            )

        self._working = WorkingMemory(
            name=f"{name}_working", persist_dir=str(persist_dir) if persist_dir else None
        )
        self._episodic = EpisodicMemory(
            name=f"{name}_episodic", persist_dir=str(persist_dir) if persist_dir else None
        )

        # Auto-save on process exit
        atexit.register(self.flush)

    # --- primary API ---

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 1.0,
    ) -> str:
        """
        Add content to the specified memory type.

        Args:
            content: The memory content (text for semantic, arbitrary for working).
            metadata: Optional metadata dict.
            memory_type: One of "semantic", "working", "episodic".
            importance: Importance score (semantic only).

        Returns:
            The entry ID as a string.
        """
        if memory_type == "semantic":
            return self._semantic.add(content, metadata=metadata, importance=importance)
        elif memory_type == "episodic":
            role = (metadata or {}).get("role", "agent")
            return self._episodic.add(role, content, metadata=metadata)
        elif memory_type == "working":
            key = (metadata or {}).get("key", str(uuid.uuid4())[:12])
            self._working.set(key, content)
            return key
        else:
            raise ValueError(f"Unknown memory type: {memory_type!r}")

    def remember(self, query: str, top_k: Optional[int] = None) -> list[MemoryEntry]:
        """
        Query semantic memory for the most relevant entries.

        Returns a list of MemoryEntry objects, sorted by relevance score descending.
        """
        return self._semantic.recall(query, top_k=top_k)

    def recall_episodic(
        self,
        query: Optional[str] = None,
        n: int = 10,
        role: Optional[str] = None,
    ) -> list[InteractionEntry]:
        """Recall recent episodic interactions, optionally filtered by query or role."""
        if query:
            return self._episodic.search(query, top_k=n)
        return self._episodic.recent(n=n, role=role)

    def forget(self, pattern: str, memory_type: str = "semantic") -> int:
        """
        Remove entries matching a content pattern from the specified memory.

        Returns number of entries removed.
        """
        if memory_type == "semantic":
            return self._semantic.forget(pattern=pattern)
        elif memory_type == "episodic":
            # EpisodicMemory doesn't support pattern forget by design (append-only)
            return 0
        elif memory_type == "working":
            keys = [k for k in self._working.keys() if pattern.lower() in str(self._working.get(k, "")).lower()]
            removed = 0
            for k in keys:
                if self._working.delete(k):
                    removed += 1
            return removed
        return 0

    def snapshot(self) -> dict[str, Any]:
        """
        Return the full memory state as a dict.

        Useful for serializing agent context for logging, debugging, or context injection.
        """
        return {
            "agent": self.name,
            "timestamp": _now(),
            "semantic": self._semantic.snapshot(),
            "working": self._working.snapshot(),
            "episodic": self._episodic.snapshot(),
        }

    def flush(self) -> None:
        """Force-persist all memory layers to disk."""
        self._semantic._maybe_save()
        self._working._save_sync()
        self._episodic._save_sync()

    def clear(self, memory_type: Optional[str] = None) -> None:
        """Clear memory. Omit type to clear all three."""
        if memory_type is None or memory_type == "semantic":
            for e in list(self._semantic._entries.keys()):
                self._semantic.forget(entry_id=e)
        if memory_type is None or memory_type == "working":
            self._working.clear()
        if memory_type is None or memory_type == "episodic":
            self._episodic.clear()

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def working(self) -> WorkingMemory:
        return self._working

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    def __repr__(self) -> str:
        return (
            f"MemoryStore(name={self.name!r}, "
            f"semantic={self._semantic.count()}, "
            f"working={self._working.count()}, "
            f"episodic={self._episodic.count()})"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Entry point for CLI testing / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(name="demo", persist_dir=td)

        store.add(
            "User asked about the deployment pipeline",
            metadata={"type": "user_message"},
        )
        store.add(
            "Found a bug in config_loader.py at line 47",
            metadata={"type": "agent_action"},
        )
        store.add(
            "Fixed the bug by adding a null check",
            metadata={"type": "agent_action"},
        )
        store.working.set("current_file", "config_loader.py")
        store.working.set("current_line", "47")
        store.episodic.agent("Thinking about the bug fix...")

        print("=== Semantic recall: 'deploy' ===")
        for r in store.remember("deploy"):
            print(f"  [{r.id}] {r.content} (score metadata: {r.metadata})")

        print("\n=== Snapshot ===")
        import json

        print(json.dumps(store.snapshot(), indent=2, default=str))
