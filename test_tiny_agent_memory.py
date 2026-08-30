"""Tests for tiny_agent_memory.py — 18 test cases covering all memory types."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from tiny_agent_memory import (
    EpisodicMemory,
    MemoryEntry,
    MemoryStore,
    SemanticMemory,
    WorkingMemory,
    _tokenise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tmp_dir():
    return tempfile.mkdtemp(prefix="tiny_agent_memory_test_")


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestTokeniser:
    def test_tokenise_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_tokenise_strips_punctuation(self):
        assert _tokenise("foo, bar! baz.") == ["foo", "bar", "baz"]

    def test_tokenise_handles_numbers(self):
        assert "2026" in _tokenise("year 2026")

    def test_tokenise_empty_string(self):
        assert _tokenise("") == []
        assert _tokenise("   ") == []


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    def test_to_dict_roundtrip(self):
        entry = MemoryEntry(
            id="abc123",
            content="test content",
            metadata={"type": "note"},
            timestamp="2026-08-30T12:00:00Z",
            importance=0.8,
        )
        data = entry.to_dict()
        restored = MemoryEntry.from_dict(data)
        assert restored.id == entry.id
        assert restored.content == entry.content
        assert restored.metadata == entry.metadata
        assert restored.importance == entry.importance


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class TestSemanticMemory:
    def test_add_returns_entry_id(self):
        mem = SemanticMemory(name="test")
        eid = mem.add("hello world")
        assert isinstance(eid, str)
        assert len(eid) == 12

    def test_add_multiple_entries(self):
        mem = SemanticMemory(name="test")
        mem.add("deploy pipeline broken")
        mem.add("config file missing")
        mem.add("database connection timeout")
        assert mem.count() == 3

    def test_recall_returns_relevant(self):
        mem = SemanticMemory(name="test", top_k=2)
        mem.add("deploy pipeline is slow", metadata={"source": "issue-42"})
        mem.add("config file format changed", metadata={"source": "issue-43"})
        mem.add("database schema needs migration", metadata={"source": "issue-44"})

        results = mem.recall("deploy pipeline")
        assert len(results) <= 2
        assert any("deploy" in r.content.lower() for r in results)

    def test_recall_empty_query(self):
        mem = SemanticMemory(name="test")
        mem.add("some content")
        results = mem.recall("")
        assert isinstance(results, list)

    def test_recall_top_k(self):
        mem = SemanticMemory(name="test")
        for i in range(10):
            mem.add(f"content number {i}")
        results = mem.recall("content", top_k=3)
        assert len(results) == 3

    def test_forget_by_pattern(self):
        mem = SemanticMemory(name="test")
        mem.add("deploy pipeline broken")
        mem.add("config file missing")
        mem.add("database timeout")

        removed = mem.forget(pattern="deploy")
        assert removed == 1
        assert mem.count() == 2

        results = mem.recall("deploy")
        assert len(results) == 0

    def test_forget_by_entry_id(self):
        mem = SemanticMemory(name="test")
        eid = mem.add("temporary note")
        assert mem.count() == 1
        removed = mem.forget(entry_id=eid)
        assert removed == 1
        assert mem.count() == 0

    def test_forget_nonexistent_returns_zero(self):
        mem = SemanticMemory(name="test")
        mem.add("existing")
        assert mem.forget(pattern="nonexistent") == 0
        assert mem.forget(entry_id="fake-id-123") == 0

    def test_snapshot_returns_dict(self):
        mem = SemanticMemory(name="test")
        mem.add("test content", metadata={"type": "test"})
        snap = mem.snapshot()
        assert snap["type"] == "semantic"
        assert snap["count"] == 1
        assert len(snap["entries"]) == 1

    def test_persistence_save_load(self):
        td = tmp_dir()
        mem1 = SemanticMemory(name="persist-test", persist_dir=td)
        mem1.add("persistent fact")
        mem1.add("another fact", metadata={"key": "value"})

        mem2 = SemanticMemory(name="persist-test", persist_dir=td)
        assert mem2.count() == 2
        assert any("persistent" in e.content for e in mem2.recall("persistent"))


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def test_set_get(self):
        wm = WorkingMemory(name="test")
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_default(self):
        wm = WorkingMemory(name="test")
        assert wm.get("missing") is None
        assert wm.get("missing", "default") == "default"

    def test_delete_existing(self):
        wm = WorkingMemory(name="test")
        wm.set("k", "v")
        assert wm.delete("k") is True
        assert wm.get("k") is None

    def test_delete_missing(self):
        wm = WorkingMemory(name="test")
        assert wm.delete("nope") is False

    def test_keys(self):
        wm = WorkingMemory(name="test")
        wm.set("a", 1)
        wm.set("b", 2)
        keys = wm.keys()
        assert "a" in keys
        assert "b" in keys

    def test_update(self):
        wm = WorkingMemory(name="test")
        wm.update({"x": 10, "y": 20})
        assert wm.get("x") == 10
        assert wm.get("y") == 20

    def test_clear(self):
        wm = WorkingMemory(name="test")
        wm.set("a", 1)
        wm.clear()
        assert wm.count() == 0

    def test_ttl_eviction(self):
        wm = WorkingMemory(name="test", ttl_seconds=0.01)
        wm.set("short-lived", "value")
        assert wm.get("short-lived") == "value"
        time.sleep(0.05)
        assert wm.get("short-lived") is None

    def test_count(self):
        wm = WorkingMemory(name="test")
        for i in range(5):
            wm.set(f"k{i}", i)
        assert wm.count() == 5

    def test_snapshot(self):
        wm = WorkingMemory(name="test")
        wm.set("file", "main.py")
        wm.set("line", 42)
        snap = wm.snapshot()
        assert snap["type"] == "working"
        assert snap["store"]["file"] == "main.py"
        assert snap["store"]["line"] == 42

    def test_persistence_save_load(self):
        td = tmp_dir()
        wm1 = WorkingMemory(name="persist-w", persist_dir=td)
        wm1.set("token", "secret123")

        wm2 = WorkingMemory(name="persist-w", persist_dir=td)
        assert wm2.get("token") == "secret123"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def test_add_returns_id(self):
        em = EpisodicMemory(name="test")
        eid = em.add("user", "hello")
        assert isinstance(eid, str)
        assert len(eid) == 12

    def test_role_shortcuts(self):
        em = EpisodicMemory(name="test")
        uid = em.user("hi")
        aid = em.agent("thinking...")
        sid = em.system("task started")
        assert em.count() == 3

    def test_invalid_role_raises(self):
        em = EpisodicMemory(name="test")
        with pytest.raises(ValueError, match="Unknown role"):
            em.add("invalid_role", "content")

    def test_recent_default(self):
        em = EpisodicMemory(name="test")
        for i in range(20):
            em.add("agent", f"action {i}")
        recent = em.recent(n=5)
        assert len(recent) == 5
        # Most recent should be action 19
        assert recent[-1].content == "action 19"

    def test_recent_by_role(self):
        em = EpisodicMemory(name="test")
        em.user("u1")
        em.agent("a1")
        em.user("u2")
        recent = em.recent(n=10, role="user")
        assert all(e.role == "user" for e in recent)
        assert len(recent) == 2

    def test_search(self):
        em = EpisodicMemory(name="test")
        em.add("user", "deploy the app to production")
        em.add("agent", "checking config file")
        em.add("agent", "found bug in deploy script")
        em.add("user", "deploy to staging first")

        results = em.search("deploy")
        assert len(results) >= 2
        assert all("deploy" in r.content.lower() for r in results)

    def test_count(self):
        em = EpisodicMemory(name="test")
        for i in range(7):
            em.add("agent", f"step {i}")
        assert em.count() == 7

    def test_clear(self):
        em = EpisodicMemory(name="test")
        em.add("agent", "something")
        em.clear()
        assert em.count() == 0

    def test_snapshot(self):
        em = EpisodicMemory(name="test")
        em.add("user", "run the tests")
        snap = em.snapshot()
        assert snap["type"] == "episodic"
        assert snap["count"] == 1

    def test_persistence_save_load(self):
        td = tmp_dir()
        em1 = EpisodicMemory(name="persist-e", persist_dir=td)
        em1.agent("first action")
        em1.agent("second action")

        em2 = EpisodicMemory(name="persist-e", persist_dir=td)
        assert em2.count() == 2


# ---------------------------------------------------------------------------
# MemoryStore — integration
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_store_creation(self):
        td = tmp_dir()
        store = MemoryStore(name="integration-test", persist_dir=td)
        assert store.name == "integration-test"

    def test_add_semantic(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        eid = store.add("user asked about the deploy pipeline", metadata={"type": "user_message"})
        assert isinstance(eid, str)

    def test_add_episodic(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        eid = store.add("agent action here", metadata={"role": "agent"})
        assert isinstance(eid, str)

    def test_remember(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("deploy pipeline failed with error", metadata={"type": "failure"})
        store.add("config file missing", metadata={"type": "error"})

        results = store.remember("deploy pipeline")
        assert len(results) >= 1
        assert any("deploy" in r.content.lower() for r in results)

    def test_remember_top_k(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        for i in range(10):
            store.add(f"content item {i}")
        results = store.remember("content", top_k=3)
        assert len(results) == 3

    def test_episodic_recall(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("agent", metadata={"role": "agent"})  # type: ignore
        store.episodic.agent("thinking about the solution")
        store.episodic.user("great!")

        recent = store.recall_episodic(n=5)
        assert len(recent) >= 2

    def test_episodic_recall_by_role(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.episodic.user("hello")
        store.episodic.agent("thinking")
        store.episodic.user("bye")

        recent = store.recall_episodic(role="user")
        assert all(e.role == "user" for e in recent)
        assert len(recent) == 2

    def test_forget_semantic(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("temporary note to remove")
        assert store.semantic.count() == 1
        removed = store.forget("temporary", memory_type="semantic")
        assert removed == 1
        assert store.semantic.count() == 0

    def test_forget_working(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.working.set("temp_key", "temp_value")
        assert store.working.count() == 1
        removed = store.forget("temp", memory_type="working")
        assert removed == 1

    def test_snapshot(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("long term fact", metadata={"type": "fact"})
        store.working.set("session", "active")
        store.episodic.agent("thinking...")

        snap = store.snapshot()
        assert "agent" in snap
        assert snap["agent"] == "test"
        assert snap["semantic"]["count"] == 1
        assert snap["working"]["count"] == 1
        assert snap["episodic"]["count"] == 1

    def test_flush(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("persistent memory")
        store.working.set("s", "v")
        store.flush()  # Should not raise

    def test_clear_all(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("fact")
        store.working.set("k", "v")
        store.episodic.agent("act")

        store.clear()
        assert store.semantic.count() == 0
        assert store.working.count() == 0
        assert store.episodic.count() == 0

    def test_clear_specific_type(self):
        td = tmp_dir()
        store = MemoryStore(name="test", persist_dir=td)
        store.add("fact")
        store.working.set("k", "v")

        store.clear(memory_type="working")
        assert store.semantic.count() == 1
        assert store.working.count() == 0

    def test_repr(self):
        td = tmp_dir()
        store = MemoryStore(name="test-repr", persist_dir=td)
        store.add("a")
        store.working.set("b", "c")
        store.episodic.agent("d")
        r = repr(store)
        assert "test-repr" in r
        assert "semantic=1" in r
        assert "working=1" in r
        assert "episodic=1" in r
