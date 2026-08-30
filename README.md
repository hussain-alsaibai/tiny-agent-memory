# tiny-agent-memory

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](tiny_agent_memory.py)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![tiny-stack](https://img.shields.io/badge/tiny-*%20ecosystem-purple.svg)](#tiny-ecosystem)

> **Zero-dependency memory system for AI agents.** Semantic + working + episodic memory in a single Python file. No numpy required — but optional for faster vector search.

```python
from tiny_agent_memory import MemoryStore

store = MemoryStore(name="my-agent", persist_dir="/tmp/agent-memory")
store.add("user asked about the deploy pipeline", metadata={"type": "user_message"})
store.add("found the bug in config_loader.py line 47", metadata={"type": "agent_action"})
results = store.remember("deploy pipeline")
store.snapshot()
```

---

## Why?

The agent memory landscape is dominated by heavyweight solutions:

| Solution | Dependencies | Lines | Cold Start | Notes |
|---|---|---|---|---|
| **tiny-agent-memory** | **None** (numpy optional) | **~400** | **Instant** | Stdlib only, audit in an afternoon |
| LangChain Memory | langchain, vectorstore, embedding model | 10K+ | Slow | Overkill for most agents |
| Mem0 | OpenAI embeddings, Qdrant/Pinecone | 5K+ | Slow | Requires cloud or self-hosted vector DB |
| recall.ai | Proprietary SDK | Closed | Slow | SaaS lock-in |
| gptme-memory | OpenAI API + file storage | ~500 | Medium | API-key required |

`tiny-agent-memory` is for developers who want **memory without ceremony** — no API keys, no external services, no 50-dependency install, no vendor lock-in. It runs anywhere Python runs, reads like a blog post, and deploys in seconds.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  MemoryStore                     │
│  (unified interface: add, remember, forget,     │
│   snapshot, recall_episodic)                    │
└───────────────┬─────────────────┬───────────────┘
                │                 │
   ┌────────────▼──────┐   ┌──────▼───────────────┐
   │  SemanticMemory   │   │    WorkingMemory      │
   │  ───────────────  │   │    ───────────────    │
   │  Long-term recall │   │  Short-term KV store  │
   │  TF-IDF / Hash    │   │  TTL support          │
   │  (numpy optional) │   │  Persists to JSON     │
   └────────────┬──────┘   └──────────────────────┘
                │
   ┌────────────▼────────────────────────────────┐
   │            EpisodicMemory                    │
   │  ──────────────────────────────────          │
   │  Append-only interaction log                  │
   │  Roles: user / agent / system / tool          │
   │  Search + recent recall                       │
   └─────────────────────────────────────────────┘
```

### The three memory types

| Memory | Lifespan | Best For | Persistence |
|---|---|---|---|
| **Semantic** | Long-term | Facts, past solutions, learnings | JSON file |
| **Working** | Session | Current task, in-flight context | JSON file (auto-save) |
| **Episodic** | Long-term | Interaction history, audit trail | JSON file (append-only) |

---

## Quick Start

### Installation

```bash
pip install tiny-agent-memory
```

Or install from source:

```bash
pip install -e .
```

For numpy-accelerated vector retrieval:

```bash
pip install tiny-agent-memory[numpy]
```

### Basic usage

```python
from tiny_agent_memory import MemoryStore

store = MemoryStore(name="my-agent", persist_dir="/tmp/agent-memory")

# --- Semantic Memory ---
# Long-term knowledge: facts, past solutions
store.add(
    "User prefers dark mode",
    metadata={"type": "preference", "source": "onboarding"},
    memory_type="semantic",
)

# --- Working Memory ---
# Current session: what file are we editing?
store.add("config_loader.py", metadata={"type": "context", "key": "current_file"}, memory_type="working")
store.add("47", metadata={"type": "context", "key": "current_line"}, memory_type="working")

# Direct working memory access
store.working.set("task", "fix the null pointer in config_loader.py")
print(store.working.get("task"))

# --- Episodic Memory ---
# Log interactions with roles
store.episodic.user("Can you fix the deploy script?")
store.episodic.agent("Looking at the deploy script now...")
store.episodic.agent("Found the bug at line 203. Patching...")

# Recall recent interactions
recent = store.episodic.recent(n=5)
for e in recent:
    print(f"[{e.role}] {e.content}")

# --- Cross-memory recall ---
results = store.remember("deploy script")
for r in results:
    print(f"  {r.content} (score: {r.metadata})")

# --- Full snapshot ---
state = store.snapshot()
# {
#   "agent": "my-agent",
#   "timestamp": "2026-08-30T13:00:00Z",
#   "semantic": {"count": 1, "entries": [...]},
#   "working":  {"count": 3, "store": {...}},
#   "episodic": {"count": 3, "entries": [...]},
# }
```

### forget() — selective deletion

```python
# Remove from semantic memory by pattern
removed = store.forget("dark mode", memory_type="semantic")
print(f"Removed {removed} entries")

# Clear working memory
store.working.clear()
```

### Auto-persistence

Memory is automatically saved to disk when a `persist_dir` is provided. On next run with the same `name` + `persist_dir`, the store reloads all prior entries.

---

## API Reference

### MemoryStore

| Method | Returns | Description |
|---|---|---|
| `add(content, metadata, memory_type, importance)` | `str` (entry ID) | Add to semantic, episodic, or working memory |
| `remember(query, top_k)` | `list[MemoryEntry]` | Semantic recall — top-K most relevant |
| `recall_episodic(query, n, role)` | `list[InteractionEntry]` | Search or list recent episodic entries |
| `forget(pattern, memory_type)` | `int` | Remove matching entries |
| `snapshot()` | `dict` | Full memory state as serialisable dict |
| `clear(memory_type)` | `None` | Clear semantic/working/episodic (or all) |
| `flush()` | `None` | Force-persist all layers |
| `semantic` | `SemanticMemory` | Direct semantic access |
| `working` | `WorkingMemory` | Direct working memory access |
| `episodic` | `EpisodicMemory` | Direct episodic access |

### SemanticMemory

| Method | Returns | Description |
|---|---|---|
| `add(content, metadata, importance)` | `str` | Add a memory entry |
| `recall(query, top_k)` | `list[MemoryEntry]` | Retrieve top-K relevant memories |
| `forget(pattern, entry_id)` | `int` | Remove by pattern or exact ID |
| `count()` | `int` | Number of entries |
| `snapshot()` | `dict` | Serialised memory state |

### WorkingMemory

| Method | Returns | Description |
|---|---|---|
| `set(key, value)` | `None` | Set a key-value pair |
| `get(key, default)` | `Any` | Get value (returns default if missing/expired) |
| `delete(key)` | `bool` | Delete a key |
| `keys()` | `list[str]` | List active keys |
| `clear()` | `None` | Clear all |
| `update(data)` | `None` | Bulk update |
| `count()` | `int` | Active key count |

### EpisodicMemory

| Method | Returns | Description |
|---|---|---|
| `add(role, content, metadata, duration_ms)` | `str` | Append an interaction |
| `user(content, **kwargs)` | `str` | Shortcut: add with role="user" |
| `agent(content, **kwargs)` | `str` | Shortcut: add with role="agent" |
| `system(content, **kwargs)` | `str` | Shortcut: add with role="system" |
| `recent(n, role)` | `list[InteractionEntry]` | N most recent (optionally by role) |
| `search(query, top_k)` | `list[InteractionEntry]` | Full-text search |
| `count()` | `int` | Total entries |
| `snapshot()` | `dict` | Serialised episodic state |

---

## Integration with tiny-workflow-engine

```python
from tiny_agent_memory import MemoryStore
from tiny_workflow_engine import WorkflowEngine, Task, pass_state

memory_store = MemoryStore(name="workflow-agent", persist_dir="/tmp/workflow-memory")

engine = WorkflowEngine(name="code-review", persist_dir="/tmp/workflows")

@engine.task()
def review_pr(state):
    pr = state["pr"]
    # Recall any past issues with this repo
    similar = memory_store.remember(f"PR review {pr['repo']}")
    if similar:
        state["past_notes"] = [r.content for r in similar[:3]]
    return state

@engine.task()
def post_review(state):
    result = state["result"]
    # Store in semantic memory for future recall
    memory_store.add(
        f"PR review complete for {result['pr']}: {result['verdict']}",
        metadata={"type": "review", "verdict": result["verdict"]},
    )
    return state

# Run
final = engine.run({"pr": {"repo": "hussain-alsaibai/dev-masterkit", "id": 42}})
snapshot = memory_store.snapshot()
print(f"Memory: {snapshot['semantic']['count']} semantic, "
      f"{snapshot['episodic']['count']} episodic entries")
```

---

## tiny-* Ecosystem

`tiny-agent-memory` is part of the `tiny-*` ecosystem — single-file, stdlib-only, auditable-in-an-afternoon libraries for AI agent primitives.

| Package | Purpose | Benchmark |
|---|---|---|
| `tiny-cli` | CLI argument parsing | 250K parse/sec |
| `tiny-log` | Structured logging | 180K logs/sec |
| `tiny-router` | HTTP request routing | 90K req/sec |
| `tiny-validator` | Schema validation | 247K validations/sec |
| `fast-cache` | In-process cache | 2.2M ops/sec (M3) |
| `tiny-workflow` | DAG orchestration | 500-step DAGs |
| **`tiny-agent-memory`** | **Agent memory** | **Zero-dep** |

All packages: audit in 30 min, cold-start from zero, no supply-chain risk.

---

## License

MIT — see [LICENSE](LICENSE).
