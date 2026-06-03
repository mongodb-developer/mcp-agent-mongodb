# AGENTS.md — guide for AI coding agents

A structured guide for AI agents working in `mcp-agent-mongodb`: how to build and test,
where key files live, and the MongoDB-specific rules to follow.

## Build and test commands

```bash
# Install (editable) + dev deps
pip install -e ".[dev]"

# Run the test suite (mongomock — no infra required)
pytest -q                # 17 tests

# Demos (need Atlas + keys; see demo/.env: ATLAS_URI, VOYAGE_API_KEY, GEMINI_API_KEY)
pip install -e ".[dev]" "mcp-agent" "google-genai" voyageai
python demo/memory_demo.py                  # bring-your-own Voyage vectors
MEMORY_MODE=auto python demo/memory_demo.py # Atlas Automated Embedding
python demo/agent_demo.py                   # Gemini agent, cross-session memory
```

## Project structure

- `src/mcp_agent_mongodb/memory.py` — `MongoMemory` (the `Memory` adapter + vector recall).
- `src/mcp_agent_mongodb/__init__.py` — public exports + `__version__`.
- `tests/test_acceptance.py` — acceptance tests (mongomock; Atlas handshake test auto-skips).
- `demo/` — runnable demos over Atlas (`memory_demo.py`, `agent_demo.py`).
- `EDD.md` — the MongoDB data model (source of truth for schema).
- `PLAN.md` — the integration plan (phases + acceptance criteria).

## Upstream extension point

mcp-agent's `Memory[MessageParamT]` base class
(`mcp_agent/workflows/llm/augmented_llm.py`) defines `extend` / `set` / `append` / `get` /
`clear`. `AugmentedLLM.__init__` assigns `self.history = SimpleMemory()`, so `MongoMemory`
is a **drop-in**: just set `llm.history = MongoMemory(...)`. Messages are arbitrary
`MessageParamT` (provider message objects such as `google.genai.types.Content`, or dicts).

## Environment variables and configuration

| Name | Required | Description |
|---|---|---|
| `ATLAS_URI` | demos / vector tests | Atlas connection string |
| `VOYAGE_API_KEY` | bring-your-own embedding demos | Voyage AI key for `voyage-3.5` |
| `GEMINI_API_KEY` | agent demo | Gemini (Google AI) key |
| `MEMORY_MODE` | optional | `auto` to use Atlas Automated Embedding in `memory_demo.py` |

## Conventions (do not break)

- The package **owns its `MongoClient`** — built from a connection string. `appName`
  (`devrel-integ-mcp-agent-python`) and the `mcp-agent-mongodb` driver-info handshake are
  always set and **non-overridable** (caller `appname`/`appName`/`driver` are stripped).
- Embeddings use **Voyage AI 3.5** (`voyage-3.5`, 1024-dim) on the bring-your-own path.
- No silent embedding fallback: `recall_semantic` raises if neither `query_vector` nor
  `auto_embed` is provided.
- `get()` preserves insertion order via the per-session `seq` counter.

## MongoDB Skills

Use the official MongoDB agent skills from https://github.com/mongodb/agent-skills
whenever the task is MongoDB-specific and a matching skill exists.

## When To Use EDD.md

Use [EDD.md](./EDD.md) as the source of truth for the MongoDB data model in this repository.

Consult [EDD.md](./EDD.md) before making changes that touch:

- The `memory` collection, document structure, or field names
- Code paths that read or write database records (`memory.py`)
- Index definitions (the `(session_id, seq)` index, optional TTL on `ts`, the Atlas Vector
  Search index)
- Validation, payloads, or anything that depends on persisted data
- Schema documentation, Mermaid diagrams, or entity modeling discussions
