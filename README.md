# mcp-agent-mongodb

> ⚠️ **ALPHA — NOT AN OFFICIAL MONGODB PRODUCT.** This integration is in **Alpha** and is **not** a supported or official MongoDB product. **Use at your own risk.**

MongoDB Atlas–backed **persistent memory** for [mcp-agent](https://github.com/lastmile-ai/mcp-agent)
(LastMile AI) — a drop-in replacement for the framework's in-process `SimpleMemory`.

- **`MongoMemory`** — implements mcp-agent's `Memory` contract
  (`extend` / `set` / `append` / `get` / `clear`), so you assign it directly to
  `llm.history`. Each message is one MongoDB document, scoped by `session_id` and ordered
  by an append `seq`, so an agent's conversation history **survives restarts** and can be
  shared across processes.
- **Semantic recall** — optional long-term recall over past messages via MongoDB Vector
  Search (`$vectorSearch`). Embedding source-agnostic: **bring your own query vector**
  (default) or enable **Atlas Automated Embedding** (server-side embeddings, no client code).

## Capabilities

| Capability | How |
|---|---|
| Persistent agent memory (MS) | `MongoMemory` as a drop-in `llm.history` |
| Semantic recall (VS) | `recall_semantic()` over MongoDB `$vectorSearch`, session-prefiltered |
| Survives restarts / multi-process | history keyed by `session_id`, stored in MongoDB |
| TTL expiry | optional TTL index on `ts` |

## Why

mcp-agent's every `AugmentedLLM` keeps conversation history in a `Memory` object; the
default `SimpleMemory` holds it in RAM and loses it when the process ends. `MongoMemory`
is a database-backed `Memory`: drop it in and the agent's history is durable, queryable,
and shareable — with optional semantic recall over everything it has seen.

## Architecture

```
AugmentedLLM.history  ──►  MongoMemory(connection_string, session_id=…)
   append/extend/set/get/clear        │
                                       ▼
                          MongoDB / Atlas  "memory" collection
                          { session_id, seq, role, message, content, embedding?, ts }
                                       │
                       recall_semantic ▼  (optional)
                          MongoDB Vector Search  $vectorSearch (session-prefiltered)
```

## Install

```bash
pip install mcp-agent-mongodb
```

## Quick start (drop-in persistent history)

```python
from google.genai import types
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent_mongodb import MongoMemory

async with agent:
    llm = await agent.attach_llm(GoogleAugmentedLLM)
    # Swap the in-process history for a MongoDB-backed one:
    llm.history = MongoMemory(
        "mongodb+srv://...",
        session_id="user-123",
        message_model=types.Content,   # rehydrate provider message objects on read
    )
    # ...subsequent runs with the same session_id reload this history from MongoDB.
```

`MongoMemory` matches the `Memory` contract exactly:
`extend(messages)`, `set(messages)`, `append(message)`, `get() -> list`, `clear()`.

### Options

| Arg | Default | Purpose |
|---|---|---|
| `connection_string` | — | MongoDB / Atlas URI (**required**) |
| `session_id` | — | Conversation/agent scope (**required**); every read/write is filtered by it |
| `database_name` | `mcp_agent` | Database name |
| `collection_name` | `memory` | Collection name |
| `message_model` | `None` | Optional pydantic model to rehydrate stored messages on `get()` |
| `vector_search_index` | `idx_agent_memory` | MongoDB Vector Search index name |
| `auto_embed` | `False` | Enable Atlas Automated Embedding (recall by query text) |
| `auto_embed_model` | `voyage-4` | Voyage model used by Automated Embedding |
| `ttl_seconds` | `None` | If set, TTL index on `ts` auto-expires idle conversations |

### Document shape

```jsonc
{
  "session_id": "user-123",
  "seq": 7,
  "role": "user",
  "message": { /* the serialized message, returned verbatim from get() */ },
  "content": "summarize the Q3 report",
  "embedding": [ /* 1024 floats — bring-your-own-vector path only */ ],
  "ts": { "$date": "..." }
}
```

## Semantic recall (MongoDB Vector Search)

The package never calls an embedding provider itself — choose one of **two first-class paths**:

**1. Bring your own vector (default).**

```python
mem = MongoMemory("mongodb+srv://...", session_id="user-123")
mem.ensure_vector_index(num_dimensions=1024)         # one-time, on Atlas

mem.append({"role": "user", "content": "I prefer window seats."},
           embedding=my_provider.embed("I prefer window seats."))

hits = mem.recall_semantic(query_vector=my_provider.embed("seating?"), k=5)
```

**2. Atlas Automated Embedding.** Atlas embeds server-side (no client embedding code):

```python
mem = MongoMemory("mongodb+srv://...", session_id="user-123", auto_embed=True)
mem.ensure_vector_index()                            # builds an `autoEmbed` index
mem.append({"role": "user", "content": "I prefer window seats."})
hits = mem.recall_semantic(query="seating preferences", k=5)
```

There is **no silent fallback**: if you neither pass a `query_vector` nor enable
`auto_embed`, `recall_semantic` raises `ValueError`.

## MCP + MongoDB synergy

mcp-agent is built on the Model Context Protocol. Pair this memory backend with
[`mongodb-partners/memory-mcp`](https://github.com/mongodb-partners/memory-mcp) to make
Atlas both the **agent memory backend** (this package) and an **MCP memory server** your
agents can call as a tool.

## Demos

- **`demo/memory_demo.py`** — persistence across two simulated processes + MongoDB Vector
  Search recall (bring-your-own Voyage vectors; `MEMORY_MODE=auto` for Automated Embedding).
- **`demo/agent_demo.py`** — a real Gemini mcp-agent whose `llm.history` is a `MongoMemory`;
  Session 2 (brand-new app/agent/LLM) answers using history reloaded from Atlas.

```bash
pip install -e ".[dev]" "mcp-agent" "google-genai" voyageai
# demo/.env: ATLAS_URI, VOYAGE_API_KEY, GEMINI_API_KEY
python demo/memory_demo.py                  # bring-your-own vectors
MEMORY_MODE=auto python demo/memory_demo.py # Atlas Automated Embedding
python demo/agent_demo.py                   # Gemini agent, cross-session memory
```

## Why MongoDB

One database for agent state: durable conversation history, semantic recall via MongoDB
Vector Search, TTL lifecycle, and flexible documents for arbitrary provider message
shapes — no separate vector store to operate.

## Conventions

- The package **owns its `MongoClient`** (built from a connection string). Connection
  `appName` = `devrel-integ-mcp-agent-python` and the `mcp-agent-mongodb` driver-info
  handshake are always set and **non-overridable** (server-side attribution).
- Embeddings use **Voyage AI 3.5** (`voyage-3.5`, 1024-dim) on the bring-your-own path.

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 17 tests, mongomock — no infra required
```

## Resources

- mcp-agent: https://github.com/lastmile-ai/mcp-agent · docs: https://docs.mcp-agent.com
- MongoDB MongoDB Vector Search: https://www.mongodb.com/docs/atlas/atlas-vector-search/
- Voyage AI embeddings: https://docs.voyageai.com

## License

MIT
