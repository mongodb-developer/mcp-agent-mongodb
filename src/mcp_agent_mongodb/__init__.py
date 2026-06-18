"""MongoDB-backed persistent memory for mcp-agent (LastMile AI).

mcp-agent stores an agent's conversation history through a :class:`Memory` base class
(``mcp_agent.workflows.llm.augmented_llm.Memory``) with a five-method contract:
``extend`` / ``set`` / ``append`` / ``get`` / ``clear``. The default is
``SimpleMemory`` (in-process, lost on restart). :class:`MongoMemory` is a drop-in
replacement that persists every message to MongoDB / Atlas — scoped by ``session_id``
and ordered by an append ``seq`` — so an agent's history survives restarts and can be
shared across processes.

It also adds optional **semantic recall** over past messages via Atlas Vector Search
(``$vectorSearch``), embedding source-agnostic: bring your own query vector (default) or
enable Atlas Automated Embedding (server-side embeddings, no client code).

Example (drop-in persistent history)::

    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
    from mcp_agent_mongodb import MongoMemory

    llm = await agent.attach_llm(OpenAIAugmentedLLM)
    # swap the in-process history for a MongoDB-backed one
    llm.history = MongoMemory("mongodb+srv://...", session_id="user-123")

Example (semantic recall — bring your own vector)::

    mem = MongoMemory("mongodb+srv://...", session_id="user-123")
    mem.append({"role": "user", "content": "I prefer window seats."},
               embedding=my_provider.embed("I prefer window seats."))
    hits = mem.recall_semantic(query_vector=my_provider.embed("seating?"))

Example (semantic recall — Atlas Automated Embedding)::

    mem = MongoMemory("mongodb+srv://...", session_id="user-123", auto_embed=True)
    mem.ensure_vector_index()                  # builds an autoEmbed index
    mem.append({"role": "user", "content": "I prefer window seats."})
    hits = mem.recall_semantic(query="seating preferences")
"""

from __future__ import annotations

__version__ = "0.1.1"

from .memory import (
    APP_NAME,
    DEFAULT_AUTO_EMBED_MODEL,
    DRIVER_NAME,
    MongoMemory,
)

__all__ = [
    "MongoMemory",
    "DEFAULT_AUTO_EMBED_MODEL",
    "APP_NAME",
    "DRIVER_NAME",
    "__version__",
]
