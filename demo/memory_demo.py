"""Demo: drop-in persistent agent memory for mcp-agent, backed by MongoDB / Atlas.

Two things are proven here without needing a full LLM run:

1. **Persistence across runs.** :class:`MongoMemory` implements the same five-method
   contract as mcp-agent's ``SimpleMemory`` (``append/extend/set/get/clear``), but the
   history lives in MongoDB. We simulate two processes (``run_a`` then ``run_b``) that
   share a ``session_id`` — the second "process" reloads the first's history from Atlas.

2. **Semantic recall via Atlas Vector Search.** Each stored message is embedded with
   Voyage AI 3.5 (bring-your-own-vector path) and recalled with ``$vectorSearch``,
   prefiltered to the session.

Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    VOYAGE_API_KEY=...        # bring-your-own-vector mode (default)

Install:
    pip install -e ".[dev]" voyageai

Run:
    python demo/memory_demo.py                    # bring-your-own Voyage vectors
    MEMORY_MODE=auto python demo/memory_demo.py   # Atlas Automated Embedding
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from mcp_agent_mongodb import MongoMemory  # noqa: E402

DEMO_DB = "mcp_agent_mem_demo"
SESSION = "mcp-agent-conversation-1"
MODE = os.environ.get("MEMORY_MODE", "byo").lower()  # "byo" | "auto"
AUTO = MODE == "auto"
VOYAGE_MODEL = "voyage-3.5"  # bring-your-own path; 1024-dim


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def embed(text: str, *, input_type: str) -> list[float]:
    """Embed with the Voyage SDK in *your* code (bring-your-own-vector path)."""
    import voyageai

    client = voyageai.Client()  # reads VOYAGE_API_KEY
    return client.embed([text], model=VOYAGE_MODEL, input_type=input_type).embeddings[0]


def _append(mem: MongoMemory, role: str, content: str) -> None:
    if AUTO:
        mem.append({"role": role, "content": content})
    else:
        mem.append(
            {"role": role, "content": content},
            embedding=embed(content, input_type="document"),
        )


def main() -> None:
    uri = os.environ.get("ATLAS_URI")
    needs = [uri] + ([] if AUTO else [os.environ.get("VOYAGE_API_KEY")])
    if not all(needs):
        extra = "" if AUTO else " + VOYAGE_API_KEY"
        print(f"This demo needs ATLAS_URI{extra} (see demo/.env).")
        sys.exit(1)

    print(f"=== mcp-agent × MongoDB persistent memory (mode={MODE}) ===")

    # --- "Process A": an agent run that records conversation history --------
    banner("RUN A — agent stores conversation history (persisted to MongoDB)")
    mem_a = MongoMemory(uri, session_id=SESSION, database_name=DEMO_DB)
    mem_a.clear()  # clean slate for a repeatable demo

    turns = [
        ("user", "I'm planning a trip to Lisbon. I'm vegetarian and avoid dairy."),
        ("assistant", "Noted — Lisbon trip, vegetarian, no dairy. I'll keep that in mind."),
        ("user", "I also strongly prefer window seats on flights."),
        ("assistant", "Got it: window seats preferred."),
    ]
    for role, content in turns:
        _append(mem_a, role, content)
        print(f"  [{role}] {content}")
    print(f"\n  [MongoDB] persisted {len(mem_a.get())} messages under session_id='{SESSION}'")
    mem_a.close()

    # --- "Process B": a fresh run reloads the same history from Atlas -------
    banner("RUN B — a new process reloads the agent's history from MongoDB")
    mem_b = MongoMemory(uri, session_id=SESSION, database_name=DEMO_DB)
    history = mem_b.get()
    print(f"  Reloaded {len(history)} messages (no in-process state shared):")
    for m in history:
        print(f"    [{m['role']}] {m['content']}")

    # --- Semantic recall via Atlas Vector Search ----------------------------
    banner("Semantic recall over the conversation via Atlas $vectorSearch")
    print("Ensuring vector index (first build can take ~1 min)...")
    ok = (
        mem_b.ensure_vector_index()
        if AUTO
        else mem_b.ensure_vector_index(num_dimensions=1024)
    )
    if not ok:
        print("Vector index did not become queryable in time.")
        sys.exit(1)
    print("Index queryable.")

    # Atlas indexing is async — wait until the stored turns are searchable.
    for _ in range(20):
        probe = (
            mem_b.recall_semantic(query="food", k=1)
            if AUTO
            else mem_b.recall_semantic(query_vector=embed("food", input_type="query"), k=1)
        )
        if probe:
            break
        time.sleep(2)

    q = "What are the traveler's dietary and seating preferences?"
    print(f"\nQuery: {q}")
    hits = (
        mem_b.recall_semantic(query=q, k=3)
        if AUTO
        else mem_b.recall_semantic(query_vector=embed(q, input_type="query"), k=3)
    )
    for h in hits:
        print(f"  • [{h.get('score', 0):.3f}] {h['content']}")

    mem_b.clear()
    mem_b.close()
    print("\nMemory demo complete — history persisted + recalled via MongoDB Atlas.")


if __name__ == "__main__":
    main()
