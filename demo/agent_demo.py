"""Agentic demo: a real mcp-agent (Gemini) agent with MongoDB-persisted memory.

This proves the headline feature — :class:`MongoMemory` is a **drop-in replacement** for
mcp-agent's in-process ``SimpleMemory``. We run two *separate* agent sessions:

- **Session 1** uses a fresh ``GoogleAugmentedLLM`` whose ``llm.history`` is swapped for a
  :class:`MongoMemory` bound to a ``session_id``. The user tells the agent a durable fact.
  The conversation turns are persisted to MongoDB / Atlas.

- **Session 2** builds a brand-new app + agent + LLM (no shared in-process state) and
  swaps in a ``MongoMemory`` for the **same** ``session_id``. Its history is reloaded from
  MongoDB, so the agent answers a follow-up that depends on Session 1 — memory survived
  the process boundary because it lives in the database, not RAM.

mcp-agent stores history as ``google.genai`` ``types.Content`` objects, so we construct
the store with ``message_model=types.Content`` to rehydrate them on read.

Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    GEMINI_API_KEY=...        # Google AI (Gemini) key

Install:
    pip install -e . "mcp-agent" "google-genai"

Run:
    python demo/agent_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from google.genai import types  # noqa: E402
from mcp_agent.agents.agent import Agent  # noqa: E402
from mcp_agent.app import MCPApp  # noqa: E402
from mcp_agent.config import GoogleSettings, Settings  # noqa: E402
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM  # noqa: E402

from mcp_agent_mongodb import MongoMemory  # noqa: E402

DEMO_DB = "mcp_agent_agent_demo"
SESSION = "mcp-agent-user-7"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _settings() -> Settings:
    """Build mcp-agent Settings with the Google (Gemini) provider from env."""
    return Settings(
        execution_engine="asyncio",
        google=GoogleSettings(
            api_key=os.environ["GEMINI_API_KEY"],
            default_model=MODEL,
        ),
    )


async def run_session(uri: str, prompt: str, *, label: str) -> str:
    """Run one agent turn whose history is a MongoDB-backed MongoMemory."""
    app = MCPApp(name=f"mongo_memory_demo_{label}", settings=_settings())
    async with app.run():
        agent = Agent(
            name="memory_assistant",
            instruction=(
                "You are a helpful personal assistant with long-term memory. "
                "Use what you remember about the user from earlier in the conversation "
                "to answer. Keep answers to 1-2 sentences."
            ),
        )
        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)

            # ---- The drop-in: swap SimpleMemory for MongoDB-backed memory. ----
            llm.history = MongoMemory(
                uri,
                session_id=SESSION,
                database_name=DEMO_DB,
                message_model=types.Content,
            )

            reloaded = len(llm.history.get())
            print(f"[{label}] reloaded {reloaded} messages from MongoDB")
            print(f"User: {prompt}")
            answer = await llm.generate_str(prompt)
            print(f"Agent: {answer}")
            print(f"[{label}] history now has {len(llm.history.get())} messages in MongoDB")
            return answer


async def main() -> None:
    uri = os.environ.get("ATLAS_URI")
    if not uri or not os.environ.get("GEMINI_API_KEY"):
        print("This demo needs ATLAS_URI + GEMINI_API_KEY (see demo/.env).")
        sys.exit(1)

    # Clean slate for a repeatable demo.
    seed = MongoMemory(uri, session_id=SESSION, database_name=DEMO_DB)
    seed.clear()
    seed.close()

    print(f"=== mcp-agent × MongoDB persistent memory (model={MODEL}) ===")

    banner("SESSION 1 — agent learns a durable fact (history persisted to MongoDB)")
    await run_session(
        uri,
        "Hi! Please remember that I'm planning a trip to Lisbon in March and I'm vegetarian.",
        label="session-1",
    )

    banner("SESSION 2 — brand-new app/agent/LLM reloads history from MongoDB")
    answer = await run_session(
        uri,
        "Based on what you know about me, suggest one thing I should book for my trip.",
        label="session-2",
    )

    banner("Result")
    print("Session 2 had NO shared in-process state with Session 1.")
    print("It answered using history reloaded from MongoDB Atlas:")
    print(f"  → {answer}")

    # Cleanup.
    cleanup = MongoMemory(uri, session_id=SESSION, database_name=DEMO_DB)
    cleanup.clear()
    cleanup.close()
    print("\nAgent demo complete — Gemini used MongoDB-persisted history across sessions.")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
