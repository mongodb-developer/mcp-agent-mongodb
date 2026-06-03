"""Shared pytest fixtures for the mcp-agent × MongoDB acceptance suite.

Non-search tests run on ``mongomock`` (no infra needed). Because the memory **always
owns its own client** (connection-string only), the offline fixture patches
``MongoClient`` in the memory module with a mongomock-backed shim that tolerates the
``appname``/``driver`` kwargs the memory always sets. The optional Atlas-backed
handshake check is skipped automatically when ATLAS_URI is unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "demo" / ".env")
except ImportError:
    pass


def make_mongomock_factory():
    """Return a ``MongoClient`` replacement backed by mongomock.

    mongomock's client ignores (and rejects) the real driver's ``appname``/``driver``
    kwargs, so we strip them before delegating.
    """
    import mongomock

    def factory(*args, **kwargs):
        kwargs.pop("appname", None)
        kwargs.pop("appName", None)
        kwargs.pop("driver", None)
        return mongomock.MongoClient(*args, **kwargs)

    return factory


@pytest.fixture()
def mock_memory(monkeypatch):
    """A MongoMemory backed by mongomock (works offline), scoped to session ``s1``."""
    import mcp_agent_mongodb.memory as memory_mod
    from mcp_agent_mongodb import MongoMemory

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://localhost:27017", session_id="s1", database_name="test_db"
    )
    yield mem


@pytest.fixture()
def atlas_uri() -> str | None:
    return os.environ.get("ATLAS_URI")
