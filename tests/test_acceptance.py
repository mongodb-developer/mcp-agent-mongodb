"""Acceptance tests for the mcp-agent × MongoDB Memory backend.

Each test maps to a numbered criterion in PLAN.md (Phase 2). Non-Atlas tests run on
``mongomock`` and need no infrastructure. The Atlas-only ``vector_recall`` /
handshake tests skip automatically when ``ATLAS_URI`` is unset.
"""

from __future__ import annotations

import os

import pytest

from mcp_agent_mongodb import APP_NAME, DRIVER_NAME, MongoMemory


def _sample_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Hello, remember my color is blue."},
        {"role": "assistant", "content": "Got it — blue."},
    ]


# Criterion 1 — append/extend then get() returns messages in insertion order.
def test_round_trip(mock_memory):
    mock_memory.append({"role": "user", "content": "first"})
    mock_memory.extend(
        [{"role": "assistant", "content": "second"}, {"role": "user", "content": "third"}]
    )

    history = mock_memory.get()
    assert [m["content"] for m in history] == ["first", "second", "third"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]


# Criterion 2 — set() overwrites prior history for the session.
def test_set_replaces(mock_memory):
    mock_memory.extend(_sample_messages())
    assert len(mock_memory.get()) == 2

    mock_memory.set([{"role": "user", "content": "only one now"}])
    history = mock_memory.get()
    assert len(history) == 1
    assert history[0]["content"] == "only one now"


# Criterion 3 — clear() removes all messages for the session.
def test_clear_empties(mock_memory):
    mock_memory.extend(_sample_messages())
    assert mock_memory.get()
    mock_memory.clear()
    assert mock_memory.get() == []


# Criterion 4 — scope isolation between session_ids.
def test_scope_isolation(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    factory = make_mongomock_factory()
    monkeypatch.setattr(memory_mod, "MongoClient", factory)

    mem_a = MongoMemory("mongodb://x", session_id="A", database_name="iso")
    # Reuse the same underlying mongomock client/db so both memories share storage.
    mem_b = MongoMemory("mongodb://x", session_id="B", database_name="iso")
    mem_b.client = mem_a.client
    mem_b.db = mem_a.db
    mem_b.memory = mem_a.memory

    mem_a.extend(_sample_messages())
    mem_b.append({"role": "user", "content": "different"})

    assert len(mem_a.get()) == 2
    assert len(mem_b.get()) == 1
    assert mem_a.get() != mem_b.get()


# Criterion 5 — pydantic-model messages serialize and reload intact.
def test_pydantic_message_round_trip(monkeypatch):
    from conftest import make_mongomock_factory
    from pydantic import BaseModel

    import mcp_agent_mongodb.memory as memory_mod

    class Msg(BaseModel):
        role: str
        content: str

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://x", session_id="pyd", database_name="pyd_db", message_model=Msg
    )
    mem.append(Msg(role="user", content="typed message"))

    history = mem.get()
    assert isinstance(history[0], Msg)
    assert history[0].role == "user"
    assert history[0].content == "typed message"


# Criterion 5b — plain string messages round-trip (wrapped + unwrapped).
def test_string_message_round_trip(mock_memory):
    mock_memory.append("just a string")
    assert mock_memory.get() == ["just a string"]


# Criterion 6 — TTL index created when ttl_seconds is set.
def test_ttl_index_created(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://x", session_id="ttl", database_name="ttl_db", ttl_seconds=3600
    )
    indexes = mem.memory.index_information()
    ttl = [v for v in indexes.values() if v.get("expireAfterSeconds") == 3600]
    assert ttl, f"expected a TTL index with expireAfterSeconds=3600, got {indexes}"


# Criterion 6b — session/seq order index exists.
def test_order_index_created(mock_memory):
    indexes = mock_memory.memory.index_information()
    assert "session_seq" in indexes


# Criterion 7 — recall_semantic enforces the embedding path (no silent fallback).
def test_byo_recall_requires_vector(mock_memory):
    with pytest.raises(ValueError, match="No `query_vector` provided"):
        mock_memory.recall_semantic(query="some text")


def test_auto_embed_rejects_byo_vector(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://x", session_id="ae", database_name="ae_db", auto_embed=True
    )
    with pytest.raises(ValueError, match="uses query text, not"):
        mem.recall_semantic(query_vector=[0.1, 0.2])
    with pytest.raises(ValueError, match="expects `query` text"):
        mem.recall_semantic()


def test_auto_embed_rejects_stored_vector(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://x", session_id="ae2", database_name="ae2_db", auto_embed=True
    )
    with pytest.raises(ValueError, match="manages embeddings server-side"):
        mem.append({"role": "user", "content": "x"}, embedding=[0.1, 0.2])


# Criterion 7b — vector index definition shapes per embedding path.
def test_vector_index_definition_byo(mock_memory):
    definition = mock_memory.vector_index_definition(num_dimensions=1024)
    fields = definition["fields"]
    vec = [f for f in fields if f["type"] == "vector"][0]
    assert vec["path"] == "embedding"
    assert vec["numDimensions"] == 1024
    assert any(f["type"] == "filter" and f["path"] == "session_id" for f in fields)


def test_vector_index_definition_auto_embed(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    mem = MongoMemory(
        "mongodb://x", session_id="ae3", database_name="ae3_db", auto_embed=True
    )
    definition = mem.vector_index_definition()
    auto = [f for f in definition["fields"] if f["type"] == "autoEmbed"][0]
    assert auto["path"] == "content"
    assert auto["model"] == "voyage-4"


# Criterion 8a — appName + driver-info constants are well-formed.
def test_appname_and_driver_constants():
    assert APP_NAME == "devrel-integ-mcp-agent-python"
    assert DRIVER_NAME == "mcp-agent-mongodb"


# Criterion 8b — required constructor args are enforced.
def test_requires_connection_string_and_session():
    with pytest.raises(ValueError, match="connection_string is required"):
        MongoMemory("", session_id="x")
    with pytest.raises(ValueError, match="session_id is required"):
        MongoMemory("mongodb://x", session_id="")


# Criterion 8c — tracking is not overridable by the caller.
def test_tracking_not_overridable(monkeypatch):
    from conftest import make_mongomock_factory

    import mcp_agent_mongodb.memory as memory_mod

    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return make_mongomock_factory()(*args, **kwargs)

    monkeypatch.setattr(memory_mod, "MongoClient", factory)
    MongoMemory(
        "mongodb://x",
        session_id="evil",
        database_name="x",
        appname="evil-app",
        appName="evil-app",
        driver="not-a-driver",
    )
    assert captured.get("appname") == APP_NAME
    assert captured.get("driver") is not None
    assert captured["driver"].name == DRIVER_NAME


# Criterion 8d (Atlas, optional) — real client carries appName + driver_info handshake.
@pytest.mark.skipif(not os.environ.get("ATLAS_URI"), reason="ATLAS_URI not set")
def test_real_client_handshake_metadata(atlas_uri):
    mem = MongoMemory(atlas_uri, session_id="handshake", database_name="mcp_agent_test")
    try:
        opts = mem.client.options
        assert opts.pool_options.metadata["application"]["name"] == APP_NAME
        driver_name = opts.pool_options.metadata["driver"]["name"]
        assert DRIVER_NAME in driver_name
    finally:
        mem.clear()
        mem.close()
