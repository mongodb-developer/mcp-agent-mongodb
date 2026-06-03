"""MongoDB / Atlas-backed :class:`Memory` for mcp-agent (LastMile AI).

mcp-agent's :class:`mcp_agent.workflows.llm.augmented_llm.Memory` base class defines a
five-method contract used by every ``AugmentedLLM`` to store its conversation history:

- ``extend(messages) -> None``
- ``set(messages) -> None``
- ``append(message) -> None``
- ``get() -> list``
- ``clear() -> None``

The framework default, ``SimpleMemory``, keeps the list in process memory and loses it
on restart. :class:`MongoMemory` implements the same contract over a MongoDB collection,
so it is a **drop-in replacement** (``llm.history = MongoMemory(...)``): each message is
one document, scoped by ``session_id`` and ordered by a monotonic ``seq``.

Messages are arbitrary ``MessageParamT`` — plain dicts or pydantic models. They are
serialized for storage (``model_dump()`` when available) under ``message`` and returned
verbatim from :meth:`get`. An optional pydantic ``message_model`` rehydrates documents
back into model instances on read.

It additionally offers **semantic recall** over stored messages through Atlas Vector
Search. Like the rest of the 100 Integs suite, it stays embedding source-agnostic with
two first-class paths and **no silent provider fallback**:

1. **Bring-your-own vector (default).** Pass ``embedding=`` to :meth:`append` /
   :meth:`extend` and ``query_vector=`` to :meth:`recall_semantic`.
2. **Atlas Automated Embedding** (``auto_embed=True``). Atlas embeds ``content``
   server-side at index- and query-time; pass ``query=`` text to recall.

Conventions applied here (100 Integs — baked in, non-overridable):
- ``appName = devrel-integ-mcp-agent-python`` so server telemetry attributes traffic.
- ``driver_info`` handshake metadata identifies the ``mcp-agent-mongodb`` library.
- The memory **always constructs and owns its own** ``MongoClient`` from a connection
  string, so these are guaranteed present on every connection with no caller opt-out.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from pymongo import ASCENDING, MongoClient
from pymongo.driver_info import DriverInfo

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

APP_NAME = "devrel-integ-mcp-agent-python"
"""MongoDB connection appName for server-side attribution (100 Integs convention)."""

DRIVER_NAME = "mcp-agent-mongodb"
"""driver_info name attached to the MongoDB handshake (distinct from appName)."""

# Default Atlas Automated Embedding model (the recommended general-text model).
DEFAULT_AUTO_EMBED_MODEL = "voyage-4"

# Resolve the package version lazily so driver_info reports the installed version.
try:  # pragma: no cover - trivial
    from . import __version__ as _PKG_VERSION
except Exception:  # pragma: no cover
    _PKG_VERSION = "0.0.0"


def _serialize(message: Any) -> dict[str, Any]:
    """Serialize an arbitrary mcp-agent message to a storable dict.

    Handles pydantic models (``model_dump``), plain dicts, and falls back to a
    ``{"value": ...}`` wrapper for primitives/strings.
    """
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(message, dict):
        return message
    return {"value": message}


def _extract_content(message: Any, doc: dict[str, Any]) -> str | None:
    """Best-effort extraction of a text ``content`` field for embedding / recall."""
    content = doc.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI/Anthropic-style content parts: join any text segments.
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("text")]
        if parts:
            return " ".join(parts)
    if isinstance(message, str):
        return message
    return None


class MongoMemory:
    """A MongoDB / Atlas-backed :class:`Memory` for mcp-agent's ``AugmentedLLM``.

    Implements the ``extend`` / ``set`` / ``append`` / ``get`` / ``clear`` contract so it
    can be assigned directly to ``llm.history``. Each message is stored as one document in
    the ``memory`` collection::

        {
          "session_id": "user-123",     # conversation / agent scope
          "seq":        7,               # monotonic order within the session
          "role":       "user",          # best-effort, extracted from the message
          "message":    { ... },         # the serialized message, returned verbatim
          "content":    "…",            # best-effort text for embedding / recall
          "embedding":  [ ... ],         # present only on the BYO-vector path
          "ts":         ISODate(...)
        }

    The memory **always constructs and owns its own** :class:`MongoClient` from the
    supplied ``connection_string``; ``appName`` and ``driver_info`` are baked in and
    cannot be overridden.

    Args:
        connection_string: MongoDB / Atlas connection URI. **Required.**
        session_id: Scope key for this conversation/agent. **Required** — every read and
            write is filtered by it, which is what makes histories isolated and resumable.
        database_name: Database to use. Default ``"mcp_agent"``.
        collection_name: Collection for message documents. Default ``"memory"``.
        message_model: Optional pydantic model class used to rehydrate stored documents
            into model instances on :meth:`get`. When ``None`` (default), the raw stored
            dict is returned.
        vector_search_index: Name of the Atlas Vector Search index. Default
            ``"idx_agent_memory"``.
        auto_embed: When True, enables the **Atlas Automated Embedding** path —
            :meth:`ensure_vector_index` builds an ``autoEmbed`` index and recall sends
            query *text*. When False (default), you supply vectors yourself.
        auto_embed_model: Voyage AI model used by Atlas Automated Embedding (only when
            ``auto_embed=True``). Default ``"voyage-4"``.
        ttl_seconds: If set, a TTL index on ``ts`` auto-expires idle conversations.
        **client_kwargs: Extra keyword arguments forwarded to :class:`MongoClient`
            (e.g. ``tls=True``). ``appname``/``appName``/``driver`` are reserved and
            overridden with the convention values.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        session_id: str,
        database_name: str = "mcp_agent",
        collection_name: str = "memory",
        message_model: type | None = None,
        vector_search_index: str = "idx_agent_memory",
        auto_embed: bool = False,
        auto_embed_model: str = DEFAULT_AUTO_EMBED_MODEL,
        ttl_seconds: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        if not connection_string:
            raise ValueError("connection_string is required")
        if not session_id:
            raise ValueError("session_id is required")

        self.session_id = session_id
        self.message_model = message_model
        self.vector_search_index = vector_search_index
        self.auto_embed = auto_embed
        self.auto_embed_model = auto_embed_model
        self.ttl_seconds = ttl_seconds

        driver_info = DriverInfo(name=DRIVER_NAME, version=_PKG_VERSION)

        # The integration owns the client. appName + driver_info are mandatory and
        # non-overridable: strip any caller-supplied values so tracking is always present.
        client_kwargs.pop("appname", None)
        client_kwargs.pop("appName", None)
        client_kwargs.pop("driver", None)

        self._owns_client = True
        self.client = MongoClient(
            connection_string,
            appname=APP_NAME,
            driver=driver_info,
            **client_kwargs,
        )

        self.db: Database = self.client[database_name]
        self.memory: Collection = self.db[collection_name]

        self._ensure_indexes()

    # -- setup -----------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the ``(session_id, seq)`` order index and the optional TTL index."""
        self.memory.create_index(
            [("session_id", ASCENDING), ("seq", ASCENDING)],
            name="session_seq",
        )
        if self.ttl_seconds is not None:
            self.memory.create_index(
                [("ts", ASCENDING)],
                name="ttl_ts",
                expireAfterSeconds=self.ttl_seconds,
            )

    def _next_seq(self) -> int:
        """Return the next monotonic ``seq`` for this session (0-based, contiguous)."""
        last = self.memory.find_one(
            {"session_id": self.session_id},
            sort=[("seq", -1)],
            projection={"seq": 1},
        )
        return (last["seq"] + 1) if last else 0

    def _build_doc(
        self,
        message: Any,
        seq: int,
        *,
        embedding: list[float] | None,
    ) -> dict[str, Any]:
        serialized = _serialize(message)
        doc: dict[str, Any] = {
            "session_id": self.session_id,
            "seq": seq,
            "role": serialized.get("role") if isinstance(serialized, dict) else None,
            "message": serialized,
            "content": _extract_content(message, serialized),
            "ts": _dt.datetime.now(_dt.timezone.utc),
        }
        if self.auto_embed and embedding is not None:
            raise ValueError(
                "auto_embed=True manages embeddings server-side; do not pass `embedding`. "
                "Atlas generates it from `content`."
            )
        if embedding is not None:
            doc["embedding"] = embedding
        return doc

    def _rehydrate(self, doc: dict[str, Any]) -> Any:
        """Return the original message for a stored document."""
        stored = doc.get("message")
        if self.message_model is not None and isinstance(stored, dict):
            return self.message_model(**stored)
        # Unwrap the primitive wrapper produced by ``_serialize``.
        if isinstance(stored, dict) and set(stored.keys()) == {"value"}:
            return stored["value"]
        return stored

    # -- Memory contract (drop-in for SimpleMemory) ----------------------------

    def append(self, message: Any, *, embedding: list[float] | None = None) -> None:
        """Persist one message at the next ``seq`` for this session.

        ``embedding`` is optional and only used for the bring-your-own-vector recall
        path. Matches mcp-agent's ``Memory.append(message)`` (the keyword is additive
        and ignored by the framework, which calls it positionally).
        """
        doc = self._build_doc(message, self._next_seq(), embedding=embedding)
        self.memory.insert_one(doc)

    def extend(
        self,
        messages: list[Any],
        *,
        embeddings: list[list[float] | None] | None = None,
    ) -> None:
        """Persist many messages with contiguous ``seq`` values (insertion order).

        ``embeddings``, when provided, must align positionally with ``messages``.
        """
        if not messages:
            return
        start = self._next_seq()
        docs: list[dict[str, Any]] = []
        for i, message in enumerate(messages):
            emb = embeddings[i] if embeddings is not None else None
            docs.append(self._build_doc(message, start + i, embedding=emb))
        self.memory.insert_many(docs)

    def set(self, messages: list[Any]) -> None:
        """Replace the entire history for this session with ``messages``."""
        self.memory.delete_many({"session_id": self.session_id})
        self.extend(messages)

    def get(self) -> list[Any]:
        """Return all messages for this session in insertion order."""
        cursor = self.memory.find({"session_id": self.session_id}).sort(
            [("seq", ASCENDING)]
        )
        return [self._rehydrate(doc) for doc in cursor]

    def clear(self) -> None:
        """Delete all messages for this session."""
        self.memory.delete_many({"session_id": self.session_id})

    # -- Atlas Vector Search index ---------------------------------------------

    def vector_index_definition(
        self,
        *,
        num_dimensions: int | None = None,
        similarity: str = "cosine",
        filter_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the Atlas Vector Search index ``definition`` for this memory.

        - ``auto_embed=True`` → an ``autoEmbed`` field on ``content`` (Atlas embeds
          server-side using ``auto_embed_model``).
        - ``auto_embed=False`` → a classic ``vector`` field on ``embedding`` (requires
          ``num_dimensions``).

        ``session_id`` is always added as a ``filter`` path, plus any ``filter_paths``.
        """
        fields: list[dict[str, Any]]
        if self.auto_embed:
            fields = [
                {
                    "type": "autoEmbed",
                    "modality": "text",
                    "path": "content",
                    "model": self.auto_embed_model,
                }
            ]
        else:
            if not num_dimensions:
                raise ValueError(
                    "num_dimensions is required for the bring-your-own-vector index "
                    "(set it to your embedding model's dimensionality, e.g. 1024)."
                )
            fields = [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": num_dimensions,
                    "similarity": similarity,
                }
            ]
        fields.append({"type": "filter", "path": "session_id"})
        for path in filter_paths or []:
            fields.append({"type": "filter", "path": path})
        return {"fields": fields}

    def ensure_vector_index(
        self,
        *,
        num_dimensions: int | None = None,
        similarity: str = "cosine",
        filter_paths: list[str] | None = None,
        wait: bool = True,
        timeout: int = 180,
    ) -> bool:
        """Create the Atlas Vector Search index if missing. Returns True once queryable.

        Requires an Atlas cluster (``createSearchIndexes`` is unsupported on local
        MongoDB / mongomock). No-op-safe if the index already exists.
        """
        from pymongo.operations import SearchIndexModel

        existing = {idx["name"] for idx in self.memory.list_search_indexes()}
        if self.vector_search_index not in existing:
            model = SearchIndexModel(
                definition=self.vector_index_definition(
                    num_dimensions=num_dimensions,
                    similarity=similarity,
                    filter_paths=filter_paths,
                ),
                name=self.vector_search_index,
                type="vectorSearch",
            )
            self.memory.create_search_index(model)

        if not wait:
            return False

        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            for idx in self.memory.list_search_indexes():
                if idx["name"] == self.vector_search_index and idx.get("queryable"):
                    return True
            time.sleep(3)
        return False

    # -- recall ----------------------------------------------------------------

    def recall_semantic(
        self,
        *,
        query: str | None = None,
        query_vector: list[float] | None = None,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic recall over this session's messages via Atlas ``$vectorSearch``.

        Two mutually exclusive query modes:

        - **Bring-your-own vector (default):** pass ``query_vector=<list[float]>``.
        - **Atlas Automated Embedding** (``auto_embed=True``): pass ``query=<text>`` and
          Atlas embeds it server-side with the index's declared model.

        The filter always pins ``session_id``, merging any extra ``filters``. Raises
        ``ValueError`` if the arguments don't match the configured embedding path —
        there is no silent provider fallback. Returns the matching documents (without the
        raw ``embedding``) plus a ``score``.
        """
        vector_stage: dict[str, Any] = {
            "index": self.vector_search_index,
            "path": "embedding",
            "numCandidates": max(k * 20, 100),
            "limit": k,
        }

        if self.auto_embed:
            if query_vector is not None:
                raise ValueError(
                    "auto_embed=True uses query text, not `query_vector`. "
                    "Pass `query=...` instead."
                )
            if query is None:
                raise ValueError(
                    "auto_embed=True expects `query` text (Atlas embeds it server-side); "
                    "got none."
                )
            vector_stage["query"] = query
        else:
            if query_vector is None:
                raise ValueError(
                    "No `query_vector` provided. Either pass a query vector produced by "
                    "your embedding provider, or construct MongoMemory with auto_embed=True "
                    "to let Atlas embed query text server-side."
                )
            vector_stage["queryVector"] = query_vector

        vs_filter: dict[str, Any] = {"session_id": self.session_id}
        if filters:
            vs_filter.update(filters)
        vector_stage["filter"] = vs_filter

        pipeline = [
            {"$vectorSearch": vector_stage},
            {
                "$project": {
                    "embedding": 0,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.memory.aggregate(pipeline))

    # -- maintenance -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying client (the memory always owns it)."""
        self.client.close()


__all__ = ["MongoMemory", "DEFAULT_AUTO_EMBED_MODEL", "APP_NAME", "DRIVER_NAME"]
