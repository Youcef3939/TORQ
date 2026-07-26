"""MCP client helpers: call the torq-knowledge MCP server over stdio.

One long-lived MCP session is held open on a background event loop and reused
across every call. The server subprocess cold-loads its embedding models once,
on the first call, then stays warm, so later calls are fast (a fresh subprocess
per call would re-load the models every time, ~200s each). Any failure tears the
session down and returns ``None`` so the caller falls back to direct retrieval;
the next call rebuilds the session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from torq.config import settings

log = logging.getLogger(__name__)


def mcp_available() -> bool:
    """Whether MCP retrieval is usable.

    Needs a networked Qdrant (``qdrant_url``): the embedded on-disk store is
    single-process (file-locked), so the server subprocess cannot open the same
    store while the parent holds the lock; MCP would then silently and
    permanently fall back to direct. With no ``qdrant_url`` the caller retrieves
    directly instead.
    """
    return settings.use_mcp and bool(settings.qdrant_url)


def _server_params() -> StdioServerParameters:
    """Build stdio launch parameters from config."""
    return StdioServerParameters(
        command=settings.mcp_server_command,
        args=settings.mcp_server_args,
    )


def _extract_list(result: Any) -> list[dict]:
    """Pull the list payload out of a CallToolResult.

    FastMCP serializes a ``list[dict]`` return as
    ``structuredContent={"result": [...]}`` and, separately, as one TextContent
    block *per item* (so ``content[0]`` is only the first record - never parse
    just that). Prefer the structured payload; rebuild from the blocks otherwise.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    return [json.loads(c.text) for c in result.content]  # type: ignore[union-attr]


class _PersistentSession:
    """One long-lived MCP stdio session on its own background event loop.

    Calls are serialized (a single MCP session handles one request at a time).
    On any failure the session is torn down and the call returns ``None`` so the
    caller falls back to direct retrieval; the next call rebuilds it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None

    def _start(self) -> None:
        """Spawn the loop thread and open the session (blocks until ready)."""
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        err: list[BaseException] = []

        async def _main() -> None:
            self._stop = asyncio.Event()
            try:
                async with stdio_client(_server_params()) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        ready.set()
                        await self._stop.wait()  # keep session open until shutdown
            except BaseException as exc:  # noqa: BLE001 — surface to _start
                err.append(exc)
                ready.set()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_main())
            loop.close()

        thread = threading.Thread(target=_runner, name="mcp-session", daemon=True)
        thread.start()
        # The MCP handshake is fast; model loading is deferred to the first tool call.
        if not ready.wait(timeout=30) or self._session is None:
            self._session = None
            raise RuntimeError(err[0] if err else "MCP session did not start in time")
        self._loop, self._thread = loop, thread

    def call(self, name: str, arguments: dict[str, Any]) -> list[dict] | None:
        with self._lock:
            if self._session is None or self._loop is None or self._loop.is_closed():
                try:
                    self._start()
                except Exception:  # noqa: BLE001 — startup failed → direct fallback
                    log.warning("MCP session failed to start; using direct", exc_info=True)
                    self._reset()
                    return None
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._session.call_tool(name, arguments), self._loop
                )
                result = fut.result(timeout=settings.mcp_call_timeout)
            except Exception:  # noqa: BLE001 — kill the session, fall back to direct
                log.warning("MCP call '%s' failed; tearing down session", name, exc_info=True)
                self.shutdown()
                return None
            if result.isError:
                log.warning("MCP tool %s returned an error: %s", name, result.content)
                return None
            return _extract_list(result)

    def shutdown(self) -> None:
        """Signal the session to close and drop references (safe to call anytime)."""
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(stop.set)
            except Exception:  # noqa: BLE001 — loop already gone
                pass
        self._reset()

    def _reset(self) -> None:
        self._loop = self._thread = self._session = self._stop = None


_SESSION = _PersistentSession()


# ── Public sync API ──────────────────────────────────────────────────────────


def mcp_search_manuals(query: str, limit: int = 4) -> list[dict] | None:
    """Search manuals via MCP.  Returns ``None`` on any failure."""
    return _SESSION.call("search_manuals", {"query": query, "limit": limit})


def mcp_search_history(
    query: str, limit: int = 4, machine: str = ""
) -> list[dict] | None:
    """Search repair history via MCP.  Returns ``None`` on any failure."""
    args: dict[str, Any] = {"query": query, "limit": limit}
    if machine:
        args["machine"] = machine
    return _SESSION.call("search_history", args)


def mcp_warmup() -> bool:
    """Prime the persistent session (triggers the one-time model cold-load).

    Meant to run at app startup, off the request path, so the first real
    diagnosis is fast. Returns whether the warmup call succeeded.
    """
    if not mcp_available():
        return False
    return mcp_search_manuals("warmup", limit=1) is not None


def mcp_shutdown() -> None:
    """Close the persistent session (called at app shutdown)."""
    _SESSION.shutdown()
