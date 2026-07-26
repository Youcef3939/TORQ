"""Persistent MCP session: warmup gating and graceful fallback when the server
subprocess cannot start. Uses a fast-failing command, so no models are loaded."""

from __future__ import annotations

from torq.mcp import client


def test_warmup_false_when_unavailable(monkeypatch):
    # No networked Qdrant -> MCP is not usable -> warmup is a no-op returning False.
    monkeypatch.setattr(client.settings, "qdrant_url", "")
    assert client.mcp_warmup() is False


def test_call_returns_none_when_server_fails(monkeypatch):
    # Point the server at a command that exits immediately: the session cannot
    # start, so call() must return None (caller then falls back to direct).
    monkeypatch.setattr(client.settings, "mcp_server_args", ["-c", "raise SystemExit(1)"])
    session = client._PersistentSession()
    try:
        assert session.call("search_manuals", {"query": "x", "limit": 1}) is None
    finally:
        session.shutdown()
