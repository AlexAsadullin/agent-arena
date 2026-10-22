from __future__ import annotations

import json
import os

import httpx
import pytest
import respx
from pydantic import SecretStr

from agent_arena.gateway.client import AdminClient, GatewayError, admin_headers, agent_headers, http_gateway_factory
from agent_arena.gateway.models import SessionBudgets, SessionInfo
from agent_arena.settings import ArenaSettings


def test_auth_headers(settings: ArenaSettings) -> None:
    gw = settings.gateway.model_copy(update={"admin_key": SecretStr("adm")})
    assert admin_headers(gw) == {"Authorization": "Bearer adm"}
    assert admin_headers(settings.gateway) == {}
    assert agent_headers(gw, SessionInfo(session_id="s1", api_key="k1")) == {"Authorization": "Bearer k1"}
    assert agent_headers(gw, SessionInfo(session_id="s1")) == {"Authorization": "Bearer adm", "X-Gateway-Session": "s1"}
    with_agent_key = gw.model_copy(update={"agent_key": SecretStr("ak")})
    assert agent_headers(with_agent_key, SessionInfo(session_id="s1")) == {"Authorization": "Bearer ak"}


@respx.mock
async def test_admin_client_against_http(settings: ArenaSettings) -> None:
    gw = settings.gateway.model_copy(update={"admin_key": SecretStr("adm"), "clock_retries": 1})
    create = respx.post(f"{gw.admin_url}/admin/v1/sessions").mock(
        return_value=httpx.Response(
            200, json={"session_id": "s1", "api_key": "k", "mcp_path": "/mcp", "status": "active"}
        )
    )
    clock = respx.post(f"{gw.admin_url}/admin/v1/sessions/s1/clock").mock(
        side_effect=[httpx.ConnectError("blip"), httpx.Response(200, json={"known_at": "2025-01-01T01:00:00Z"})]
    )
    async with httpx.AsyncClient(base_url=gw.admin_url) as http:
        admin = AdminClient(http, gw)
        info = await admin.create_session("a:b", "2025-01-01T00:00:00Z", SessionBudgets(capital=100.0))
        moved = await admin.set_clock("s1", "2025-01-01T01:00:00Z")
    sent = create.calls.last.request
    assert sent.headers["authorization"] == "Bearer adm"
    body = json.loads(sent.content)
    assert body["mode"] == "paper"
    assert body["budgets"] == {"capital": 100.0}
    assert info.api_key == "k"
    assert moved.known_at == "2025-01-01T01:00:00Z"
    assert clock.call_count == 2


@respx.mock
async def test_admin_client_errors(settings: ArenaSettings) -> None:
    gw = settings.gateway.model_copy(update={"clock_retries": 0})
    respx.post(f"{gw.admin_url}/admin/v1/sessions").mock(
        side_effect=[httpx.Response(422, json={}), httpx.Response(200, json=[1])]
    )
    respx.post(f"{gw.admin_url}/admin/v1/sessions/s/clock").mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient(base_url=gw.admin_url) as http:
        admin = AdminClient(http, gw)
        for _ in range(2):
            with pytest.raises(GatewayError):
                await admin.create_session("a", "2025-01-01T00:00:00Z", SessionBudgets())
        with pytest.raises(GatewayError, match="clock"):
            await admin.set_clock("s", "2025-01-01T00:00:00Z")


@pytest.mark.live
async def test_live_gateway_session_roundtrip(settings: ArenaSettings) -> None:
    """Needs a running I9 gateway and ARENA_GATEWAY__ADMIN_KEY; run with ``pytest -m live``."""
    key = os.environ.get("ARENA_GATEWAY__ADMIN_KEY")
    if not key:
        pytest.skip("ARENA_GATEWAY__ADMIN_KEY not set")
    gw = settings.gateway.model_copy(update={"admin_key": SecretStr(key)})
    async with http_gateway_factory(gw)(0) as backend:
        session = await backend.admin.create_session("arena:live-test", "2025-06-01T00:00:00Z", SessionBudgets())
        async with backend.tools(session) as tools:
            clock = await tools.clock()
    assert not isinstance(clock, Exception)
