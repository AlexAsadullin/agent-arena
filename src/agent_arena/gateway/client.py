"""Typed client for the I9 gateway: MCP tools (contract 6.6) plus the admin HTTP API.

The arena never passes ``known_at`` to a tool; it moves the session clock through the admin API
and lets the gateway inject it, exactly as the contract requires.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

import httpx
import httpx2
import structlog
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from agent_arena.gateway.models import (
    Account,
    BacktestResult,
    BudgetStatus,
    ClockInfo,
    ClockRequest,
    CreateSessionRequest,
    Features,
    Forecast,
    MarketData,
    OrderResult,
    Positions,
    Refusal,
    SessionBudgets,
    SessionInfo,
    UsageAck,
    VerdictResult,
)
from agent_arena.settings import GatewaySettings

log = structlog.get_logger(__name__)


class GatewayError(RuntimeError):
    """The gateway could not be reached or answered outside the contract."""


class ToolCaller(Protocol):
    """Anything that can call an MCP tool and return its JSON result or a refusal."""

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | Refusal: ...


def parse_tool_result(result: CallToolResult) -> dict[str, Any] | Refusal:
    """Turn an MCP tool result into a JSON object or a :class:`Refusal`.

    Refusals are recognised in three shapes: an MCP error result (``isError``), an object with
    ``"refused": true``, or an object with ``"status": "refused"``.
    """
    text = " ".join(c.text for c in result.content if isinstance(c, TextContent))
    if result.is_error:
        return Refusal(code="tool_error", reason=text or "tool error")
    data: Any = result.structured_content
    if data is None:
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise GatewayError(f"tool returned non-JSON content: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise GatewayError(f"tool returned {type(data).__name__}, expected an object")
    if data.get("refused") is True or data.get("status") == "refused":
        return Refusal(
            code=str(data.get("code", "refused")),
            reason=str(data.get("reason", "")),
            frozen=bool(data.get("frozen", False)),
        )
    return data


class McpToolCaller:
    """:class:`ToolCaller` over a connected ``mcp.Client``."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | Refusal:
        result = await self._client.call_tool(name, arguments)
        return parse_tool_result(result)


def _typed[M: BaseModel](model: type[M], raw: dict[str, Any] | Refusal) -> M | Refusal:
    return raw if isinstance(raw, Refusal) else model.model_validate(raw)


class GatewayTools:
    """One method per contract-6.6 tool, each returning a typed result or a :class:`Refusal`."""

    def __init__(self, caller: ToolCaller) -> None:
        self._caller = caller

    async def clock(self) -> ClockInfo | Refusal:
        return _typed(ClockInfo, await self._caller.call("clock", {}))

    async def market_data(self, venue: str, symbol: str, channel: str, start: str, end: str) -> MarketData | Refusal:
        args = {"venue": venue, "symbol": symbol, "channel": channel, "start": start, "end": end}
        return _typed(MarketData, await self._caller.call("market_data", args))

    async def features(self, names: list[str], symbols: list[str], start: str, end: str) -> Features | Refusal:
        args = {"names": names, "symbols": symbols, "start": start, "end": end}
        return _typed(Features, await self._caller.call("features", args))

    async def forecast(self, model: str, symbols: list[str], horizon: int) -> Forecast | Refusal:
        args = {"model": model, "symbols": symbols, "horizon": horizon}
        return _typed(Forecast, await self._caller.call("forecast", args))

    async def run_backtest(
        self,
        strategy: str,
        params: dict[str, Any],
        universe: list[str],
        start: str,
        end: str,
        fidelity: str,
    ) -> BacktestResult | Refusal:
        args = {
            "strategy": strategy,
            "params": params,
            "universe": universe,
            "start": start,
            "end": end,
            "fidelity": fidelity,
        }
        return _typed(BacktestResult, await self._caller.call("run_backtest", args))

    async def verdict(self, strategy_id: str) -> VerdictResult | Refusal:
        return _typed(VerdictResult, await self._caller.call("verdict", {"strategy_id": strategy_id}))

    async def place_order(
        self,
        venue: str,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        strategy_id: str,
        rationale: str,
        price: float | None = None,
    ) -> OrderResult | Refusal:
        args: dict[str, Any] = {
            "venue": venue,
            "symbol": symbol,
            "side": side,
            "type": type_,
            "qty": qty,
            "strategy_id": strategy_id,
            "rationale": rationale,
        }
        if price is not None:
            args["price"] = price
        return _typed(OrderResult, await self._caller.call("place_order", args))

    async def cancel_order(self, order_id: str) -> OrderResult | Refusal:
        return _typed(OrderResult, await self._caller.call("cancel_order", {"order_id": order_id}))

    async def positions(self) -> Positions | Refusal:
        return _typed(Positions, await self._caller.call("positions", {}))

    async def account(self) -> Account | Refusal:
        return _typed(Account, await self._caller.call("account", {}))

    async def budget(self) -> BudgetStatus | Refusal:
        return _typed(BudgetStatus, await self._caller.call("budget", {}))

    async def report_usage(self, tokens_in: int, tokens_out: int, latency_ms: float, model: str) -> UsageAck | Refusal:
        args = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "model": model,
        }
        return _typed(UsageAck, await self._caller.call("report_usage", args))


class AdminClient:
    """Admin HTTP API of the gateway: create sessions and move their clock."""

    def __init__(self, http: httpx.AsyncClient, settings: GatewaySettings) -> None:
        self._http = http
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        key = self._settings.admin_key
        return {self._settings.admin_key_header: key.get_secret_value()} if key else {}

    async def create_session(self, agent_id: str, known_at: str, budgets: SessionBudgets) -> SessionInfo:
        body = CreateSessionRequest(agent_id=agent_id, mode=self._settings.mode, known_at=known_at, budgets=budgets)
        resp = await self._post("/admin/v1/sessions", body.model_dump(mode="json"))
        return SessionInfo.model_validate(resp)

    async def set_clock(self, session_id: str, known_at: str) -> ClockInfo:
        body = ClockRequest(known_at=known_at).model_dump(mode="json")
        path = f"/admin/v1/sessions/{session_id}/clock"
        last: Exception | None = None
        for attempt in range(self._settings.clock_retries + 1):
            try:
                return ClockInfo.model_validate(await self._post(path, body))
            except httpx.TransportError as exc:
                last = exc
                log.warning("clock_retry", session_id=session_id, attempt=attempt)
        raise GatewayError(f"clock advance failed for {session_id}") from last

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._http.post(path, json=body, headers=self._headers())
        if resp.status_code >= 400:
            raise GatewayError(f"POST {path} -> {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if not isinstance(data, dict):
            raise GatewayError(f"POST {path} returned a non-object body")
        return data


class GatewayBackend(Protocol):
    """A gateway the runner can drive: admin API plus per-session MCP tool access."""

    @property
    def admin(self) -> AdminClient: ...

    def tools(self, session: SessionInfo) -> AbstractAsyncContextManager[GatewayTools]: ...


GatewayFactory = Callable[[int, int], AbstractAsyncContextManager[GatewayBackend]]
"""Yields the gateway backend for a season cell ``(episode, seed)``."""


class HttpGateway:
    """The real I9 gateway over streamable HTTP (MCP) and HTTP (admin)."""

    def __init__(self, settings: GatewaySettings, admin_http: httpx.AsyncClient) -> None:
        self._settings = settings
        self._admin = AdminClient(admin_http, settings)

    @property
    def admin(self) -> AdminClient:
        return self._admin

    @asynccontextmanager
    async def tools(self, session: SessionInfo) -> AsyncIterator[GatewayTools]:
        headers = {self._settings.session_header: session.session_id}
        key = session.api_key or (self._settings.agent_key.get_secret_value() if self._settings.agent_key else None)
        if key:
            headers[self._settings.agent_auth_header] = f"{self._settings.agent_auth_scheme} {key}".strip()
        async with (
            httpx2.AsyncClient(headers=headers, timeout=self._settings.timeout_s) as http,
            Client(streamable_http_client(self._settings.mcp_url, http_client=http)) as client,
        ):
            yield GatewayTools(McpToolCaller(client))


def http_gateway_factory(settings: GatewaySettings) -> GatewayFactory:
    """Every cell uses the same real gateway; the market is whatever I2 serves at ``known_at``."""

    @asynccontextmanager
    async def factory(_episode: int, _seed: int) -> AsyncIterator[GatewayBackend]:
        async with httpx.AsyncClient(base_url=settings.admin_url, timeout=settings.timeout_s) as admin_http:
            yield HttpGateway(settings, admin_http)

    return factory
