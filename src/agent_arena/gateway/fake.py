"""In-process fake of the I9 gateway, implementing the full contract-6.6 tool set.

It is a real MCP server (``mcp`` SDK ``MCPServer``) reached through the SDK's in-process
transport, plus a FastAPI admin app reached through ``httpx.ASGITransport``, so the arena's
production client code is exercised end to end without sockets. It enforces what the arena
relies on from the real gateway: time-travel refusals, per-session budgets that freeze the
session on breach, pre-trade limits and fills at the session clock.
"""

from __future__ import annotations

import math
import secrets
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from mcp import Client
from mcp.server.mcpserver import MCPServer
from scipy.stats import norm

from agent_arena.gateway.client import AdminClient, GatewayBackend, GatewayFactory, GatewayTools
from agent_arena.gateway.client import McpToolCaller as _McpToolCaller
from agent_arena.gateway.market import SyntheticMarket
from agent_arena.gateway.models import (
    ClockInfo,
    ClockRequest,
    CreateSessionRequest,
    SessionBudgets,
    SessionInfo,
)
from agent_arena.season import SeasonConfig
from agent_arena.settings import GatewaySettings, ModelPrice
from agent_arena.timeutil import iso, parse_iso

# httpx needs an absolute base URL even for an in-process ASGI transport; it is never resolved.
_IN_PROCESS_BASE_URL = "http://in-process"
TOKENS_PER_MILLION = 1_000_000
BPS = 10_000
# Quantile levels returned by the fake forecast tool.
_FORECAST_QUANTILES = (0.1, 0.5, 0.9)


def refusal(code: str, reason: str, *, frozen: bool = False) -> dict[str, Any]:
    """Structured refusal in the shape the arena client recognises."""
    return {
        "schema_version": "1",
        "refused": True,
        "code": code,
        "reason": reason,
        "frozen": frozen,
    }


@dataclass
class RestingOrder:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float


@dataclass
class FakeSession:
    session_id: str
    agent_id: str
    mode: str
    known_at: datetime
    budgets: SessionBudgets
    api_key: str
    cash: float
    peak_equity: float
    positions: dict[str, float] = field(default_factory=dict)
    orders_this_step: int = 0
    token_spend_usd: float = 0.0
    frozen_reason: str | None = None
    resting: dict[str, RestingOrder] = field(default_factory=dict)


@dataclass
class AuditEntry:
    session_id: str
    known_at: str
    tool: str
    decision: str
    detail: dict[str, Any]


class FakeGateway:
    """Fake gateway for one season episode (one synthetic market)."""

    def __init__(
        self,
        season: SeasonConfig,
        market: SyntheticMarket,
        pricing: Mapping[str, ModelPrice],
        admin_key: str | None = None,
        admin_key_header: str = "X-Admin-Key",
    ) -> None:
        self.season = season
        self.market = market
        self.pricing = dict(pricing)
        self.admin_key = admin_key
        self.admin_key_header = admin_key_header
        self.sessions: dict[str, FakeSession] = {}
        self.audit: list[AuditEntry] = []
        self._servers: dict[str, MCPServer] = {}
        self.app = self._build_admin_app()

    # ----------------------------------------------------------------- admin API
    def _build_admin_app(self) -> FastAPI:
        app = FastAPI(title="fake-i9-admin")

        def check(request: Request) -> None:
            if self.admin_key is not None and request.headers.get(self.admin_key_header) != self.admin_key:
                raise HTTPException(status_code=401, detail="bad admin key")

        @app.post("/admin/v1/sessions")
        def create_session(body: CreateSessionRequest, request: Request) -> SessionInfo:
            check(request)
            known_at = parse_iso(body.known_at)
            session = FakeSession(
                session_id=uuid.uuid4().hex,
                agent_id=body.agent_id,
                mode=body.mode,
                known_at=known_at,
                budgets=body.budgets,
                api_key=secrets.token_hex(16),
                cash=body.budgets.capital,
                peak_equity=body.budgets.capital,
            )
            self.sessions[session.session_id] = session
            return SessionInfo(
                session_id=session.session_id,
                agent_id=session.agent_id,
                known_at=iso(known_at),
                api_key=session.api_key,
            )

        @app.post("/admin/v1/sessions/{session_id}/clock")
        def set_clock(session_id: str, body: ClockRequest, request: Request) -> ClockInfo:
            check(request)
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session")
            new = parse_iso(body.known_at)
            if new < session.known_at:
                raise HTTPException(status_code=409, detail="clock can only move forward")
            session.known_at = new
            session.orders_this_step = 0
            self._fill_resting(session)
            self._check_drawdown(session)
            return ClockInfo(known_at=iso(new))

        return app

    # ----------------------------------------------------------------- accounting
    def equity(self, session: FakeSession) -> float:
        marks = sum(qty * self.market.price_at(sym, session.known_at) for sym, qty in session.positions.items())
        return session.cash + marks

    def _check_drawdown(self, session: FakeSession) -> None:
        equity = self.equity(session)
        session.peak_equity = max(session.peak_equity, equity)
        drawdown = 1.0 - equity / session.peak_equity
        if drawdown > session.budgets.max_drawdown and session.frozen_reason is None:
            session.frozen_reason = "max_drawdown"
            self._audit(session, "budget", "freeze", {"drawdown": drawdown})

    def _fill(self, session: FakeSession, symbol: str, side: str, qty: float, price: float) -> float:
        signed = qty if side == "buy" else -qty
        fee = abs(qty * price) * self.season.market.fee_bps / BPS
        session.positions[symbol] = session.positions.get(symbol, 0.0) + signed
        session.cash -= signed * price + fee
        return fee

    def _fill_resting(self, session: FakeSession) -> None:
        for order_id, order in list(session.resting.items()):
            price = self.market.price_at(order.symbol, session.known_at)
            marketable = price <= order.price if order.side == "buy" else price >= order.price
            if marketable:
                self._fill(session, order.symbol, order.side, order.qty, order.price)
                del session.resting[order_id]

    def _audit(self, session: FakeSession, tool: str, decision: str, detail: dict[str, Any]) -> None:
        self.audit.append(AuditEntry(session.session_id, iso(session.known_at), tool, decision, detail))

    # ----------------------------------------------------------------- MCP server
    def server_for(self, session_id: str) -> MCPServer:
        """The MCP server bound to one session (the real gateway resolves it from auth)."""
        if session_id not in self._servers:
            self._servers[session_id] = self._build_server(self.sessions[session_id])
        return self._servers[session_id]

    def _build_server(self, session: FakeSession) -> MCPServer:
        # Tools are async so the SDK calls them inline instead of hopping to a worker thread.
        server = MCPServer("fake-i9-gateway")
        market = self.market
        season = self.season

        def frozen() -> dict[str, Any] | None:
            if session.frozen_reason is None:
                return None
            return refusal("session_frozen", session.frozen_reason, frozen=True)

        def window(start: str, end: str) -> tuple[datetime, datetime] | dict[str, Any]:
            lo, hi = parse_iso(start), parse_iso(end)
            if hi > session.known_at:
                self._audit(session, "data", "refuse", {"end": end})
                return refusal("time_travel", f"end {end} is after known_at {iso(session.known_at)}")
            if lo > hi:
                return refusal("bad_request", "start is after end")
            return lo, hi

        @server.tool(description="Session clock: the 'as known at' time (ISO-8601 UTC).")
        async def clock() -> dict[str, Any]:
            return {"schema_version": "1", "known_at": iso(session.known_at)}

        @server.tool(description="Trades (I1 schema) for one symbol in [start, end], end <= clock.")
        async def market_data(venue: str, symbol: str, channel: str, start: str, end: str) -> dict[str, Any]:
            if venue != season.venue or symbol not in season.symbols:
                return refusal("unknown_instrument", f"{venue}/{symbol}")
            if channel != season.channel:
                return refusal("unsupported_channel", channel)
            span = window(start, end)
            if isinstance(span, dict):
                return span
            rows = [
                {
                    "ts_exchange": t.ts_exchange,
                    "price": t.price,
                    "size": t.size,
                    "side": t.side,
                    "trade_id": t.trade_id,
                }
                for t in market.trades(symbol, *span)
            ]
            return {
                "schema_version": "1",
                "venue": venue,
                "symbol": symbol,
                "channel": channel,
                "rows": rows,
            }

        @server.tool(
            description="Features at `end`: ret_<n> (n-bar log return), vol_<n> (n-bar stdev of "
            "log returns), z_<n> (price z-score vs n-bar mean). Unknown names return null."
        )
        async def features(names: list[str], symbols: list[str], start: str, end: str) -> dict[str, Any]:
            span = window(start, end)
            if isinstance(span, dict):
                return span
            out = [
                {"symbol": s, "name": n, "value": _feature(market.closes_until(s, span[1]), n)}
                for s in symbols
                if s in season.symbols
                for n in names
            ]
            return {"schema_version": "1", "features": out}

        @server.tool(description="Naive drift forecast of log return over `horizon` bars.")
        async def forecast(model: str, symbols: list[str], horizon: int) -> dict[str, Any]:
            items = []
            for s in symbols:
                closes = market.closes_until(s, session.known_at)[-season.lookback_bars :]
                rets = np.diff(np.log(closes))
                mu = float(rets.mean()) * horizon if rets.size else 0.0
                sd = float(rets.std(ddof=1)) * math.sqrt(horizon) if rets.size > 1 else 0.0
                quantiles = {str(q): mu + sd * float(norm.ppf(q)) for q in _FORECAST_QUANTILES}
                items.append({"symbol": s, "horizon": horizon, "mean": mu, "quantiles": quantiles})
            return {
                "schema_version": "1",
                "forecasts": items,
                "model_cutoff": iso(session.known_at),
                "model": model,
            }

        @server.tool(description="Vectorised backtest of ts_momentum(lookback) over [start, end].")
        async def run_backtest(
            strategy: str,
            params: dict[str, Any],
            universe: list[str],
            start: str,
            end: str,
            fidelity: str,
        ) -> dict[str, Any]:
            if (blocked := frozen()) is not None:
                return blocked
            span = window(start, end)
            if isinstance(span, dict):
                return span
            if strategy != "ts_momentum":
                return refusal("unknown_strategy", strategy)
            lookback = int(params.get("lookback", season.lookback_bars))
            pnl: list[float] = []
            for s in universe:
                closes = market.closes_until(s, span[1])
                rets = np.diff(np.log(closes))
                signal = np.sign(np.convolve(rets, np.ones(lookback), "valid"))[:-1]
                pnl.extend((signal * rets[lookback:]).tolist())
            arr = np.asarray(pnl)
            sharpe = float(arr.mean() / arr.std(ddof=1)) if arr.size > 1 and arr.std() > 0 else 0.0
            self._audit(session, "run_backtest", "run", {"strategy": strategy})
            return {
                "schema_version": "1",
                "run_id": uuid.uuid4().hex,
                "trial_id": None,
                "metrics": {
                    "sharpe_per_bar": sharpe,
                    "n_bars": int(arr.size),
                    "fidelity": fidelity,
                },
            }

        @server.tool(description="S6 verdict for a strategy (the fake has no ledger).")
        async def verdict(strategy_id: str) -> dict[str, Any]:
            return {
                "schema_version": "1",
                "strategy_id": strategy_id,
                "passed": False,
                "dsr": None,
                "reasons": ["fake gateway: no ledger attached"],
            }

        @server.tool(
            description="Place an order at the session clock. qty in base units (> 0); side "
            "buy|sell; type market|limit (limit needs price). Pre-trade checked against budgets."
        )
        async def place_order(
            venue: str,
            symbol: str,
            side: str,
            type: str,
            qty: float,
            strategy_id: str,
            rationale: str,
            price: float | None = None,
        ) -> dict[str, Any]:
            if (blocked := frozen()) is not None:
                return blocked
            budgets = session.budgets

            def reject(reason: str) -> dict[str, Any]:
                self._audit(session, "place_order", "reject", {"symbol": symbol, "reason": reason})
                return {
                    "schema_version": "1",
                    "order_id": None,
                    "status": "rejected",
                    "reject_reason": reason,
                }

            if venue != season.venue or symbol not in season.symbols:
                return reject("instrument_not_whitelisted")
            if side not in ("buy", "sell") or type not in ("market", "limit") or not qty > 0:
                return reject("bad_request")
            if type == "limit" and price is None:
                return reject("limit_without_price")
            if session.orders_this_step >= budgets.max_orders_per_step:
                return reject("order_rate")
            mark = market.price_at(symbol, session.known_at)
            if qty * mark > budgets.max_order_notional:
                return reject("max_order_notional")
            signed = qty if side == "buy" else -qty
            new_position = session.positions.get(symbol, 0.0) + signed
            if abs(new_position) * mark > budgets.max_position_notional:
                return reject("position_limit")
            session.orders_this_step += 1
            order_id = uuid.uuid4().hex
            if type == "limit" and price is not None:
                marketable = mark <= price if side == "buy" else mark >= price
                if not marketable:
                    session.resting[order_id] = RestingOrder(order_id, symbol, side, qty, price)
                    self._audit(session, "place_order", "rest", {"symbol": symbol})
                    return {"schema_version": "1", "order_id": order_id, "status": "accepted"}
            fee = self._fill(session, symbol, side, qty, mark)
            self._audit(
                session,
                "place_order",
                "fill",
                {"symbol": symbol, "side": side, "qty": qty, "rationale": rationale[:200]},
            )
            return {
                "schema_version": "1",
                "order_id": order_id,
                "status": "filled",
                "filled_qty": qty,
                "avg_price": mark,
                "fee": fee,
                "strategy_id": strategy_id,
            }

        @server.tool(description="Cancel a resting order.")
        async def cancel_order(order_id: str) -> dict[str, Any]:
            if session.resting.pop(order_id, None) is None:
                return {
                    "schema_version": "1",
                    "order_id": order_id,
                    "status": "rejected",
                    "reject_reason": "unknown_order",
                }
            return {"schema_version": "1", "order_id": order_id, "status": "cancelled"}

        @server.tool(description="Open positions (base units).")
        async def positions() -> dict[str, Any]:
            items = [{"symbol": s, "qty": q} for s, q in sorted(session.positions.items()) if q != 0.0]
            return {"schema_version": "1", "positions": items}

        @server.tool(description="Account equity and cash in account currency.")
        async def account() -> dict[str, Any]:
            return {
                "schema_version": "1",
                "account_id": session.session_id,
                "equity": self.equity(session),
                "cash": session.cash,
            }

        @server.tool(description="Budget limits, usage so far and whether the session is frozen.")
        async def budget() -> dict[str, Any]:
            equity = self.equity(session)
            return {
                "schema_version": "1",
                "limits": session.budgets.model_dump(exclude={"schema_version"}),
                "used": {
                    "token_spend_usd": session.token_spend_usd,
                    "drawdown": 1.0 - equity / session.peak_equity,
                    "orders_this_step": float(session.orders_this_step),
                },
                "frozen": session.frozen_reason is not None,
                "frozen_reason": session.frozen_reason,
            }

        @server.tool(description="Report LLM usage (tokens, latency in ms, model id) for budgeting.")
        async def report_usage(tokens_in: int, tokens_out: int, latency_ms: float, model: str) -> dict[str, Any]:
            if (blocked := frozen()) is not None:
                return blocked
            price = self.pricing.get(model)
            if price is None:
                return refusal("unknown_model", model)
            cost = (tokens_in * price.input + tokens_out * price.output) / TOKENS_PER_MILLION
            session.token_spend_usd += cost
            if session.token_spend_usd > session.budgets.token_spend_usd:
                session.frozen_reason = "token_budget"
            elif latency_ms > session.budgets.max_latency_ms:
                session.frozen_reason = "latency_budget"
            if session.frozen_reason is not None:
                self._audit(session, "report_usage", "freeze", {"reason": session.frozen_reason})
                return refusal("budget_breach", session.frozen_reason, frozen=True)
            return {
                "schema_version": "1",
                "accepted": True,
                "token_spend_usd": session.token_spend_usd,
            }

        return server

    # ----------------------------------------------------------------- backend protocol
    @asynccontextmanager
    async def tools(self, session: SessionInfo) -> AsyncIterator[GatewayTools]:
        async with Client(self.server_for(session.session_id)) as client:
            yield GatewayTools(_McpToolCaller(client))


def _feature(closes: np.ndarray[Any, np.dtype[np.float64]], name: str) -> float | None:
    """Compute a fake-gateway feature from closes up to the query end."""
    kind, _, n_text = name.partition("_")
    if not n_text.isdigit():
        return None
    n = int(n_text)
    if closes.size <= n:
        return None
    window = closes[-(n + 1) :]
    if kind == "ret":
        return float(math.log(window[-1] / window[0]))
    if kind == "vol":
        return float(np.diff(np.log(window)).std(ddof=1))
    if kind == "z":
        sd = float(window[1:].std(ddof=1))
        return float((window[-1] - window[1:].mean()) / sd) if sd > 0 else 0.0
    return None


class FakeGatewayBackend:
    """:class:`GatewayBackend` view of a :class:`FakeGateway` with an in-process admin client."""

    def __init__(self, fake: FakeGateway, admin: AdminClient) -> None:
        self.fake = fake
        self._admin = admin

    @property
    def admin(self) -> AdminClient:
        return self._admin

    def tools(self, session: SessionInfo) -> AbstractAsyncContextManager[GatewayTools]:
        return self.fake.tools(session)


def fake_gateway_factory(
    season: SeasonConfig,
    pricing: Mapping[str, ModelPrice],
    gateway_settings: GatewaySettings,
    registry: dict[int, FakeGateway] | None = None,
) -> GatewayFactory:
    """One fake gateway per episode, each serving that episode's synthetic market.

    ``registry`` (optional) collects the fakes so tests can inspect sessions and the audit log.
    """
    admin_key = gateway_settings.admin_key.get_secret_value() if gateway_settings.admin_key else None

    @asynccontextmanager
    async def factory(episode: int) -> AsyncIterator[GatewayBackend]:
        fake = FakeGateway(
            season,
            SyntheticMarket(season, episode),
            pricing,
            admin_key,
            gateway_settings.admin_key_header,
        )
        if registry is not None:
            registry[episode] = fake
        transport = httpx.ASGITransport(app=fake.app)
        async with httpx.AsyncClient(transport=transport, base_url=_IN_PROCESS_BASE_URL) as http:
            yield FakeGatewayBackend(fake, AdminClient(http, gateway_settings))

    return factory
