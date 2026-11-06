"""In-process fake of the I9 gateway, implementing the full contract-6.6 tool set.

It is a real MCP server (``mcp`` SDK ``MCPServer``) reached through the SDK's in-process
transport, plus a FastAPI admin app reached through ``httpx.ASGITransport``, so the arena's
production client code runs end to end without sockets. It mirrors the I9 behaviour the arena
relies on: bearer-key admin auth, forward-only session clocks, refusals as MCP errors with the
I9 refusal body, budgets that refuse (exposure, order notional, order rate) or freeze the session
(drawdown, tokens, token cost, latency), research sessions that cannot trade, and a short list of
tools that still answer when frozen. Market orders fill at the session clock's last price.
"""

from __future__ import annotations

import json
import math
import secrets
import uuid
from collections import deque
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
from mcp.types import CallToolResult, TextContent
from scipy.stats import norm

from agent_arena.fees import FeeBook, Liquidity
from agent_arena.gateway.client import AdminClient, GatewayBackend, GatewayFactory, GatewayTools, McpToolCaller
from agent_arena.gateway.market import SyntheticMarket
from agent_arena.gateway.models import ClockInfo, ClockRequest, CreateSessionRequest, SessionBudgets, SessionInfo
from agent_arena.season import SeasonConfig
from agent_arena.settings import GatewaySettings, ModelPrice
from agent_arena.timeutil import iso, parse_iso, to_ns

# httpx needs an absolute base URL even for an in-process ASGI transport; it is never resolved.
_IN_PROCESS_BASE_URL = "http://in-process"
# I9 serves MCP under this path and returns it at session creation.
_MCP_PATH = "/mcp"
# Tools that I9 still answers on a frozen session (I9 config budgets.frozen_allowed_tools).
_FROZEN_ALLOWED = frozenset({"clock", "budget", "positions", "account", "verdict"})
_FORECAST_QUANTILES = (0.1, 0.5, 0.9)
TOKENS_PER_MILLION = 1_000_000
BPS = 10_000


def ok(payload: dict[str, Any]) -> CallToolResult:
    body = {"schema_version": "1", **payload}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(body))], structured_content=body)


def refused(code: str, message: str, status: str, details: dict[str, Any] | None = None) -> CallToolResult:
    """I9 refusal: an MCP tool error whose structured body is ``{refused, code, message, ...}``."""
    body = {
        "schema_version": "1",
        "refused": True,
        "code": code,
        "message": message,
        "details": details or {},
        "session_status": status,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(body))], structured_content=body, is_error=True
    )


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
    order_times: deque[datetime] = field(default_factory=deque)
    tokens_in: int = 0
    tokens_out: int = 0
    token_cost_usd: float = 0.0
    latency_total_ms: float = 0.0
    latency_max_ms: float = 0.0
    decisions: int = 0
    orders_submitted: int = 0
    freeze: dict[str, Any] | None = None
    resting: dict[str, RestingOrder] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "frozen" if self.freeze is not None else "active"


@dataclass
class AuditEntry:
    session_id: str
    known_at: str
    tool: str
    decision: str
    detail: dict[str, Any]


class FakeGateway:
    """Fake gateway for one season episode (one synthetic market, shared by all sessions)."""

    def __init__(
        self,
        season: SeasonConfig,
        market: SyntheticMarket,
        pricing: Mapping[str, ModelPrice],
        settings: GatewaySettings,
        fees: FeeBook | None = None,
    ) -> None:
        self.season = season
        self.market = market
        self.pricing = dict(pricing)
        self.settings = settings
        self.fees = fees
        self.admin_key = settings.admin_key.get_secret_value() if settings.admin_key else None
        self.sessions: dict[str, FakeSession] = {}
        self.audit: list[AuditEntry] = []
        self._servers: dict[str, MCPServer] = {}
        self.app = self._build_admin_app()

    # ----------------------------------------------------------------- admin API
    def _build_admin_app(self) -> FastAPI:
        app = FastAPI(title="fake-i9-admin")
        expected = None if self.admin_key is None else f"{self.settings.admin_auth_scheme} {self.admin_key}".strip()

        def check(request: Request) -> None:
            if expected is not None and request.headers.get(self.settings.admin_auth_header) != expected:
                raise HTTPException(status_code=401, detail="unauthenticated")

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
                cash=self.season.capital,
                peak_equity=self.season.capital,
            )
            self.sessions[session.session_id] = session
            return SessionInfo(
                session_id=session.session_id,
                agent_id=session.agent_id,
                known_at=iso(known_at),
                status=session.status,
                api_key=session.api_key,
                mcp_path=_MCP_PATH,
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
            self._fill_resting(session)
            self._check_drawdown(session)
            return ClockInfo(known_at=iso(new), status=session.status)

        return app

    # ----------------------------------------------------------------- accounting
    def mark(self, symbol: str, session: FakeSession) -> float:
        return self.market.price_at(symbol, session.known_at)

    def equity(self, session: FakeSession) -> float:
        return session.cash + sum(q * self.mark(s, session) for s, q in session.positions.items())

    def gross(self, session: FakeSession, positions: Mapping[str, float]) -> float:
        return sum(abs(q) * self.mark(s, session) for s, q in positions.items())

    def _freeze(self, session: FakeSession, kind: str, detail: dict[str, Any]) -> None:
        if session.freeze is None:
            session.freeze = {"reason": kind, "known_at": iso(session.known_at), **detail}
            self._audit(session, "budget", "freeze", {"kind": kind, **detail})

    def _check_drawdown(self, session: FakeSession) -> None:
        limit = session.budgets.max_drawdown
        equity = self.equity(session)
        session.peak_equity = max(session.peak_equity, equity)
        drawdown = 1.0 - equity / session.peak_equity
        if limit is not None and drawdown > limit:
            self._freeze(session, "drawdown", {"drawdown": drawdown})

    def fee(self, symbol: str, qty: float, price: float, liquidity: Liquidity) -> float:
        """Fee of one fill: the venue's schedule (as I9's mirror sandbox charges), else ``market.fee_bps``."""
        if self.fees is not None and self.fees.has_schedule(self.season.venue):
            return float(self.fees.fee(self.season.venue, symbol, qty, price, liquidity, 0.0))
        if self.season.market.fee_bps is None:
            raise ValueError(f"venue {self.season.venue!r} has no fee schedule and the season sets no market.fee_bps")
        return abs(qty * price) * self.season.market.fee_bps / BPS

    def _fill(
        self, session: FakeSession, symbol: str, side: str, qty: float, price: float, liquidity: Liquidity
    ) -> float:
        signed = qty if side == "buy" else -qty
        fee = self.fee(symbol, qty, price, liquidity)
        session.positions[symbol] = session.positions.get(symbol, 0.0) + signed
        session.cash -= signed * price + fee
        return fee

    def _fill_resting(self, session: FakeSession) -> None:
        for order_id, order in list(session.resting.items()):
            price = self.mark(order.symbol, session)
            if (price <= order.price) if order.side == "buy" else (price >= order.price):
                self._fill(session, order.symbol, order.side, order.qty, order.price, "maker")
                del session.resting[order_id]

    def _audit(self, session: FakeSession, tool: str, decision: str, detail: dict[str, Any]) -> None:
        self.audit.append(AuditEntry(session.session_id, iso(session.known_at), tool, decision, detail))

    def _usage(self, session: FakeSession) -> dict[str, Any]:
        return {
            "tokens_in": session.tokens_in,
            "tokens_out": session.tokens_out,
            "token_cost_usd": session.token_cost_usd,
            "latency_total_ms": session.latency_total_ms,
            "latency_max_ms": session.latency_max_ms,
            "decisions": session.decisions,
            "orders_submitted": session.orders_submitted,
        }

    def _remaining(self, session: FakeSession) -> dict[str, float]:
        b = session.budgets
        out: dict[str, float] = {}
        if b.max_token_cost_usd is not None:
            out["token_cost_usd"] = b.max_token_cost_usd - session.token_cost_usd
        if b.max_tokens is not None:
            out["tokens"] = float(b.max_tokens - session.tokens_in - session.tokens_out)
        if b.max_total_latency_ms is not None:
            out["total_latency_ms"] = b.max_total_latency_ms - session.latency_total_ms
        return out

    # ----------------------------------------------------------------- MCP server
    def server_for(self, session_id: str) -> MCPServer:
        """The MCP server bound to one session (the real gateway resolves the session from the bearer key)."""
        if session_id not in self._servers:
            self._servers[session_id] = self._build_server(self.sessions[session_id])
        return self._servers[session_id]

    def _build_server(self, session: FakeSession) -> MCPServer:
        # Tools are async so the SDK calls them inline instead of hopping to a worker thread.
        server = MCPServer("fake-i9-gateway")
        market = self.market
        season = self.season

        def blocked(tool: str) -> CallToolResult | None:
            if session.freeze is not None and tool not in _FROZEN_ALLOWED:
                return refused("session_frozen", f"session frozen: {session.freeze['reason']}", session.status)
            return None

        def window(start: str, end: str) -> tuple[datetime, datetime] | CallToolResult:
            lo, hi = parse_iso(start), parse_iso(end)
            if hi > session.known_at:
                self._audit(session, "data", "refuse", {"end": end})
                return refused(
                    "time_travel",
                    f"end {end} is after the session clock {iso(session.known_at)}",
                    session.status,
                    {"known_at": iso(session.known_at)},
                )
            if lo > hi:
                return refused("invalid_request", "start is after end", session.status)
            return lo, hi

        @server.tool(description="Session clock: the 'as known at' time (ISO-8601 UTC) that every answer uses.")
        async def clock() -> CallToolResult:
            return ok(
                {
                    "session_id": session.session_id,
                    "agent_id": session.agent_id,
                    "mode": session.mode,
                    "known_at": iso(session.known_at),
                    "known_at_source": "session",
                    "status": session.status,
                }
            )

        @server.tool(description="Rows of one I1 channel for venue/symbol in [start, end]; end must be <= clock.")
        async def market_data(venue: str, symbol: str, channel: str, start: str, end: str) -> CallToolResult:
            if (stop := blocked("market_data")) is not None:
                return stop
            if venue != season.venue or symbol not in season.symbols or channel != season.channel:
                return refused("invalid_request", f"unknown {venue}/{symbol}/{channel}", session.status)
            span = window(start, end)
            if isinstance(span, CallToolResult):
                return span
            rows = [
                {
                    "symbol": symbol,
                    "venue": venue,
                    "valid_time": t.ts_exchange,
                    "knowledge_time": t.ts_exchange,
                    "price": t.price,
                    "size": t.size,
                    "side": t.side,
                }
                for t in market.trades(symbol, *span)
            ]
            return ok(
                {
                    "dataset": f"tape.{channel}",
                    "venue": venue,
                    "symbol": symbol,
                    "channel": channel,
                    "start": start,
                    "end": end,
                    "known_at": iso(session.known_at),
                    "clamped": False,
                    "n_rows": len(rows),
                    "truncated": False,
                    "rows": rows,
                }
            )

        @server.tool(
            description="Long-format features at `end`: ret_<n> (n-bar log return), vol_<n> (n-bar stdev of "
            "log returns), z_<n> (price z-score vs n-bar mean). Unknown names give null values."
        )
        async def features(names: list[str], symbols: list[str], start: str, end: str) -> CallToolResult:
            if (stop := blocked("features")) is not None:
                return stop
            span = window(start, end)
            if isinstance(span, CallToolResult):
                return span
            # Wide rows, as I2 answers: one row per symbol at the query end, one column per feature.
            stamp = to_ns(span[1])
            rows = [
                {
                    "symbol": s,
                    "valid_time": stamp,
                    "knowledge_time": stamp,
                    **{n: _feature(market.closes_until(s, span[1]), n) for n in names},
                }
                for s in symbols
                if s in season.symbols
            ]
            return ok({"names": names, "symbols": symbols, "known_at": iso(session.known_at), "rows": rows})

        @server.tool(description="Naive drift forecast of the log return over `horizon` bars (fraction).")
        async def forecast(model: str, symbols: list[str], horizon: int) -> CallToolResult:
            if (stop := blocked("forecast")) is not None:
                return stop
            items = []
            for s in symbols:
                closes = market.closes_until(s, session.known_at)[-season.lookback_bars :]
                rets = np.diff(np.log(closes))
                mu = float(rets.mean()) * horizon if rets.size else 0.0
                sd = float(rets.std(ddof=1)) * math.sqrt(horizon) if rets.size > 1 else 0.0
                quantiles = {str(q): mu + sd * float(norm.ppf(q)) for q in _FORECAST_QUANTILES}
                items.append({"symbol": s, "horizon": horizon, "mean": mu, "quantiles": quantiles})
            known = iso(session.known_at)
            return ok(
                {"model": model, "horizon": horizon, "known_at": known, "model_cutoff": known, "forecasts": items}
            )

        @server.tool(description="Vectorised backtest of ts_momentum(lookback) over [start, end] (end <= clock).")
        async def run_backtest(
            strategy: str, params: dict[str, Any], universe: list[str], start: str, end: str, fidelity: str
        ) -> CallToolResult:
            if (stop := blocked("run_backtest")) is not None:
                return stop
            span = window(start, end)
            if isinstance(span, CallToolResult):
                return span
            if strategy != "ts_momentum":
                return refused("invalid_request", f"unknown strategy {strategy}", session.status)
            lookback = int(params.get("lookback", season.lookback_bars))
            pnl: list[float] = []
            for s in universe:
                rets = np.diff(np.log(market.closes_until(s, span[1])))
                signal = np.sign(np.convolve(rets, np.ones(lookback), "valid"))[:-1]
                pnl.extend((signal * rets[lookback:]).tolist())
            arr = np.asarray(pnl)
            sharpe = float(arr.mean() / arr.std(ddof=1)) if arr.size > 1 and arr.std() > 0 else 0.0
            self._audit(session, "run_backtest", "run", {"strategy": strategy})
            return ok(
                {
                    "run_id": uuid.uuid4().hex,
                    "strategy": strategy,
                    "known_at": iso(session.known_at),
                    "start": start,
                    "end": end,
                    "fidelity": fidelity,
                    "metrics": {"sharpe_per_bar": sharpe, "n_bars": int(arr.size)},
                }
            )

        @server.tool(description="S6 verdict plus S7 leakage status for a strategy (the fake has no ledger).")
        async def verdict(strategy_id: str) -> CallToolResult:
            return ok(
                {
                    "strategy_id": strategy_id,
                    "verdict": {"strategy_id": strategy_id, "passed": False, "reasons": ["fake gateway: no ledger"]},
                    "llm_driven": session.decisions > 0,
                    "leakage_status": "not_audited" if session.decisions > 0 else "not_required",
                    "leakage_reports": [],
                    "promotable": False,
                    "reasons": ["fake gateway: no ledger attached"],
                }
            )

        @server.tool(
            description="Place an order at the session clock. qty in base units (> 0); side buy|sell; type "
            "market|limit (limit needs price). Checked against exposure, notional and order-rate budgets."
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
        ) -> CallToolResult:
            if (stop := blocked("place_order")) is not None:
                return stop
            budgets = session.budgets

            def reject(code: str, message: str, check: str) -> CallToolResult:
                self._audit(session, "place_order", "refuse", {"symbol": symbol, "check": check})
                if code == "budget_breached":
                    details: dict[str, Any] = {
                        "breaches": [{"kind": check, "message": message}],
                        "freezes_session": False,
                    }
                elif code == "pre_trade_rejected":
                    details = {"checks": [{"name": check, "passed": False}]}
                else:
                    details = {}
                return refused(code, message, session.status, details)

            if session.mode == "research":
                return reject("mode_forbidden", "research sessions cannot trade", "mode")
            if venue != season.venue or symbol not in season.symbols:
                return reject("pre_trade_rejected", f"{venue}/{symbol} is not whitelisted", "instrument_whitelist")
            if side not in ("buy", "sell") or type not in ("market", "limit") or not qty > 0:
                return reject("invalid_request", "bad side, type or qty", "request")
            if type == "limit" and price is None:
                return reject("invalid_request", "limit order needs a price", "request")
            mark = self.mark(symbol, session)
            notional = qty * mark
            if budgets.max_order_notional is not None and notional > budgets.max_order_notional:
                return reject("budget_breached", "order notional above budget", "order_notional")
            signed = qty if side == "buy" else -qty
            after = {**session.positions, symbol: session.positions.get(symbol, 0.0) + signed}
            if budgets.capital is not None and self.gross(session, after) > budgets.capital:
                return reject("budget_breached", "gross exposure above capital budget", "capital")
            if budgets.max_orders_per_window is not None and budgets.order_window_s is not None:
                horizon = budgets.order_window_s
                while session.order_times and (session.known_at - session.order_times[0]).total_seconds() >= horizon:
                    session.order_times.popleft()
                if len(session.order_times) >= budgets.max_orders_per_window:
                    return reject("budget_breached", "order rate above budget", "order_rate")
            session.order_times.append(session.known_at)
            session.orders_submitted += 1
            order_id = uuid.uuid4().hex
            order: dict[str, Any] = {"order_id": order_id, "symbol": symbol, "side": side, "qty": qty, "price": price}
            if type == "limit" and price is not None and not ((mark <= price) if side == "buy" else (mark >= price)):
                session.resting[order_id] = RestingOrder(order_id, symbol, side, qty, price)
                self._audit(session, "place_order", "rest", {"symbol": symbol})
                return ok({"order": {**order, "status": "accepted"}, "reference_price": mark, "notional": notional})
            fee = self._fill(session, symbol, side, qty, mark, "taker")
            self._audit(session, "place_order", "fill", {"symbol": symbol, "side": side, "qty": qty})
            # Like I9's mirror sandbox: the order reports its fill (quantity, average price and fee).
            filled = {**order, "status": "filled", "price": mark, "filled_qty": qty, "avg_price": mark, "fee": fee}
            return ok({"order": filled, "reference_price": mark, "notional": notional, "routed_to": "i6"})

        @server.tool(description="Cancel a resting order.")
        async def cancel_order(order_id: str) -> CallToolResult:
            if (stop := blocked("cancel_order")) is not None:
                return stop
            if session.resting.pop(order_id, None) is None:
                return refused("not_found", f"no resting order {order_id}", session.status)
            return ok({"order": {"order_id": order_id, "status": "cancelled"}, "routed_to": "i6"})

        @server.tool(description="Open positions (base units) with mark prices at the session clock.")
        async def positions() -> CallToolResult:
            items = [
                {"venue": season.venue, "symbol": s, "qty": q, "mark_price": self.mark(s, session)}
                for s, q in sorted(session.positions.items())
                if q != 0.0
            ]
            return ok({"account_id": session.session_id, "positions": items})

        @server.tool(description="Account equity and balance (quote currency) with drawdown from peak.")
        async def account() -> CallToolResult:
            equity = self.equity(session)
            return ok(
                {
                    "account": {"account_id": session.session_id, "equity": equity, "balance": session.cash},
                    "peak_equity": session.peak_equity,
                    "drawdown": 1.0 - equity / session.peak_equity,
                    "max_drawdown": session.budgets.max_drawdown,
                }
            )

        @server.tool(description="Budget limits, usage so far, what remains, and whether the session is frozen.")
        async def budget() -> CallToolResult:
            return ok(
                {
                    "status": session.status,
                    "freeze": session.freeze,
                    "limits": session.budgets.model_dump(exclude_none=True),
                    "usage": self._usage(session),
                    "remaining": self._remaining(session),
                }
            )

        @server.tool(description="Report one LLM decision: tokens in/out, latency in ms and model id.")
        async def report_usage(tokens_in: int, tokens_out: int, latency_ms: float, model: str) -> CallToolResult:
            if (stop := blocked("report_usage")) is not None:
                return stop
            price = self.pricing.get(model)
            if price is None:
                return refused("invalid_request", f"no price for model {model}", session.status)
            cost = (tokens_in * price.input + tokens_out * price.output) / TOKENS_PER_MILLION
            session.tokens_in += tokens_in
            session.tokens_out += tokens_out
            session.token_cost_usd += cost
            session.latency_total_ms += latency_ms
            session.latency_max_ms = max(session.latency_max_ms, latency_ms)
            session.decisions += 1
            b = session.budgets
            breaches = [
                ("token_cost", b.max_token_cost_usd is not None and session.token_cost_usd > b.max_token_cost_usd),
                ("tokens", b.max_tokens is not None and session.tokens_in + session.tokens_out > b.max_tokens),
                ("latency", b.max_latency_ms is not None and latency_ms > b.max_latency_ms),
                (
                    "total_latency",
                    b.max_total_latency_ms is not None and session.latency_total_ms > b.max_total_latency_ms,
                ),
            ]
            for kind, hit in breaches:
                if hit:
                    self._freeze(session, kind, {"model": model})
                    details = {
                        "breaches": [{"kind": kind, "message": f"{kind} budget breached"}],
                        "freezes_session": True,
                        "freeze_budget": kind,
                    }
                    return refused("budget_breached", f"{kind} budget breached", session.status, details)
            return ok(
                {"model": model, "cost_usd": cost, "usage": self._usage(session), "remaining": self._remaining(session)}
            )

        return server

    # ----------------------------------------------------------------- backend protocol
    @asynccontextmanager
    async def tools(self, session: SessionInfo) -> AsyncIterator[GatewayTools]:
        async with Client(self.server_for(session.session_id)) as client:
            yield GatewayTools(McpToolCaller(client))


def _feature(closes: np.ndarray[Any, np.dtype[np.float64]], name: str) -> float | None:
    """A fake-gateway feature from closes up to the query end (None if unknown or undefined)."""
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
    fees: FeeBook | None = None,
) -> GatewayFactory:
    """One fake gateway per episode, each serving that episode's synthetic market.

    ``registry`` (optional) collects the fakes so tests can inspect sessions and the audit log.
    """

    @asynccontextmanager
    async def factory(episode: int) -> AsyncIterator[GatewayBackend]:
        fake = FakeGateway(season, SyntheticMarket(season, episode), pricing, gateway_settings, fees)
        if registry is not None:
            registry[episode] = fake
        transport = httpx.ASGITransport(app=fake.app)
        async with httpx.AsyncClient(transport=transport, base_url=_IN_PROCESS_BASE_URL) as http:
            yield FakeGatewayBackend(fake, AdminClient(http, gateway_settings))

    return factory
