"""Contract-6.6 behaviour of the in-process fake gateway, exercised through the arena's own clients."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from mcp import Client
from pydantic import SecretStr

from agent_arena.gateway.client import AdminClient, GatewayError, GatewayTools, McpToolCaller
from agent_arena.gateway.fake import FakeGateway
from agent_arena.gateway.market import SyntheticMarket
from agent_arena.gateway.models import Account, BudgetStatus, OrderResult, Refusal, SessionBudgets, SessionInfo
from agent_arena.season import SeasonConfig
from agent_arena.settings import ArenaSettings
from agent_arena.timeutil import iso

CONTRACT_TOOLS = {
    "clock",
    "market_data",
    "features",
    "forecast",
    "run_backtest",
    "verdict",
    "place_order",
    "cancel_order",
    "positions",
    "account",
    "budget",
    "report_usage",
}


class Harness:
    def __init__(self, fake: FakeGateway, admin: AdminClient, season: SeasonConfig) -> None:
        self.fake = fake
        self.admin = admin
        self.season = season

    async def session(self, budgets: SessionBudgets | None = None, step: int = 0) -> SessionInfo:
        t = self.season.step_time(0, step)
        return await self.admin.create_session("agent:test", iso(t), budgets or SessionBudgets())


@pytest.fixture
async def harness(settings: ArenaSettings, season: SeasonConfig) -> AsyncIterator[Harness]:
    gw = settings.gateway.model_copy(update={"admin_key": SecretStr("adm")})
    fake = FakeGateway(season, SyntheticMarket(season, 0), settings.pricing, gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app), base_url="http://t") as http:
        yield Harness(fake, AdminClient(http, gw), season)


async def test_tool_list_matches_contract(harness: Harness) -> None:
    session = await harness.session()
    async with Client(harness.fake.server_for(session.session_id)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        schema = next(t for t in (await client.list_tools()).tools if t.name == "place_order").input_schema
    assert names == CONTRACT_TOOLS
    assert "known_at" not in str(schema)
    assert session.api_key
    assert session.mcp_path == "/mcp"


async def test_admin_auth_and_forward_only_clock(harness: Harness, settings: ArenaSettings) -> None:
    unauth = AdminClient(harness.admin._http, settings.gateway)
    with pytest.raises(GatewayError, match="401"):
        await unauth.create_session("x", iso(harness.season.start), SessionBudgets())
    session = await harness.session(step=2)
    with pytest.raises(GatewayError, match="409"):
        await harness.admin.set_clock(session.session_id, iso(harness.season.step_time(0, 1)))
    with pytest.raises(GatewayError, match="404"):
        await harness.admin.set_clock("missing", iso(harness.season.step_time(0, 3)))
    moved = await harness.admin.set_clock(session.session_id, iso(harness.season.step_time(0, 3)))
    assert moved.known_at == iso(harness.season.step_time(0, 3))


async def test_time_travel_is_refused(harness: Harness) -> None:
    session = await harness.session()
    s, t0 = harness.season, harness.season.step_time(0, 0)
    async with Client(harness.fake.server_for(session.session_id)) as client:
        tools = GatewayTools(McpToolCaller(client))
        future = await tools.market_data(s.venue, s.symbols[0], s.channel, iso(t0), iso(t0 + s.bar))
        assert isinstance(future, Refusal)
        assert future.code == "time_travel"
        assert not future.frozen
        past = await tools.market_data(s.venue, s.symbols[0], s.channel, iso(t0 - s.bar), iso(t0))
        assert not isinstance(past, Refusal)
        assert past.rows
        bad = await tools.market_data(s.venue, "NOPE", s.channel, iso(t0 - s.bar), iso(t0))
        assert isinstance(bad, Refusal)
        inverted = await tools.features(["ret_3"], s.symbols, iso(t0), iso(t0 - s.bar))
        assert isinstance(inverted, Refusal)
        feats = await tools.features(["ret_3", "vol_3", "z_3", "nope", "ret_x"], s.symbols, iso(t0 - s.bar), iso(t0))
        assert not isinstance(feats, Refusal)
        values = {(f.symbol, f.name): f.value for f in feats.values()}
        assert values[(s.symbols[0], "nope")] is None
        assert values[(s.symbols[0], "vol_3")] is not None
        clock = await tools.clock()
        assert not isinstance(clock, Refusal)
        assert clock.known_at == iso(t0)


async def test_orders_budgets_and_freeze(harness: Harness) -> None:
    s = harness.season
    budgets = SessionBudgets(
        capital=6000, max_order_notional=5000, max_orders_per_window=2, order_window_s=3600, max_token_cost_usd=0.001
    )
    session = await harness.session(budgets)
    sym = s.symbols[1]
    price = harness.fake.market.price_at(sym, s.step_time(0, 0))
    async with Client(harness.fake.server_for(session.session_id)) as client:
        tools = GatewayTools(McpToolCaller(client))
        big = await tools.place_order(s.venue, sym, "buy", "market", 6000 / price, "sid", "why")
        assert isinstance(big, Refusal)
        assert big.details["check"] == "order_notional"
        filled = await tools.place_order(s.venue, sym, "buy", "market", 4000 / price, "sid", "why")
        assert isinstance(filled, OrderResult)
        assert filled.order.status == "filled"
        assert filled.order.fee == pytest.approx(4000 * s.market.fee_bps / 10_000)
        over = await tools.place_order(s.venue, sym, "buy", "market", 3000 / price, "sid", "why")
        assert isinstance(over, Refusal)
        assert over.details["check"] == "capital"
        resting = await tools.place_order(s.venue, sym, "buy", "limit", 0.1, "sid", "why", price=price * 0.5)
        assert isinstance(resting, OrderResult)
        assert resting.order.status == "accepted"
        rate = await tools.place_order(s.venue, sym, "sell", "market", 0.1, "sid", "why")
        assert isinstance(rate, Refusal)
        assert rate.details["check"] == "order_rate"
        cancelled = await tools.cancel_order(resting.order.order_id)
        assert isinstance(cancelled, OrderResult)
        assert cancelled.order.status == "cancelled"
        assert isinstance(await tools.cancel_order("nope"), Refusal)
        for bad in (
            await tools.place_order("X", sym, "buy", "market", 1.0, "sid", "r"),
            await tools.place_order(s.venue, sym, "hold", "market", 1.0, "sid", "r"),
            await tools.place_order(s.venue, sym, "buy", "limit", 1.0, "sid", "r"),
        ):
            assert isinstance(bad, Refusal)

        positions = await tools.positions()
        assert not isinstance(positions, Refusal)
        assert positions.positions[0].qty == pytest.approx(4000 / price)
        account = await tools.account()
        assert isinstance(account, Account)
        assert account.account.equity == pytest.approx(s.capital - 4000 * s.market.fee_bps / 10_000)

        breach = await tools.report_usage(100_000, 10_000, 100.0, "scripted/trend-chaser")
        assert isinstance(breach, Refusal)
        assert breach.frozen
        assert breach.details["kind"] == "token_cost"
        frozen_data = await tools.market_data(s.venue, sym, s.channel, iso(s.start), iso(s.step_time(0, 0)))
        assert isinstance(frozen_data, Refusal)
        assert frozen_data.code == "session_frozen"
        status = await tools.budget()
        assert isinstance(status, BudgetStatus)
        assert status.frozen
        assert status.frozen_reason == "token_cost"
        assert not isinstance(await tools.positions(), Refusal)
        verdict = await tools.verdict("sid")
        assert not isinstance(verdict, Refusal)
        assert verdict.leakage_status == "not_audited"
        assert verdict.promotable is False
    decisions = [a.decision for a in harness.fake.audit]
    assert "freeze" in decisions
    assert "fill" in decisions


async def test_usage_latency_and_unknown_model(harness: Harness) -> None:
    session = await harness.session(SessionBudgets(max_latency_ms=1000))
    async with Client(harness.fake.server_for(session.session_id)) as client:
        tools = GatewayTools(McpToolCaller(client))
        assert isinstance(await tools.report_usage(1, 1, 1.0, "unknown-model"), Refusal)
        ack = await tools.report_usage(1000, 100, 500.0, "claude-opus-5-5")
        assert not isinstance(ack, Refusal)
        assert ack.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
        slow = await tools.report_usage(1, 1, 5000.0, "claude-opus-5-5")
        assert isinstance(slow, Refusal)
        assert slow.details["kind"] == "latency"


async def test_drawdown_freezes_on_clock_advance(harness: Harness) -> None:
    s = harness.season
    session = await harness.session(SessionBudgets(max_drawdown=0.0001))
    sym = s.symbols[0]
    price = harness.fake.market.price_at(sym, s.step_time(0, 0))
    async with Client(harness.fake.server_for(session.session_id)) as client:
        tools = GatewayTools(McpToolCaller(client))
        await tools.place_order(s.venue, sym, "sell", "market", 9000 / price, "sid", "r")
        for step in range(1, s.steps_per_episode):
            await harness.admin.set_clock(session.session_id, iso(s.step_time(0, step)))
            if harness.fake.sessions[session.session_id].freeze is not None:
                break
        status = await tools.budget()
        assert isinstance(status, BudgetStatus)
        assert status.frozen_reason == "drawdown"


async def test_research_mode_cannot_trade_and_backtest_tools(harness: Harness, settings: ArenaSettings) -> None:
    s = harness.season
    research = AdminClient(harness.admin._http, harness.admin._settings.model_copy(update={"mode": "research"}))
    session = await research.create_session("agent:r", iso(s.step_time(0, 0)), SessionBudgets())
    async with Client(harness.fake.server_for(session.session_id)) as client:
        tools = GatewayTools(McpToolCaller(client))
        refusal = await tools.place_order(s.venue, s.symbols[0], "buy", "market", 0.01, "sid", "r")
        assert isinstance(refusal, Refusal)
        assert refusal.code == "mode_forbidden"
        t0 = s.step_time(0, 0)
        bt = await tools.run_backtest("ts_momentum", {"lookback": 3}, s.symbols, iso(s.start), iso(t0), "vectorised")
        assert not isinstance(bt, Refusal)
        assert bt.metrics["n_bars"] > 0
        assert isinstance(
            await tools.run_backtest("other", {}, s.symbols, iso(s.start), iso(t0), "vectorised"), Refusal
        )
        assert isinstance(
            await tools.run_backtest("ts_momentum", {}, s.symbols, iso(s.start), iso(t0 + s.bar), "vectorised"), Refusal
        )
        fc = await tools.forecast("naive", s.symbols, 2)
        assert not isinstance(fc, Refusal)
        assert {f.symbol for f in fc.forecasts} == set(s.symbols)
        assert set(fc.forecasts[0].quantiles) == {"0.1", "0.5", "0.9"}
