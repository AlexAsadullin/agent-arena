from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from agent_arena.gateway.client import GatewayError, parse_tool_result
from agent_arena.gateway.market import SyntheticMarket, cell_seed
from agent_arena.gateway.models import Refusal, TradeRow
from agent_arena.runner import bar_closes
from agent_arena.season import SeasonConfig, load_season
from agent_arena.settings import ArenaSettings, load_settings
from agent_arena.timeutil import from_ns, iso, parse_iso, to_ns

FIXTURE = Path("tests/fixtures/tiny_season.yaml")


def test_settings_yaml_env_and_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_STATS__BOOTSTRAP_SAMPLES", "777")
    monkeypatch.setenv("ARENA_GATEWAY__ADMIN_KEY", "secret")
    s = load_settings()
    assert s.stats.bootstrap_samples == 777
    assert s.gateway.admin_key is not None
    assert s.gateway.admin_key.get_secret_value() == "secret"
    assert "secret" not in repr(s.gateway)
    assert s.service.port == 8607
    assert s.arena_dir == s.data_root / "agent-arena"
    assert "claude-opus-5-5" in s.pricing


def test_settings_reject_bad_values() -> None:
    with pytest.raises(ValidationError):
        ArenaSettings(_env_file=None, stats={"bootstrap_samples": 1})


def test_time_roundtrip() -> None:
    ts = datetime(2025, 1, 2, 3, 4, 5, 6, tzinfo=UTC)
    assert from_ns(to_ns(ts)) == ts
    assert parse_iso(iso(ts)) == ts
    assert iso(ts).endswith("Z")
    assert parse_iso("2025-01-01T03:00:00+03:00") == datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="naive"):
        to_ns(datetime(2025, 1, 1))
    with pytest.raises(ValueError, match="timezone"):
        parse_iso("2025-01-01T00:00:00")


def test_season_layout(season: SeasonConfig) -> None:
    assert season.episode_start(0) == season.start + season.lookback_bars * season.bar
    # Episodes are back to back: the next one starts one bar after the previous final mark.
    assert season.episode_start(1) == season.step_time(0, season.steps_per_episode) + season.bar
    assert season.end == season.step_time(1, season.steps_per_episode)
    assert season.total_bars > season.episodes * season.steps_per_episode
    assert season.regime(5).name == season.market.regimes[1].name
    assert len(season.llm_agents) == 3
    assert len(season.baseline_agents) == 4


def test_season_validation() -> None:
    raw = load_season(FIXTURE).model_dump()
    with pytest.raises(ValidationError, match="unique"):
        SeasonConfig.model_validate({**raw, "agents": [raw["agents"][0], raw["agents"][0]]})
    with pytest.raises(ValidationError, match="initial_prices"):
        SeasonConfig.model_validate({**raw, "symbols": ["X"]})
    with pytest.raises(ValidationError, match="timezone"):
        SeasonConfig.model_validate({**raw, "start": "2025-01-01T00:00:00"})
    shifted = SeasonConfig.model_validate(
        {**raw, "start": datetime(2025, 1, 1, 3, tzinfo=timezone(timedelta(hours=3)))}
    )
    assert shifted.start.tzinfo == UTC


def test_market_is_deterministic_and_consistent(season: SeasonConfig) -> None:
    a, b = SyntheticMarket(season, 0), SyntheticMarket(season, 0)
    other = SyntheticMarket(season, 1)
    sym = season.symbols[0]
    assert (a.closes[sym] == b.closes[sym]).all()
    assert not (a.closes[sym] == other.closes[sym]).all()
    t = season.step_time(0, 3)
    trades = a.trades(sym, t - season.bar, t)
    assert trades[-1].price == pytest.approx(a.price_at(sym, t))
    assert all(t - season.bar <= from_ns(x.ts_exchange) <= t for x in trades)
    assert a.closes_until(sym, t)[-1] == a.price_at(sym, t)
    with pytest.raises(ValueError, match="no data"):
        a.price_at(sym, season.start - 10 * season.bar)
    assert cell_seed(1, "x") == cell_seed(1, "x") != cell_seed(2, "x")


def test_bar_closes_takes_last_trade_per_bar() -> None:
    hour = 3_600_000_000_000
    end = from_ns(10 * hour)
    rows = [
        TradeRow(ts_exchange=8 * hour - 5, price=1.0, size=1),
        TradeRow(ts_exchange=8 * hour, price=2.0, size=1),
        TradeRow(ts_local=10 * hour, price=4.0, size=1),
        TradeRow(ts_exchange=9 * hour + 1, price=3.0, size=1),
    ]
    # Bar ending at 9h has no trades, so it carries the 8h close forward.
    assert bar_closes(rows, end, 3, hour) == [2.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="timestamp"):
        _ = TradeRow(price=1.0, size=1).ts


def test_parse_tool_result_shapes() -> None:
    def result(text: str, structured: object = None, error: bool = False) -> CallToolResult:
        return CallToolResult(
            content=[TextContent(type="text", text=text)], structured_content=structured, is_error=error
        )

    assert parse_tool_result(result("", {"a": 1})) == {"a": 1}
    assert parse_tool_result(result('{"b": 2}')) == {"b": 2}
    refused = parse_tool_result(
        result("", {"refused": True, "code": "budget_breached", "message": "m", "session_status": "frozen"})
    )
    assert isinstance(refused, Refusal)
    assert refused.frozen
    body = '{"refused": true, "code": "time_travel", "message": "late", "details": {}, "session_status": "active"}'
    text_refusal = parse_tool_result(result(body, error=True))
    assert isinstance(text_refusal, Refusal)
    assert (text_refusal.code, text_refusal.frozen) == ("time_travel", False)
    assert parse_tool_result(result("")) == {}
    err = parse_tool_result(result("bad", error=True))
    assert isinstance(err, Refusal)
    assert err.code == "tool_error"
    with pytest.raises(GatewayError):
        parse_tool_result(result("not json"))
    with pytest.raises(GatewayError):
        parse_tool_result(result("", [1, 2]))
