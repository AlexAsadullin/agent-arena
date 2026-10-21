"""Arena runner: season -> episodes -> runs (agent x seed x prompt variant) -> steps.

For every run the runner opens a gateway session through the admin API, connects to the MCP
tools for that session, and for each step:

1. reads market data, features, forecasts and budget through MCP tools at the session clock;
2. asks the agent for target weights;
3. reports LLM usage to the gateway (``report_usage``) and charges token cost to the ledger;
4. sends market orders through ``place_order`` (the gateway's pre-trade checks and budgets
   apply exactly as they would to a human), mirrors the fills, and charges latency slippage;
5. advances the session clock one bar through the admin API and marks the book to market.

Once the gateway freezes a session (budget breach), the agent is no longer consulted and the
book is carried to the end of the episode, so a frozen run still bears the risk it holds.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import structlog

from agent_arena.accounting import BPS, Ledger, dec, latency_cost, token_cost
from agent_arena.agents.base import Agent, Decision, Observation
from agent_arena.agents.factory import AgentFactory
from agent_arena.gateway.client import GatewayBackend, GatewayError, GatewayFactory, GatewayTools
from agent_arena.gateway.models import Refusal, SessionBudgets, TradeRow
from agent_arena.results import RunResult, StepRecord
from agent_arena.season import AgentConfig, SeasonConfig
from agent_arena.settings import ArenaSettings
from agent_arena.timeutil import iso, to_ns, utcnow

log = structlog.get_logger(__name__)

# Rationale text forwarded to the gateway's audit log is truncated to keep orders small.
_RATIONALE_CHARS = 500


@dataclass(frozen=True)
class RunSpec:
    agent: AgentConfig
    variant: str | None
    episode: int
    seed: int


def strategy_id(season_id: str, agent_id: str) -> str:
    """Strategy id used for orders and S6 trials: one strategy per arena agent."""
    return f"agent-arena/{season_id}/{agent_id}"


def bar_closes(rows: list[TradeRow], end: datetime, n_bars: int, bar_ns: int) -> list[float]:
    """Close of each bar ending at ``end - k * bar`` for k = n_bars-1 .. 0 (last trade in the bar)."""
    ordered = sorted(rows, key=lambda r: r.ts)
    end_ns = to_ns(end)
    closes: list[float] = []
    idx = 0
    last: float | None = None
    for k in range(n_bars - 1, -1, -1):
        boundary = end_ns - k * bar_ns
        last_in_bar: float | None = None
        while idx < len(ordered) and ordered[idx].ts <= boundary:
            if ordered[idx].ts > boundary - bar_ns:
                last_in_bar = ordered[idx].price
            idx += 1
        if last_in_bar is not None:
            last = last_in_bar
        if last is not None:
            closes.append(last)
    return closes


@dataclass
class _Execution:
    traded_notional: Decimal
    n_orders: int = 0
    n_rejected: int = 0
    n_clamped: int = 0
    frozen_reason: str | None = None

    def __post_init__(self) -> None:
        self.reasons: list[str] = []


class ArenaRunner:
    def __init__(
        self,
        season: SeasonConfig,
        settings: ArenaSettings,
        gateway_factory: GatewayFactory,
        agents: AgentFactory,
    ) -> None:
        self.season = season
        self.settings = settings
        self.gateway_factory = gateway_factory
        self.agents = agents
        self._bar_ns = season.bar_seconds * 1_000_000_000

    def plan(self) -> list[RunSpec]:
        specs: list[RunSpec] = []
        for episode in range(self.season.episodes):
            for seed in self.season.seeds:
                for agent in self.season.agents:
                    variants: list[str | None] = list(self.season.prompt_variants) if agent.kind == "llm" else [None]
                    specs.extend(RunSpec(agent, v, episode, seed) for v in variants)
        return specs

    async def run_season(self) -> list[RunResult]:
        self.agents.validate(self.season)
        specs = self.plan()
        limiter = asyncio.Semaphore(self.settings.max_concurrent_runs)
        episodes = sorted({s.episode for s in specs})

        async def run_episode(episode: int) -> list[RunResult]:
            episode_specs = [s for s in specs if s.episode == episode]
            async with self.gateway_factory(episode) as gateway:

                async def guarded(spec: RunSpec) -> RunResult:
                    async with limiter:
                        return await self.run_one(gateway, spec)

                return list(await asyncio.gather(*(guarded(s) for s in episode_specs)))

        nested = await asyncio.gather(*(run_episode(e) for e in episodes))
        results = [r for episode_results in nested for r in episode_results]
        results.sort(key=lambda r: (r.agent_id, r.variant or "", r.episode, r.seed))
        log.info("season_done", season=self.season.season_id, runs=len(results))
        return results

    async def run_one(self, gateway: GatewayBackend, spec: RunSpec) -> RunResult:
        season = self.season
        agent = self.agents.build(spec.agent, spec.variant, spec.seed)
        started = utcnow()
        t0 = season.step_time(spec.episode, 0)
        session_agent_id = f"{spec.agent.id}/{spec.variant or 'baseline'}/e{spec.episode}/s{spec.seed}"
        session = await gateway.admin.create_session(
            session_agent_id, iso(t0), SessionBudgets(**season.budgets.model_dump(exclude={"schema_version"}))
        )
        ledger = Ledger(season.budgets.capital)
        steps: list[StepRecord] = []
        frozen_reason: str | None = None
        model = spec.agent.model if spec.agent.kind == "llm" else None
        async with gateway.tools(session) as tools:
            closes = await self._window(tools, t0)
            for step in range(season.steps_per_episode):
                record, frozen_reason, closes = await self._step(
                    gateway, tools, session.session_id, agent, spec, step, ledger, closes, frozen_reason
                )
                steps.append(record)
        return RunResult(
            run_id=uuid.uuid4().hex,
            season_id=season.season_id,
            agent_id=spec.agent.id,
            is_llm=agent.is_llm,
            model=model,
            variant=spec.variant,
            episode=spec.episode,
            seed=spec.seed,
            session_id=session.session_id,
            started_at=iso(started),
            finished_at=iso(utcnow()),
            frozen_reason=frozen_reason,
            steps=steps,
        )

    async def _step(
        self,
        gateway: GatewayBackend,
        tools: GatewayTools,
        session_id: str,
        agent: Agent,
        spec: RunSpec,
        step: int,
        ledger: Ledger,
        closes: dict[str, list[float]],
        frozen_reason: str | None,
    ) -> tuple[StepRecord, str | None, dict[str, list[float]]]:
        season = self.season
        t = season.step_time(spec.episode, step)
        prices = {s: c[-1] for s, c in closes.items()}
        equity = ledger.equity(prices)
        weights_before = ledger.weights(prices)
        if frozen_reason is None and equity <= 0:
            frozen_reason = "bankrupt"
        observed: Observation | str | None = None
        if frozen_reason is None:
            observed = await self._observe(tools, spec, step, t, closes, ledger)
            if isinstance(observed, str):
                frozen_reason = observed
        decision: Decision
        if isinstance(observed, Observation):
            decision = await agent.decide(observed)
        else:
            decision = Decision(targets=weights_before, status="frozen")
        fees_before = ledger.fees

        tokens = Decimal(0)
        if decision.usage is not None:
            tokens = token_cost(decision.usage, self.settings.pricing)
            ledger.charge_tokens(tokens)
            ack = await tools.report_usage(
                decision.usage.tokens_in,
                decision.usage.tokens_out,
                decision.usage.latency_ms,
                decision.usage.model,
            )
            if isinstance(ack, Refusal) and ack.frozen:
                frozen_reason = ack.reason or ack.code

        execution = _Execution(traded_notional=Decimal(0))
        if frozen_reason is None and decision.status == "ok":
            execution = await self._execute(tools, spec, decision, prices, float(equity), ledger)
            frozen_reason = execution.frozen_reason
        latency = Decimal(0)
        if decision.usage is not None and execution.traded_notional > 0:
            latency = latency_cost(execution.traded_notional, decision.usage.latency_ms, self.settings.latency)
            ledger.charge_latency(latency)
        weights_after = ledger.weights(prices)

        t_next = season.step_time(spec.episode, step + 1)
        await gateway.admin.set_clock(session_id, iso(t_next))
        closes_next = await self._window(tools, t_next)
        equity_next = ledger.equity({s: c[-1] for s, c in closes_next.items()})
        e_now = float(equity)
        ret_net = float(equity_next - equity) / e_now if e_now > 0 else 0.0
        ret_gross = float(equity_next - equity + tokens + latency) / e_now if e_now > 0 else 0.0
        record = StepRecord(
            step=step,
            ts=iso(t),
            equity=e_now,
            equity_next=float(equity_next),
            ret_net=ret_net,
            ret_gross=ret_gross,
            fees=float(ledger.fees - fees_before),
            token_cost=float(tokens),
            latency_cost=float(latency),
            weights_before=weights_before,
            targets=decision.targets,
            weights_after=weights_after,
            turnover=sum(abs(weights_after[s] - weights_before[s]) for s in season.symbols),
            traded_notional=float(execution.traded_notional),
            n_orders=execution.n_orders,
            n_rejected=execution.n_rejected,
            n_clamped=execution.n_clamped,
            reject_reasons=execution.reasons,
            status=decision.status,
            tokens_in=decision.usage.tokens_in if decision.usage else 0,
            tokens_out=decision.usage.tokens_out if decision.usage else 0,
            latency_ms=decision.usage.latency_ms if decision.usage else 0.0,
            frozen=frozen_reason is not None,
        )
        return record, frozen_reason, closes_next

    async def _observe(
        self,
        tools: GatewayTools,
        spec: RunSpec,
        step: int,
        t: datetime,
        closes: dict[str, list[float]],
        ledger: Ledger,
    ) -> Observation | str:
        """Build the observation, or return the freeze reason if the gateway froze the session."""
        season = self.season
        budget = await tools.budget()
        if isinstance(budget, Refusal):
            return budget.reason or budget.code
        if budget.frozen:
            return budget.frozen_reason or "frozen"
        start = iso(t - season.lookback_bars * season.bar)
        prices = {s: c[-1] for s, c in closes.items()}
        features: dict[str, dict[str, float | None]] = {}
        if season.features:
            feats = await tools.features(season.features, season.symbols, start, iso(t))
            if not isinstance(feats, Refusal):
                for f in feats.features:
                    features.setdefault(f.symbol, {})[f.name] = f.value
        forecasts: dict[str, float] = {}
        if season.forecast_model is not None:
            fc = await tools.forecast(season.forecast_model, season.symbols, season.forecast_horizon)
            if not isinstance(fc, Refusal):
                forecasts = {item.symbol: item.mean for item in fc.forecasts}
        limit = budget.limits.get("token_spend_usd")
        used = budget.used.get("token_spend_usd", 0.0)
        equity = float(ledger.equity(prices))
        return Observation(
            season_id=season.season_id,
            episode=spec.episode,
            step=step,
            seed=spec.seed,
            known_at=t,
            symbols=season.symbols,
            closes=closes,
            features=features,
            forecasts=forecasts,
            weights=ledger.weights(prices),
            equity=equity,
            cash=float(ledger.cash),
            max_weight=season.max_weight,
            token_budget_left_usd=None if limit is None else limit - used,
        )

    async def _execute(
        self,
        tools: GatewayTools,
        spec: RunSpec,
        decision: Decision,
        prices: dict[str, float],
        equity: float,
        ledger: Ledger,
    ) -> _Execution:
        season = self.season
        settings = self.settings.accounting
        result = _Execution(traded_notional=Decimal(0))
        sid = strategy_id(season.season_id, spec.agent.id)
        for symbol in season.symbols:
            target = decision.targets.get(symbol, ledger.weights(prices)[symbol])
            bounded = max(-season.max_weight, min(season.max_weight, target))
            if bounded != target:
                result.n_clamped += 1
            price = prices[symbol]
            delta = bounded * equity / price - ledger.position(symbol)
            if abs(delta) * price < settings.min_order_notional:
                continue
            side = "buy" if delta > 0 else "sell"
            order = await tools.place_order(
                season.venue,
                symbol,
                side,
                "market",
                abs(delta),
                sid,
                decision.rationale[:_RATIONALE_CHARS],
            )
            result.n_orders += 1
            if isinstance(order, Refusal):
                result.n_rejected += 1
                result.reasons.append(order.code)
                if order.frozen:
                    result.frozen_reason = order.reason or order.code
                    break
                continue
            if order.status == "rejected":
                result.n_rejected += 1
                result.reasons.append(order.reject_reason or "rejected")
                continue
            if order.status not in ("filled", "partially_filled"):
                continue
            qty = order.filled_qty if order.filled_qty is not None else abs(delta)
            fill_price = order.avg_price if order.avg_price is not None else price
            fee = order.fee if order.fee is not None else qty * fill_price * settings.fallback_fee_bps / float(BPS)
            ledger.apply_fill(symbol, qty if side == "buy" else -qty, fill_price, fee)
            result.traded_notional += dec(qty) * dec(fill_price)
        return result

    async def _window(self, tools: GatewayTools, t: datetime) -> dict[str, list[float]]:
        """Bar closes per symbol for the look-back window ending at ``t`` (last close = price at ``t``)."""
        season = self.season
        start = iso(t - season.lookback_bars * season.bar)
        out: dict[str, list[float]] = {}
        for symbol in season.symbols:
            data = await tools.market_data(season.venue, symbol, season.channel, start, iso(t))
            if isinstance(data, Refusal):
                raise GatewayError(f"market_data refused at {iso(t)}: {data.code} {data.reason}")
            closes = bar_closes(data.rows, t, season.lookback_bars + 1, self._bar_ns)
            if not closes:
                raise GatewayError(f"no trades for {symbol} in the window ending {iso(t)}")
            out[symbol] = closes
        return out
