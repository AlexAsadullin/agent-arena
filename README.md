# agent-arena (S8)

Agent Arena ranks LLM trading agents and quant baselines with statistics that account for luck and cost. Every
agent trades through the I9 MCP gateway under the same budgets and pre-trade checks as a human, pays for its
tokens and its decision latency, and is ranked by net Sharpe ratio with bootstrap confidence intervals over
market episodes, seeds and prompt variants, a deflated Sharpe ratio and pairwise win probabilities. On
synthetic agents with known skill, a single run per agent (the Alpha Arena and LiveTradeBench rule) named the
wrong winner in 51.7% of 300 seasons; the arena protocol named a wrong winner in 0.0% (95% upper bound 1.3%)
and abstained when the evidence was too weak. Specification: `../quant_platform_plan_full.md`, project S8.

## Quickstart

```bash
cd agent-arena
cp .env.example .env          # set TRADEQ_DATA_DIR to the absolute path of Tradeq/data
make setup                    # uv venv --python 3.13 .venv && uv sync --frozen
make lint typecheck test      # offline, 25 to 55 s depending on load
make reproduce                # synthetic season against the in-process fake gateway, about 1 min
make report                   # experiments behind docs/report.md, about 10 s
make integration              # opt-in: a small MOEX season through the real I9 gateway, about 5 s
```

`make reproduce` runs 3 scripted LLM doubles and 4 baselines over 6 episodes x 3 seeds x 3 prompt variants
(234 runs) and writes `$TRADEQ_DATA_DIR/agent-arena/seasons/synthetic-s1/` (`leaderboard.md`,
`leaderboard.html`, `leaderboard.json`, `runs.jsonl`, `steps.parquet`, `meta.json`). Trials go to the S6
ledger, or to its offline inbox `$TRADEQ_DATA_DIR/research-ledger/inbox/s8-<yyyymmdd>.jsonl`.

CLI: `arena run-season --season <yaml> [--gateway fake|live] [--gateway-url http://host:port]`,
`arena leaderboard --season <yaml>` (recompute from stored runs), `arena report --season <yaml>` (re-render
Markdown and HTML). `--gateway-url` points the admin API and MCP at a running I9 and implies `--gateway live`.

### Running against the real I9

`make integration` runs `tests/integration/test_real_i9.py` (marker `integration`, skipped by `make test`). It
starts `../risk-gated-mcp-gateway/.venv/bin/gateway serve --offline` (override the checkout with
`ARENA_I9_REPO`) as a subprocess on a free loopback port, so I9 runs with its own in-process fakes of its
downstream services. I9 gets a scratch
`TRADEQ_DATA_DIR` under the gitignored `tmp/` and the admin key `ARENA_GATEWAY__ADMIN_KEY` (random per run if
unset). The season is `config/seasons/i9-offline.yaml`: 2 agents, 2 episodes, 1 seed, 30 one-minute steps.

Against an I9 started with `make -C ../risk-gated-mcp-gateway serve-offline`, run
`ARENA_GATEWAY__ADMIN_KEY=<key> arena run-season --season config/seasons/i9-offline.yaml --gateway-url
http://127.0.0.1:8608`. The season trades SBER and GAZP on `moex-tbank` in whole lots; I9 routes paper orders
to its mirror sandbox, which matches them against I2's order book. The test checks that every fill is routed
to `mirror` within 5 ticks of I9's reference price, that each fill's fee equals the arena's fee schedule,
that the arena books I9's fees once, and that the only rejects are exposure and order-notional limit hits.
Last run (2026-11-05): 4 runs, 209 orders filled, ts-momentum net Sharpe -959 [-1221, -697] against the
trend chaser at -1193 [-1273, -1113] with 168.8 bps of intelligence cost (one-minute bars annualised over
525,600 periods; a smoke test, not evidence).

## Configuration

Defaults live in `config/settings.yaml`; env variables override them with the prefix `ARENA_` and `__` for
nesting (`ARENA_STATS__BOOTSTRAP_SAMPLES=5000`). Seasons (instruments, episodes, seeds, prompt variants,
budgets, agents) are YAML files in `config/seasons/`; prompt templates are in `config/prompts/variants.yaml`.

| Variable | Meaning |
|---|---|
| `TRADEQ_DATA_DIR` | platform data root; this project writes only under `agent-arena/` (plus the S6 inbox fallback) |
| `ARENA_GATEWAY__ADMIN_KEY` | I9 admin key, sent as `Authorization: Bearer <key>` |
| `ARENA_GATEWAY__MCP_URL`, `ARENA_GATEWAY__ADMIN_URL` | I9 endpoints (default port 8608) |
| `ARENA_PROVIDERS__ANTHROPIC__API_KEY`, `ARENA_PROVIDERS__OPENAI_COMPAT__API_KEY` | keys for real LLM agents |
| `ARENA_LEDGER__BASE_URL`, `ARENA_LEAKAGE__BASE_URL` | S6 and S7 services |
| `ARENA_ACCOUNTING__FILL_PRICE_SOURCE` | `order` (default: the order's fill price) or `reference` (I9's I2 reference price at the session clock) |

Fees: `fees.venues.<venue>` in `config/settings.yaml` holds the fee schedule per venue. The MOEX venues use the
T-Invest basic tariff per user, 2026-11-05: 0.25% of trade value per fill, maker and taker alike, rounded
half-up to the kopeck, and 0% for T-Bank's own funds (T-Capital ETFs; config tickers merged with I1's
`$TRADEQ_DATA_DIR/market-tape-recorder/reference/tbank_zero_fee_funds.json` when it exists, otherwise a
warning). This is the same model as I9's mirror sandbox. The arena books the fee that the gateway reports
for an order (I9 and the fake both report it) and charges nothing more; only when a gateway reports no fee
does it apply the schedule, or `accounting.fallback_fee_bps` on a venue without one. The fake gateway charges
the schedule too, or the season's `market.fee_bps` on a venue without one (the synthetic crypto season).

Model ids are set per agent in the season file; prices (USD per million tokens) are in `pricing`, and a model
without a price cannot enter a season. Latency slippage is `latency.slippage_bps_per_second` (default 2).

## Architecture

```
season YAML -> ArenaRunner -> per episode: GatewayBackend (I9 over HTTP, or the in-process fake)
                 |  per run: admin API creates a paper session; per step: set clock -> MCP tools
                 |  (market_data, features, forecast, budget) -> agent.decide -> report_usage -> place_order
                 v
            RunResult (step records) -> S6 trials -> S7 leakage audit -> leaderboard -> md / html / json
```

| Module | Role |
|---|---|
| `gateway/client.py`, `gateway/models.py` | typed MCP and admin client for contract 6.6 (I9 result shapes) |
| `gateway/fake.py`, `gateway/market.py` | in-process I9 fake (`mcp` MCPServer + FastAPI admin) and the synthetic market |
| `agents/` | one `Agent` protocol: LLM agents (Anthropic and OpenAI-compatible adapters over httpx, a scripted double) and baselines |
| `accounting.py` | Decimal ledger; token cost from the price table, latency slippage on traded notional |
| `runner.py` | seasons, episodes, runs and steps; frozen sessions carry their book to the episode end |
| `stats/` | Sharpe, PSR, DSR; paired two-level bootstrap; win probabilities, Bradley-Terry, Holm |
| `diagnostics.py` | over-trading, seed and prompt agreement, limit hits, concentration, martingale score, refusal and error rates |
| `leaderboard.py`, `report/`, `storage.py` | leaderboard model, Markdown and HTML rendering, artefact files |
| `clients/` | S6 ledger and S7 leakage clients with offline fallbacks |
| `experiments.py` | `make report` experiments |

## Contracts

Consumed (conventions section 6): 6.6 MCP gateway tools and admin API, 6.1 S6 `POST /v1/trials/batch` with
the inbox fallback and `GET /v1/strategies/{id}/verdict`, 6.5 S7 `POST /v1/probe` with fallback to report
files and then to the status `not_audited`. Result shapes of the 6.6 tools follow the I9 README: refusals are
MCP errors with `{refused, code, message, details, session_status}`; MCP calls carry the session `api_key`
as a bearer token, or the admin key plus `X-Gateway-Session`. Arena sessions are `paper` sessions with a
session clock, because I9 research sessions cannot trade.

Produced: leaderboard artefacts under `$TRADEQ_DATA_DIR/agent-arena/seasons/<season_id>/`. `leaderboard.json`
(`schema_version: "1"`) has one row per agent with `net_sharpe`, `gross_sharpe`, `net_return_pct`,
`intelligence_cost_bps` (each `{value, ci_low, ci_high}`), `dsr` and `dsr_source` (`s6` or `local-fallback`),
`p_vs_best_baseline`, `p_holm`, `bt_strength`, `leakage` and `diagnostics`, plus the two pairwise win
probability matrices. `steps.parquet` has one row per run and step (`ts_decision` in int64 ns).

Contract notes: no contract field was added. The arena reads optional order fill details (`filled_qty`,
`avg_price`, `fee`) when the gateway reports them; otherwise it fills market orders at the order price (or
`reference_price`) with the schedule fee.

## Key results

Synthetic data only (see `docs/report.md`, which `make report` regenerates); 95% intervals in brackets. Rerun
on 2026-11-06 on the MOEX demo world (SBER, GAZP, TMOS in roubles, whole lots, 0.25% commission, 0% for TMOS).
The E1/E1b/E2 rows are simulations of the protocol and did not change; the demo-season rows did.

| Result | Value |
|---|---|
| Wrong winner, single run per agent, 5 agents with known skill | 51.7% [46.0, 57.3] of 300 seasons |
| Wrong winner, arena protocol, same seasons | 0.0% [0.0, 1.3]; correct 71.3% [66.0, 76.2], abstains otherwise |
| False passes without skill: PSR >= 0.95 vs DSR >= 0.95 | 8.2% [7.4, 9.0] vs 0.0% [0.0, 0.1] |
| Correct calls: 6 x 3 x 3 vs 12 x 3 x 3 design (episodes x seeds x variants) | 72.7% [67.4, 77.4] vs 92.7% [89.1, 95.1] |
| Martingale double, gross vs net Sharpe in the demo season | -17.81 [-24.22, -11.47] vs -31.01 [-36.18, -25.90] |
| Demo-season leader (net Sharpe) | buy-and-hold 7.87 [-2.86, 19.69]; no agent beats it significantly |

## Limitations and roadmap

All evidence is synthetic; no real LLM and no real market data have been run yet. The demo season
(`config/seasons/synthetic.yaml`) is a MOEX world with continuous hourly bars (no session gaps or auctions);
the v0.1 crypto season is kept as `config/seasons/synthetic-crypto.yaml`. With a 0.25% commission, every
frequently rebalancing agent loses, so the demo ranking mostly measures turnover. The latency slippage rate is an assumption until it is calibrated on fills. The real I9 has run a small
MOEX season (`make integration`) through its mirror sandbox with I9's offline fakes of I2 and the other
services, not with recorded MOEX books. I9 prices the scripted test doubles with its
default row, so its token budget for them differs from the arena's cost table; the Claude rates agree.

Roadmap: a calibrated latency model from I6 fills; a season with real models through the Anthropic and
OpenAI-compatible adapters; minimum-detectable-effect planning per season from the measured variance
components; leaderboard publishing to I8.
