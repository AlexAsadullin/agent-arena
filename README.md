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
make lint typecheck test      # offline, about 30 s
make reproduce                # synthetic season against the in-process fake gateway, about 1 min
make report                   # experiments behind docs/report.md, about 10 s
```

`make reproduce` runs 3 scripted LLM doubles and 4 baselines over 6 episodes x 3 seeds x 3 prompt variants
(234 runs) and writes `$TRADEQ_DATA_DIR/agent-arena/seasons/synthetic-s1/` (`leaderboard.md`,
`leaderboard.html`, `leaderboard.json`, `runs.jsonl`, `steps.parquet`, `meta.json`). Trials go to the S6
ledger, or to its offline inbox `$TRADEQ_DATA_DIR/research-ledger/inbox/s8-<yyyymmdd>.jsonl`.

CLI: `arena run-season --season <yaml> [--gateway fake|live]`, `arena leaderboard --season <yaml>` (recompute
from stored runs), `arena report --season <yaml>` (re-render Markdown and HTML).

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
`avg_price`, `fee`) when the gateway reports them; otherwise it fills market orders at `reference_price` with
the configured fallback fee.

## Key results

Synthetic data only (see `docs/report.md`, which `make report` regenerates); 95% intervals in brackets.

| Result | Value |
|---|---|
| Wrong winner, single run per agent, 5 agents with known skill | 51.7% [46.0, 57.3] of 300 seasons |
| Wrong winner, arena protocol, same seasons | 0.0% [0.0, 1.3]; correct 71.3% [66.0, 76.2], abstains otherwise |
| False passes without skill: PSR >= 0.95 vs DSR >= 0.95 | 8.2% [7.4, 9.0] vs 0.0% [0.0, 0.1] |
| Correct calls: 6 x 3 x 3 vs 12 x 3 x 3 design (episodes x seeds x variants) | 72.7% [67.4, 77.4] vs 92.7% [89.1, 95.1] |
| Martingale double, gross vs net Sharpe in the demo season | -3.83 [-12.53, 5.22] vs -25.27 [-33.80, -16.51] |

## Limitations and roadmap

All evidence is synthetic; no real LLM and no real market data have been run yet. The latency slippage rate
is an assumption until it is calibrated on I6 fills. The live gateway path is covered by unit tests with
mocked HTTP and one test behind `-m live`, but it has not yet run a full season against a running I9.

Roadmap: a calibrated latency model from I6 fills; a season with real models through the Anthropic and
OpenAI-compatible adapters; minimum-detectable-effect planning per season from the measured variance
components; leaderboard publishing to I8.
