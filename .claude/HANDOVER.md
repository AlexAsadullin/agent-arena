# HANDOVER: agent-arena (Tradeq S8)

Last updated 2026-10-23 by the S8 agent, after adapting to the fixed I9 gateway.

## Goal

Agent Arena evaluates LLM trading agents with statistics that survive scrutiny. It deflates Sharpe ratios for the
number of configurations tried, bootstraps across episodes, seeds and prompt variants, charges token cost and
decision latency against PnL, and sends every order through the same I9 risk path that humans use. The brief is
`../prompts/07-agent-arena.md`, and the default port is 8607. The project consumes contracts 6.6 (I9), 6.1 (S6)
and 6.5 (S7 leakage reports), and it produces leaderboard artefacts.

## Stack

Python 3.13 with a uv `.venv` and pins from `../constraints.txt`. Libraries: the `mcp` 2.3.0 SDK (client and the
in-process fake server), httpx and httpx2, FastAPI (the fake admin app), numpy, scipy, pandas and pyarrow,
pydantic-settings (`ARENA_*`, `config/settings.yaml`, season files in `config/seasons/`, prompt templates in
`config/prompts/`), typer (the `arena` CLI) and structlog. Transitive pins live in
`[tool.uv] constraint-dependencies`.

## Architecture and data flow

`season.py` expands a season into episodes and runs (agent by seed by prompt variant). For each run, `runner.py`
opens a paper session through the I9 admin API (`gateway/client.py`) and moves the session clock forward one
bar per step. It drives the agent over MCP, so the agent reads market_data, features, forecast and budget,
reports usage and places orders. When a session freezes, the runner stops asking the agent but keeps marking the
book until the episode ends. `agents/` holds the LLM agents (Anthropic and OpenAI-compatible providers in
`providers.py`), a deterministic scripted LLM double with three behaviours, and four baselines.
`accounting.py` keeps Decimal balances and charges token cost and latency slippage. `stats/` and
`leaderboard.py` compute net Sharpe with a two-level paired bootstrap, the DSR from S6 or a labelled local
fallback, pairwise win probabilities (bootstrap and Bradley-Terry) and Holm-corrected tests against the best
baseline. `diagnostics.py` measures over-trading, consistency, sizing and refusals. `gateway/fake.py` is a full
in-process fake of I9 that tests and the demo use.

## Files

| Path | Purpose |
|---|---|
| src/agent_arena/runner.py, season.py, pipeline.py | Season, episode, run and step orchestration |
| src/agent_arena/gateway/{client,fake,market,models}.py | I9 MCP and admin client, the fake gateway, the synthetic market |
| src/agent_arena/agents/{base,llm,providers,scripted,baselines,factory}.py | Agents and LLM provider adapters |
| src/agent_arena/accounting.py | Net-of-intelligence-cost PnL |
| src/agent_arena/stats/{bootstrap,pairwise,sharpe}.py, leaderboard.py | Leaderboard statistics |
| src/agent_arena/diagnostics.py | Behavioural diagnostics |
| src/agent_arena/clients/{ledger,leakage}.py | S6 and S7 clients with fallbacks |
| src/agent_arena/experiments.py, report/render.py | make report |

## State at handover

1. The repo is finished and green. Lint and `mypy --strict` are clean, 78 tests pass and 1 live test is deselected, and coverage is 96%. The tree is clean, with 10 commits.
2. `make reproduce` runs 234 runs in about 1 minute, and `make report` takes about 10 s. Outputs are in `$TRADEQ_DATA_DIR/agent-arena/seasons/` and `reports/`.
3. docs/report.md is committed, and the orchestrator checked it against the writing rules.
4. Headline results (synthetic): a single run per agent named the wrong winner in 51.7% of 300 seasons (95% CI 46.0% to 57.3%), while the arena protocol did so in 0.0% (CI 0.0% to 1.3%). Among agents with no skill, 8.2% of configurations passed PSR of at least 0.95, against 0.0% for DSR. A design of 12 episodes, 3 seeds and 3 variants made 92.7% correct calls, against 72.7% for 6 episodes. Latency slippage, not token cost, moved the martingale double from 4th place to last.
5. The fake gateway matches the finished I9: bearer auth with the session api_key or the admin key plus `X-Gateway-Session`, the I9 budget names, and refusal bodies with `isError`. Arena sessions are paper sessions, because I9 forbids trading in research mode.
6. The real I9 runs a small season: `make integration` (marker `integration`, skipped by `make test`) starts `../risk-gated-mcp-gateway/.venv/bin/gateway serve --offline` on a free loopback port with a scratch TRADEQ_DATA_DIR under the gitignored `tmp/` and the admin key from `ARENA_GATEWAY__ADMIN_KEY` (random if unset). Season `config/seasons/i9-offline.yaml`: 2 agents, 2 episodes, 1 seed, 30 one-minute steps, order rate 4 per 60 s on the session clock. Last result (2026-10-23): 4 runs, 226 orders filled, rejects only `capital` and `order_notional`, every fill price equal to I9's reference price, no frozen run; ts-momentum net Sharpe -586 above the trend chaser at -984, which paid 199.4 bps of intelligence cost per run. About 4 to 12 s.
7. `arena run-season --gateway-url http://host:port` points the client at a running I9 (for example `make -C ../risk-gated-mcp-gateway serve-offline`). `accounting.fill_price_source` stays `order`, because I9 now forwards the session clock to I6 (`X-Tradeq-Known-At`).
8. Fixes from the real I9: I2-shaped trade rows (`valid_time`, int ns or ISO), wide feature rows, the order's `price` used as the fill price, refusal causes from `details.breaches[].kind` and `details.checks[]` (`Refusal.reason`), httpx2 request logs silenced, and strategy ids joined with `:` (`agent-arena:<season>:<agent>`), because S6's verdict route cannot match a slash; the S6 client refuses ids with a slash. The fake gateway returns the same shapes.
9. Price tables: the Claude rates in `config/settings.yaml` match I9's (claude-opus-5-5 4.00/20.00, claude-sonnet-5-5 2.00/10.00, claude-haiku-5-5 0.10/0.50 USD per million tokens). I9 prices the scripted doubles with its default row (4/20), so I9's token budget for them is stricter than the arena's cost.

## Mistakes and lessons

1. The first version of the fake gateway was written before I9 existed, and its auth, budget names and refusal bodies differed from I9. Commit bfd3b21 aligned them. Check the fake against the real I9 README whenever I9 changes.
2. uv first resolved newer cffi, idna and pycparser than `constraints.txt` lists. Pin transitive dependencies through `[tool.uv] constraint-dependencies`.
3. The fake gateway first ran tools in worker threads, which made runs slow and nondeterministic. They now run inline (commit 753d2e3).
4. `make reproduce` and `make report` write synthetic demo trials into the shared S6 inbox in the real data root. Set `ARENA_LEDGER__OFFLINE_INBOX=false` to stop this when it is not wanted.
5. The latency slippage rate (2 bps per second, capped at 50 bps) is an assumption until it is calibrated on I6 fills. No run has used a real LLM yet.
6. The fake was aligned with the I9 README, not with I9 output, so three shapes were still wrong until the first real run: trade rows carry `valid_time` instead of `ts_exchange`, features come in wide rows, and refusal causes sit in `details.breaches` and `details.checks`. Validate the fake against a real run (make integration) after every I9 change, not against documentation.
7. Four I9 issues found by the first real run were fixed in I9 on 2026-10-23: the order-rate window now runs on the session clock, the offline I6 fake fills at the session clock, `gateway serve --offline` exists (the arena's launcher script was deleted), and the claude-opus-5-5 price is 4/20. The arena's strategy ids contained slashes until the same day, which S6's verdict route cannot match; check ids against every route that embeds them.
8. The shared scratchpad directory is shared with other agents, and one of them overwrote a profiling script. Keep scratch files under this repo's `tmp/`.
