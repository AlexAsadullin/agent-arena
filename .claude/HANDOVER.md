# HANDOVER: agent-arena (Tradeq S8)

Last updated 2026-10-23 by the orchestrating session. It was written from the finishing agent's final report and
from the repository.

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

## Mistakes and lessons

1. The first version of the fake gateway was written before I9 existed, and its auth, budget names and refusal bodies differed from I9. Commit bfd3b21 aligned them. Check the fake against the real I9 README whenever I9 changes.
2. uv first resolved newer cffi, idna and pycparser than `constraints.txt` lists. Pin transitive dependencies through `[tool.uv] constraint-dependencies`.
3. The fake gateway first ran tools in worker threads, which made runs slow and nondeterministic. They now run inline (commit 753d2e3).
4. `make reproduce` and `make report` write synthetic demo trials into the shared S6 inbox in the real data root. Set `ARENA_LEDGER__OFFLINE_INBOX=false` to stop this when it is not wanted.
5. The latency slippage rate (2 bps per second, capped at 50 bps) is an assumption until it is calibrated on I6 fills. No run has used the real I9 or a real LLM yet.
