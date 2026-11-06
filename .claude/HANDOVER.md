# HANDOVER: agent-arena (Tradeq S8)

Last updated 2026-11-05 (T-Invest fees for MOEX). Earlier sections: 2026-10-23 (fixed I9), 2026-10-29 (MOEX pivot).

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

1. Lint and `mypy --strict` are clean; 88 tests pass and 2 are deselected (`live` in test_gateway_http.py and the `integration` real-I9 test), coverage 96%, 16 commits. The fee work of 2026-11-05 is not committed yet.
2. `make reproduce` runs 234 runs in about 1 minute, and `make report` takes about 10 s. Outputs are in `$TRADEQ_DATA_DIR/agent-arena/seasons/` and `reports/`.
3. docs/report.md is committed, and the orchestrator checked it against the writing rules.
4. Headline results (synthetic): a single run per agent named the wrong winner in 51.7% of 300 seasons (95% CI 46.0% to 57.3%), while the arena protocol did so in 0.0% (CI 0.0% to 1.3%). Among agents with no skill, 8.2% of configurations passed PSR of at least 0.95, against 0.0% for DSR. A design of 12 episodes, 3 seeds and 3 variants made 92.7% correct calls, against 72.7% for 6 episodes. Latency slippage, not token cost, moved the martingale double from 4th place to last.
5. The fake gateway matches I9's auth (session api_key, or the admin key plus `X-Gateway-Session`), budget names and refusal bodies with `isError`. Since 2026-11-05 a market fill also reports `filled_qty`, `avg_price` and `fee` the way I9's mirror sandbox does. It still serves its own synthetic market and answers `routed_to: i6`; it has no order book. Arena sessions are paper sessions, because I9 forbids trading in research mode.
6. The real I9 runs a small MOEX season: `make integration` (marker `integration`, skipped by `make test`) starts `../risk-gated-mcp-gateway/.venv/bin/gateway serve --offline` on a free loopback port with a scratch TRADEQ_DATA_DIR under the gitignored `tmp/` and the admin key from `ARENA_GATEWAY__ADMIN_KEY` (random if unset). Season `config/seasons/i9-offline.yaml`: moex-tbank SBER and GAZP in whole lots, 2 agents, 2 episodes, 1 seed, 30 one-minute steps. I9 routes the orders to its mirror sandbox. Last result (2026-11-05): 4 runs, 209 orders filled, rejects only `capital` and `order_notional`, fills within 5 ticks of the reference price, I9's per-fill fees equal to the arena's schedule, and the arena's booked fees equal to I9's; ts-momentum net Sharpe -959 above the trend chaser at -1193 (168.8 bps of intelligence cost). About 5 s.
7. `arena run-season --gateway-url http://host:port` points the client at a running I9 (for example `make -C ../risk-gated-mcp-gateway serve-offline`). `accounting.fill_price_source` stays `order`; with the mirror the order's `avg_price` is used anyway.
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

## MOEX pivot (2026-10-29)

The platform is MOEX only now, and I9 routes paper sessions to its mirror sandbox (contract 6.8), which matches
orders against the order book in I2. Changes here, made by the session that did the pivot:
1. `config/seasons/i9-offline.yaml`: venue `moex-tbank`, symbols SBER and GAZP, rouble capital and budgets,
   `lot_sizes` (new optional season field).
2. `runner.py`: `round_to_lots` rounds every order toward zero to whole lots (MOEX rejects fractions; I9 refuses
   them with `qty_whole_lots`); a `cancelled` market order that carries `filled_qty > 0` is booked as the partial
   fill it is (the mirror cancels what the book cannot fill).
3. `tests/integration/test_real_i9.py`: fills are now routed to `mirror` and sit within 5 ticks of I9's reference
   price, instead of exactly at it. Last run (2026-10-30): 193 orders filled, rejects only `capital` and
   `order_notional`, about 5 s.
4. The arena's own fake gateway still uses its crypto symbols and routes to `i6`; it is a self-contained fake and
   was left as is. The report (`docs/report.md`) is unchanged, because it uses the fake.

Use: the arena's main role is strategy against strategy (quant baselines, RL policies, scripted agents); real LLM
agents are optional and should run under a token budget, because every bar of every run is an LLM call.

## T-Invest fees for MOEX (2026-11-05)

User decision: local MOEX simulations charge the T-Invest basic tariff, 0.25% of trade value per fill, and 0% for
T-Bank's own funds (T-Capital ETFs); there is no broker. I9's mirror sandbox got the same model in parallel
(`brokers/commission.py`).
1. `fees.py` (`FeeBook`) reads `fees.venues.<venue>` (moex-tbank, moex-algopack, moex-iss share one YAML anchor;
   `source` cites the decision): `maker_fee`, `taker_fee`, `rounding` (0.01: each fill's fee rounded half-up to the
   kopeck on the exact decimal `qty * price`, as I9 does), `zero_fee_funds` (tickers matching I9's list, instrument
   uids, I1's reference file `market-tape-recorder/reference/tbank_zero_fee_funds.json`; an absent file logs
   `zero_fee_reference_missing` and falls back to config; a malformed one raises). Symbols match by ticker (SECID).
2. The runner books the fee that the gateway reports and charges nothing more; without a reported fee it applies
   the schedule (taker), else `accounting.fallback_fee_bps`. The fake charges the schedule (taker for market,
   maker for resting fills), else the season's `market.fee_bps`, now optional and removed from `i9-offline.yaml`.
3. Tests: 88 passed (6 new: `tests/unit/test_fees.py`, a MOEX season on the fake in `test_runner_pipeline.py`);
   the live test checks I9's per-fill fees against `FeeBook` and the arena total against I9's total.
4. The synthetic demo season, `make report` and `docs/report.md` are unchanged (SYNTH venue, 5 bps).

## MOEX demo world (2026-11-06)

`config/seasons/synthetic.yaml` is now MOEX: venue moex-tbank, SBER-RUB-TQBR and GAZP-RUB-TQBR (lot 10) and
TMOS-RUB-TQTF (lot 1, 0% commission), 1,000,000 RUB, prices of 2026-11-05, fees from `fees.venues.moex-tbank`
(no `market.fee_bps`). The crypto season is `config/seasons/synthetic-crypto.yaml` (optional). `make reproduce`
and `make report` were rerun with `ARENA_LEDGER__OFFLINE_INBOX=false` (no demo trials in the S6 inbox): 234
runs; buy-and-hold leads at net Sharpe 7.87 [-2.86, 19.69]; martingale gross -17.81 vs net -31.01. The protocol
experiments (E1, E1b, E2) do not depend on the world and are unchanged. 88 tests pass.
