# Agent Arena: choosing a statistical evaluation protocol for LLM trading agents

This is the research report of the present project (S8, agent-arena), version 0.1. All numbers come from synthetic data, which the Data section describes, so they test the evaluation protocol and are not evidence about any market or any real language model.

## 1. Question

The present study asks which evaluation protocol the arena should use to rank LLM trading agents, because the public analogs rank agents in ways that their own commentators do not trust. The spec section on S8 states that commentators found Alpha Arena results regime-dependent and not statistically significant, and that later seasons produced trade counts from 158 to 1,418 under identical prompts. The study therefore asks three narrower questions. The first question is how often a ranking that rests on a single run per agent, as in Alpha Arena and LiveTradeBench, names the wrong winner when the true skill of every agent is known. The second question is how many market episodes, seeds and prompt variants the arena needs before it can name a winner. The third question is how much the ranking changes when the arena charges each agent for its tokens and for its decision latency, which the analogs do not charge.

## 2. Data

The study uses two synthetic data sets, because the arena has no live market feed or paid model access yet and because question one needs known ground truth. Neither data set is market evidence.

The first data set is a simulation of agents with known skill, which the experiments module of the present project generates (make report, tables E1 and E2). Each of five agents produces per-step returns r = mu_k + beta_k m(e, t) + u(k, e, run) + eps. In this model mu_k is the true skill, m is a market factor that all agents share within an episode, u is a persistent offset that stays fixed for one seed and prompt variant, and eps is unit noise. The configured annualised Sharpe ratios of the five agents are 12, 6, 0, -6 and -12, and the factor loadings are 0.2, 1.0, -0.5, 0.6 and -1.0, so the ranking of the agents changes from regime to regime. The factor has a standard deviation of 0.6, the persistent offset has a standard deviation of 0.03, and each run has 72 hourly steps with 8,760 periods per year. A null scenario sets every mu_k to zero, so that no agent has skill and any declared winner is wrong.

The second data set is the stored demo season synthetic-s1, which make reproduce creates when it runs the full arena against the in-process fake of the I9 gateway (tables E0 and E3, and the season leaderboard). The fake gateway serves a synthetic market of two instruments, which follows an autoregressive log-return process with four regimes (trend up, mean reversion, trend down and momentum). The season has 6 episodes of 72 hourly decisions, 3 seeds and 3 prompt variants (neutral, risk averse and aggressive), which gives 234 runs. Three scripted LLM doubles take part, and each one reads the rendered prompt and answers with text. The trend chaser trades large noisy momentum positions, the cautious contrarian holds small mean-reversion positions inside a no-trade band, and the martingale gambler doubles its size after each loss and sometimes refuses or answers with malformed text. Four quant baselines (buy and hold, time-series momentum, mean reversion and a random agent) compete under the same gateway budgets.

## 3. Methods

The arena protocol ranks agents by the annualised net Sharpe ratio of each run, and it attaches uncertainty with a two-level bootstrap, which resamples market episodes first and then the seed and prompt replicates inside each drawn episode. Every agent shares the same episode draws in each bootstrap replicate, so the comparisons stay paired on the market path, as Davison and Hinkley (1997, section 3.8) describe for two-stage resampling. The protocol declares a winner only when the bootstrap win probability of the top agent, which is the share of replicates in which its mean beats a rival, reaches 0.95 against every rival. Otherwise the protocol abstains, and it also abstains when the season has fewer than two episodes.

The net Sharpe ratio uses per-step returns after trading fees and after the intelligence cost, which the arena computes from two parts. The token cost multiplies the reported input and output tokens by the configured price table in USD per million tokens. The latency cost charges the traded notional with a slippage of 2 basis points per second of decision latency, capped at 50 basis points, because the arena freezes the market clock while the agent thinks and a live agent would trade later at a moved price. The gross Sharpe ratio adds both charges back.

The leaderboard also reports the deflated Sharpe ratio (DSR), which corrects the Sharpe ratio for the number of strategies that the researcher tried. The probabilistic Sharpe ratio (PSR) of Bailey and Lopez de Prado (2012) is the probability that the true Sharpe ratio exceeds a benchmark, given the sample length, skewness and kurtosis. The DSR of Bailey and Lopez de Prado (2014) is the PSR whose benchmark is the expected maximum Sharpe ratio across all agent and prompt-variant configurations of the season. The arena takes the DSR from the S6 research ledger when the ledger answers, and otherwise it computes the local fallback and labels it as such.

Significance against the baselines uses a one-sided paired bootstrap test of the hypothesis that the mean run Sharpe of an agent does not exceed that of the best baseline, with the shifted-null p-value of Efron and Tibshirani (1993, section 16.4) and the Holm (1979) step-down correction across agents. Pairwise strengths come from a Bradley-Terry model (Bradley and Terry 1952), which the arena fits to per-episode head-to-head outcomes with the minorisation-maximisation algorithm of Hunter (2004). The study compares the arena protocol with two alternatives. The single-run protocol takes one run per agent and ranks by total return, which is how Alpha Arena and LiveTradeBench rank models. The mean-only protocol averages the run Sharpe over all runs and always names the top agent, so it uses the same data as the arena but reports no uncertainty. All rates carry 95% Wilson score intervals, and every experiment uses fixed seeds.

## 4. Experiments and results

### 4.1 E1: the single-run protocol names the wrong winner about half of the time

The single-run protocol named the wrong winner in 51.7% of 300 replications (Wilson 95% interval 46.0% to 57.3%), while the arena protocol named a wrong winner in 0.0% of them (interval 0.0% to 1.3%), as table E1 of make report shows. The two-proportion z-test for this difference equals 14.5, while the critical value at 95% confidence is 1.96, so the hypothesis that both protocols err equally often is rejected. The arena paid for this accuracy with abstention, because it declared no winner in 28.7% of the replications (23.8% to 34.0%) and named the true best agent in 71.3% (66.0% to 76.2%). The mean-only protocol named the wrong winner in 0.7% of the skill replications, which shows that most of the gain comes from the use of 54 runs per agent rather than one.

| Scenario | Single run wrong | Mean only wrong | Arena wrong | Arena abstains | Arena correct |
|---|---|---|---|---|---|
| Skill (true Sharpe 11.9, 5.1, 0.0, -5.6, -10.3) | 51.7% [46.0, 57.3] | 0.7% [0.2, 2.4] | 0.0% [0.0, 1.3] | 28.7% [23.8, 34.0] | 71.3% [66.0, 76.2] |
| Null (no agent has skill) | 100.0% [98.7, 100.0] | 100.0% [98.7, 100.0] | 0.3% [0.1, 1.9] | 99.7% [98.1, 99.9] | not applicable |

The null scenario shows why abstention matters, because both protocols that always name a winner were wrong in every replication when no agent had skill, while the arena declared a winner in 0.3% of them (0.1% to 1.9%). This rate is far below the 5% that a test at 95% confidence would allow, so the requirement of a 0.95 win probability against every rival is a conservative rule.

Deflation removed the false passes that the plain PSR produced, as table E1b shows. Among 4,500 agent and variant configurations without skill, 8.2% (7.4% to 9.0%) reached a PSR of at least 0.95, which exceeds the nominal 5%, because the persistent seed and prompt offsets make the pooled returns of a configuration dependent, which the PSR formula does not assume. A one-sample z-test of the 8.2% rate against 5% equals 9.8, while the critical value is 1.96, so the excess is not random. With 15 configurations per season, the chance that at least one of them passes the PSR rule is 72.2%, so a season leaderboard that used the PSR would usually show a false winner. The DSR rule passed 0.0% of the configurations (0.0% to 0.1%).

### 4.2 E2: the arena needs about 54 runs per agent, and episodes matter most

The correct-call rate of the arena rose from 30.0% with 6 episodes and one run each to 72.7% with 6 episodes, 3 seeds and 3 variants, and to 92.7% with 12 episodes, 3 seeds and 3 variants, as table E2 shows. The wrong-call rate was 7.0% (4.6% to 10.5%) with 3 episodes, 2.3% with 6 episodes and one replicate, and 0.0% (upper bound 1.3%) for every design that had at least 18 runs per agent. The small-sample result means that the percentile bootstrap understates the uncertainty when it resamples only three episodes, so the arena should not publish a winner from fewer than six episodes.

| Design (episodes x seeds x variants) | Runs per agent | Arena correct | Arena wrong | CI half-width of the best agent's Sharpe |
|---|---|---|---|---|
| 1 x 1 x 1 | 1 | 0.0% (abstains) | 0.0% | not defined |
| 3 x 1 x 1 | 3 | 21.0% [16.8, 26.0] | 7.0% [4.6, 10.5] | 9.78 (standard error 0.30) |
| 6 x 1 x 1 | 6 | 30.0% [25.1, 35.4] | 2.3% [1.1, 4.7] | 7.51 (0.15) |
| 6 x 3 x 1 | 18 | 37.7% [32.4, 43.3] | 0.0% [0.0, 1.3] | 6.42 (0.08) |
| 6 x 1 x 3 | 18 | 43.0% [37.5, 48.7] | 0.0% [0.0, 1.3] | 6.48 (0.08) |
| 6 x 3 x 3 | 54 | 72.7% [67.4, 77.4] | 0.0% [0.0, 1.3] | 4.06 (0.04) |
| 12 x 3 x 3 | 108 | 92.7% [89.1, 95.1] | 0.0% [0.0, 1.3] | 2.94 (0.02) |
| 6 x 5 x 5 | 150 | 93.0% [89.5, 95.4] | 0.0% [0.0, 1.3] | 2.57 (0.03) |

An extra episode buys more accuracy than an extra replicate, because 12 episodes with 3 seeds and 3 variants reached 92.7% with 108 runs per agent, while 6 episodes with 5 seeds and 5 variants reached 93.0% only with 150 runs. The demo season supports the same conclusion, because the standard deviation of run Sharpe between episodes was 11.5 to 13.3 for the three LLM doubles, which is larger than the standard deviation across seeds (0.01 to 5.9) and across prompt variants (2.5 to 11.0), as table E0 shows. In the demo season the prompt variants moved run Sharpe more than the seeds for all three doubles, for example 10.96 against 5.88 for the martingale gambler, so a season with real LLM agents needs at least three prompt variants.

| LLM double | SD between episodes | SD across seeds | SD across variants | SD total |
|---|---|---|---|---|
| Trend chaser | 13.28 | 4.97 | 7.39 | 13.87 |
| Cautious contrarian | 12.47 | 0.01 | 2.52 | 11.67 |
| Martingale gambler | 11.50 | 5.88 | 10.96 | 14.20 |

### 4.3 E3: intelligence cost reverses the rank of the over-trading agents

Charging the intelligence cost moved the martingale gambler from fourth to last place, because its gross Sharpe ratio of -3.83 (95% interval -12.53 to 5.22) fell to a net Sharpe ratio of -25.27 (-33.80 to -16.51) in the demo season, as the season leaderboard and table E3 show. Its intelligence cost was 614.5 basis points of capital per run (545.0 to 680.4), and the trend chaser paid 325.1 basis points (292.9 to 356.8), while the cautious contrarian paid 12.1 basis points (9.2 to 15.4). Kendall's tau between the gross and the net ranking of the seven agents equals 0.62 with a p-value of 0.069, so the hypothesis that the two rankings are unrelated is not rejected at 95% confidence, although the two over-trading doubles changed places with the baselines.

Latency slippage, and not token spend, produced almost all of the intelligence cost in this season. The token share of the cost was 0.03% to 0.05% for the three doubles, because a prompt of about 1,500 tokens at the configured prices costs a fraction of a cent per decision on a 10,000 USD account, as table E3 shows. Table E3b supports this conclusion, because a 300-fold token volume lowered the net Sharpe of the martingale gambler only from -25.3 to -28.5 at 2 basis points per second, while the removal of the latency charge raised it from -25.3 to -3.8. With no latency charge, which matches the cost-free accounting of LiveTradeBench, the martingale gambler ranked fourth, ahead of buy and hold at -7.4.

| Agent | Gross Sharpe | Net Sharpe | Intelligence cost, bps per run |
|---|---|---|---|
| Cautious contrarian | 0.12 [-8.55, 9.38] | -0.80 [-9.28, 8.19] | 12.1 |
| Mean reversion (baseline) | -2.56 [-12.87, 8.78] | -2.56 [-12.87, 8.78] | 0.0 |
| Time-series momentum (baseline) | -3.60 [-11.20, 4.89] | -3.60 [-11.20, 4.89] | 0.0 |
| Martingale gambler | -3.83 [-12.33, 4.74] | -25.27 [-33.39, -17.33] | 614.5 |
| Buy and hold (baseline) | -7.43 [-15.83, 0.93] | -7.43 [-15.83, 0.93] | 0.0 |
| Trend chaser | -11.33 [-17.87, -4.01] | -22.12 [-31.04, -12.47] | 325.1 |
| Random (baseline) | -11.59 [-18.85, -5.83] | -11.59 [-18.85, -5.83] | 0.0 |

The intervals in this table come from table E3, which uses 400 bootstrap resamples, so they differ slightly from the leaderboard intervals, which use 2,000.

No LLM double beat the best baseline in the demo season, although the cautious contrarian had the highest net Sharpe ratio. The one-sided paired bootstrap test of the contrarian against mean reversion gives a p-value of 0.145, which becomes 0.873 after the Holm correction, while the critical level is 0.05, so the hypothesis that the contrarian is no better than mean reversion is not rejected. The behavioural diagnostics of the leaderboard identified the designed behaviours. The martingale gambler had a martingale score of 0.66, which is the rise in the probability that an agent adds exposure after a loss, and a limit-hit rate of 0.77 per step, while the trend chaser turned over 2.69 times the median baseline turnover.

### 4.4 Parameters that changed during the study

Three parameters changed during the study, and each change followed an observed fact. The gateway exposure budget rose from 9,000 to 15,000 USD and the order-notional budget rose from 8,000 to 12,000 USD, because the buy-and-hold baseline at a weight of 0.5 per instrument hit the exposure limit on 12 of 23 orders in a test season, which would have penalised a baseline for the arena's own budget choice. The synthetic market changed from 4 trades per bar and a look-back of 48 bars to 2 trades per bar and 24 bars, because the measured cost of about 6 milliseconds per decision step, which mostly went into the serialisation of market data through MCP, made the full season take several minutes. The arena protocol started to abstain with fewer than two episodes, because the first run of table E2 showed a wrong-call rate of 48.3% for the one-episode design, where the bootstrap has no spread to resample.

## 5. Comparison with analogs

The present study could not run the analogs themselves, because Alpha Arena was a closed real-money event, LiveTradeBench needs 50 days of live data and paid model access, and TradingAgents needs paid LLM calls for every decision, so the study reimplemented their evaluation protocols on the synthetic data. The spec section on S8 states that Alpha Arena Season 1 ranked six models on one run each and that four of the six lost money, so the single-run protocol of table E1 reproduces its ranking rule, and that rule named the wrong winner in 51.7% of the skill replications and in every null replication. The spec section also states that LiveTradeBench ran a 50-day live evaluation of 21 models with a fixed agent template and without cost or latency economics, so its protocol corresponds to a single prompt variant with no intelligence cost, and table E3b shows that cost-free accounting would have ranked the martingale gambler fourth rather than last. The spec section states that TradingAgents is non-deterministic and that a third-party review advises against it when reproducible decisions are needed, and table E0 measures the same issue in the arena, where the prompt variant alone moved run Sharpe by a standard deviation of up to 10.96.

| Protocol | Runs per agent | Uncertainty | Intelligence cost | Wrong winner, skill scenario | Wrong winner, null scenario |
|---|---|---|---|---|---|
| Single run, ranked by PnL (Alpha Arena and LiveTradeBench rule) | 1 | none | not charged | 51.7% [46.0, 57.3] | 100.0% [98.7, 100.0] |
| Mean over all runs (simple baseline) | 54 | none | charged | 0.7% [0.2, 2.4] | 100.0% [98.7, 100.0] |
| Arena: two-level bootstrap, 0.95 win probability, DSR | 54 | bootstrap CI, DSR, Holm | charged | 0.0% [0.0, 1.3] | 0.3% [0.1, 1.9] |

## 6. Decision

The present project adopts the arena protocol, which ranks by net Sharpe ratio, resamples episodes and then replicates, names a winner only at a win probability of 0.95 against every rival, and deflates for every agent and prompt configuration. The evidence for this decision is that the protocol named no wrong winner in table E1, while the single-run rule of the analogs was wrong about half of the time, and that only the DSR kept false passes below 1% in table E1b. The project also requires at least 6 episodes, 3 seeds and 3 prompt variants for a published season, because table E2 shows wrong calls of 7.0% with three episodes and a correct-call rate of 72.7% with the 6 x 3 x 3 design. The arena charges both tokens and latency, because table E3 shows that the latency charge decided the rank of the two over-trading doubles. The latency rate of 2 basis points per second remains an assumption, so the leaderboard shows the gross Sharpe ratio next to the net one.

## 7. Limitations

The results rest on synthetic data, so they show how the protocol behaves under a known generating process and say nothing about real markets or real language models. The skill simulation uses Gaussian noise and one market factor, and real returns with fat tails or serial correlation would widen the intervals that the bootstrap reports. The latency charge depends on a configured slippage rate, which the project has not yet calibrated against fills from the I6 shadow exchange, and table E3b shows that the ranking is sensitive to that rate. The scripted doubles answer from fixed rules, so their seed and prompt sensitivity in table E0 reflects the configured noise rather than the behaviour of any real model. The real I9 gateway does not let research sessions trade, so the arena opens paper sessions that are pinned to a session clock, and the fills then come from I6, which must replay the same history for the marks to match.

## 8. References

Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2).

Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. Journal of Portfolio Management 40(5).

Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs. Biometrika 39.

Davison, A. C. and Hinkley, D. V. (1997). Bootstrap Methods and their Application. Cambridge University Press.

Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6.

Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models. Annals of Statistics 32.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association 22.

The spec sources, as quant_platform_plan_full.md cites them in project S8, are TradingAgents (arxiv.org/html/2412.20138v4) with its third-party review (x-cmd.com/install/tradingagents), LiveTradeBench (arxiv.org/html/2511.03628v1), the Alpha Arena Season 1 report (forklog.com), the commentary on significance (innobu.com) and the report on later seasons (wallstreetcn.com/articles/3771681).

The present project produces its evidence with two targets: make report writes tables E0 to E3 to $TRADEQ_DATA_DIR/agent-arena/reports/report_tables.md, and make reproduce writes the season leaderboard to $TRADEQ_DATA_DIR/agent-arena/seasons/synthetic-s1/leaderboard.md.
