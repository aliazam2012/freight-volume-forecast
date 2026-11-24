# ADR-0008: Forecasting models, backtest and intervals

Status: accepted 2026-11-23.

## Table of Contents

- [Context](#context)
- [Decision](#decision)
- [Alternatives](#alternatives)
- [Consequences](#consequences)

---

## Context

volume-forecast has to show classic machine learning done the way a planning team would trust it: a baseline every model must beat, a backtest that cannot leak the future, ranges whose coverage is measured, and an agent on top that cannot invent a number. Twelve lanes of daily data over nearly three years is a small problem; the choices should fit that size.

## Decision

- **Baseline first.** Seasonal naive (same weekday last week) is the bar, and MASE is scaled by its in-sample error, so a MASE below 1 means a model beats it.
- **Two real models.** Holt-Winters per lane (damped trend, weekly season, on log volume) as the classical statistical model, and one global HistGradientBoostingRegressor across all lanes. The global model predicts volume as a multiple of the lane's level 28 to 55 days back, from the calendar (weekday, day of year, holiday, announced promotion), lags of 28 days or more, and last year's seasonal shape. Lags of at least the horizon mean one model serves every day of a 28-day forecast without feeding its own forecasts back in.
- **Rolling-origin backtest.** Eight origins 28 days apart; at each, every model is fitted only on data up to the origin. A test changes all later volumes and checks that the features do not move.
- **Split conformal intervals.** Bounds come from the earlier half of the origins, per model and week ahead, on the log ratio of actual to forecast; coverage is measured on the later half. The forward forecast's ranges use all eight origins.
- **Runs with ids.** A run id hashes the data, configuration and package version. The report and every agent answer cite it.
- **Agent over tools.** The model calls read-only tools and quotes their figures; it never computes. A grader checks every number against the tool results.

## Alternatives

- **Prophet.** It handles holidays and seasonality well, but brings a Stan toolchain for a problem the global model already covers with plain features.
- **A neural forecaster (N-BEATS, a temporal fusion transformer).** Built for thousands of series; with twelve it adds training cost and tuning without a likely gain, and is harder to explain to a planner.
- **One gradient-boosted model per lane.** About a twelfth of the training rows each, and a holiday seen only a handful of times per lane. The global model learns the holiday effect across all lanes at once.
- **Quantile regression for ranges.** Needs a model per quantile and gives no coverage guarantee; conformal ranges wrap any model and their coverage is checked directly.

## Consequences

Results depend only on code and a seed and need no API key. On one machine a rerun gives the same report; across processor types the model arithmetic can move a last digit (the first CI run showed it), so the planner's golden set grades a committed copy of the run its answers describe rather than a rebuilt one. The data is synthetic, so the report shows the method, not accuracy on a real network, and says so. The backtest windows miss the November and December peak; the report states that, and a later backtest with origins in the fall would test it. Adding a model means adding one class with `fit` and `predict`; the backtest, intervals, report and tools take it unchanged.
