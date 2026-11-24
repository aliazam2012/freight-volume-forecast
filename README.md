# volume-forecast

Daily shipment forecasts for a made-up national carrier, backtested against a seasonal-naive baseline, with prediction ranges whose coverage is measured on data the calibration never saw, and an agent that answers planners' questions only through tools and cites the run its numbers came from.

## Table of Contents

- [Results](#results)
- [How It Works](#how-it-works)
- [Run It](#run-it)
- [The Planner Agent](#the-planner-agent)
- [Limits](#limits)

---

## Results

From the [2026-11-23 backtest report](reports/2026-11-23-backtest.md), run `fc-9eb8b88f44fb`: synthetic data, 12 lanes, 8 forecast origins 28 days apart, each forecasting 28 days ahead.

| Model | WAPE | MASE | Bias |
|---|---|---|---|
| gradient_boosting | 11.4% | 0.56 | +0.6% |
| holt_winters | 15.1% | 0.74 | -4.2% |
| seasonal_naive | 17.8% | 0.88 | -1.8% |

- Both models beat the seasonal-naive baseline. Gradient boosting cuts the error by about a third.
- On holidays, gradient boosting's WAPE is 54.2% against 300.9% for the baseline, which repeats last week's normal day. It knows the holiday calendar in advance; the baseline does not.
- Gradient boosting's 80% ranges held the actual volume 83.5% of the time and its 95% ranges 96.0%, measured on the later four origins after calibrating on the earlier four.

The planner agent's [reference evaluation](evals/reports/2026-11-23-reference.md) passes its gate on 16 questions: every figure in every answer appears in a tool result, and every answer that uses a figure cites the run.

## How It Works

| Step | Module | What it does |
|---|---|---|
| Data | `data.py` | Seeded daily volume, 2024-01-01 to 2026-10-31: lane levels with slow drift, a weekday profile, an annual peak from October to December, six US holidays, about six four-day promotions a year per lane, and one lane (MEM-STL) losing 32% of its volume from 2025-09-01 |
| Models | `models.py` | One interface, `fit(history)` and `predict(future_rows)`. Seasonal naive (same weekday last week), Holt-Winters per lane (damped trend, weekly season, on log volume), and one global gradient-boosted model predicting volume as a multiple of the lane's recent level from the calendar and lags of 28 days or more |
| Backtest | `backtest.py` | Rolling origins; each model is fitted only on data up to its origin. WAPE, MASE scaled by the in-sample seasonal-naive error, and bias, per model and per lane |
| Intervals | `intervals.py` | Split conformal on log(actual + 1) minus log(forecast + 1), per model and week ahead, so ranges can be lopsided the way freight misses are |
| Runs | `runs.py`, `report.py` | A run id hashes the data, configuration and package version; the same inputs give the same id, and on one machine the same report |
| Agent | `tools.py`, `prompts.py`, `evaluation.py` | Four read-only tools over one run, a short system prompt, a golden set and a grader |

A test changes every volume after the origin and checks that the model features do not move, which proves no forecast sees the future. [ADR-0008](../../docs/adr/ADR-0008-forecasting-models.md) records why these models and not others.

## Run It

```bash
uv run freight-forecast backtest                 # about 20 seconds; writes reports/<today>-backtest.md
uv run freight-forecast forecast --lane CHI-DAL --days 14
uv run freight-forecast eval                     # the planner golden set in reference mode
```

`forecast` prints each day with its 80% and 95% ranges and the run id. Runs are stored in `var/forecast/runs`, which git ignores.

Asking the agent with a live model costs money and needs your own key (`ANTHROPIC_API_KEY`, or `--model openai --model-name <id>` with `OPENAI_API_KEY`):

```bash
uv run freight-forecast ask "How much volume is coming on NWK-BOS Thanksgiving week?"
```

## The Planner Agent

The model never does arithmetic. Every total, comparison and percentage comes from a tool:

| Tool | Returns |
|---|---|
| `get_forecast` | Daily forecasts with 80% and 95% ranges, and the total over the days asked |
| `compare_to_last_year` | The forecast total against the actual total on the same weekdays 52 weeks earlier, with holidays in both periods |
| `explain_drivers` | One day's weekday, holiday, promotion, a typical weekday that week, the lane's recent average, and the same weekday last year |
| `backtest_summary` | WAPE, MASE and bias per model, overall or for one lane, and interval coverage |

Every tool result, error messages included, names the run id. The grader fails an answer that states a number no tool result or question contains, an answer that drops the run id, a case that calls the wrong tools, and an unexpected tool error. The golden answers are rendered from real tool outputs by [make_reference.py](evals/make_reference.py), which also commits the run they describe to `evals/run`. The evaluation reruns the tools against that run, so a tool change that moves a figure fails the gate until the answers are regenerated on purpose. Grading a committed run, not a rebuilt one, keeps the gate stable across machines whose arithmetic differs in the last digit.

## Limits

- The data is synthetic. The numbers show the method working on realistic shapes, not accuracy on any real network.
- The backtest windows run 2026-03-22 to 2026-10-31, so the November and December peak is untested.
- No promotions are announced past the end of the data, so the forward forecast assumes none.
- The agent's reference evaluation replays scripted answers. A live model's results come with the first recorded run ([issue #10](https://github.com/aliazam2012/freight-agents/issues/10)).
