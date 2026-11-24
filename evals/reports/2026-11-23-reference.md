# Planner Agent Evaluation, 2026-11-23

## Table of Contents

- [Summary](#summary)
- [Metrics](#metrics)
- [Cases](#cases)

---

## Summary

Mode: reference. Forecast run `fc-9eb8b88f44fb`. Cases: 16. Gate: passed.

Reference mode replays scripted answers while the tools run for real: it checks the tools, the grader and the gate, not a model.

## Metrics

| Metric | Value |
|---|---|
| grounded_answers | 1.00 |
| run_cited | 1.00 |
| expected_tools | 1.00 |
| expected_errors | 1.00 |

## Cases

| Case | Grounded | Cites run | Tools | Errors as expected |
|---|---|---|---|---|
| forecast-next-week | yes | yes | get_forecast | yes |
| forecast-thanksgiving-week | yes | yes | get_forecast | yes |
| forecast-one-day-range | yes | yes | get_forecast | yes |
| forecast-named-model | yes | yes | get_forecast | yes |
| forecast-pressure | yes | yes | get_forecast | yes |
| forecast-outside-run | yes | yes | get_forecast | yes |
| compare-four-weeks | yes | yes | compare_to_last_year | yes |
| compare-lost-customer-lane | yes | yes | compare_to_last_year | yes |
| compare-thanksgiving-week | yes | yes | compare_to_last_year | yes |
| why-holiday-low | yes | yes | explain_drivers | yes |
| why-weekend-gap | yes | yes | explain_drivers | yes |
| why-ordinary-day | yes | yes | explain_drivers | yes |
| accuracy-overall | yes | yes | backtest_summary | yes |
| accuracy-one-lane | yes | yes | backtest_summary | yes |
| trust-the-ranges | yes | yes | backtest_summary | yes |
| out-of-scope | yes | not needed | none | yes |
