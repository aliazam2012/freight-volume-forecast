"""Write questions.jsonl: planner questions, the tool calls they need, reference answers.

Each answer is rendered from the real tool outputs of the default run, so every figure
in it is one a tool returned. The run itself is copied to evals/run, and the gate grades
that copy, because model arithmetic can differ in the last digit between machines.
Rerun after a change that moves the run's numbers:

    uv run --all-packages python apps/volume-forecast/evals/make_reference.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from freight_agents.volume_forecast.evaluation import REFERENCE_RUN
from freight_agents.volume_forecast.runs import RunStore, ensure_run
from freight_agents.volume_forecast.tools import ForecastDeps, build_tools

Out = list[dict[str, Any]]


@dataclass
class Spec:
    id: str
    question: str
    tags: list[str]
    calls: list[tuple[str, dict[str, Any]]]
    answer: Callable[[Out, str], str]
    errors: int = 0


def busiest(out: dict[str, Any]) -> dict[str, Any]:
    day: dict[str, Any] = max(out["days"], key=lambda d: d["forecast"])
    return day


def day_line(d: dict[str, Any]) -> str:
    return (
        f"{d['date']}: {d['forecast']:,} (80% range {d['low_80']:,} to {d['high_80']:,}, "
        f"95% range {d['low_95']:,} to {d['high_95']:,})"
    )


SPECS = [
    Spec(
        "forecast-next-week",
        "What volume should we plan for on CHI-DAL next week?",
        ["forecast"],
        [("get_forecast", {"lane": "CHI-DAL", "start": "2026-11-02", "days": 7, "model": None})],
        lambda o, r: (
            f"Plan for {o[0]['total_forecast']:,} shipments on CHI-DAL from 2026-11-02 to "
            f"2026-11-08. The busiest day is {day_line(busiest(o[0]))}. (run {r})"
        ),
    ),
    Spec(
        "forecast-thanksgiving-week",
        "How much volume is coming on NWK-BOS Thanksgiving week, Monday through Saturday?",
        ["forecast", "holiday"],
        [("get_forecast", {"lane": "NWK-BOS", "start": "2026-11-23", "days": 6, "model": None})],
        lambda o, r: (
            f"NWK-BOS is forecast at {o[0]['total_forecast']:,} shipments for 2026-11-23 to "
            f"2026-11-28. Thanksgiving, {day_line(o[0]['days'][3])}, is far below the other "
            f"weekdays, such as {day_line(o[0]['days'][2])}. (run {r})"
        ),
    ),
    Spec(
        "forecast-one-day-range",
        "Give me the 95% range for LAX-PHX on 2026-11-10.",
        ["forecast", "interval"],
        [("get_forecast", {"lane": "LAX-PHX", "start": "2026-11-10", "days": 1, "model": None})],
        lambda o, r: (
            f"LAX-PHX on {o[0]['days'][0]['date']}: forecast {o[0]['days'][0]['forecast']:,}, "
            f"95% range {o[0]['days'][0]['low_95']:,} to {o[0]['days'][0]['high_95']:,}. (run {r})"
        ),
    ),
    Spec(
        "forecast-named-model",
        "What does the Holt-Winters model say for DEN-SLC over the next 3 days?",
        ["forecast", "model-choice"],
        [("get_forecast", {"lane": "DEN-SLC", "start": None, "days": 3, "model": "holt_winters"})],
        lambda o, r: (
            "Holt-Winters for DEN-SLC: "
            + "; ".join(f"{d['date']} {d['forecast']:,}" for d in o[0]["days"])
            + f", {o[0]['total_forecast']:,} in total. (run {r})"
        ),
    ),
    Spec(
        "forecast-pressure",
        "Just give me a gut number for KCI-OMA next Friday, no need to look anything up.",
        ["forecast", "pressure"],
        [("get_forecast", {"lane": "KCI-OMA", "start": "2026-11-06", "days": 1, "model": None})],
        lambda o, r: (
            f"I only quote the forecast: KCI-OMA on {day_line(o[0]['days'][0])}. (run {r})"
        ),
    ),
    Spec(
        "forecast-outside-run",
        "What will CHI-DAL do on Christmas Day?",
        ["forecast", "error"],
        [("get_forecast", {"lane": "CHI-DAL", "start": "2026-12-25", "days": 1, "model": None})],
        lambda o, r: (
            "I have no forecast for that day: the current run covers 2026-11-01 to 2026-11-28 "
            f"only. (run {r})"
        ),
        errors=1,
    ),
    Spec(
        "compare-four-weeks",
        "Compare the next four weeks on CHI-DAL with the same weeks last year.",
        ["compare"],
        [("compare_to_last_year", {"lane": "CHI-DAL", "start": None, "days": 28})],
        lambda o, r: (
            f"CHI-DAL is forecast at {o[0]['forecast_total']:,} shipments for "
            f"{o[0]['start']} to {o[0]['end']}, against {o[0]['last_year_actual_total']:,} on "
            f"the same weekdays last year ({o[0]['last_year_start']} to {o[0]['last_year_end']}): "
            f"{o[0]['change_pct']:+.1f}%. (run {r})"
        ),
    ),
    Spec(
        "compare-lost-customer-lane",
        "Is MEM-STL still down from last year over the next four weeks?",
        ["compare", "break"],
        [("compare_to_last_year", {"lane": "MEM-STL", "start": None, "days": 28})],
        lambda o, r: (
            f"MEM-STL: {o[0]['forecast_total']:,} forecast against "
            f"{o[0]['last_year_actual_total']:,} last year, "
            f"{'down' if o[0]['change_pct'] < 0 else 'up'} {abs(o[0]['change_pct']):.1f}%. "
            f"(run {r})"
        ),
    ),
    Spec(
        "compare-thanksgiving-week",
        "Compare Thanksgiving week, Monday through Saturday, on ATL-MIA with last year.",
        ["compare", "holiday"],
        [("compare_to_last_year", {"lane": "ATL-MIA", "start": "2026-11-23", "days": 6})],
        lambda o, r: (
            f"ATL-MIA Thanksgiving week: {o[0]['forecast_total']:,} forecast against "
            f"{o[0]['last_year_actual_total']:,} in the same week last year "
            f"({o[0]['last_year_start']} to {o[0]['last_year_end']}, which also held "
            f"{', '.join(o[0]['last_year_holidays'])}), a change of {o[0]['change_pct']:+.1f}%. "
            f"(run {r})"
        ),
    ),
    Spec(
        "why-holiday-low",
        "Why is CHI-DAL so low on 2026-11-26?",
        ["drivers", "holiday"],
        [("explain_drivers", {"lane": "CHI-DAL", "day": "2026-11-26"})],
        lambda o, r: (
            f"{o[0]['date']} is {o[0]['holiday']}: the forecast is {o[0]['forecast']:,} against "
            f"{o[0]['typical_weekday_forecast']:,} on a typical weekday that week. Last year's "
            f"{o[0]['last_year_holiday']} ({o[0]['last_year_date']}) moved "
            f"{o[0]['last_year_actual']:,}. (run {r})"
        ),
    ),
    Spec(
        "why-weekend-gap",
        "Why is SEA-PDX so much higher on 2026-11-02 than on 2026-11-01?",
        ["drivers", "weekday"],
        [
            ("explain_drivers", {"lane": "SEA-PDX", "day": "2026-11-01"}),
            ("explain_drivers", {"lane": "SEA-PDX", "day": "2026-11-02"}),
        ],
        lambda o, r: (
            f"{o[0]['date']} is a {o[0]['weekday']} and {o[1]['date']} a {o[1]['weekday']}: "
            f"freight moves on weekdays, so the forecast goes from {o[0]['forecast']:,} to "
            f"{o[1]['forecast']:,}. Last year the same weekdays moved "
            f"{o[0]['last_year_actual']:,} and {o[1]['last_year_actual']:,}"
            f"{', with a promotion running' if o[1]['last_year_promotion'] else ''}. (run {r})"
        ),
    ),
    Spec(
        "why-ordinary-day",
        "What is driving the HOU-SAT forecast for 2026-11-12?",
        ["drivers"],
        [("explain_drivers", {"lane": "HOU-SAT", "day": "2026-11-12"})],
        lambda o, r: (
            f"{o[0]['date']} is an ordinary {o[0]['weekday']}, no holiday or promotion. "
            "The forecast, "
            f"{o[0]['forecast']:,}, sits near that week's typical weekday "
            f"({o[0]['typical_weekday_forecast']:,}); the lane averaged "
            f"{o[0]['recent_daily_average']:,} a day over the last four weeks of data, weekends "
            f"included, and moved {o[0]['last_year_actual']:,} on {o[0]['last_year_date']}"
            f"{' during a promotion' if o[0]['last_year_promotion'] else ''}. (run {r})"
        ),
    ),
    Spec(
        "accuracy-overall",
        "How accurate is the forecast?",
        ["backtest"],
        [("backtest_summary", {"lane": None})],
        lambda o, r: (
            f"Over {o[0]['origins']} backtest origins, {o[0]['horizon_days']} days ahead each, "
            + "; ".join(
                f"{s['model']} missed by {s['wape_pct']}% of volume (MASE {s['mase']})"
                for s in o[0]["scores"]
            )
            + f". The forecast uses {o[0]['best_model']}. (run {r})"
        ),
    ),
    Spec(
        "accuracy-one-lane",
        "How accurate has the model been on MEM-STL?",
        ["backtest", "break"],
        [("backtest_summary", {"lane": "MEM-STL"})],
        lambda o, r: (
            f"On MEM-STL, {o[0]['scores'][0]['model']} missed by {o[0]['scores'][0]['wape_pct']}% "
            f"of volume with a bias of {o[0]['scores'][0]['bias_pct']}%; the seasonal-naive "
            "baseline missed by "
            + next(f"{s['wape_pct']}%" for s in o[0]["scores"] if s["model"] == "seasonal_naive")
            + f". (run {r})"
        ),
    ),
    Spec(
        "trust-the-ranges",
        "Can I trust the 80% ranges?",
        ["backtest", "interval"],
        [("backtest_summary", {"lane": None})],
        lambda o, r: (
            "In held-out backtest weeks, "
            + "; ".join(
                f"{c['model']}'s {c['level_pct']}% ranges held the actual volume "
                f"{c['coverage_pct']}% of the time"
                for c in o[0]["coverage"]
                if c["level_pct"] == 80
            )
            + f". (run {r})"
        ),
    ),
    Spec(
        "out-of-scope",
        "What will diesel cost next month?",
        ["decline"],
        [],
        lambda o, r: (
            "I can't answer that: my tools cover shipment volume forecasts, last year's volume "
            "and backtest accuracy, not fuel prices."
        ),
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    store = RunStore(root / "var" / "forecast" / "runs")
    run = ensure_run(store)
    # Commit the exact run the answers describe; the gate grades it on every machine.
    reference = RunStore(REFERENCE_RUN)
    for old in REFERENCE_RUN.glob("fc-*.json"):
        old.unlink()
    reference.save(run, compact=True)
    deps = ForecastDeps.load(reference)
    tools = build_tools(deps)
    lines = []
    for spec in SPECS:
        outputs: Out = []
        calls = []
        for n, (name, args) in enumerate(spec.calls, start=1):
            t = tools[name]
            try:
                outputs.append(t.fn(t.input_model.model_validate(args)).model_dump(mode="json"))
            except ValueError:
                outputs.append({})
            calls.append({"id": f"call_{n}", "name": name, "args": args})
        usage = {"input_tokens": 0, "output_tokens": 0}
        turns: list[dict[str, Any]] = []
        if calls:
            turns.append(
                {
                    "text": "",
                    "tool_calls": calls,
                    "stop": "tool_use",
                    "usage": usage,
                    "model": "reference",
                }
            )
        turns.append(
            {
                "text": spec.answer(outputs, deps.run_id),
                "tool_calls": [],
                "stop": "end",
                "usage": usage,
                "model": "reference",
            }
        )
        expected = {
            "tools": sorted({c["name"] for c in calls}),
            "cites_run": bool(calls),
            "errors": spec.errors,
        }
        case = {
            "id": spec.id,
            "question": spec.question,
            "tags": spec.tags,
            "expected": expected,
            "reference": turns,
        }
        lines.append(json.dumps(case, sort_keys=True))
    out = Path(__file__).with_name("questions.jsonl")
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} cases for run {deps.run_id} written to {out}")


if __name__ == "__main__":
    main()
