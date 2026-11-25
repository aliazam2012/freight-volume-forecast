"""freight-forecast: run the backtest and write its report, or print a stored forecast."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from freight_agents.volume_forecast.report import write
from freight_agents.volume_forecast.runs import FORECAST_DAYS, RunStore, build_run, ensure_run

DEFAULT_RUNS = Path("var/forecast/runs")
DEFAULT_REPORTS = Path("apps/volume-forecast/reports")
DEFAULT_EVAL_REPORTS = Path("apps/volume-forecast/evals/reports")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freight-forecast")
    commands = parser.add_subparsers(dest="command", required=True)

    backtest = commands.add_parser("backtest", help="backtest every model and write the report")
    backtest.add_argument("--seed", type=int, default=7)
    backtest.add_argument("--origins", type=int, default=8)
    backtest.add_argument("--day", default=date.today().isoformat(), help="report date")
    backtest.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)

    forecast = commands.add_parser("forecast", help="print a lane's forecast from a stored run")
    forecast.add_argument("--lane", required=True)
    forecast.add_argument("--days", type=int, default=14)
    forecast.add_argument("--model", help="defaults to the run's best model")
    forecast.add_argument("--run", default="latest", help="run id, or latest")

    evaluate = commands.add_parser("eval", help="grade the planner agent's golden set")
    evaluate.add_argument("--day", default=date.today().isoformat(), help="report date")
    evaluate.add_argument("--reports", type=Path, default=DEFAULT_EVAL_REPORTS)
    evaluate.add_argument(
        "--mode", choices=["reference", "recorded", "record"], default="reference"
    )
    evaluate.add_argument(
        "--cassettes", type=Path, default=Path("apps/volume-forecast/evals/cassettes")
    )
    evaluate.add_argument("--model-name", default="claude-opus-5-5")
    evaluate.add_argument("--max-usd", type=float, default=0.0, help="record: the approved cap")
    evaluate.add_argument("--limit", type=int, help="record: only the first N questions")

    ask = commands.add_parser("ask", help="ask the planner agent a question (live model)")
    ask.add_argument("question")
    ask.add_argument("--model", choices=["anthropic", "openai"], default="anthropic")
    ask.add_argument("--model-name")

    for sub in (backtest, forecast, ask):
        sub.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="run store directory")

    args = parser.parse_args(argv)
    if args.command == "eval":
        # The golden set grades the committed reference run, not the local run store.
        return _evaluate(args)
    store = RunStore(args.runs)

    if args.command == "backtest":
        if args.origins < 2:
            parser.error("--origins needs at least 2: one half calibrates, the other is checked")
        run = build_run(seed=args.seed, count=args.origins, day=args.day)
        store.save(run)
        path = write(run, args.reports)
        print(f"run {run['run_id']}  best model {run['best_model']}")
        for row in sorted(run["backtest"]["overall"], key=lambda r: r["wape"]):
            print(f"  {row['model']:<18} WAPE {row['wape']:.1%}  MASE {row['mase']:.2f}")
        print(f"report: {path}")
        return 0

    if args.command == "ask":
        return _ask(store, args.question, args.model, args.model_name)

    try:
        run = store.load(args.run)
    except (FileNotFoundError, ValueError):
        print(f"no run {args.run!r} in {args.runs}; run `freight-forecast backtest` first")
        return 2
    if not 1 <= args.days <= FORECAST_DAYS:
        print(f"--days must be between 1 and {FORECAST_DAYS}", file=sys.stderr)
        return 2
    model = args.model or run["best_model"]
    rows = [r for r in run["forecast"] if r["lane"] == args.lane and r["model"] == model]
    if not rows:
        lanes = sorted({r["lane"] for r in run["forecast"]})
        print(f"no forecast for lane {args.lane!r}; lanes: {', '.join(lanes)}", file=sys.stderr)
        return 2
    print(f"{args.lane}, {model}, run {run['run_id']} (data through {run['data']['end']})")
    print(f"{'date':<11} {'forecast':>8}  {'80% range':>13}  {'95% range':>13}")
    for r in sorted(rows, key=lambda r: r["date"])[: args.days]:
        note = "  holiday" if r["holiday"] else ""
        print(
            f"{r['date']:<11} {r['forecast']:>8.0f}  {r['lo_80']:>6.0f}-{r['hi_80']:<6.0f}"
            f"  {r['lo_95']:>6.0f}-{r['hi_95']:<6.0f}{note}"
        )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from freight_agents.evals.recording import StaleCassette
    from freight_agents.llm import Model
    from freight_agents.volume_forecast.evaluation import (
        FLOORS,
        QUESTIONS,
        load_cases,
        record,
        reference_deps,
        run_recorded,
        run_reference,
        write_report,
    )

    deps, cases = reference_deps(), load_cases(QUESTIONS)
    if args.mode == "record":
        from freight_agents.llm.anthropic_adapter import AnthropicModel

        def live(case_id: str) -> Model:
            return AnthropicModel(model=args.model_name, effort="low")

        return record(cases, deps, args.cassettes, live, args.max_usd, args.limit).code
    try:
        result = (
            run_reference(cases, deps)
            if args.mode == "reference"
            else run_recorded(cases, deps, args.cassettes, args.model_name)
        )
    except StaleCassette as exc:
        print(f"FAIL {exc}")
        return 2
    path = write_report(result, deps, args.reports, args.day)
    for name, value in result.metrics.items():
        print(f"{name}: {value:.2f}")
    failures = result.failures(FLOORS)
    print(f"gate: {'passed' if not failures else 'failed: ' + '; '.join(failures)}")
    print(f"report: {path}")
    return 1 if failures else 0


def _ask(store: RunStore, question: str, kind: str, name: str | None) -> int:
    from datetime import UTC, datetime

    from freight_agents.agentcore.hooks import HookContext
    from freight_agents.llm import Model, TextPart
    from freight_agents.volume_forecast.evaluation import planner
    from freight_agents.volume_forecast.tools import ForecastDeps

    model: Model
    if kind == "anthropic":
        from freight_agents.llm.anthropic_adapter import AnthropicModel

        model = AnthropicModel(model=name) if name else AnthropicModel()
    else:
        if not name:
            print("--model openai needs --model-name", file=sys.stderr)
            return 2
        from freight_agents.llm.openai_adapter import OpenAIModel

        model = OpenAIModel(model=name)
    run = ensure_run(store)
    deps = ForecastDeps.load(store, run["run_id"])
    ctx = HookContext(run_id="ask", agent="forecast-planner", now=datetime.now(UTC))
    result = planner(model, deps).run([TextPart(text=question)], ctx)
    print(result.text)
    for outcome in result.outcomes:
        print(f"  tool {outcome.tool}: {outcome.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
