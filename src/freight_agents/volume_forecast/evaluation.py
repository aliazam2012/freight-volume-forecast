"""The planner agent's golden set: questions, reference answers, and the grader.

Modes: ``reference`` replays each case's scripted turns. The tools still run for
real against the committed reference run, so a tool change that moves a figure
fails the gate. A live model run goes through ``freight-forecast ask`` and, once
recorded, through the same grader.

Per case the grader checks four things: every number in the answer appears in a
tool result or the question; the answer cites the run id when it uses a figure;
the tools called are the ones the question needs; tool errors are only the ones
the case expects.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freight_agents.agentcore.agent import Agent, Budget
from freight_agents.agentcore.guards import cap_size
from freight_agents.agentcore.hooks import HookContext
from freight_agents.agentcore.tools import ToolRegistry
from freight_agents.evals.recording import RecordResult, record_cases, replay_model
from freight_agents.llm import Model, ModelTurn, ScriptedModel, TextPart
from freight_agents.volume_forecast.grounding import ungrounded
from freight_agents.volume_forecast.prompts import PLANNER
from freight_agents.volume_forecast.runs import RunStore
from freight_agents.volume_forecast.tools import ForecastDeps, build_tools
from pydantic import BaseModel


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(p for p in here.parents if (p / "uv.lock").exists())


EVALS = repo_root() / "apps" / "volume-forecast" / "evals"
QUESTIONS = EVALS / "questions.jsonl"
# The run the reference answers were rendered from. Model arithmetic can differ in the last
# digit between machines, so the golden set grades this committed run, never a rebuilt one.
REFERENCE_RUN = EVALS / "run"
ASKED_AT = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)  # the morning after the data ends

Floor = tuple[str, float]
FLOORS: dict[str, Floor] = {
    "grounded_answers": (">=", 1.0),
    "run_cited": (">=", 1.0),
    "expected_tools": (">=", 1.0),
    "expected_errors": (">=", 1.0),
}


class PlannerCase(BaseModel):
    id: str
    question: str
    tags: list[str]
    expected: dict[str, Any]
    reference: list[dict[str, Any]]


def reference_deps() -> ForecastDeps:
    return ForecastDeps.load(RunStore(REFERENCE_RUN))


def load_cases(path: Path) -> list[PlannerCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PlannerCase.model_validate_json(line) for line in lines if line.strip()]


def planner(model: Model, deps: ForecastDeps) -> Agent:
    registry = ToolRegistry()
    for t in build_tools(deps).values():
        registry.register(t)
    return Agent(
        name="forecast-planner",
        model=model,
        registry=registry,
        system=PLANNER,
        post=[cap_size()],
        budget=Budget(max_turns=4, max_tool_calls=6),
    )


@dataclass
class EvalResult:
    mode: str
    model: str
    metrics: dict[str, float]
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def failures(self, floors: Mapping[str, Floor]) -> list[str]:
        out = []
        for name, (op, bound) in floors.items():
            value = self.metrics[name]
            if not (value >= bound if op == ">=" else value <= bound):
                out.append(f"{name} {value:.2f} is not {op} {bound:.2f}")
        return out


def grade(case: PlannerCase, deps: ForecastDeps, model: Model) -> dict[str, Any]:
    ctx = HookContext(run_id=f"ask-{case.id}", agent="forecast-planner", now=ASKED_AT)
    result = planner(model, deps).run([TextPart(text=case.question)], ctx)
    sources = [o.as_result().content for o in result.outcomes] + [case.question]
    invented = sorted(ungrounded(result.text, sources))
    called = sorted({o.tool for o in result.outcomes})
    errors = sum(o.status in ("error", "denied") for o in result.outcomes)
    return {
        "id": case.id,
        "grounded": not invented,
        "invented": invented,
        "cited": deps.run_id in result.text,
        "needs_cite": bool(case.expected["cites_run"]),
        "tools_ok": called == sorted(case.expected["tools"]),
        "called": called,
        "errors_ok": errors == case.expected.get("errors", 0),
        "answer": result.text,
    }


def run_suite(
    cases: list[PlannerCase],
    deps: ForecastDeps,
    model_for: Callable[[PlannerCase], Model],
    mode: str,
    model_name: str,
) -> EvalResult:
    rows = [grade(case, deps, model_for(case)) for case in cases]
    cited = [r["cited"] for r in rows if r["needs_cite"]]

    def share(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 1.0

    metrics = {
        "grounded_answers": share([r["grounded"] for r in rows]),
        "run_cited": share(cited),
        "expected_tools": share([r["tools_ok"] for r in rows]),
        "expected_errors": share([r["errors_ok"] for r in rows]),
    }
    return EvalResult(mode, model_name, metrics, rows)


def run_reference(cases: list[PlannerCase], deps: ForecastDeps) -> EvalResult:
    def scripted(case: PlannerCase) -> Model:
        turns = [ModelTurn.model_validate(t) for t in case.reference]
        return ScriptedModel(turns, name="reference")

    return run_suite(cases, deps, scripted, "reference", "reference")


def run_recorded(
    cases: list[PlannerCase], deps: ForecastDeps, cassettes: Path, model_name: str
) -> EvalResult:
    """The same questions, tools and grader, answered by a real model's recorded turns."""
    return run_suite(
        cases,
        deps,
        lambda case: replay_model(case.id, cassettes, model_name),
        "recorded",
        model_name,
    )


def record(
    cases: list[PlannerCase],
    deps: ForecastDeps,
    cassettes: Path,
    live: Callable[[str], Model],
    cap: float,
    limit: int | None = None,
    out: Callable[[str], object] = print,
) -> RecordResult:
    """Record a live model answering every question, the tools running for real, under the cap."""
    by_id = {case.id: case for case in cases}
    ids = list(by_id)
    return record_cases(
        ids[:limit] if limit else ids,
        lambda case_id, model: grade(by_id[case_id], deps, model),
        live,
        cassettes,
        cap,
        total=len(ids),
        out=out,
    )


def write_report(result: EvalResult, deps: ForecastDeps, directory: Path, day: str) -> Path:
    failures = result.failures(FLOORS)
    lines = [
        f"# Planner Agent Evaluation, {day}",
        "",
        "## Table of Contents",
        "",
        "- [Summary](#summary)",
        "- [Metrics](#metrics)",
        "- [Cases](#cases)",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"Mode: {result.mode}. Forecast run `{deps.run_id}`. Cases: {len(result.per_case)}. "
        f"Gate: {'passed' if not failures else 'failed: ' + '; '.join(failures)}.",
        "",
        (
            "Reference mode replays scripted answers while the tools run for real: it checks "
            "the tools, the grader and the gate, not a model."
            if result.mode == "reference"
            else f"Answers replay the recorded turns of {result.model}; the tools run for real "
            "and every figure is checked against their results."
        ),
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v:.2f} |" for k, v in result.metrics.items()],
        "",
        "## Cases",
        "",
        "| Case | Grounded | Cites run | Tools | Errors as expected |",
        "|---|---|---|---|---|",
    ]
    for r in result.per_case:
        cites = ("yes" if r["cited"] else "no") if r["needs_cite"] else "not needed"
        lines.append(
            f"| {r['id']} | {'yes' if r['grounded'] else 'no: ' + ', '.join(r['invented'])} "
            f"| {cites} | {', '.join(r['called']) or 'none'} | "
            f"{'yes' if r['errors_ok'] else 'no'} |"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{day}-{result.mode}.md"
    path.write_text("\n".join(lines) + "\n")
    latest = {"date": day, "mode": result.mode, "run_id": deps.run_id, "metrics": result.metrics}
    (directory / "latest.json").write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
    return path
