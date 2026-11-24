from pathlib import Path

from freight_agents.volume_forecast.evaluation import (
    FLOORS,
    QUESTIONS,
    load_cases,
    run_reference,
)
from freight_agents.volume_forecast.runs import RunStore
from freight_agents.volume_forecast.tools import ForecastDeps


def test_the_golden_set_passes_the_gate_in_reference_mode(default_store: RunStore) -> None:
    cases = load_cases(QUESTIONS)
    assert len(cases) >= 12
    result = run_reference(cases, ForecastDeps.load(default_store))
    broken = [
        (r["id"], r["invented"], r["called"])
        for r in result.per_case
        if not (r["grounded"] and r["tools_ok"] and r["errors_ok"])
        or (r["needs_cite"] and not r["cited"])
    ]
    assert result.failures(FLOORS) == [], broken


def test_the_grader_catches_an_invented_number_and_a_missing_run_id(
    default_store: RunStore, tmp_path: Path
) -> None:
    deps = ForecastDeps.load(default_store)
    case = next(c for c in load_cases(QUESTIONS) if c.expected["cites_run"])
    final = case.reference[-1]
    invented = case.model_copy(
        update={"reference": [*case.reference[:-1], {**final, "text": final["text"] + " 987654"}]}
    )
    uncited = case.model_copy(
        update={
            "reference": [
                *case.reference[:-1],
                {**final, "text": final["text"].replace(deps.run["run_id"], "the run")},
            ]
        }
    )
    result = run_reference([invented, uncited], deps)
    assert result.metrics["grounded_answers"] == 0.5
    assert result.metrics["run_cited"] == 0.5
