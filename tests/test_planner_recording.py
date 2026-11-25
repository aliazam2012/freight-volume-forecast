from pathlib import Path

import pytest
from freight_agents.evals.recording import StaleCassette
from freight_agents.llm import Model, ModelTurn, ScriptedModel, Usage
from freight_agents.volume_forecast.evaluation import (
    FLOORS,
    QUESTIONS,
    load_cases,
    record,
    reference_deps,
    run_recorded,
    write_report,
)

CASES = load_cases(QUESTIONS)


def live_reference(case_id: str) -> Model:
    (case,) = [c for c in CASES if c.id == case_id]
    usage = Usage(input_tokens=4_000, output_tokens=300)
    turns = [
        ModelTurn.model_validate(t).model_copy(update={"model": "claude-opus-5-5", "usage": usage})
        for t in case.reference
    ]
    return ScriptedModel(turns, name="claude-opus-5-5")


@pytest.fixture(scope="module")
def cassettes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cassettes")
    assert record(CASES, reference_deps(), out, live_reference, cap=20.0).code == 0
    return out


def test_every_question_is_recorded(cassettes: Path) -> None:
    assert {p.stem for p in cassettes.glob("*.jsonl")} == {c.id for c in CASES}


def test_recorded_mode_replays_through_the_same_grader_and_gate(
    cassettes: Path, tmp_path: Path
) -> None:
    result = run_recorded(CASES, reference_deps(), cassettes, "claude-opus-5-5")
    assert result.mode == "recorded"
    assert result.failures(FLOORS) == []
    path = write_report(result, reference_deps(), tmp_path, "2026-11-25")
    assert path.name == "2026-11-25-recorded.md"
    text = path.read_text()
    assert "claude-opus-5-5" in text
    assert "Reference mode replays" not in text


def test_a_missing_cassette_fails_naming_the_question(cassettes: Path, tmp_path: Path) -> None:
    victim, *rest = sorted(cassettes.glob("*.jsonl"))
    for path in rest:
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with pytest.raises(StaleCassette, match=victim.stem):
        run_recorded(CASES, reference_deps(), tmp_path, "claude-opus-5-5")


def test_the_eval_command_refuses_to_record_without_a_cap(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from freight_agents.volume_forecast.cli import main

    assert main(["eval", "--mode", "record", "--cassettes", str(tmp_path)]) == 2
    assert "--max-usd" in capsys.readouterr().out
