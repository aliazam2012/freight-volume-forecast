from pathlib import Path

import pytest
from freight_agents.volume_forecast.cli import main
from freight_agents.volume_forecast.data import generate
from freight_agents.volume_forecast.report import render
from freight_agents.volume_forecast.runs import RunStore, data_hash, run_id

FAST = ["--origins", "2", "--day", "2026-11-23"]


def test_run_id_depends_on_data_config_and_version() -> None:
    h = data_hash(generate(seed=7))
    base = run_id(h, {"origins": 8}, "0.1.0")
    assert base == run_id(h, {"origins": 8}, "0.1.0")
    assert base.startswith("fc-")
    assert base != run_id(h, {"origins": 7}, "0.1.0")
    assert base != run_id(h, {"origins": 8}, "0.1.1")
    assert base != run_id(data_hash(generate(seed=8)), {"origins": 8}, "0.1.0")


def test_run_store_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save({"run_id": "fc-abc", "x": 1})
    assert store.load("fc-abc") == {"run_id": "fc-abc", "x": 1}
    assert store.load("latest")["run_id"] == "fc-abc"
    with pytest.raises(FileNotFoundError):
        store.load("fc-missing")


def test_backtest_is_reproducible_and_reports_its_run(tmp_path: Path) -> None:
    args = ["backtest", "--runs", str(tmp_path / "runs"), "--reports", str(tmp_path / "rep"), *FAST]
    assert main(args) == 0
    first = (tmp_path / "rep" / "2026-11-23-backtest.md").read_text()
    assert main(args) == 0
    again = (tmp_path / "rep" / "2026-11-23-backtest.md").read_text()
    assert first == again
    run = RunStore(tmp_path / "runs").load("latest")
    assert run["run_id"] in first
    assert (tmp_path / "rep" / "latest.json").exists()
    assert {row["model"] for row in run["backtest"]["overall"]} == {
        "seasonal_naive",
        "holt_winters",
        "gradient_boosting",
    }


def test_forecast_prints_days_with_intervals_and_the_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = str(tmp_path / "runs")
    main(["backtest", "--runs", runs, "--reports", str(tmp_path / "rep"), *FAST])
    capsys.readouterr()
    assert main(["forecast", "--runs", runs, "--lane", "CHI-DAL", "--days", "5"]) == 0
    out = capsys.readouterr().out
    run = RunStore(tmp_path / "runs").load("latest")
    assert run["run_id"] in out
    assert out.count("2026-11-0") == 5
    assert "80%" in out
    assert "95%" in out
    assert main(["forecast", "--runs", runs, "--lane", "XXX-YYY", "--days", "5"]) == 2


def _run(gb_wape: float) -> dict[str, object]:
    overall = [
        {"model": "seasonal_naive", "wape": 0.18, "mase": 0.9, "bias": 0.0, "rows": 10},
        {"model": "gradient_boosting", "wape": gb_wape, "mase": 0.6, "bias": 0.0, "rows": 10},
    ]
    return {
        "run_id": "fc-test",
        "version": "0.1.0",
        "day": "2026-11-23",
        "data": {"seed": 7, "hash": "h", "start": "2024-01-01", "end": "2026-10-31", "lanes": 12},
        "config": {"origins": ["2026-10-03"], "horizon": 28, "levels": [0.8, 0.95]},
        "best_model": "gradient_boosting",
        "backtest": {"overall": overall, "by_lane": [], "holidays": [], "coverage": []},
    }


def test_report_says_so_when_a_model_does_not_beat_the_baseline() -> None:
    assert "does not beat" in render(_run(gb_wape=0.20))
    assert "beats seasonal_naive" in render(_run(gb_wape=0.11))


def test_report_says_when_the_backtest_misses_peak_season() -> None:
    assert "peak-season accuracy is untested" in render(_run(gb_wape=0.11))
    run = _run(gb_wape=0.11)
    run["config"] = {"origins": ["2025-11-01"], "horizon": 28, "levels": [0.8, 0.95]}
    assert "peak-season" not in render(run)


def test_eval_command_grades_the_golden_set_and_writes_a_report(tmp_path: Path) -> None:
    assert main(["eval", "--reports", str(tmp_path), "--day", "2026-11-23"]) == 0
    assert (tmp_path / "2026-11-23-reference.md").exists()
    assert (tmp_path / "latest.json").exists()
