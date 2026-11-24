from datetime import date
from typing import Any

import pandas as pd
import pytest
from freight_agents.agentcore.tools import Tool
from freight_agents.volume_forecast.data import holiday_names
from freight_agents.volume_forecast.runs import RunStore, planned_run_id
from freight_agents.volume_forecast.tools import ForecastDeps, build_tools


@pytest.fixture(scope="module")
def tools(default_store: RunStore) -> dict[str, Tool]:
    return build_tools(ForecastDeps.load(default_store))


def call(tools: dict[str, Tool], name: str, **args: Any) -> dict[str, Any]:
    t = tools[name]
    out: dict[str, Any] = t.fn(t.input_model.model_validate(args)).model_dump(mode="json")
    return out


def test_holiday_names_include_thanksgiving() -> None:
    names = holiday_names(2025, 2026)
    assert names[pd.Timestamp("2026-11-26")] == "Thanksgiving"
    assert names[pd.Timestamp("2025-11-27")] == "Thanksgiving"


def test_the_default_run_id_is_known_before_building(default_run: dict[str, Any]) -> None:
    assert planned_run_id() == default_run["run_id"]


def test_every_tool_names_the_run(tools: dict[str, Tool], default_run: dict[str, Any]) -> None:
    outputs = [
        call(tools, "get_forecast", lane="CHI-DAL", start=None, days=3, model=None),
        call(tools, "compare_to_last_year", lane="CHI-DAL", start=None, days=7),
        call(tools, "explain_drivers", lane="CHI-DAL", day="2026-11-02"),
        call(tools, "backtest_summary", lane=None),
    ]
    assert all(o["run_id"] == default_run["run_id"] for o in outputs)


def test_get_forecast_returns_days_inside_their_intervals(tools: dict[str, Tool]) -> None:
    out = call(tools, "get_forecast", lane="ATL-MIA", start="2026-11-03", days=5, model=None)
    assert out["days"][0]["date"] == "2026-11-03"
    assert len(out["days"]) == 5
    for d in out["days"]:
        assert d["low_95"] <= d["low_80"] <= d["forecast"] <= d["high_80"] <= d["high_95"]
    assert out["total_forecast"] == pytest.approx(sum(d["forecast"] for d in out["days"]), abs=3)


def test_get_forecast_refuses_days_outside_the_run(tools: dict[str, Tool]) -> None:
    t = tools["get_forecast"]
    with pytest.raises(ValueError, match="2026-11-01 to 2026-11-28"):
        t.fn(
            t.input_model.model_validate(
                {"lane": "ATL-MIA", "start": "2026-11-20", "days": 14, "model": None}
            )
        )
    with pytest.raises(ValueError, match="lanes"):
        t.fn(
            t.input_model.model_validate({"lane": "NOPE", "start": None, "days": 3, "model": None})
        )


def test_compare_to_last_year_lines_up_weekdays(tools: dict[str, Tool]) -> None:
    out = call(tools, "compare_to_last_year", lane="CHI-DAL", start="2026-11-02", days=7)
    assert out["last_year_start"] == "2025-11-03"
    assert date.fromisoformat(out["last_year_start"]).weekday() == date(2026, 11, 2).weekday()
    expected = (out["forecast_total"] / out["last_year_actual_total"] - 1) * 100
    assert out["change_pct"] == pytest.approx(expected, abs=0.05)


def test_explain_drivers_names_the_holiday(tools: dict[str, Tool]) -> None:
    out = call(tools, "explain_drivers", lane="CHI-DAL", day="2026-11-26")
    assert out["holiday"] == "Thanksgiving"
    assert out["weekday"] == "Thursday"
    assert out["last_year_date"] == "2025-11-27"
    assert out["last_year_holiday"] == "Thanksgiving"
    assert out["forecast"] < 0.5 * out["typical_weekday_forecast"]


def test_backtest_summary_overall_and_by_lane(
    tools: dict[str, Tool], default_run: dict[str, Any]
) -> None:
    out = call(tools, "backtest_summary", lane=None)
    assert out["best_model"] == default_run["best_model"]
    assert {s["model"] for s in out["scores"]} == {
        "seasonal_naive",
        "holt_winters",
        "gradient_boosting",
    }
    assert {c["level_pct"] for c in out["coverage"]} == {80, 95}
    lane = call(tools, "backtest_summary", lane="MEM-STL")
    assert lane["lane"] == "MEM-STL"
    assert lane["scores"] != out["scores"]


def test_explain_drivers_flags_last_years_promotion(
    tools: dict[str, Tool], default_store: RunStore
) -> None:
    data = ForecastDeps.load(default_store).data
    promo = data[
        data["promo"]
        & (data["date"] >= "2025-11-02")
        & (data["date"] <= "2025-11-29")
        & ~data["holiday"]
    ].iloc[0]
    day = (promo["date"] + pd.Timedelta(days=364)).date().isoformat()
    out = call(tools, "explain_drivers", lane=promo["lane"], day=day)
    assert out["last_year_promotion"] is True
    assert out["promotion"] is False  # no promotion is announced past the data


def test_tool_errors_name_the_run(tools: dict[str, Tool], default_run: dict[str, Any]) -> None:
    t = tools["get_forecast"]
    with pytest.raises(ValueError, match=default_run["run_id"]):
        t.fn(
            t.input_model.model_validate(
                {"lane": "CHI-DAL", "start": "2026-12-25", "days": 1, "model": None}
            )
        )
