import itertools
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from freight_agents.volume_forecast.backtest import (
    bias,
    mase,
    origins,
    run_backtest,
    seasonal_scale,
    summarize,
    wape,
)
from freight_agents.volume_forecast.data import generate
from freight_agents.volume_forecast.models import SeasonalNaive


def test_wape_and_bias_on_known_errors() -> None:
    actual = np.array([10.0, 20.0])
    assert wape(actual, np.array([12.0, 18.0])) == pytest.approx(4 / 30)
    assert bias(actual, np.array([12.0, 24.0])) == pytest.approx(6 / 30)
    assert bias(actual, np.array([8.0, 16.0])) == pytest.approx(-6 / 30)


def test_mase_scales_by_in_sample_seasonal_naive_error() -> None:
    # A weekly pattern that rises by 2 each week: the lag-7 error is always 2.
    days = pd.date_range("2026-01-01", periods=28, freq="D")
    history = pd.DataFrame(
        {"date": days, "lane": "A", "volume": [10 + (i % 7) + 2 * (i // 7) for i in range(28)]}
    )
    scale = seasonal_scale(history)
    assert scale["A"] == pytest.approx(2.0)
    assert mase(np.array([5.0, 7.0]), np.array([6.0, 6.0]), np.array([2.0, 2.0])) == 0.5


def test_origins_are_28_days_apart_and_end_a_horizon_before_the_last_date() -> None:
    data = generate(seed=7)
    got = origins(data, count=8, step=28, horizon=28)
    assert len(got) == 8
    assert all((b - a).days == 28 for a, b in itertools.pairwise(got))
    assert got[-1] + pd.Timedelta(days=28) == data["date"].max()


class Spy(SeasonalNaive):
    seen: ClassVar[list[pd.Timestamp]] = []

    def fit(self, history: pd.DataFrame) -> None:
        Spy.seen.append(history["date"].max())
        super().fit(history)


def test_backtest_fits_each_model_only_on_data_up_to_its_origin() -> None:
    data = generate(seed=7)
    cut = origins(data, count=2, step=28, horizon=28)
    Spy.seen = []
    frame = run_backtest(data, [Spy], cut, horizon=28)
    assert Spy.seen == cut
    assert len(frame) == 2 * 28 * 12
    assert set(frame.columns) >= {"origin", "date", "lane", "model", "actual", "forecast", "scale"}
    assert (frame["date"] > frame["origin"]).all()
    assert (frame["date"] <= frame["origin"] + pd.Timedelta(days=28)).all()


def test_summary_reports_each_model_overall_and_by_lane() -> None:
    frame = pd.DataFrame(
        {
            "model": ["m", "m", "m", "m"],
            "lane": ["A", "A", "B", "B"],
            "actual": [10.0, 10.0, 20.0, 20.0],
            "forecast": [11.0, 9.0, 22.0, 22.0],
            "scale": [1.0, 1.0, 2.0, 2.0],
        }
    )
    overall, by_lane = summarize(frame)
    row = overall.set_index("model").loc["m"]
    assert row["wape"] == pytest.approx(6 / 60)
    assert row["bias"] == pytest.approx(4 / 60)
    assert row["mase"] == pytest.approx(1.0)
    assert by_lane.set_index(["model", "lane"]).loc[("m", "B"), "bias"] == pytest.approx(0.1)
