import numpy as np
import pandas as pd
import pytest
from freight_agents.volume_forecast.intervals import (
    conformal_table,
    coverage,
    horizon_week,
    split_conformal,
    with_intervals,
)


def _errors(ratios: list[float], model: str = "m") -> pd.DataFrame:
    origin = pd.Timestamp("2026-01-01")
    forecast = np.full(len(ratios), 99.0)
    return pd.DataFrame(
        {
            "model": model,
            "origin": origin,
            "date": origin + pd.Timedelta(days=1),
            "actual": np.array(ratios) * 100 - 1,
            "forecast": forecast,
        }
    )


def test_horizon_week_buckets_days_ahead() -> None:
    origin = pd.Series([pd.Timestamp("2026-01-01")] * 4)
    dates = origin + pd.to_timedelta([1, 7, 8, 40], unit="D")
    assert list(horizon_week(origin, dates, weeks=4)) == [1, 1, 2, 4]


def test_conformal_bounds_are_finite_sample_quantiles_of_log_ratios() -> None:
    ratios = [0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.5]
    table = conformal_table(_errors(ratios), levels=(0.8,))
    row = table.set_index(["model", "week", "level"]).loc[("m", 1, 0.8)]
    # n = 9, tail 0.1 each side: rank ceil((n + 1) * 0.9) = 9 from the top and the bottom.
    assert row["hi"] == pytest.approx(np.log(1.5))
    assert row["lo"] == pytest.approx(np.log(0.5))


def test_intervals_wrap_the_forecast_on_the_log_scale() -> None:
    table = pd.DataFrame(
        {"model": ["m"], "week": [1], "level": [0.8], "lo": [np.log(0.9)], "hi": [np.log(1.1)]}
    )
    forecasts = pd.DataFrame(
        {
            "model": ["m"],
            "origin": [pd.Timestamp("2026-01-01")],
            "date": [pd.Timestamp("2026-01-02")],
            "forecast": [99.0],
        }
    )
    out = with_intervals(forecasts, table)
    assert out.loc[0, "lo_80"] == pytest.approx(89.0)
    assert out.loc[0, "hi_80"] == pytest.approx(109.0)


def test_coverage_counts_actuals_inside_their_interval() -> None:
    frame = pd.DataFrame(
        {
            "model": "m",
            "actual": [10.0, 12.0, 10.0, 9.0],
            "lo_80": [9.0, 9.0, 9.0, 9.0],
            "hi_80": [11.0, 11.0, 11.0, 11.0],
        }
    )
    got = coverage(frame, levels=(0.8,)).set_index(["model", "level"])
    assert got.loc[("m", 0.8), "coverage"] == pytest.approx(0.75)


def test_split_conformal_calibrates_on_early_origins_and_covers_later_ones() -> None:
    rng = np.random.default_rng(3)
    starts = pd.date_range("2026-01-01", periods=4, freq="28D")
    rows = []
    for origin in starts:
        forecast = rng.uniform(100, 1000, 600)
        actual = np.expm1(np.log1p(forecast) + rng.normal(0, 0.1, 600))
        rows.append(
            pd.DataFrame(
                {
                    "model": "m",
                    "origin": origin,
                    "date": origin + pd.to_timedelta(rng.integers(1, 29, 600), unit="D"),
                    "actual": actual,
                    "forecast": forecast,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    table, cover = split_conformal(frame, levels=(0.8, 0.95))
    assert set(table["week"]) == {1, 2, 3, 4}
    got = cover.set_index(["model", "level"])["coverage"]
    assert abs(got[("m", 0.8)] - 0.8) < 0.04
    assert abs(got[("m", 0.95)] - 0.95) < 0.03
