"""Rolling-origin backtest.

Each origin is a day the forecast could have been made. Every model is fitted on
the history up to that day and forecasts the next `horizon` days, which are then
scored against what happened. Origins step back from the end of the data, so the
last forecast window ends on the last day.

Metrics, over all rows of a model (or of a model on one lane):

* WAPE: total absolute error over total volume. A planner's error rate.
* MASE: mean absolute error, each row divided by its lane's in-sample error of
  "same weekday last week" before the origin. Below 1 beats that naive rule.
* Bias: total forecast over total actual, minus one. Positive means too high.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from freight_agents.volume_forecast.models import Forecaster, future_frame

Factory = Callable[[], Forecaster]


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(actual - forecast).sum() / np.abs(actual).sum())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(forecast.sum() / actual.sum() - 1)


def mase(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast) / scale))


def seasonal_scale(history: pd.DataFrame, season: int = 7) -> pd.Series:
    """Per lane, the mean absolute error of the same-weekday-last-week rule in `history`."""
    wide = history.pivot(index="date", columns="lane", values="volume").astype(float)
    return wide.diff(season).abs().mean()


def origins(
    data: pd.DataFrame, count: int = 8, step: int = 28, horizon: int = 28
) -> list[pd.Timestamp]:
    last = data["date"].max() - pd.Timedelta(days=horizon)
    return [last - pd.Timedelta(days=step * i) for i in reversed(range(count))]


def run_backtest(
    data: pd.DataFrame, factories: Sequence[Factory], cut: Sequence[pd.Timestamp], horizon: int = 28
) -> pd.DataFrame:
    """One row per origin, model, lane and day: the forecast, the actual, and the MASE scale."""
    frames = []
    for origin in cut:
        history = data[data["date"] <= origin]
        future = future_frame(data, origin, horizon)
        actual = future.merge(data[["date", "lane", "volume"]], on=["date", "lane"], how="left")
        scale = future["lane"].map(seasonal_scale(history)).to_numpy(dtype=float)
        for factory in factories:
            model = factory()
            model.fit(history)
            forecast = model.predict(future).to_numpy(dtype=float)
            frames.append(
                future.assign(
                    origin=origin,
                    model=model.name,
                    actual=actual["volume"].to_numpy(dtype=float),
                    forecast=forecast,
                    scale=scale,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["forecast"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "wape": wape(actual, forecast),
                "mase": mase(actual, forecast, group["scale"].to_numpy(dtype=float)),
                "bias": bias(actual, forecast),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics per model, and per model and lane."""
    return _metrics(frame, ["model"]), _metrics(frame, ["model", "lane"])
