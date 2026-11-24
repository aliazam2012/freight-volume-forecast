"""Three forecasters behind one interface.

Every model is fitted on the history up to a forecast origin and then asked for
the future rows: date, lane, and the calendar a planner knows in advance
(holidays, announced promotions). None of them ever sees a volume after the
origin; `features` enforces that for the one model that builds lags.

* SeasonalNaive: the same weekday last week. The baseline every other model has
  to beat, and the scale for MASE.
* HoltWinters: exponential smoothing per lane with a damped trend and a weekly
  season, fitted on log volume so the weekday swing scales with the level.
* GradientBoosting: one model for all lanes. It predicts volume as a multiple of
  the lane's recent level, from the calendar, the promotion flag, and lags of at
  least 28 days, so a 28-day forecast never needs a value it does not have.
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
import pandas as pd
from freight_agents.volume_forecast.data import holidays
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MIN_LAG = 28
LAGS = (28, 35, 42, 49)
LEVEL_DAYS = 28
FEATURES = [
    "lane_code",
    "day_of_week",
    "day_of_year",
    "holiday",
    "promo",
    "r28",
    "r35",
    "r42",
    "r49",
    "ly_ratio",
]


class Forecaster(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> None: ...

    def predict(self, future: pd.DataFrame) -> pd.Series: ...


def future_frame(data: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """The rows to forecast: every lane, the `horizon` days after `origin`, no volume.

    Promotions come from the data's calendar where it covers the day (announced
    ahead), otherwise none; holidays come from the holiday calendar.
    """
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    lanes = list(dict.fromkeys(data["lane"]))
    grid = pd.DataFrame({"date": np.tile(dates, len(lanes)), "lane": np.repeat(lanes, len(dates))})
    known = data[["date", "lane", "promo"]]
    out = grid.merge(known, on=["date", "lane"], how="left")
    out["promo"] = out["promo"].astype("boolean").fillna(False).astype(bool)
    out["holiday"] = out["date"].isin(holidays(dates[0].year, dates[-1].year))
    return out


def _wide(rows: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Volume as a date by lane table over the rows' full date range, blank after the origin."""
    known = rows.loc[rows["date"] <= origin, ["date", "lane", "volume"]]
    wide = known.pivot(index="date", columns="lane", values="volume").astype(float)
    days = pd.date_range(rows["date"].min(), rows["date"].max(), freq="D")
    lanes = list(dict.fromkeys(rows["lane"]))
    return wide.reindex(index=days, columns=lanes)


def _lookup(wide: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    i = wide.index.get_indexer(pd.Index(rows["date"]))
    j = wide.columns.get_indexer(pd.Index(rows["lane"]))
    return np.asarray(wide.to_numpy()[i, j], dtype=float)


def features(rows: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Model inputs for `rows`, built only from volumes on or before `origin`.

    `level` is the lane's mean over days t-55 to t-28; lags are expressed as
    ratios to it, so one model serves lanes of any size. `ly_ratio` is how far
    the same week last year sat above the level that preceded it, which carries
    the annual season into the forecast.
    """
    wide = _wide(rows, origin)
    level_wide = wide.shift(MIN_LAG).rolling(LEVEL_DAYS, min_periods=20).mean()
    level = _lookup(level_wide, rows)
    out = pd.DataFrame(index=rows.index)
    out["lane"] = rows["lane"].to_numpy()
    out["day_of_week"] = rows["date"].dt.day_of_week.to_numpy()
    out["day_of_year"] = rows["date"].dt.day_of_year.to_numpy()
    out["holiday"] = rows["holiday"].astype(int).to_numpy()
    out["promo"] = rows["promo"].astype(int).to_numpy()
    out["level"] = level
    for k in LAGS:
        out[f"r{k}"] = _lookup(wide.shift(k), rows) / level
    week_last_year = wide.rolling(7, center=True, min_periods=5).mean().shift(364)
    level_last_year = level_wide.shift(364)
    out["ly_ratio"] = _lookup(week_last_year, rows) / _lookup(level_last_year, rows)
    return out


class SeasonalNaive:
    """The same weekday in the last week before the origin."""

    name = "seasonal_naive"

    def __init__(self, season: int = 7) -> None:
        self.season = season
        self._history = pd.DataFrame()

    def fit(self, history: pd.DataFrame) -> None:
        self._history = history[["date", "lane", "volume"]]

    def predict(self, future: pd.DataFrame) -> pd.Series:
        origin = self._history["date"].max()
        ahead = (future["date"] - origin).dt.days.to_numpy()
        back = self.season * np.ceil(ahead / self.season).astype(int)
        source = future["date"] - pd.to_timedelta(back, unit="D")
        wide = _wide(self._history, origin)
        values = _lookup(wide, pd.DataFrame({"date": source, "lane": future["lane"]}))
        return pd.Series(values, index=future.index, name=self.name)


class HoltWinters:
    """Damped additive trend and additive weekly season on log volume, one fit per lane."""

    name = "holt_winters"

    def __init__(self, fit_days: int = 728) -> None:
        self.fit_days = fit_days
        self._fits: dict[str, object] = {}
        self._origin = pd.Timestamp(0)

    def fit(self, history: pd.DataFrame) -> None:
        self._origin = history["date"].max()
        recent = history[history["date"] > self._origin - pd.Timedelta(days=self.fit_days)]
        self._fits = {}
        for lane, rows in recent.groupby("lane", sort=False):
            series = np.log1p(rows.sort_values("date")["volume"].to_numpy(dtype=float))
            model = ExponentialSmoothing(
                series, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
            )
            with warnings.catch_warnings():
                # Optimizer chatter on some lanes; the backtest judges the fit.
                warnings.simplefilter("ignore")
                self._fits[str(lane)] = model.fit()

    def predict(self, future: pd.DataFrame) -> pd.Series:
        ahead = (future["date"] - self._origin).dt.days.to_numpy()
        out = np.zeros(len(future))
        for lane in future["lane"].unique():
            mask = (future["lane"] == lane).to_numpy()
            path = np.asarray(self._fits[str(lane)].forecast(int(ahead[mask].max())))  # type: ignore[attr-defined]
            out[mask] = np.expm1(path[ahead[mask] - 1])
        return pd.Series(np.clip(out, 0, None), index=future.index, name=self.name)


class GradientBoosting:
    """One global model of volume relative to the lane's level 28 to 55 days back."""

    name = "gradient_boosting"

    def __init__(self, seed: int = 7, max_iter: int = 300, learning_rate: float = 0.05) -> None:
        self.seed = seed
        self._model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            early_stopping=False,
            categorical_features=[0],
            random_state=seed,
        )
        self._history = pd.DataFrame()
        self._lanes: list[str] = []

    def _matrix(self, feats: pd.DataFrame) -> np.ndarray:
        codes = pd.Categorical(feats["lane"], categories=self._lanes).codes
        table = feats.assign(lane_code=codes)[FEATURES]
        return np.asarray(table.to_numpy(dtype=float))

    def fit(self, history: pd.DataFrame) -> None:
        self._history = history[["date", "lane", "volume", "promo", "holiday"]]
        self._lanes = sorted(history["lane"].unique())
        feats = features(self._history, history["date"].max())
        usable = feats["level"].notna().to_numpy()
        volume = history["volume"].to_numpy(dtype=float)
        target = volume[usable] / feats["level"].to_numpy()[usable]
        self._model.fit(self._matrix(feats[usable]), target)

    def predict(self, future: pd.DataFrame) -> pd.Series:
        origin = self._history["date"].max()
        rows = pd.concat([self._history, future.assign(volume=np.nan)], ignore_index=True)
        feats = features(rows, origin).iloc[len(self._history) :]
        ratio = self._model.predict(self._matrix(feats))
        values = np.clip(ratio * feats["level"].to_numpy(), 0, None)
        return pd.Series(values, index=future.index, name=self.name)
