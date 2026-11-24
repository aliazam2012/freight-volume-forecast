"""Prediction intervals by split conformal calibration.

The backtest's earlier origins are the calibration set: for each model and week
ahead, the spread of log(actual + 1) - log(forecast + 1) gives a lower and an
upper bound at each level. The later origins never touch the calibration; how
often their actuals land inside the bounds is the measured coverage. Working on
the log scale makes one table serve lanes of any size and lets the interval be
lopsided, which freight volume is (a holiday is a much bigger miss below than
anything above).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

LEVELS = (0.8, 0.95)
WEEKS = 4


def _pct(level: float) -> int:
    return round(level * 100)


def horizon_week(origin: pd.Series, date: pd.Series, weeks: int = WEEKS) -> np.ndarray:
    """1 for days 1 to 7 ahead, 2 for 8 to 14, and so on; later days share the last week."""
    days = (date - origin).dt.days.to_numpy()
    return np.asarray(np.clip(np.ceil(days / 7).astype(int), 1, weeks))


def conformal_table(
    errors: pd.DataFrame, levels: Sequence[float] = LEVELS, weeks: int = WEEKS
) -> pd.DataFrame:
    """Bounds on the log ratio per model, week ahead and level, with the finite-sample rank."""
    scores = np.log1p(errors["actual"].to_numpy(dtype=float)) - np.log1p(
        errors["forecast"].to_numpy(dtype=float)
    )
    frame = pd.DataFrame(
        {
            "model": errors["model"].to_numpy(),
            "week": horizon_week(errors["origin"], errors["date"], weeks),
            "score": scores,
        }
    )
    rows = []
    for (model, _), group in frame.groupby(["model", "week"], sort=True):
        week = int(group["week"].iloc[0])
        ordered = np.sort(group["score"].to_numpy())
        n = len(ordered)
        for level in levels:
            rank = math.ceil((n + 1) * (1 - (1 - level) / 2))
            if rank > n:
                raise ValueError(f"{n} calibration rows are too few for a {level:.0%} interval")
            rows.append(
                {
                    "model": model,
                    "week": week,
                    "level": level,
                    "lo": float(ordered[n - rank]),
                    "hi": float(ordered[rank - 1]),
                    "calibration_rows": n,
                }
            )
    return pd.DataFrame(rows)


def with_intervals(forecasts: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """`forecasts` with lo_<pct> and hi_<pct> columns for each level in `table`."""
    out = forecasts.copy()
    weeks = int(table["week"].max())
    out["week"] = horizon_week(out["origin"], out["date"], weeks)
    base = np.log1p(out["forecast"].to_numpy(dtype=float))
    for level in sorted(table["level"].unique()):
        rows = table[table["level"] == level]
        bounds = out[["model", "week"]].merge(rows, on=["model", "week"], how="left")
        pct = _pct(float(level))
        out[f"lo_{pct}"] = np.clip(np.expm1(base + bounds["lo"].to_numpy()), 0, None)
        out[f"hi_{pct}"] = np.expm1(base + bounds["hi"].to_numpy())
    return out


def coverage(frame: pd.DataFrame, levels: Sequence[float] = LEVELS) -> pd.DataFrame:
    """Share of actuals inside their interval, and the interval width relative to volume."""
    rows = []
    for (model,), group in frame.groupby(["model"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        for level in levels:
            pct = _pct(level)
            lo = group[f"lo_{pct}"].to_numpy(dtype=float)
            hi = group[f"hi_{pct}"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "level": level,
                    "coverage": float(np.mean((actual >= lo) & (actual <= hi))),
                    "relative_width": float((hi - lo).sum() / actual.sum()),
                    "rows": len(group),
                }
            )
    return pd.DataFrame(rows)


def split_conformal(
    frame: pd.DataFrame, levels: Sequence[float] = LEVELS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate on the earlier half of the origins; measure coverage on the later half."""
    cut = sorted(frame["origin"].unique())
    early = frame[frame["origin"].isin(cut[: len(cut) // 2])]
    late = frame[frame["origin"].isin(cut[len(cut) // 2 :])]
    table = conformal_table(early, levels)
    return table, coverage(with_intervals(late, table), levels)
