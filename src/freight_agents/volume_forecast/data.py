"""Synthetic daily shipment volume for a made-up national carrier.

Twelve lanes named by city codes, every day from 2024-01-01 to 2026-10-31. The
volume on a lane is a level that drifts slowly, times a weekday profile (freight
moves on weekdays), an annual season that peaks from October to December, dips
on US holidays, lifts during announced promotions, and, on one lane, a lost
customer from 2025-09-01. Counts are drawn with extra-Poisson noise. Everything
is seeded, so the same seed always gives the same data.

The data is invented; the shapes are the ones real freight volume shows, so the
models face the problems they would face on real data: seasonality, calendar
effects, a known-in-advance promotion calendar, and a break the history does not
predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START, END = "2024-01-01", "2026-10-31"
LANES = [
    "CHI-DAL",
    "ATL-MIA",
    "LAX-PHX",
    "NWK-BOS",
    "DEN-SLC",
    "SEA-PDX",
    "HOU-SAT",
    "MEM-STL",
    "CLT-RDU",
    "IND-CMH",
    "OAK-RNO",
    "KCI-OMA",
]
BREAK_LANE = "MEM-STL"
BREAK_DATE = pd.Timestamp("2025-09-01")
BREAK_FACTOR = 0.68
WEEKDAY_PROFILE = np.array([1.10, 1.15, 1.12, 1.08, 1.05, 0.45, 0.35])
HOLIDAY_FACTOR = 0.25
PROMO_FACTOR = 1.30


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.day_of_week) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.day_of_week - weekday) % 7)


def holiday_names(first_year: int, last_year: int) -> dict[pd.Timestamp, str]:
    """New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas."""
    names: dict[pd.Timestamp, str] = {}
    for y in range(first_year, last_year + 1):
        names[pd.Timestamp(year=y, month=1, day=1)] = "New Year's Day"
        names[_last_weekday(y, 5, 0)] = "Memorial Day"
        names[pd.Timestamp(year=y, month=7, day=4)] = "Independence Day"
        names[_nth_weekday(y, 9, 0, 1)] = "Labor Day"
        names[_nth_weekday(y, 11, 3, 4)] = "Thanksgiving"
        names[pd.Timestamp(year=y, month=12, day=25)] = "Christmas"
    return names


def holidays(first_year: int, last_year: int) -> list[pd.Timestamp]:
    return list(holiday_names(first_year, last_year))


def _promotions(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """About six four-day promotions a year, announced ahead, so a model may use them."""
    flags = np.zeros(len(dates), dtype=bool)
    starts = rng.choice(len(dates) - 4, size=max(1, int(len(dates) / 365 * 6)), replace=False)
    for s in starts:
        flags[s : s + 4] = True
    return flags


def generate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    days = np.arange(len(dates))
    weekday = WEEKDAY_PROFILE[dates.day_of_week]
    annual = 1 + 0.30 * np.cos(2 * np.pi * (dates.day_of_year.to_numpy() - 335) / 365.25)
    holiday = dates.isin(holidays(dates[0].year, dates[-1].year))
    frames = []
    for lane in LANES:
        level = rng.uniform(300, 1500)
        growth = np.exp(rng.uniform(-0.00005, 0.0003) * days)
        promo = _promotions(rng, dates)
        mean = level * growth * weekday * annual
        mean = mean * np.where(holiday, HOLIDAY_FACTOR, 1.0) * np.where(promo, PROMO_FACTOR, 1.0)
        if lane == BREAK_LANE:
            mean = mean * np.where(dates >= BREAK_DATE, BREAK_FACTOR, 1.0)
        noisy = mean * rng.gamma(shape=60.0, scale=1 / 60.0, size=len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "lane": lane,
                    "volume": rng.poisson(noisy).astype(int),
                    "promo": promo,
                    "holiday": holiday,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
