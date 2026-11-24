"""A run: one backtest plus a forward forecast, saved under an id that names its inputs.

The id is a hash of the data, the configuration and the package version, so the
same inputs always give the same id and any change gives a new one. Every number
the report prints and every figure the agent's tools return comes from one run
and carries its id.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from freight_agents.volume_forecast.backtest import origins, run_backtest, summarize
from freight_agents.volume_forecast.data import generate
from freight_agents.volume_forecast.intervals import (
    LEVELS,
    conformal_table,
    split_conformal,
    with_intervals,
)
from freight_agents.volume_forecast.models import (
    Forecaster,
    GradientBoosting,
    HoltWinters,
    SeasonalNaive,
    future_frame,
)

VERSION = version("freight-agents-volume-forecast")
STEP_DAYS = 28
FORECAST_DAYS = 28
BASELINE = "seasonal_naive"
RUN_REF = re.compile(r"fc-[a-z0-9]+")

Run = dict[str, Any]


def data_hash(data: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(data, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def run_id(data_digest: str, config: dict[str, Any], package_version: str) -> str:
    blob = json.dumps(
        {"data": data_digest, "config": config, "version": package_version}, sort_keys=True
    )
    return "fc-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows: dates as YYYY-MM-DD, floats to six decimals, missing as null."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = json.loads(out.to_json(orient="records", double_precision=6))
    return rows


class RunStore:
    """runs/<id>.json, plus a LATEST file naming the newest."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def save(self, run: Run, compact: bool = False) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{run['run_id']}.json"
        text = json.dumps(run, sort_keys=True, indent=None if compact else 1)
        path.write_text(text + "\n")
        (self.directory / "LATEST").write_text(run["run_id"] + "\n")
        return path

    def load(self, ref: str = "latest") -> Run:
        if ref == "latest":
            ref = (self.directory / "LATEST").read_text().strip()
        if not RUN_REF.fullmatch(ref):
            raise ValueError(f"not a run id: {ref!r}")
        run: Run = json.loads((self.directory / f"{ref}.json").read_text())
        return run


def _factories(gb_seed: int) -> dict[str, Callable[[], Forecaster]]:
    return {
        "seasonal_naive": SeasonalNaive,
        "holt_winters": HoltWinters,
        "gradient_boosting": lambda: GradientBoosting(seed=gb_seed),
    }


def run_config(
    seed: int = 7, count: int = 8, horizon: int = 28, gb_seed: int = 7
) -> tuple[pd.DataFrame, list[pd.Timestamp], dict[str, Any]]:
    """The data, the origins and the configuration a run with these settings uses."""
    data = generate(seed=seed)
    cut = origins(data, count=count, step=STEP_DAYS, horizon=horizon)
    config = {
        "seed": seed,
        "origins": [d.strftime("%Y-%m-%d") for d in cut],
        "horizon": horizon,
        "levels": list(LEVELS),
        "models": list(_factories(gb_seed)),
        "gb_seed": gb_seed,
    }
    return data, cut, config


def planned_run_id(seed: int = 7, count: int = 8, horizon: int = 28, gb_seed: int = 7) -> str:
    """The id a run with these settings will get, without running it."""
    data, _, config = run_config(seed, count, horizon, gb_seed)
    return run_id(data_hash(data), config, VERSION)


def ensure_run(store: RunStore, seed: int = 7, count: int = 8, day: str = "") -> Run:
    """The stored run for these settings, built and saved first if it is not there."""
    try:
        return store.load(planned_run_id(seed, count))
    except FileNotFoundError:
        run = build_run(seed=seed, count=count, day=day)
        store.save(run)
        return run


def build_run(
    seed: int = 7, count: int = 8, horizon: int = 28, gb_seed: int = 7, day: str = ""
) -> Run:
    """Backtest every model, calibrate intervals, and forecast the days after the data."""
    data, cut, config = run_config(seed, count, horizon, gb_seed)
    factories = _factories(gb_seed)
    frame = run_backtest(data, list(factories.values()), cut, horizon)
    overall, by_lane = summarize(frame)
    holidays, _ = summarize(frame[frame["holiday"]])
    _, coverage = split_conformal(frame)
    best = str(overall.sort_values("wape").iloc[0]["model"])

    # Forward intervals use every origin; coverage above was measured on held-out ones.
    table = conformal_table(frame)
    last = data["date"].max()
    future = future_frame(data, last, FORECAST_DAYS)
    forward = []
    for name, factory in factories.items():
        model = factory()
        model.fit(data)
        forward.append(future.assign(model=name, origin=last, forecast=model.predict(future)))
    forecast = with_intervals(pd.concat(forward, ignore_index=True), table)

    return {
        "run_id": run_id(data_hash(data), config, VERSION),
        "version": VERSION,
        "day": day,
        "data": {
            "seed": seed,
            "hash": data_hash(data),
            "start": data["date"].min().strftime("%Y-%m-%d"),
            "end": last.strftime("%Y-%m-%d"),
            "lanes": int(data["lane"].nunique()),
        },
        "config": config,
        "best_model": best,
        "backtest": {
            "overall": records(overall),
            "by_lane": records(by_lane),
            "holidays": records(holidays),
            "coverage": records(coverage),
        },
        "intervals": records(table),
        "forecast": records(forecast.drop(columns=["origin", "week"])),
    }
