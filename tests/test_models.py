import numpy as np
import pandas as pd
from freight_agents.volume_forecast.data import generate
from freight_agents.volume_forecast.models import (
    GradientBoosting,
    HoltWinters,
    SeasonalNaive,
    features,
    future_frame,
)

DATA = generate(seed=7)
ORIGIN = pd.Timestamp("2026-06-30")
HISTORY = DATA[DATA["date"] <= ORIGIN]
FUTURE = future_frame(DATA, ORIGIN, horizon=28)


def test_future_frame_has_known_calendar_but_no_volume() -> None:
    assert len(FUTURE) == 28 * 12
    assert FUTURE["date"].min() == ORIGIN + pd.Timedelta(days=1)
    assert "volume" not in FUTURE.columns
    assert {"promo", "holiday"} <= set(FUTURE.columns)


def test_seasonal_naive_repeats_last_week() -> None:
    model = SeasonalNaive()
    model.fit(HISTORY)
    pred = model.predict(FUTURE)
    lane = FUTURE["lane"] == "CHI-DAL"
    first = FUTURE[lane].iloc[0]
    last_week = HISTORY[
        (HISTORY["lane"] == "CHI-DAL") & (HISTORY["date"] == first["date"] - pd.Timedelta(days=7))
    ]["volume"].item()
    assert pred[lane].iloc[0] == last_week
    assert len(pred) == len(FUTURE)


def test_features_ignore_every_volume_after_the_origin() -> None:
    changed = DATA.copy()
    changed.loc[changed["date"] > ORIGIN, "volume"] = 10**6
    a = features(DATA, ORIGIN)
    b = features(changed, ORIGIN)
    pd.testing.assert_frame_equal(a, b)
    # Not vacuous: the first weeks after the origin do get history-based features.
    soon = (DATA["date"] > ORIGIN) & (DATA["date"] <= ORIGIN + pd.Timedelta(days=28))
    assert a.loc[soon, "level"].notna().all()


def test_models_return_one_non_negative_forecast_per_row() -> None:
    for model in (SeasonalNaive(), HoltWinters(), GradientBoosting(seed=7)):
        model.fit(HISTORY)
        pred = model.predict(FUTURE)
        assert len(pred) == len(FUTURE), model.name
        assert (pred >= 0).all(), model.name
        assert np.isfinite(pred).all(), model.name


def test_gradient_boosting_uses_the_known_calendar() -> None:
    model = GradientBoosting(seed=7)
    model.fit(HISTORY)
    base = FUTURE[(FUTURE["lane"] == "CHI-DAL") & (FUTURE["date"].dt.day_of_week == 2)].iloc[[1]]
    base = base.assign(holiday=False, promo=False)
    holiday, promo = base.assign(holiday=True), base.assign(promo=True)
    plain = model.predict(base).item()
    assert model.predict(holiday).item() < 0.7 * plain
    assert model.predict(promo).item() > 1.1 * plain
