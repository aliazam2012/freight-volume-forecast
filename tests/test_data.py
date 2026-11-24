import pandas as pd
from freight_agents.volume_forecast.data import BREAK_DATE, BREAK_LANE, generate, holidays

DATA = generate(seed=7)


def _lane_mean(frame: pd.DataFrame, mask: pd.Series) -> float:
    return float(frame.loc[mask, "volume"].mean())


def test_shape_and_columns() -> None:
    assert list(DATA.columns) == ["date", "lane", "volume", "promo", "holiday"]
    assert DATA["lane"].nunique() == 12
    assert DATA["date"].min() == pd.Timestamp("2024-01-01")
    assert DATA["date"].max() == pd.Timestamp("2026-10-31")
    assert (DATA["volume"] >= 0).all()


def test_same_seed_same_data() -> None:
    pd.testing.assert_frame_equal(DATA, generate(seed=7))
    assert not DATA["volume"].equals(generate(seed=8)["volume"])


def test_weekdays_carry_more_than_weekends() -> None:
    dow = DATA["date"].dt.day_of_week
    assert _lane_mean(DATA, dow < 5) > 1.5 * _lane_mean(DATA, dow >= 5)


def test_peak_season_beats_february() -> None:
    # Within one year, so growth between years does not blur the season.
    month, year = DATA["date"].dt.month, DATA["date"].dt.year
    assert _lane_mean(DATA, (month == 12) & (year == 2025)) > 1.15 * _lane_mean(
        DATA, (month == 2) & (year == 2025)
    )


def test_holidays_dip_below_their_weekday_neighbors() -> None:
    days = [d for d in holidays(2024, 2026) if d.day_of_week < 5]
    hol = DATA[DATA["date"].isin(days)]
    week_before = DATA[DATA["date"].isin([d - pd.Timedelta(days=7) for d in days])]
    assert hol["volume"].mean() < 0.6 * week_before["volume"].mean()
    assert DATA.loc[DATA["date"].isin(days), "holiday"].all()


def test_the_structural_break_is_visible() -> None:
    # Year over year on the same weeks, against the other lanes, so the season cancels out.
    def yoy(lane: str) -> float:
        rows = DATA[DATA["lane"] == lane]
        now = rows[
            (rows["date"] >= BREAK_DATE) & (rows["date"] < BREAK_DATE + pd.Timedelta(days=60))
        ]
        then = rows[
            (rows["date"] >= BREAK_DATE - pd.DateOffset(years=1))
            & (rows["date"] < BREAK_DATE - pd.DateOffset(years=1) + pd.Timedelta(days=60))
        ]
        return float(now["volume"].mean() / then["volume"].mean())

    others = sorted(yoy(lane) for lane in DATA["lane"].unique() if lane != BREAK_LANE)
    assert yoy(BREAK_LANE) < 0.8 * others[len(others) // 2]


def test_promotions_lift_volume() -> None:
    weekday = DATA["date"].dt.day_of_week < 5
    promo = _lane_mean(DATA, weekday & DATA["promo"])
    plain = _lane_mean(DATA, weekday & ~DATA["promo"])
    assert DATA["promo"].sum() > 0
    assert promo > 1.1 * plain
