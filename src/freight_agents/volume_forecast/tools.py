"""The planner agent's tools. Each reads one stored run and returns figures with its id.

The agent states only numbers these tools return, so every total, comparison and
percentage is computed here, in code, and never by the model.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from freight_agents.agentcore.tools import Risk, Tool, tool
from freight_agents.volume_forecast.data import generate, holiday_names
from freight_agents.volume_forecast.runs import Run, RunStore, data_hash
from pydantic import BaseModel

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SAME_WEEKDAY_LAST_YEAR = dt.timedelta(days=364)


@dataclass(frozen=True)
class ForecastDeps:
    run: Run
    data: pd.DataFrame

    @classmethod
    def load(cls, store: RunStore, ref: str = "latest") -> ForecastDeps:
        run = store.load(ref)
        data = generate(seed=run["data"]["seed"])
        if data_hash(data) != run["data"]["hash"]:
            raise ValueError(f"run {run['run_id']} was built on different data")
        return cls(run, data)

    @cached_property
    def forecast(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.run["forecast"])
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    @cached_property
    def holidays(self) -> dict[pd.Timestamp, str]:
        return holiday_names(self.data["date"].min().year, self.forecast["date"].max().year)

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])

    def lanes(self) -> list[str]:
        return sorted(self.forecast["lane"].unique())

    def rows(self, lane: str, model: str | None = None) -> pd.DataFrame:
        model = model or self.run["best_model"]
        if lane not in self.lanes():
            raise ValueError(
                f"run {self.run_id}: unknown lane {lane}; lanes: {', '.join(self.lanes())}"
            )
        models = sorted(self.forecast["model"].unique())
        if model not in models:
            raise ValueError(
                f"run {self.run_id}: unknown model {model}; models: {', '.join(models)}"
            )
        picked = self.forecast[(self.forecast["lane"] == lane) & (self.forecast["model"] == model)]
        return picked.sort_values("date")

    def window(self, rows: pd.DataFrame, start: dt.date | None, days: int) -> pd.DataFrame:
        first, last = rows["date"].min(), rows["date"].max()
        begin = pd.Timestamp(start) if start else first
        end = begin + pd.Timedelta(days=days - 1)
        if days < 1 or begin < first or end > last:
            raise ValueError(
                f"run {self.run_id} forecasts {first:%Y-%m-%d} to {last:%Y-%m-%d}; "
                f"ask for 1 to {len(rows)} days inside that"
            )
        return rows[(rows["date"] >= begin) & (rows["date"] <= end)]

    def actual(self, lane: str, begin: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        d = self.data
        return d[(d["lane"] == lane) & (d["date"] >= begin) & (d["date"] <= end)]


class ForecastIn(BaseModel):
    lane: str
    start: dt.date | None
    days: int
    model: str | None


class ForecastDay(BaseModel):
    date: dt.date
    weekday: str
    forecast: int
    low_80: int
    high_80: int
    low_95: int
    high_95: int
    holiday: str | None


class ForecastOut(BaseModel):
    run_id: str
    model: str
    lane: str
    days: list[ForecastDay]
    total_forecast: int


class CompareIn(BaseModel):
    lane: str
    start: dt.date | None
    days: int


class CompareOut(BaseModel):
    run_id: str
    model: str
    lane: str
    start: dt.date
    end: dt.date
    forecast_total: int
    holidays: list[str]
    last_year_start: dt.date
    last_year_end: dt.date
    last_year_actual_total: int
    last_year_holidays: list[str]
    change_pct: float


class DriversIn(BaseModel):
    lane: str
    day: dt.date


class DriversOut(BaseModel):
    run_id: str
    model: str
    lane: str
    date: dt.date
    weekday: str
    forecast: int
    holiday: str | None
    promotion: bool
    typical_weekday_forecast: int
    recent_daily_average: int
    last_year_date: dt.date
    last_year_actual: int
    last_year_holiday: str | None
    last_year_promotion: bool


class SummaryIn(BaseModel):
    lane: str | None


class ModelScore(BaseModel):
    model: str
    wape_pct: float
    mase: float
    bias_pct: float


class CoverageRow(BaseModel):
    model: str
    level_pct: int
    coverage_pct: float


class SummaryOut(BaseModel):
    run_id: str
    origins: int
    horizon_days: int
    lane: str | None
    best_model: str
    scores: list[ModelScore]
    coverage: list[CoverageRow]


def build_tools(deps: ForecastDeps) -> dict[str, Tool]:
    def holiday(day: pd.Timestamp) -> str | None:
        return deps.holidays.get(pd.Timestamp(day))

    @tool(
        name="get_forecast",
        description="Daily shipment forecast for one lane with 80% and 95% ranges, and the "
        "total over the days asked. start null means the first forecast day; model null means "
        "the run's best model.",
        risk=Risk.READ,
    )
    def get_forecast(args: ForecastIn) -> ForecastOut:
        rows = deps.window(deps.rows(args.lane, args.model), args.start, args.days)
        days = [
            ForecastDay(
                date=r["date"].date(),
                weekday=WEEKDAYS[r["date"].weekday()],
                forecast=round(r["forecast"]),
                low_80=round(r["lo_80"]),
                high_80=round(r["hi_80"]),
                low_95=round(r["lo_95"]),
                high_95=round(r["hi_95"]),
                holiday=holiday(r["date"]),
            )
            for r in rows.to_dict("records")
        ]
        return ForecastOut(
            run_id=deps.run_id,
            model=str(rows["model"].iloc[0]),
            lane=args.lane,
            days=days,
            total_forecast=round(rows["forecast"].sum()),
        )

    @tool(
        name="compare_to_last_year",
        description="Forecast total for a lane over some days against the actual total on "
        "the same weekdays 52 weeks earlier, with the change in percent and any holidays in "
        "either period.",
        risk=Risk.READ,
    )
    def compare_to_last_year(args: CompareIn) -> CompareOut:
        rows = deps.window(deps.rows(args.lane), args.start, args.days)
        begin, end = rows["date"].min(), rows["date"].max()
        then_begin, then_end = begin - SAME_WEEKDAY_LAST_YEAR, end - SAME_WEEKDAY_LAST_YEAR
        then = deps.actual(args.lane, then_begin, then_end)
        now_total, then_total = round(rows["forecast"].sum()), int(then["volume"].sum())
        return CompareOut(
            run_id=deps.run_id,
            model=str(rows["model"].iloc[0]),
            lane=args.lane,
            start=begin.date(),
            end=end.date(),
            forecast_total=now_total,
            holidays=[h for d in rows["date"] if (h := holiday(d))],
            last_year_start=then_begin.date(),
            last_year_end=then_end.date(),
            last_year_actual_total=then_total,
            last_year_holidays=[h for d in then["date"] if (h := holiday(d))],
            change_pct=round((now_total / then_total - 1) * 100, 1),
        )

    @tool(
        name="explain_drivers",
        description="What shapes one day's forecast on a lane: weekday, holiday, announced "
        "promotion, a typical weekday that week, the lane's recent daily average, and the same "
        "weekday last year with its holiday and promotion.",
        risk=Risk.READ,
    )
    def explain_drivers(args: DriversIn) -> DriversOut:
        rows = deps.rows(args.lane)
        day = pd.Timestamp(args.day)
        row = deps.window(rows, args.day, 1).iloc[0]
        week = rows[
            (rows["date"] >= day - pd.Timedelta(days=day.weekday()))
            & (rows["date"] < day + pd.Timedelta(days=7 - day.weekday()))
        ]
        typical = week[(week["date"].dt.day_of_week < 5) & ~week["date"].map(holiday).notna()]
        if typical.empty:
            typical = rows[(rows["date"].dt.day_of_week < 5) & ~rows["date"].map(holiday).notna()]
        end = deps.data["date"].max()
        recent = deps.actual(args.lane, end - pd.Timedelta(days=27), end)
        then = day - SAME_WEEKDAY_LAST_YEAR
        last_year = deps.actual(args.lane, then, then)
        return DriversOut(
            run_id=deps.run_id,
            model=str(row["model"]),
            lane=args.lane,
            date=args.day,
            weekday=WEEKDAYS[day.weekday()],
            forecast=round(row["forecast"]),
            holiday=holiday(day),
            promotion=bool(row["promo"]),
            typical_weekday_forecast=round(typical["forecast"].mean()),
            recent_daily_average=round(recent["volume"].mean()),
            last_year_date=then.date(),
            last_year_actual=int(last_year["volume"].sum()),
            last_year_holiday=holiday(then),
            last_year_promotion=bool(last_year["promo"].any()),
        )

    @tool(
        name="backtest_summary",
        description="How accurate each model was in the run's backtest (WAPE and bias in "
        "percent, MASE below 1 beats repeating last week), overall or for one lane, and how "
        "often the 80% and 95% ranges held the actual volume.",
        risk=Risk.READ,
    )
    def backtest_summary(args: SummaryIn) -> SummaryOut:
        bt = deps.run["backtest"]
        if args.lane is None:
            rows = bt["overall"]
        else:
            deps.rows(args.lane)
            rows = [r for r in bt["by_lane"] if r["lane"] == args.lane]
        scores = [
            ModelScore(
                model=r["model"],
                wape_pct=round(r["wape"] * 100, 1),
                mase=round(r["mase"], 2),
                bias_pct=round(r["bias"] * 100, 1),
            )
            for r in sorted(rows, key=lambda r: r["wape"])
        ]
        coverage = [
            CoverageRow(
                model=r["model"],
                level_pct=round(r["level"] * 100),
                coverage_pct=round(r["coverage"] * 100, 1),
            )
            for r in bt["coverage"]
        ]
        return SummaryOut(
            run_id=deps.run_id,
            origins=len(deps.run["config"]["origins"]),
            horizon_days=deps.run["config"]["horizon"],
            lane=args.lane,
            best_model=deps.run["best_model"],
            scores=scores,
            coverage=coverage,
        )

    tools = [get_forecast, compare_to_last_year, explain_drivers, backtest_summary]
    return {t.name: t for t in tools}
