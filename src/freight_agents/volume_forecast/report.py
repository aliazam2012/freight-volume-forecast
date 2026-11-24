"""The dated backtest report: every number comes from one run and names its id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from freight_agents.volume_forecast.runs import BASELINE, Run


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _metric_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [r["model"], f"{r['wape']:.1%}", f"{r['mase']:.2f}", f"{r['bias']:+.1%}", str(r["rows"])]
        for r in sorted(rows, key=lambda r: r["wape"])
    ]


def _peak_note(origins: list[str], horizon: int) -> list[str]:
    """Say so when no forecast window reaches the November and December peak."""
    days = pd.date_range(
        pd.Timestamp(origins[0]) + pd.Timedelta(days=1),
        pd.Timestamp(origins[-1]) + pd.Timedelta(days=horizon),
    )
    if days.month.isin([11, 12]).any():
        return []
    return [
        f"- Forecast windows run {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, so the November "
        "and December peak is not in this backtest: peak-season accuracy is untested."
    ]


def verdicts(run: Run) -> list[str]:
    overall = {r["model"]: r for r in run["backtest"]["overall"]}
    base = overall[BASELINE]
    out = []
    for name, row in sorted(overall.items(), key=lambda kv: kv[1]["wape"]):
        if name == BASELINE:
            continue
        verb = "beats" if row["wape"] < base["wape"] else "does not beat"
        out.append(
            f"- {name} {verb} {BASELINE}: WAPE {row['wape']:.1%} against {base['wape']:.1%}, "
            f"MASE {row['mase']:.2f}."
        )
    return out


def render(run: Run) -> str:
    data, config, bt = run["data"], run["config"], run["backtest"]
    lines = [
        f"# Volume Forecast Backtest, {run['day']}",
        "",
        "## Table of Contents",
        "",
        "- [Summary](#summary)",
        "- [Accuracy](#accuracy)",
        "- [Intervals](#intervals)",
        "- [By Lane](#by-lane)",
        "- [Setup](#setup)",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"Run `{run['run_id']}`. Best model by WAPE: {run['best_model']}.",
        "",
        *verdicts(run),
        "",
        "## Accuracy",
        "",
        f"{len(config['origins'])} forecast origins, {config['horizon']} days ahead each, "
        f"{data['lanes']} lanes. WAPE is total absolute error over total volume; MASE below "
        f"1 beats repeating last week; bias above zero means forecasts ran high.",
        "",
        *_table(["Model", "WAPE", "MASE", "Bias", "Rows"], _metric_rows(bt["overall"])),
        "",
    ]
    if bt["holidays"]:
        lines += [
            "Holiday days only:",
            "",
            *_table(["Model", "WAPE", "MASE", "Bias", "Rows"], _metric_rows(bt["holidays"])),
            "",
        ]
    lines += [
        "## Intervals",
        "",
        "Split conformal: bounds calibrated on the earlier half of the origins, coverage "
        "measured on the later half, which the calibration never saw.",
        "",
        *_table(
            ["Model", "Level", "Coverage", "Width over volume", "Rows"],
            [
                [
                    r["model"],
                    f"{r['level']:.0%}",
                    f"{r['coverage']:.1%}",
                    f"{r['relative_width']:.2f}",
                    str(r["rows"]),
                ]
                for r in bt["coverage"]
            ],
        ),
        "",
        "## By Lane",
        "",
    ]
    by_lane: dict[str, dict[str, float]] = {}
    for r in bt["by_lane"]:
        by_lane.setdefault(r["lane"], {})[r["model"]] = r["wape"]
    best = run["best_model"]
    lines += _table(
        ["Lane", f"{best} WAPE", f"{BASELINE} WAPE"],
        [[lane, f"{m[best]:.1%}", f"{m[BASELINE]:.1%}"] for lane, m in sorted(by_lane.items())],
    )
    lines += [
        "",
        "## Setup",
        "",
        f"- Data: synthetic, seed {data['seed']}, hash `{data['hash']}`, "
        f"{data['start']} to {data['end']}. Invented lanes; the shapes (weekday, peak season, "
        "holidays, promotions, one lane losing a customer) are the ones real freight shows.",
        f"- Origins: {', '.join(config['origins'])}.",
        *_peak_note(config["origins"], config["horizon"]),
        "- Models: default settings chosen before the first backtest; nothing was tuned "
        "against these origins.",
        f"- Package version {run['version']}. Reproduce: `freight-forecast backtest "
        f"--seed {data['seed']} --origins {len(config['origins'])}`.",
        "",
    ]
    return "\n".join(lines)


def write(run: Run, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run['day']}-backtest.md"
    path.write_text(render(run))
    latest = {
        "date": run["day"],
        "run_id": run["run_id"],
        "best_model": run["best_model"],
        "overall": run["backtest"]["overall"],
        "coverage": run["backtest"]["coverage"],
    }
    (directory / "latest.json").write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
    return path
