from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_WEATHER_COLUMNS = [
    "date",
    "weather_source",
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "humidity_mean_pct",
    "rainfall_mm",
    "wind_mean_mps",
    "cdd_18",
    "hdd_18",
    "is_rainy",
]

ACTUAL_COLUMN_ALIASES = {
    "date": ("date", "tm"),
    "temp_mean_c": ("temp_mean_c", "avgTa", "avg_temp_c"),
    "temp_max_c": ("temp_max_c", "maxTa", "max_temp_c"),
    "temp_min_c": ("temp_min_c", "minTa", "min_temp_c"),
    "humidity_mean_pct": ("humidity_mean_pct", "avgRhm", "avg_humidity_pct"),
    "rainfall_mm": ("rainfall_mm", "sumRn", "sum_rainfall_mm"),
    "wind_mean_mps": ("wind_mean_mps", "avgWs", "avg_wind_mps"),
}


def normalize_actual_weather(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    for target, aliases in ACTUAL_COLUMN_ALIASES.items():
        source = _first_existing_column(frame, aliases)
        if source is None:
            result[target] = np.nan
        else:
            result[target] = frame[source]

    result["date"] = pd.to_datetime(result["date"])
    result["weather_source"] = "actual"
    for column in [
        "temp_mean_c",
        "temp_min_c",
        "temp_max_c",
        "humidity_mean_pct",
        "rainfall_mm",
        "wind_mean_mps",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["rainfall_mm"] = result["rainfall_mm"].fillna(0.0)
    return add_degree_day_features(result)[STANDARD_WEATHER_COLUMNS]


def select_forecast_snapshots(
    frame: pd.DataFrame,
    cutoff_hour: int = 0,
) -> pd.DataFrame:
    forecast = frame.copy()
    forecast["target_date"] = pd.to_datetime(forecast["target_date"]).dt.normalize()
    forecast["issued_at"] = pd.to_datetime(forecast["issued_at"])
    forecast["forecast_datetime"] = pd.to_datetime(forecast["forecast_datetime"])
    forecast["_cutoff_at"] = forecast["target_date"] + pd.to_timedelta(
        cutoff_hour,
        unit="h",
    )
    eligible = forecast[forecast["issued_at"] < forecast["_cutoff_at"]].copy()
    if eligible.empty:
        return eligible.drop(columns=["_cutoff_at"], errors="ignore")

    latest = (
        eligible.groupby("target_date", as_index=False)["issued_at"]
        .max()
        .rename(columns={"issued_at": "_selected_issued_at"})
    )
    selected = eligible.merge(latest, on="target_date", how="inner")
    selected = selected[selected["issued_at"] == selected["_selected_issued_at"]].copy()
    selected["weather_source"] = "forecast"
    return selected.drop(columns=["_cutoff_at", "_selected_issued_at"], errors="ignore")


def aggregate_forecast_weather(frame: pd.DataFrame) -> pd.DataFrame:
    forecast = frame.copy()
    forecast["target_date"] = pd.to_datetime(forecast["target_date"]).dt.normalize()
    forecast["issued_at"] = pd.to_datetime(forecast["issued_at"])
    for column in ["temp_c", "humidity_pct", "rainfall_mm", "wind_mps"]:
        if column not in forecast.columns:
            forecast[column] = np.nan
        forecast[column] = pd.to_numeric(forecast[column], errors="coerce")

    grouped = forecast.groupby(["target_date", "issued_at"], dropna=False, sort=True)
    daily = grouped.agg(
        temp_mean_c=("temp_c", "mean"),
        temp_min_c=("temp_c", "min"),
        temp_max_c=("temp_c", "max"),
        humidity_mean_pct=("humidity_pct", "mean"),
        rainfall_mm=("rainfall_mm", "sum"),
        wind_mean_mps=("wind_mps", "mean"),
    ).reset_index()
    daily = daily.rename(columns={"target_date": "date"})
    daily["weather_source"] = "forecast"
    daily = add_degree_day_features(daily)
    return daily[
        [
            "date",
            "issued_at",
            "weather_source",
            "temp_mean_c",
            "temp_min_c",
            "temp_max_c",
            "humidity_mean_pct",
            "rainfall_mm",
            "wind_mean_mps",
            "cdd_18",
            "hdd_18",
            "is_rainy",
        ]
    ]


def add_degree_day_features(
    frame: pd.DataFrame,
    base_temp_c: float = 18.0,
) -> pd.DataFrame:
    result = frame.copy()
    temp = pd.to_numeric(result["temp_mean_c"], errors="coerce")
    result["cdd_18"] = (temp - base_temp_c).clip(lower=0)
    result["hdd_18"] = (base_temp_c - temp).clip(lower=0)
    result["rainfall_mm"] = pd.to_numeric(result["rainfall_mm"], errors="coerce").fillna(0.0)
    result["is_rainy"] = (result["rainfall_mm"] > 0).astype(int)
    return result


def normalize_kma_forecast_items(items: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows_by_key: dict[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp], dict[str, object]] = {}
    for item in items:
        category = str(item.get("category", "")).strip()
        target = _parse_forecast_datetime(item)
        issued = _parse_issue_datetime(item)
        if pd.isna(target) or pd.isna(issued):
            continue
        key = (target.normalize(), issued, target)
        row = rows_by_key.setdefault(
            key,
            {
                "target_date": target.normalize(),
                "issued_at": issued,
                "forecast_datetime": target,
            },
        )
        value = _parse_forecast_value(item.get("fcstValue"))
        if category == "TMP":
            row["temp_c"] = value
        elif category == "REH":
            row["humidity_pct"] = value
        elif category in {"PCP", "RN1"}:
            row["rainfall_mm"] = value
        elif category == "WSD":
            row["wind_mps"] = value
        elif category == "POP":
            row["rain_prob_pct"] = value
    return pd.DataFrame(rows_by_key.values())


def _first_existing_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _parse_forecast_datetime(item: dict[str, object]) -> pd.Timestamp:
    date = str(item.get("fcstDate", "")).strip()
    time = str(item.get("fcstTime", "0000")).strip().zfill(4)
    if not date:
        return pd.NaT
    return pd.to_datetime(f"{date}{time}", format="%Y%m%d%H%M", errors="coerce")


def _parse_issue_datetime(item: dict[str, object]) -> pd.Timestamp:
    date = str(item.get("baseDate", item.get("tmfc", ""))).strip()
    time = str(item.get("baseTime", "")).strip().zfill(4)
    if not date:
        return pd.NaT
    if len(date) == 12 and not time:
        return pd.to_datetime(date, format="%Y%m%d%H%M", errors="coerce")
    return pd.to_datetime(f"{date}{time}", format="%Y%m%d%H%M", errors="coerce")


def _parse_forecast_value(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "강수없음", "없음"}:
        return 0.0
    if "1mm 미만" in text:
        return 0.5
    if text.startswith("30.0~50.0"):
        return 40.0
    digits = "".join(ch if ch.isdigit() or ch in ".-" else " " for ch in text)
    parts = [part for part in digits.split() if part]
    if not parts:
        return np.nan
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return np.nan
    if "~" in text and len(values) >= 2:
        return float(sum(values[:2]) / 2.0)
    if math.isfinite(values[0]):
        return float(values[0])
    return np.nan
