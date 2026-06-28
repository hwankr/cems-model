from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.kma_weather_api import (
    DEFAULT_CAMPUS_LAT,
    DEFAULT_CAMPUS_LON,
    build_apihub_url,
    fetch_text,
    get_kma_auth_key,
    kma_grid_from_lonlat,
)
from modeling.weather_features import (
    aggregate_forecast_weather,
    normalize_actual_weather,
    normalize_kma_forecast_items,
    select_forecast_snapshots,
)


DEFAULT_ASOS_STATION_ID = "143"
DEFAULT_ACTUAL_ENDPOINT = "/api/typ01/url/kma_sfcdd.php"
DEFAULT_FORECAST_ENDPOINT = "/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
SHORT_FORECAST_VARIABLES = {
    "TMP": "temp_c",
    "REH": "humidity_pct",
    "PCP": "rainfall_mm",
    "WSD": "wind_mps",
}


def parse_apihub_table(text: str) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line.lstrip("#").strip().split()
            if {"TM", "STN"}.issubset(set(candidate)) or {"TMFC", "TMEF"}.issubset(
                set(candidate),
            ):
                header = candidate
            continue
        rows.append(line.split())

    if not rows:
        return pd.DataFrame()
    if header and all(len(row) == len(header) for row in rows):
        return pd.DataFrame(rows, columns=header)
    return pd.DataFrame(rows)


def fetch_actual_weather(
    start: str,
    end: str,
    station_id: str,
    output_path: Path,
    endpoint: str = DEFAULT_ACTUAL_ENDPOINT,
) -> pd.DataFrame:
    auth_key = get_kma_auth_key()
    rows = []
    for date in _date_range(start, end):
        url = build_apihub_url(
            endpoint,
            {
                "tm": date.strftime("%Y%m%d"),
                "stn": station_id,
                "disp": "0",
                "help": "1",
            },
            auth_key,
        )
        table = parse_apihub_table(fetch_text(url))
        if table.empty:
            continue
        rows.append(table)

    if not rows:
        raise RuntimeError("No ASOS actual weather rows were returned.")

    raw = pd.concat(rows, ignore_index=True)
    actual = normalize_actual_weather(build_actual_weather_input(raw))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual.to_csv(output_path, index=False, encoding="utf-8-sig")
    return actual


def fetch_forecast_weather(
    start: str,
    end: str,
    output_path: Path,
    lon: float = DEFAULT_CAMPUS_LON,
    lat: float = DEFAULT_CAMPUS_LAT,
    endpoint: str = DEFAULT_FORECAST_ENDPOINT,
    hour_step: int = 6,
) -> pd.DataFrame:
    auth_key = get_kma_auth_key()
    grid = kma_grid_from_lonlat(lon=lon, lat=lat)
    rows_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for target_date in _date_range(start, end):
        issue_date = target_date - timedelta(days=1)
        # 23:00 is the main leakage-safe cutoff rule used by this project.
        issue_param = issue_date.strftime("%Y%m%d23")
        issue_at = issue_date.strftime("%Y%m%d2300")
        for hour in range(0, 24, hour_step):
            target_param = target_date.strftime("%Y%m%d") + f"{hour:02d}"
            target_at = target_date.strftime("%Y%m%d") + f"{hour:02d}00"
            key = (issue_at, target_at)
            row = rows_by_key.setdefault(
                key,
                {
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "issued_at": pd.to_datetime(issue_at, format="%Y%m%d%H%M"),
                    "forecast_datetime": pd.to_datetime(target_at, format="%Y%m%d%H%M"),
                },
            )
            for variable, column in SHORT_FORECAST_VARIABLES.items():
                url = build_apihub_url(
                    endpoint,
                    {
                        "tmfc": issue_param,
                        "tmef": target_param,
                        "vars": variable,
                    },
                    auth_key,
                )
                value = extract_grid_value(fetch_text(url), x=grid.x, y=grid.y)
                if column == "rainfall_mm" and pd.notna(value) and value < 0:
                    value = 0.0
                row[column] = value

    normalized = pd.DataFrame(rows_by_key.values())
    if normalized.empty:
        raise RuntimeError(
            "No forecast rows could be normalized. Save the raw API response and check "
            "whether this account has the short-term forecast archive API enabled.",
        )
    selected = select_forecast_snapshots(normalized)
    forecast = aggregate_forecast_weather(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False, encoding="utf-8-sig")
    return forecast


def _forecast_table_to_items(
    table: pd.DataFrame,
    issue_time: str,
    target_time: str,
) -> list[dict[str, object]]:
    if table.empty:
        return []
    items: list[dict[str, object]] = []
    lower_columns = {str(column).lower(): column for column in table.columns}
    category_col = lower_columns.get("category") or lower_columns.get("elem") or lower_columns.get("var")
    value_col = lower_columns.get("fcstvalue") or lower_columns.get("value") or lower_columns.get("val")
    if category_col is None or value_col is None:
        return items
    for _, row in table.iterrows():
        items.append(
            {
                "baseDate": issue_time[:8],
                "baseTime": issue_time[8:],
                "fcstDate": target_time[:8],
                "fcstTime": target_time[8:],
                "category": row[category_col],
                "fcstValue": row[value_col],
            },
        )
    return items


def extract_grid_value(text: str, x: int, y: int, nx: int = 149, ny: int = 253) -> float:
    values = [float(value) for value in re.findall(r"[-+]?\d+\.\d+", text)]
    expected_count = nx * ny
    if len(values) != expected_count:
        raise RuntimeError(
            f"Unexpected KMA grid size: expected {expected_count} values, got {len(values)}.",
        )
    if x < 1 or x > nx or y < 1 or y > ny:
        raise ValueError(f"KMA grid point out of range: x={x}, y={y}.")
    value = values[(y - 1) * nx + (x - 1)]
    return value


def build_actual_weather_input(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tm": _pick_column(raw, ["TM", "tm", "date", 0]),
            "avgTa": _pick_column(raw, ["TA_AVG", "avgTa", "AVG_TA", 10]),
            "maxTa": _pick_column(raw, ["TA_MAX", "maxTa", "MAX_TA", 11]),
            "minTa": _pick_column(raw, ["TA_MIN", "minTa", "MIN_TA", 13]),
            "sumRn": _pick_column(raw, ["RN_DAY", "sumRn", "SUM_RN", 38]),
            "avgRhm": _pick_column(raw, ["HM_AVG", "avgRhm", "AVG_RHM", 18]),
            "avgWs": _pick_column(raw, ["WS_AVG", "avgWs", "AVG_WS", 2]),
        },
    )


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    return pd.Series([pd.NA] * len(frame))


def _date_range(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    while current <= last:
        yield current
        current += timedelta(days=1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch KMA weather data for CEMS models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    actual = subparsers.add_parser("actual", help="Fetch ASOS actual daily weather.")
    actual.add_argument("--start", default="2025-07-01")
    actual.add_argument("--end", default="2026-06-23")
    actual.add_argument("--station-id", default=DEFAULT_ASOS_STATION_ID)
    actual.add_argument("--output", type=Path, default=Path("outputs/weather_actual_daily.csv"))
    actual.add_argument("--endpoint", default=DEFAULT_ACTUAL_ENDPOINT)

    forecast = subparsers.add_parser("forecast", help="Fetch short-term forecast weather.")
    forecast.add_argument("--start", default="2026-04-01")
    forecast.add_argument("--end", default="2026-06-23")
    forecast.add_argument("--lon", type=float, default=DEFAULT_CAMPUS_LON)
    forecast.add_argument("--lat", type=float, default=DEFAULT_CAMPUS_LAT)
    forecast.add_argument("--output", type=Path, default=Path("outputs/weather_forecast_daily.csv"))
    forecast.add_argument("--endpoint", default=DEFAULT_FORECAST_ENDPOINT)
    forecast.add_argument(
        "--hour-step",
        type=int,
        default=6,
        choices=[3, 6],
        help="Forecast valid-hour interval. Use 6 for faster daily features or 3 for full short-term steps.",
    )

    grid = subparsers.add_parser("grid", help="Print campus forecast grid x/y.")
    grid.add_argument("--lon", type=float, default=DEFAULT_CAMPUS_LON)
    grid.add_argument("--lat", type=float, default=DEFAULT_CAMPUS_LAT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "actual":
            data = fetch_actual_weather(
                args.start,
                args.end,
                args.station_id,
                args.output,
                args.endpoint,
            )
            print(f"wrote={args.output}")
            print(f"rows={len(data)}")
        elif args.command == "forecast":
            data = fetch_forecast_weather(
                args.start,
                args.end,
                args.output,
                args.lon,
                args.lat,
                args.endpoint,
                args.hour_step,
            )
            print(f"wrote={args.output}")
            print(f"rows={len(data)}")
        elif args.command == "grid":
            grid = kma_grid_from_lonlat(lon=args.lon, lat=args.lat)
            print(f"x={grid.x}")
            print(f"y={grid.y}")
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
