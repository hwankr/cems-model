from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AREA_FEATURE_COLUMNS = [
    "area_total_sqm",
    "room_count",
    "floor_count",
    "area_missing",
    "profile_kwh_per_sqm_month",
    "rooms_per_1000sqm",
]


BUILDING_AREA_ALIASES = {
    "사범대학": ["사범대학", "사범대학 신관"],
    "생활과학대": ["생활과학대학본관", "생활과학대학별관"],
    "음대": ["예술대학 음악관"],
    "제1공장형실습장": ["기계실습공장(1)"],
}


def load_building_area_features(
    path: Path,
    building_names: list[str],
) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path, sheet_name=0, header=2)
        raw = _normalize_excel_area_columns(raw)
    return build_building_area_table(raw, building_names)


def build_building_area_table(
    raw_area_rows: pd.DataFrame,
    building_names: list[str],
) -> pd.DataFrame:
    raw = _normalize_area_columns(raw_area_rows)
    for column in ["private_area_sqm", "common_area_sqm", "total_area_sqm", "room_count"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)

    grouped = (
        raw.groupby("building", dropna=False)
        .agg(
            area_private_sqm=("private_area_sqm", "sum"),
            area_common_sqm=("common_area_sqm", "sum"),
            area_total_sqm=("total_area_sqm", "sum"),
            room_count=("room_count", "sum"),
            floor_count=("floor", "count"),
        )
        .reset_index()
    )

    rows = []
    for building in building_names:
        names = BUILDING_AREA_ALIASES.get(building, [building])
        selected_parts = []
        for name in names:
            hit = grouped[
                grouped["building"].fillna("").astype(str).str.contains(name, regex=False)
            ]
            if not hit.empty:
                selected_parts.append(hit)
        if not selected_parts:
            rows.append({"building_name_recent": building, "area_missing": 1})
            continue

        selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(
            subset=["building"],
        )
        sums = selected[
            ["area_private_sqm", "area_common_sqm", "area_total_sqm", "room_count", "floor_count"]
        ].sum()
        rows.append(
            {
                "building_name_recent": building,
                "area_private_sqm": sums["area_private_sqm"],
                "area_common_sqm": sums["area_common_sqm"],
                "area_total_sqm": sums["area_total_sqm"],
                "room_count": sums["room_count"],
                "floor_count": sums["floor_count"],
                "area_missing": 0,
            }
        )
    return pd.DataFrame(rows)


def add_building_area_features(
    panel: pd.DataFrame,
    area_features: pd.DataFrame,
) -> pd.DataFrame:
    result = panel.merge(area_features, on="building_name_recent", how="left")
    result["area_missing"] = pd.to_numeric(result.get("area_missing"), errors="coerce").fillna(1)

    for column in ["area_total_sqm", "room_count", "floor_count"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        median = result[column].median()
        if not np.isfinite(median):
            median = 1.0
        result[column] = result[column].fillna(median)

    result["profile_kwh_per_sqm_month"] = (
        pd.to_numeric(result["profile_monthly_kwh_mean"], errors="coerce")
        / result["area_total_sqm"].replace(0, np.nan)
    ).fillna(0)
    result["rooms_per_1000sqm"] = (
        result["room_count"] / result["area_total_sqm"].replace(0, np.nan) * 1000.0
    ).fillna(0)
    return result


def _normalize_excel_area_columns(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy()
    result.columns = [f"c{index}" for index in range(len(result.columns))]
    return result.rename(
        columns={
            "c0": "building_no",
            "c1": "building",
            "c2": "zone",
            "c3": "floor",
            "c4": "private_area_sqm",
            "c5": "common_area_sqm",
            "c6": "total_area_sqm",
            "c7": "room_count",
        }
    )


def _normalize_area_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if {"building", "floor", "private_area_sqm", "common_area_sqm", "total_area_sqm", "room_count"}.issubset(
        raw.columns,
    ):
        return raw.copy()
    return _normalize_excel_area_columns(raw)
