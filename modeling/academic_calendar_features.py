from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


CALENDAR_BASE_URL = "https://www.yu.ac.kr/main/bachelor/calendar.do"


ACADEMIC_FEATURE_COLUMNS = [
    "academic_is_instruction_period",
    "academic_is_vacation",
    "academic_is_midterm",
    "academic_is_final",
    "academic_is_makeup_class_day",
    "academic_is_course_eval",
    "academic_is_education_practice",
    "academic_days_since_semester_start",
    "academic_days_to_final",
    "academic_semester_week",
]


def fetch_calendar_html(year: int) -> str:
    query = urlencode({"mode": "calendar", "srYear": int(year)})
    url = f"{CALENDAR_BASE_URL}?{query}"
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def extract_calendar_events_from_html(html: str) -> pd.DataFrame:
    match = re.search(r"var\s+calData\s*=\s*(\[.*?\]);", html, flags=re.S)
    if not match:
        raise ValueError("Could not find calData in Yeungnam calendar HTML.")

    events = json.loads(match.group(1))
    rows = []
    for event in events:
        if not event.get("startDt") or not event.get("endDt"):
            continue
        rows.append(
            {
                "start_date": pd.to_datetime(event["startDt"]),
                "end_date": pd.to_datetime(event["endDt"]),
                "event_text": str(event.get("text") or "").strip(),
                "source_article_no": event.get("articleNo"),
            }
        )
    return pd.DataFrame(rows).sort_values(["start_date", "end_date", "event_text"]).reset_index(
        drop=True,
    )


def build_academic_daily_features(
    events: pd.DataFrame,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    normalized = _normalize_events(events)
    result = pd.DataFrame(
        {
            "date": pd.date_range(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                freq="D",
            )
        }
    )
    for column in ACADEMIC_FEATURE_COLUMNS:
        result[column] = 0
    result["academic_days_since_semester_start"] = 999
    result["academic_days_to_final"] = 999

    _mark_event_ranges(result, normalized, "중간시험", "academic_is_midterm")
    _mark_event_ranges(result, normalized, "기말시험", "academic_is_final")
    _mark_event_ranges(result, normalized, "공휴일수업대체지정일", "academic_is_makeup_class_day")
    _mark_event_ranges(result, normalized, "강의평가", "academic_is_course_eval")
    _mark_event_ranges(result, normalized, "교육실습", "academic_is_education_practice")

    semester_starts = _event_start_dates(normalized, exact_text="개강")
    vacation_starts = _event_start_dates(normalized, contains_text="방학")
    final_starts = _event_start_dates(normalized, contains_text="기말시험")

    for index, row in result.iterrows():
        current = row["date"]
        past_semesters = [day for day in semester_starts if day <= current]
        past_vacations = [day for day in vacation_starts if day <= current]
        latest_semester = max(past_semesters) if past_semesters else None
        latest_vacation = max(past_vacations) if past_vacations else None

        in_vacation = latest_vacation is not None and (
            latest_semester is None or latest_vacation > latest_semester
        )
        in_instruction = latest_semester is not None and not in_vacation
        result.at[index, "academic_is_vacation"] = int(in_vacation)
        result.at[index, "academic_is_instruction_period"] = int(in_instruction)
        if latest_semester is not None:
            days_since = int((current - latest_semester).days)
            result.at[index, "academic_days_since_semester_start"] = days_since
            result.at[index, "academic_semester_week"] = max(days_since // 7 + 1, 1)

        future_finals = [day for day in final_starts if day >= current]
        if future_finals:
            result.at[index, "academic_days_to_final"] = int((min(future_finals) - current).days)

    return result[["date", *ACADEMIC_FEATURE_COLUMNS]]


def save_academic_calendar_outputs(
    years: list[int],
    event_output_path: Path,
    daily_output_path: Path,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.concat(
        [extract_calendar_events_from_html(fetch_calendar_html(year)) for year in years],
        ignore_index=True,
    ).drop_duplicates(subset=["start_date", "end_date", "event_text"])
    daily = build_academic_daily_features(events, start_date=start_date, end_date=end_date)
    event_output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(event_output_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_output_path, index=False, encoding="utf-8-sig")
    return events, daily


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["start_date"] = pd.to_datetime(result["start_date"])
    result["end_date"] = pd.to_datetime(result["end_date"])
    result["event_text"] = result["event_text"].fillna("").astype(str)
    return result


def _mark_event_ranges(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    contains_text: str,
    column: str,
) -> None:
    matches = events[events["event_text"].str.contains(contains_text, regex=False, na=False)]
    for _, event in matches.iterrows():
        mask = (daily["date"] >= event["start_date"]) & (daily["date"] <= event["end_date"])
        daily.loc[mask, column] = 1


def _event_start_dates(
    events: pd.DataFrame,
    contains_text: str | None = None,
    exact_text: str | None = None,
) -> list[pd.Timestamp]:
    if exact_text is not None:
        matches = events[events["event_text"] == exact_text]
    elif contains_text is not None:
        matches = events[events["event_text"].str.contains(contains_text, regex=False, na=False)]
    else:
        matches = events.iloc[0:0]
    return sorted(pd.to_datetime(matches["start_date"]).tolist())
