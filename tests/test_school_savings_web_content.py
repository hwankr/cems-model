"""
TDD tests for school_savings_web static demo site.
Run: pytest tests/test_school_savings_web_content.py -v
"""

import json
import math
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent
WEB_DIR = BASE / "school_savings_web"
DATA_FILE = WEB_DIR / "data" / "savings.json"

# ---------------------------------------------------------------------------
# 1. File existence checks
# ---------------------------------------------------------------------------

def test_index_html_exists():
    assert (WEB_DIR / "index.html").exists(), "school_savings_web/index.html missing"

def test_styles_css_exists():
    assert (WEB_DIR / "styles.css").exists(), "school_savings_web/styles.css missing"

def test_app_js_exists():
    assert (WEB_DIR / "app.js").exists(), "school_savings_web/app.js missing"

# ---------------------------------------------------------------------------
# 2. index.html content checks
# ---------------------------------------------------------------------------

def _html():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")

def test_html_contains_jeolgam():
    assert "절감" in _html()

def test_html_contains_baseline():
    assert "베이스라인" in _html()

def test_html_contains_area():
    assert "면적당" in _html()

def test_html_contains_leaderboard():
    assert "리더보드" in _html()

def test_html_contains_accuracy():
    assert "정확도" in _html()

def test_html_contains_glossary():
    assert "용어" in _html()

# ---------------------------------------------------------------------------
# 3. app.js content checks
# ---------------------------------------------------------------------------

def _js():
    return (WEB_DIR / "app.js").read_text(encoding="utf-8")

def test_js_fetches_savings_json():
    assert "data/savings.json" in _js()

def test_js_references_series():
    assert "series" in _js()

def test_js_references_daily():
    assert "daily" in _js()

def test_js_references_leaderboard():
    assert "leaderboard" in _js()

def test_js_references_accuracy():
    assert "accuracy" in _js()

def test_js_references_scorecard():
    assert "scorecard" in _js()

def test_js_references_avoided():
    assert "avoided" in _js()

def test_js_references_p50():
    assert "p50" in _js()

def test_js_references_p90():
    assert "p90" in _js()

def test_js_references_glossary():
    assert "glossary" in _js()

# ---------------------------------------------------------------------------
# 4. JSON contract tests
# ---------------------------------------------------------------------------

def _data():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)

def test_json_top_level_keys():
    d = _data()
    required = {"meta", "accuracy", "scorecard", "series", "daily", "leaderboard", "glossary"}
    missing = required - set(d.keys())
    assert not missing, f"Missing top-level keys: {missing}"

def test_json_series_length():
    d = _data()
    assert len(d["series"]) == 480, f"Expected 480 series rows, got {len(d['series'])}"

def test_json_daily_length():
    d = _data()
    assert len(d["daily"]) == 20, f"Expected 20 daily rows, got {len(d['daily'])}"

def test_json_leaderboard_length():
    d = _data()
    assert len(d["leaderboard"]) == 3, f"Expected 3 leaderboard rows, got {len(d['leaderboard'])}"

def test_json_series_keys():
    d = _data()
    expected = {
        "timestamp", "report_date", "hour", "actual",
        "p10", "p50", "p90", "avoided_kwh", "avoided_pct",
        "is_confirmed_saving", "is_overuse"
    }
    actual = set(d["series"][0].keys())
    assert actual == expected, f"series[0] keys mismatch. Extra: {actual - expected}, Missing: {expected - actual}"

def test_json_leaderboard_keys():
    d = _data()
    expected = {
        "round", "days", "baseline_kwh", "actual_kwh",
        "avoided_kwh", "avoided_pct", "avoided_per_sqm_kwh", "rank"
    }
    actual = set(d["leaderboard"][0].keys())
    assert actual == expected, f"leaderboard[0] keys mismatch. Got: {actual}"

def test_json_series_values_are_finite():
    d = _data()
    for i, row in enumerate(d["series"]):
        for field in ("actual", "p10", "p50", "p90", "avoided_kwh"):
            val = row[field]
            assert math.isfinite(val), f"series[{i}].{field} = {val} is not finite"

def test_json_glossary_has_nine_entries():
    d = _data()
    assert len(d["glossary"]) == 9, f"Expected 9 glossary entries, got {len(d['glossary'])}"

def test_json_glossary_keys():
    d = _data()
    for i, entry in enumerate(d["glossary"]):
        assert "term" in entry, f"glossary[{i}] missing 'term'"
        assert "desc" in entry, f"glossary[{i}] missing 'desc'"

def test_json_scorecard_keys():
    d = _data()
    sc = d["scorecard"]
    required = {
        "baseline_sum_kwh", "actual_sum_kwh", "avoided_sum_kwh",
        "avoided_pct", "avoided_per_sqm_kwh", "confirmed_saving_hours",
        "overuse_hours", "n_rows", "heldout_wape", "heldout_coverage",
        "area_sqm", "avoided_pct_display"
    }
    missing = required - set(sc.keys())
    assert not missing, f"scorecard missing keys: {missing}"

def test_json_accuracy_has_baselines():
    d = _data()
    acc = d["accuracy"]
    assert "baselines" in acc
    assert len(acc["baselines"]) >= 2
    assert acc["baselines"][0]["model"] == "frozen P50"
    assert acc["baselines"][1]["model"] == "naive_same_dow_hour_profile"
