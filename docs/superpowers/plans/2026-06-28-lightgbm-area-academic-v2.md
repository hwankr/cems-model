# LightGBM Area Academic V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add building-area and coarse academic-calendar feature candidates to the current LightGBM weather pipeline, then select the best validated champion model automatically.

**Architecture:** Keep the existing LightGBM walk-forward structure. Add focused feature modules for building area and academic calendar extraction, then train separate LightGBM variants for weather, weather+area, weather+academic, and weather+area+academic. Outputs include all candidate metrics plus a champion JSON selected by lowest WAPE.

**Tech Stack:** Python, pandas, LightGBM, unittest, PowerShell, static HTML/JS dashboard.

## Global Constraints

- Do not use 2026-04 onward actual weather as model input.
- Do not add raw academic event text one-hot features to the model.
- Use only coarse academic operation-state features.
- Keep existing LightGBM baseline outputs available for comparison.
- Building area source is `inputs/yu_building_area.xlsx`, copied from the user-provided workbook.
- Academic calendar source is the Yeungnam University calendar page for 2024-2026.

---

### Task 1: Academic Calendar Features

**Files:**
- Create: `modeling/academic_calendar_features.py`
- Create: `tests/test_academic_calendar_features.py`
- Create: `scripts/fetch_yu_academic_calendar.py`

**Interfaces:**
- Produces: `extract_calendar_events_from_html(html)`, `build_academic_daily_features(events, start_date, end_date)`, `fetch_calendar_html(year)`.

- [x] Write tests for parsing `calData` JSON and generating coarse daily features.
- [x] Run `python -m unittest tests.test_academic_calendar_features -v` and confirm missing-module failure.
- [x] Implement parser, feature builder, and fetch script.
- [x] Rerun academic tests and confirm pass.

### Task 2: Building Area Features

**Files:**
- Create: `modeling/building_area_features.py`
- Create: `tests/test_building_area_features.py`

**Interfaces:**
- Produces: `build_building_area_table(raw_area_rows)`, `add_building_area_features(panel, area_features)`, `load_building_area_features(path)`.

- [x] Write tests for floor-level aggregation, alias matching, missing-area fallback, and intensity features.
- [x] Run `python -m unittest tests.test_building_area_features -v` and confirm missing-module failure.
- [x] Implement the feature module.
- [x] Rerun building area tests and confirm pass.

### Task 3: Candidate LightGBM Variants

**Files:**
- Modify: `modeling/ml_baseline_model.py`
- Modify: `tests/test_ml_baseline_model.py`
- Modify: `scripts/run_ml_baseline_model.py`

**Interfaces:**
- Produces prediction columns `pred_lightgbm_weather_area_q50_kwh`, `pred_lightgbm_weather_academic_q50_kwh`, and `pred_lightgbm_weather_area_academic_q50_kwh`.
- Produces: `outputs/ml_baseline_champion.json`.

- [x] Write tests proving area and academic files add candidate model columns.
- [x] Run ML tests and confirm expected failures.
- [x] Implement candidate variant training and champion selection.
- [x] Rerun ML tests and confirm pass.

### Task 4: Data Generation, Docs, and Web

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `docs/03_experiment_log.md`
- Create: `docs/runs/2026-06-28-lightgbm-area-academic-v2.md`
- Modify: `outputs/README.md`

**Interfaces:**
- Dashboard loads `ml_baseline_champion.json` and defaults to the champion prediction column.

- [x] Fetch 2024-2026 academic calendar data.
- [x] Run `scripts/run_ml_baseline_model.py`.
- [x] Update docs and web text with actual champion metrics.
- [x] Run full tests, JS check, model run, and browser render verification.
