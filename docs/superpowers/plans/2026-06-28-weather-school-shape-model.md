# Weather School Shape Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe weather model that trains only on pre-2026-04 school-wide daily demand and building monthly profiles, then evaluates 2026-04 onward building daily usage with forecast weather.

**Architecture:** Weather ingestion is separated from modeling. `modeling/weather_features.py` normalizes actual and forecast weather into daily feature tables. `modeling/weather_school_model.py` trains a school-level daily multiplier model and allocates predictions to buildings using fixed monthly profile bases.

**Tech Stack:** Python, pandas, numpy, LightGBM, standard-library HTTP/XML/CSV utilities.

## Global Constraints

- Do not train on 2026-04 onward building daily labels.
- For target date `D`, the main forecast experiment may use only forecasts issued before `D 00:00` KST.
- Historical training weather can use actual daily observations through `2026-03-31`.
- 2026-04 onward actual weather is allowed only in the oracle comparison, not the primary forecast experiment.
- KMA credentials must be read from `.env` or the environment and never printed.
- Keep `.env` ignored by Git.

---

### Task 1: Weather Feature Normalization

**Files:**
- Create: `tests/test_weather_features.py`
- Create: `modeling/weather_features.py`

**Interfaces:**
- Produces: `normalize_actual_weather(frame)`, `select_forecast_snapshots(frame, cutoff_hour=0)`, `aggregate_forecast_weather(frame)`, `add_degree_day_features(frame, base_temp_c=18.0)`.

- [x] Write failing tests for actual weather feature normalization, forecast cutoff selection, and daily aggregation.
- [x] Run `python -m unittest tests.test_weather_features -v` and confirm failures are missing imports/functions.
- [x] Implement the minimal weather feature functions.
- [x] Rerun `python -m unittest tests.test_weather_features -v` and confirm pass.

### Task 2: School Multiplier Model

**Files:**
- Create: `tests/test_weather_school_model.py`
- Create: `modeling/weather_school_model.py`

**Interfaces:**
- Consumes: normalized weather feature frames from Task 1.
- Produces: `build_school_training_frame(...)`, `build_weather_school_predictions(...)`, `calculate_weather_model_metrics(...)`, `run_weather_school_model(...)`.

- [x] Write failing tests proving training rows stop before 2026-04, forecast rows use `weather_source=forecast`, and building predictions sum to the predicted school total.
- [x] Run `python -m unittest tests.test_weather_school_model -v` and confirm failures.
- [x] Implement the school multiplier model, using LightGBM when available and a deterministic median fallback for tests.
- [x] Rerun model tests and confirm pass.

### Task 3: KMA Weather Fetching

**Files:**
- Create: `scripts/fetch_kma_weather.py`
- Modify: `requirements.txt` only if a new dependency is unavoidable.

**Interfaces:**
- Produces CLI outputs `outputs/weather_actual_daily.csv` and `outputs/weather_forecast_daily.csv`.

- [x] Add tests through existing weather normalization functions instead of hitting the network.
- [x] Implement `.env` loading without printing secrets.
- [x] Implement KMA APIHub ASOS daily actual and short-term forecast archive fetch commands with explicit endpoint URLs and safe error messages.
- [x] If the API key is not authorized, fail with an actionable message and keep the model runnable from CSV inputs.

### Task 4: Runner and Documentation

**Files:**
- Create: `scripts/run_weather_school_model.py`
- Modify: `docs/03_experiment_log.md`
- Create: `docs/runs/2026-06-28-weather-school-shape-v1.md`
- Modify: `outputs/README.md`

**Interfaces:**
- CLI accepts `--actual-weather`, `--forecast-weather`, `--output-dir`, and writes comparison CSVs.

- [x] Implement a runner around `run_weather_school_model`.
- [x] Run unit tests.
- [x] Run the model if weather CSVs exist.
- [x] Document the leakage contract, required weather files, and API key setup.
