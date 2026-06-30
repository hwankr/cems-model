# School Hourly Model Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate school-wide hourly electricity forecasting model and a separate static web viewer for predictions, errors, and validation metrics.

**Architecture:** Add a new `modeling/school_hourly_model.py` module that consumes `school_power_usage_split/ml_ready/power_usage_1hour_ml.csv`, creates leakage-safe lag/calendar features, trains a LightGBM regression model, and writes CSV/JSON outputs under `outputs/`. Add a separate `school_hourly_web/` static viewer that reads those outputs without touching the existing building-level `web/` dashboard.

**Tech Stack:** Python, pandas, numpy, LightGBM with a median fallback, unittest, static HTML/CSS/JavaScript.

## Global Constraints

- Official validation window is `2026-06-01 00:00` through `2026-06-29 23:00`.
- `2026-06-30` is excluded from official scoring because the source day is incomplete.
- Training rows must be strictly before `2026-06-01 00:00`.
- The model is an operational one-hour-resolution forecast that may use observations available before the predicted timestamp through shifted lag features.
- Do not modify the existing 17-building model modules or the existing `web/` dashboard.
- Write tests before production model code and verify red-green behavior.

---

### Task 1: Model Interfaces And Feature Tests

**Files:**
- Create: `tests/test_school_hourly_model.py`
- Create: `modeling/school_hourly_model.py`

**Interfaces:**
- Produces: `load_hourly_usage(path: Path) -> pandas.DataFrame`
- Produces: `add_hourly_features(frame: pandas.DataFrame) -> pandas.DataFrame`
- Produces: `split_train_validation(frame: pandas.DataFrame, validation_start: str, validation_end: str) -> tuple[pandas.DataFrame, pandas.DataFrame]`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run tests and confirm import/function failures**
- [ ] **Step 3: Implement the minimal data loading, lag feature, and split code**
- [ ] **Step 4: Run tests and confirm they pass**

### Task 2: Training, Metrics, And Outputs

**Files:**
- Modify: `modeling/school_hourly_model.py`
- Create: `scripts/run_school_hourly_model.py`
- Test: `tests/test_school_hourly_model.py`

**Interfaces:**
- Consumes: Task 1 interfaces
- Produces: `run_school_hourly_model(data_path: Path, output_dir: Path) -> SchoolHourlyRunResult`
- Produces: `outputs/school_hourly_predictions.csv`
- Produces: `outputs/school_hourly_model_comparison.csv`
- Produces: `outputs/school_hourly_metrics_by_hour.csv`
- Produces: `outputs/school_hourly_metrics_by_day.csv`
- Produces: `outputs/school_hourly_top_errors.csv`
- Produces: `outputs/school_hourly_run_summary.json`

- [ ] **Step 1: Write failing output and metrics tests**
- [ ] **Step 2: Run tests and confirm expected missing behavior**
- [ ] **Step 3: Implement baselines, model training, metrics, and output saving**
- [ ] **Step 4: Run tests and model script**

### Task 3: Separate Static Web Viewer

**Files:**
- Create: `school_hourly_web/index.html`
- Create: `school_hourly_web/styles.css`
- Create: `school_hourly_web/app.js`
- Modify: `tests/test_web_content.py`

**Interfaces:**
- Consumes: Task 2 output CSV/JSON paths
- Produces: separate static viewer at `school_hourly_web/index.html`

- [ ] **Step 1: Write failing web-content tests**
- [ ] **Step 2: Run tests and confirm expected missing phrases/files**
- [ ] **Step 3: Implement the static web viewer**
- [ ] **Step 4: Run tests and browser verification through a local static server**

### Task 4: Run Record And Verification

**Files:**
- Create: `docs/runs/2026-06-30-school-hourly-v1.md`

**Interfaces:**
- Consumes: generated metrics and run summary
- Produces: Korean-first run documentation with validation window, input data, champion model, and caveats

- [ ] **Step 1: Generate model outputs**
- [ ] **Step 2: Write run record from actual metrics**
- [ ] **Step 3: Run full verification: unit tests, model script, web content tests, and browser smoke check**
