# CEMS 기준 모델 구현 계획

> **작업자 지침:** 이 계획을 구현할 때는 `superpowers:subagent-driven-development` 또는 `superpowers:executing-plans` 흐름을 사용해 작업 단위별로 진행한다. 진행 상황은 체크박스(`- [ ]`)로 추적한다.

**목표:** 전력 데이터만 사용하는 Colab 우선 기준 모델을 만들고, 데이터 확인부터 프로젝트 적용 준비까지의 기록 체계를 갖춘다.

**구조:** `10_baseline_ready_panel_primary_reliable.csv`를 단일 메인 패널로 사용한다. 예측 계산식과 성능 지표 계산은 노트북 함수로 작성해, 이후 `modeling/baseline_model.py`로 옮겨도 동작이 바뀌지 않게 한다.

**기술 스택:** Python, pandas, numpy, matplotlib, Colab notebook, CSV outputs.

## 전체 제약 조건

- 1차 기준 모델은 전력 사용량 데이터만 사용한다.
- 메인 입력 파일은 `10_baseline_ready_panel_primary_reliable.csv`다.
- 타깃 컬럼은 `usage_kwh_clean`이다.
- 검증 행은 `is_validation_target_clean == True` 조건을 사용한다.
- 기본 검증 구간은 `2026-04-01`부터 `2026-06-23`까지다.
- `2026-06-24` 부분 일자는 평가에서 제외한다.
- `2026-06-25`부터 `2026-06-30`까지의 placeholder 날짜는 평가에서 제외한다.
- 모델 산출물은 `outputs/` 아래에 저장한다.
- 1차 기준 모델에서는 scikit-learn을 필수 의존성으로 두지 않는다.

---

## 파일 구조

- 생성: `cems_baseline_model_colab.ipynb`
  - 데이터 로드, 기준 모델 예측, 성능 지표 계산, 오차 시각화, CSV 저장을 수행하는 Colab 노트북.
- 생성: `outputs/`
  - 예측값과 성능 지표 CSV 파일을 저장하는 폴더.
- 수정: `docs/03_experiment_log.md`
  - 노트북 실행 이후 완료된 실행 정보를 추가한다.
- 생성: `docs/runs/2026-06-28-baseline-v1.md`
  - 첫 완료 실행의 상세 기록을 남긴다.

### 작업 1: Colab 노트북 뼈대 만들기

**파일:**
- 생성: `cems_baseline_model_colab.ipynb`

**인터페이스:**
- 입력: `10_baseline_ready_panel_primary_reliable.csv`
- 출력: `load_panel`, `add_baseline_predictions`, `calculate_metrics`, `save_outputs` 함수가 포함된 노트북 섹션

- [ ] **1단계: 노트북 섹션 만들기**

다음 제목을 사용한다.

```markdown
# CEMS 전력 데이터 기반 기준 모델
## 1. 환경 설정
## 2. 데이터 로드
## 3. 스키마와 검증 구간 확인
## 4. 기준 모델 예측 함수
## 5. 성능 지표 계산 함수
## 6. 성능 평가
## 7. 오차 분석
## 8. 산출물 저장
## 9. 다음 모델링 단계
```

- [ ] **2단계: 기본 import 추가**

다음 코드를 사용한다.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

- [ ] **3단계: Colab 파일 업로드 fallback 추가**

다음 코드를 사용한다.

```python
DATA_PATH = Path("10_baseline_ready_panel_primary_reliable.csv")

if not DATA_PATH.exists():
    try:
        from google.colab import files
        uploaded = files.upload()
        DATA_PATH = Path(next(iter(uploaded.keys())))
    except ModuleNotFoundError:
        raise FileNotFoundError("10_baseline_ready_panel_primary_reliable.csv 파일을 노트북과 같은 위치에 두세요.")
```

### 작업 2: 데이터 확인 로직 추가

**파일:**
- 수정: `cems_baseline_model_colab.ipynb`

**인터페이스:**
- 입력: `DATA_PATH`
- 출력: `panel` 데이터프레임

- [ ] **1단계: 데이터 로드 함수 추가**

```python
def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel
```

- [ ] **2단계: 필수 컬럼과 데이터 형태 확인 추가**

```python
required_columns = {
    "date",
    "report_month",
    "building_name_recent",
    "usage_kwh_clean",
    "is_validation_target_clean",
    "profile_monthly_kwh_mean",
    "calendar_days_in_month",
    "school_total_day_weight_observed_month",
}
missing_columns = required_columns - set(panel.columns)
assert not missing_columns, f"필수 컬럼이 없습니다: {sorted(missing_columns)}"
assert panel["building_name_recent"].nunique() == 17
assert panel["date"].min().strftime("%Y-%m-%d") == "2026-04-01"
assert panel["date"].max().strftime("%Y-%m-%d") == "2026-06-30"
```

### 작업 3: 기준 모델 예측값 추가

**파일:**
- 수정: `cems_baseline_model_colab.ipynb`

**인터페이스:**
- 입력: `panel` 데이터프레임
- 출력: `pred_uniform_daily_kwh`, `pred_school_shape_kwh` 컬럼이 추가된 데이터프레임

- [ ] **1단계: 예측 함수 추가**

```python
def add_baseline_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["pred_uniform_daily_kwh"] = (
        result["profile_monthly_kwh_mean"] / result["calendar_days_in_month"]
    )
    result["pred_school_shape_kwh"] = (
        result["profile_monthly_kwh_mean"] * result["school_total_day_weight_observed_month"]
    )
    return result
```

- [ ] **2단계: 예측값 음수 여부 확인 추가**

```python
prediction_columns = ["pred_uniform_daily_kwh", "pred_school_shape_kwh"]
assert (panel[prediction_columns] >= 0).all().all()
```

### 작업 4: 성능 지표와 평가표 추가

**파일:**
- 수정: `cems_baseline_model_colab.ipynb`

**인터페이스:**
- 입력: 예측 컬럼이 추가된 데이터프레임
- 출력: 성능 지표 테이블

- [ ] **1단계: 성능 지표 계산 함수 추가**

```python
def calculate_metrics(frame: pd.DataFrame, pred_col: str, group_cols: list[str]) -> pd.DataFrame:
    scored = frame.dropna(subset=["usage_kwh_clean", pred_col]).copy()
    scored["error"] = scored[pred_col] - scored["usage_kwh_clean"]
    scored["abs_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["ape"] = np.where(
        scored["usage_kwh_clean"] > 0,
        scored["abs_error"] / scored["usage_kwh_clean"],
        np.nan,
    )

    rows = []
    for keys, group in scored.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "prediction_method": pred_col,
            "n_rows": len(group),
            "actual_sum_kwh": group["usage_kwh_clean"].sum(),
            "pred_sum_kwh": group[pred_col].sum(),
            "mae_kwh": group["abs_error"].mean(),
            "rmse_kwh": np.sqrt(group["squared_error"].mean()),
            "mape": group["ape"].mean(),
            "bias_kwh_mean": group["error"].mean(),
            "bias_pct_sum": (
                group["error"].sum() / group["usage_kwh_clean"].sum()
                if group["usage_kwh_clean"].sum() else np.nan
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **2단계: 검증 데이터프레임 만들기**

```python
validation = panel[panel["is_validation_target_clean"].astype(str).str.lower() == "true"].copy()
```

- [ ] **3단계: 건물별, 월별 성능표 만들기**

```python
metric_frames = []
for pred_col in ["pred_uniform_daily_kwh", "pred_school_shape_kwh"]:
    metric_frames.append(calculate_metrics(validation, pred_col, ["building_name_recent"]))
metrics_by_building = pd.concat(metric_frames, ignore_index=True)

metric_frames = []
for pred_col in ["pred_uniform_daily_kwh", "pred_school_shape_kwh"]:
    metric_frames.append(calculate_metrics(validation, pred_col, ["report_month"]))
metrics_by_month = pd.concat(metric_frames, ignore_index=True)
```

### 작업 5: 결과 저장과 실행 기록 작성

**파일:**
- 수정: `cems_baseline_model_colab.ipynb`
- 생성: `outputs/baseline_predictions.csv`
- 생성: `outputs/baseline_metrics_by_building.csv`
- 생성: `outputs/baseline_metrics_by_month.csv`
- 생성: `outputs/baseline_error_rankings.csv`
- 수정: `docs/03_experiment_log.md`
- 생성: `docs/runs/2026-06-28-baseline-v1.md`

**인터페이스:**
- 입력: 예측 데이터프레임과 성능 지표 데이터프레임
- 출력: 재사용 가능한 CSV 산출물과 실행 기록 문서

- [ ] **1단계: 산출물 저장 함수 추가**

```python
def save_outputs(panel: pd.DataFrame, validation: pd.DataFrame, metrics_by_building: pd.DataFrame, metrics_by_month: pd.DataFrame) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    panel.to_csv(output_dir / "baseline_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_by_building.to_csv(output_dir / "baseline_metrics_by_building.csv", index=False, encoding="utf-8-sig")
    metrics_by_month.to_csv(output_dir / "baseline_metrics_by_month.csv", index=False, encoding="utf-8-sig")

    ranked = validation.copy()
    ranked["abs_error_school_shape"] = (
        ranked["pred_school_shape_kwh"] - ranked["usage_kwh_clean"]
    ).abs()
    ranked = ranked.sort_values("abs_error_school_shape", ascending=False)
    ranked.head(100).to_csv(output_dir / "baseline_error_rankings.csv", index=False, encoding="utf-8-sig")
```

- [ ] **2단계: 실행 기록 문서 업데이트**

노트북 출력에서 실제 성능 지표 값을 확인한 뒤 `docs/03_experiment_log.md`에 완료된 실행을 추가하고, `docs/runs/2026-06-28-baseline-v1.md`를 작성한다.
