# 실험 로그

이 파일은 모델 실험 실행 기록의 인덱스입니다. 상세 실행 기록은 `docs/runs/` 아래에 보관합니다.

## 실행 기록

| 날짜 | 실행 ID | 모델 | 입력 파일 | 출력 폴더 | 상태 | 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-28 | baseline-v0-planned | 전력 데이터 기반 기준 모델 | `10_baseline_ready_panel_primary_reliable.csv` | `outputs/` | 계획됨 | Colab 우선 기준 모델 구조를 정의 |
| 2026-06-28 | baseline-v1 | 전력량 기반 기본 검증 모델 | `10_baseline_ready_panel_primary_reliable.csv` | `outputs/` | 참고용 | 학교 전체 일별 관측 흐름을 쓰는 방식은 실시간 예측 기준선에서 제외 |
| 2026-06-28 | baseline-v2-realtime | 실시간 기준 월 프로파일 균등 일할 | `10_baseline_ready_panel_primary_reliable.csv` | `outputs/` | 완료 | 2026년 4월 이후 실제 학교 전체 흐름을 예측 입력에서 제거 |
| 2026-06-28 | weather-school-shape-v1 | 학교 전체 multiplier + 기상 예보 + 건물 월 프로파일 배분 | `13_school_total_daily_clean.csv`, `06_monthly_profile_primary_reliable.csv`, `outputs/weather_*.csv` | `outputs/` | 비교용, 주력 제외 | 전체 MAE 162.31 kWh, WAPE 14.55%. LightGBM보다 낮지 않아 제외 |
| 2026-06-28 | lightgbm-weather-v1 | LightGBM q50 walk-forward + 예보 기상 feature | `10_baseline_ready_panel_primary_reliable.csv`, `outputs/weather_forecast_daily.csv` | `outputs/` | 완료 | MAE 131.98 kWh, WAPE 11.87%. 기존 LightGBM q50보다 개선 |
| 2026-06-28 | lightgbm-area-academic-v2 | LightGBM + 예보 기상 + 건물 면적 + 학사일정 후보 비교 | `10_baseline_ready_panel_primary_reliable.csv`, `outputs/weather_forecast_daily.csv`, `inputs/yu_building_area.xlsx`, `outputs/yu_academic_calendar_daily_features.csv` | `outputs/` | 완료 | champion은 LightGBM + weather + area. MAE 131.46 kWh, WAPE 11.82%, bias -0.23% |

## 의사결정 기록

| 날짜 | 결정 | 이유 |
| --- | --- | --- |
| 2026-06-28 | LightGBM보다 전력 데이터 기반 기준 모델을 먼저 만든다 | 복잡한 모델의 성능을 공정하게 판단하려면 단순 비교 기준이 먼저 필요하다 |
| 2026-06-28 | 실시간 예측 기준선에서는 2026년 4월 이후 학교 전체 일별 관측 흐름을 쓰지 않는다 | 예측 대상 기간 내부의 실제 학교 전체 총량이나 일별 비중을 입력으로 넣으면 데이터 누수가 된다 |
| 2026-06-28 | 기상 모델의 검증 입력은 예측일 전날까지 발표된 예보값만 사용한다 | 실제 운영 시점에서 알 수 없는 실제 기상값을 4월 이후 입력으로 쓰면 과하게 유리한 평가가 된다 |
| 2026-06-28 | 주력 고도화 모델은 LightGBM에 예보 기상 feature를 직접 결합한다 | school multiplier 방식은 LightGBM보다 성능이 낮고, 기존 LightGBM 구조가 더 안정적이다 |
| 2026-06-28 | 건물 면적 feature는 champion 후보로 채택한다 | weather만 쓴 LightGBM 대비 WAPE가 11.87%에서 11.82%로 소폭 개선됐다 |
| 2026-06-28 | 학사일정 feature는 만들고 후보 비교하되 champion에서는 제외한다 | 현재 검증에서는 weather+area보다 WAPE가 나빠졌다. 학사일정은 과적합 위험이 있어 coarse flag만 유지한다 |
