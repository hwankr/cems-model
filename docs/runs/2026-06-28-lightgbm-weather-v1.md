# 2026-06-28 lightgbm-weather-v1 실행 기록

## 목적

기존 LightGBM walk-forward 모델을 버리지 않고, KMA 단기예보 기반 기상 feature를 직접 추가해 더 고도화된 모델을 만든다.

## 모델 구조

1. 기존 `ml_baseline_model.py`의 LightGBM q50 walk-forward 구조를 유지한다.
2. 기존 feature인 날짜 정보, 월 프로파일, 최근 사용량 lag, 요일/최근사용량 휴리스틱 예측값을 그대로 사용한다.
3. 2026년 4월 이후 검증 구간에는 실제 기상값이 아니라 대상일 00:00 KST 이전에 발표된 예보 feature만 붙인다.
4. 짧은 학습 구간에서 과적합을 줄이기 위해 학습 feature에는 `weather_cdd_18`, `weather_hdd_18`, `weather_rainfall_mm`, `weather_is_rainy`를 사용한다.
5. 기존 LightGBM 예측 컬럼은 유지하고, 기상 결합 모델은 `pred_lightgbm_weather_q50_kwh` 및 분위수 컬럼으로 별도 저장한다.

## 실행 명령

```powershell
& 'C:\Users\fabro\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_ml_baseline_model.py
```

기본 입력:

- `10_baseline_ready_panel_primary_reliable.csv`
- `outputs/weather_actual_daily.csv`
- `outputs/weather_forecast_daily.csv`

주요 출력:

- `outputs/ml_baseline_predictions.csv`
- `outputs/ml_baseline_model_comparison.csv`
- `outputs/ml_baseline_metrics_by_month.csv`
- `outputs/ml_baseline_metrics_by_building.csv`
- `outputs/ml_baseline_coverage.csv`
- `outputs/ml_baseline_pinball_loss.csv`

## 전체 성능

| 모델 | MAE | RMSE | WAPE | 총량 bias |
| --- | ---: | ---: | ---: | ---: |
| 기존 LightGBM q50 | 134.3123 kWh | 228.3113 kWh | 0.1208 | -0.5257% |
| LightGBM + weather q50 | 131.9848 kWh | 221.8092 kWh | 0.1187 | -0.4063% |

## 월별 성능

| 월 | 기존 LightGBM MAE | LightGBM + weather MAE | 기존 WAPE | weather WAPE |
| --- | ---: | ---: | ---: | ---: |
| 2026-04 | 78.3220 kWh | 75.3471 kWh | 0.0741 | 0.0713 |
| 2026-05 | 139.3672 kWh | 136.9551 kWh | 0.1330 | 0.1307 |
| 2026-06 | 166.4489 kWh | 164.6859 kWh | 0.1345 | 0.1331 |

## 판정

`weather-school-shape-v1`은 기존 LightGBM보다 낮은 성능이라 주력에서 제외한다. 현재 주력 고도화 모델은 `LightGBM + weather q50`이다.
