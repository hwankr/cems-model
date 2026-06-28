# 2026-06-28 lightgbm-area-academic-v2 실행 기록

## 목적

현재 LightGBM + weather 모델에 건물 면적 데이터와 학사일정 feature를 추가 후보로 붙여, 실제 실시간 운영 가정에서 성능이 개선되는 조합만 champion으로 선택한다.

## 데이터 소스

- 전력 패널: `10_baseline_ready_panel_primary_reliable.csv`
- 예보 기상: `outputs/weather_forecast_daily.csv`
- 건물 면적: `inputs/yu_building_area.xlsx`
- 학사일정 원천: `https://www.yu.ac.kr/main/bachelor/calendar.do?mode=calendar&srYear=2026`
- 학사일정 산출: `outputs/yu_academic_calendar_events.csv`, `outputs/yu_academic_calendar_daily_features.csv`

## feature 후보

| 후보 | 추가 feature | 판단 |
| --- | --- | --- |
| weather | 냉방도일, 난방도일, 강수량, 강수 여부 | 기존 v1 기준 |
| weather+area | 전용면적, 공용면적, 총면적, 호실수, 층수, 면적당 월 사용량 | champion |
| weather+academic | 수업 기간, 방학, 중간/기말, 보강일, 강의평가, 학기 주차 | champion 제외 |
| weather+area+academic | area와 academic 동시 사용 | champion 제외 |

학사일정은 과거 일정을 그대로 외우는 방향의 과적합을 피하기 위해 세부 행사명을 직접 쓰지 않고 coarse flag와 거리형 feature만 사용했다.

## 실행 명령

```powershell
& 'C:\Users\fabro\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/fetch_yu_academic_calendar.py
& 'C:\Users\fabro\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_ml_baseline_model.py
```

## 전체 성능

검증 구간은 2026-04-15부터 2026-06-23까지이며, 17개 건물 1,190행이다.

| 모델 | MAE | RMSE | WAPE | 총량 bias |
| --- | ---: | ---: | ---: | ---: |
| LightGBM q50 | 134.31 kWh | 228.31 kWh | 12.08% | -0.53% |
| LightGBM + weather q50 | 131.98 kWh | 221.81 kWh | 11.87% | -0.41% |
| LightGBM + weather + area q50 | 131.46 kWh | 221.95 kWh | 11.82% | -0.23% |
| LightGBM + weather + academic q50 | 132.90 kWh | 224.16 kWh | 11.95% | -0.17% |
| LightGBM + weather + area + academic q50 | 133.72 kWh | 225.38 kWh | 12.02% | -0.27% |

## 결정

현재 champion은 `lightgbm_weather_area_quantile_q50_walk_forward`이고 예측 컬럼은 `pred_lightgbm_weather_area_q50_kwh`이다.

학사일정 feature는 생성과 검증은 완료했지만 현재 데이터에서는 WAPE가 악화되어 champion에서 제외했다. 다만 향후 1년 이상 일별 실측이 쌓이면 학사일정 효과를 다시 검증할 수 있도록 산출 파이프라인은 유지한다.

## 산출물

- `outputs/ml_baseline_predictions.csv`
- `outputs/ml_baseline_model_comparison.csv`
- `outputs/ml_baseline_champion.json`
- `outputs/ml_baseline_coverage.csv`
- `outputs/ml_baseline_pinball_loss.csv`
- `outputs/ml_baseline_group_bias_by_building.csv`
- `web/index.html`
- `web/app.js`
