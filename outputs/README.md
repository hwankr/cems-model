# 기준 모델 산출물

이 폴더에는 기준 모델 노트북과 스크립트에서 생성한 결과 파일을 저장합니다. CSV는 직접 수정하지 않고, 입력 데이터나 모델 코드가 바뀌면 스크립트를 다시 실행해 재생성합니다.

## 주요 파일

| 파일 | 내용 |
| --- | --- |
| `baseline_predictions.csv` | 하루-건물 단위 실제값, 기준 예측값, 검증 플래그, 오차 |
| `baseline_metrics_by_building.csv` | 기존 기준 모델의 건물별 MAE, RMSE, MAPE, bias |
| `baseline_metrics_by_month.csv` | 기존 기준 모델의 월별 MAE, RMSE, MAPE, bias |
| `baseline_error_rankings.csv` | 오차 분석을 위한 상위 오차 행 |
| `ml_baseline_predictions.csv` | LightGBM 기본형, weather, weather+area, weather+academic, weather+area+academic 후보의 일별 예측 |
| `ml_baseline_model_comparison.csv` | 모든 기준 모델과 LightGBM 후보의 전체 성능 비교 |
| `ml_baseline_champion.json` | 현재 champion 모델과 예측 컬럼명. 웹 대시보드가 이 파일을 읽어 기본 모델을 선택 |
| `ml_baseline_metrics_by_month.csv` | 모델별 월별 성능 |
| `ml_baseline_metrics_by_building.csv` | 모델별 건물별 성능 |
| `ml_baseline_coverage.csv` | LightGBM 후보별 80%, 90% 예측구간 coverage |
| `ml_baseline_pinball_loss.csv` | LightGBM 후보별 분위수 pinball loss |
| `ml_baseline_residual_correlations.csv` | LightGBM 잔차와 주요 feature의 상관 |
| `ml_baseline_group_bias_by_building.csv` | champion 기준 건물별 bias 진단 |
| `weather_actual_daily.csv` | KMA 관측 기상값을 일별 feature로 정규화한 파일 |
| `weather_forecast_daily.csv` | 예측일 전날까지 발표된 KMA 예보만 선택해 만든 일별 예보 feature |
| `weather_school_predictions.csv` | 비교용 school-shape 모델의 건물별 일별 예측 |
| `weather_school_model_comparison.csv` | weather school-shape 모델의 전체 성능 |
| `weather_school_metrics_by_month.csv` | weather school-shape 모델의 월별 성능 |
| `weather_school_metrics_by_building.csv` | weather school-shape 모델의 건물별 성능 |
| `yu_academic_calendar_events.csv` | 영남대 학사일정 페이지에서 추출한 2024-2026 이벤트 |
| `yu_academic_calendar_daily_features.csv` | 학사일정 이벤트를 일별 coarse feature로 변환한 파일 |

## 현재 champion

`ml_baseline_champion.json` 기준 현재 champion은 `pred_lightgbm_weather_area_q50_kwh`입니다.

| 모델 | MAE | RMSE | WAPE | 총량 bias |
| --- | ---: | ---: | ---: | ---: |
| LightGBM + weather + area q50 | 131.46 kWh | 221.95 kWh | 11.82% | -0.23% |

학사일정 feature는 생성해 후보로 비교했지만 현재 검증에서는 weather+area보다 WAPE가 나빠 champion에서 제외했습니다.
