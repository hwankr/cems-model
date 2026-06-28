# 2026-06-28 weather-school-shape-v1 실행 기록

## 목적

기상 데이터를 추가하되, 2026년 4월 이후 검증 구간에서 실제 기상값이나 당일 이후 정보를 입력으로 쓰지 않는 모델 경로를 만든다.

## 모델 구조

1. 2026-03-31까지의 학교 전체 일별 전력 사용량을 학습 타깃으로 사용한다.
2. 건물별 월 프로파일을 일할해 날짜별 건물 기준 사용량을 만든다.
3. 학교 전체 월내 일별 사용 비중(`school_total_day_weight_observed_month`)을 월 일수로 환산한 `school_multiplier`를 학습한다.
4. 학습 feature는 날짜, 요일, 주말 여부, 이전 학교 multiplier lag, 실제 일별 기상 feature다.
5. 2026-04-01 이후 예측 feature는 대상일 00:00 KST 이전에 발표된 예보 기상 feature만 사용한다.
6. 예측된 학교 multiplier를 건물별 월 프로파일 기준 사용량에 곱해 건물별 일별 예측값을 만든다.

## 누수 방지 조건

- 2026년 4월 이후 건물 일별 사용량은 학습에 쓰지 않는다.
- `school_total_daily_kwh`, `school_total_day_weight_observed_month` 같은 4월 이후 당일 학교 전체 실측 흐름은 예측 입력으로 쓰지 않는다.
- 학교 전체 lag feature는 대상일 이전 날짜의 값만 사용한다.
- 4월 이후 실제 기상값은 메인 실험 입력으로 쓰지 않는다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `modeling/weather_features.py` | 실제/예보 기상 feature 정규화 |
| `modeling/kma_weather_api.py` | `.env` 로더, KMA 격자 변환, APIHub URL helper |
| `modeling/weather_school_model.py` | 학교 multiplier 학습, 건물별 배분, 성능 산출 |
| `scripts/fetch_kma_weather.py` | KMA 실제/예보 기상 CSV 수집 |
| `scripts/run_weather_school_model.py` | weather-school-shape 모델 실행 |

## 실행 명령

기상 CSV 수집:

```powershell
python scripts/fetch_kma_weather.py actual --start 2025-07-01 --end 2026-06-23
python scripts/fetch_kma_weather.py forecast --start 2026-04-01 --end 2026-06-23 --hour-step 6
```

이 환경에서는 `python` 별칭 대신 번들 Python을 사용한다.

```powershell
& 'C:\Users\fabro\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_weather_school_model.py
```

## 현재 상태

KMA APIHub ASOS/단기예보 API 승인 후 기상 CSV를 생성했고, weather-school-shape 모델 실행까지 완료했다.
이후 기존 LightGBM에 예보 기상 feature를 직접 결합한 `lightgbm-weather-v1`이 더 좋은 성능을 보여, 이 모델은 주력에서 제외하고 비교용 기록으로 남긴다.

생성 파일:

- `outputs/weather_actual_daily.csv`: 2025-07-01부터 2026-06-23까지 358행
- `outputs/weather_forecast_daily.csv`: 2026-04-01부터 2026-06-23까지 84행
- `outputs/weather_school_predictions.csv`: 2026년 4월 이후 17개 건물 일별 예측 1428행
- `outputs/weather_school_model_comparison.csv`
- `outputs/weather_school_metrics_by_month.csv`
- `outputs/weather_school_metrics_by_building.csv`

전체 성능:

| 모델 | MAE | RMSE | WAPE | 총량 bias |
| --- | ---: | ---: | ---: | ---: |
| weather-school-shape-v1 | 162.3123 kWh | 241.1261 kWh | 0.1455 | -2.7459% |

월별 성능:

| 월 | MAE | RMSE | WAPE | 총량 bias |
| --- | ---: | ---: | ---: | ---: |
| 2026-04 | 125.2272 kWh | 175.4137 kWh | 0.1146 | -6.0354% |
| 2026-05 | 169.9515 kWh | 235.4736 kWh | 0.1622 | -10.1522% |
| 2026-06 | 200.3880 kWh | 312.2101 kWh | 0.1620 | 9.5003% |
