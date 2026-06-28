# 기준 모델 산출물

이 폴더에는 기준 모델 노트북 또는 스크립트에서 생성한 파일을 저장한다.

예상 파일:

| 파일 | 내용 |
| --- | --- |
| `baseline_predictions.csv` | 행 단위 실제값, 예측값, 검증 플래그, 잔차 |
| `baseline_metrics_by_building.csv` | 건물별 MAE, RMSE, MAPE, bias |
| `baseline_metrics_by_month.csv` | 월별 MAE, RMSE, MAPE, bias |
| `baseline_error_rankings.csv` | 오차 분석을 위한 상위 오차 행 |

생성된 CSV 파일은 직접 수정하지 않는다. 결과가 바뀌어야 한다면 노트북이나 스크립트를 다시 실행하고 `docs/runs/` 아래에 실행 기록을 남긴다.
