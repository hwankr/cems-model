const DEFAULT_METHOD = "pred_lightgbm_weather_area_q50_kwh";

const state = {
  predictions: [],
  mlPredictions: [],
  monthMetrics: [],
  buildingMetrics: [],
  fairnessModelComparison: [],
  fairnessCoverage: [],
  fairnessPinball: [],
  fairnessResidualCorrelations: [],
  fairnessGroupBias: [],
  champion: null,
  selectedMethod: DEFAULT_METHOD,
  selectedBuilding: "all",
  selectedMonth: "all",
};

const methodLabels = {
  pred_realtime_uniform_daily_kwh: "실시간 기준 월 프로파일 균등 일할",
  pred_weekday_recent_kwh: "요일+최근사용량 모델",
  pred_lightgbm_q50_kwh: "LightGBM q50 walk-forward",
  pred_lightgbm_weather_q50_kwh: "LightGBM + weather 모델",
  pred_lightgbm_weather_area_q50_kwh: "LightGBM + weather + area 모델",
  pred_lightgbm_weather_academic_q50_kwh: "LightGBM + weather + academic 후보",
  pred_lightgbm_weather_area_academic_q50_kwh: "LightGBM + weather + area + academic 후보",
};

const files = {
  predictions: "../outputs/baseline_predictions.csv",
  mlPredictions: "../outputs/ml_baseline_predictions.csv",
  monthMetrics: "../outputs/baseline_metrics_by_month.csv",
  buildingMetrics: "../outputs/baseline_metrics_by_building.csv",
  mlMonthMetrics: "../outputs/ml_baseline_metrics_by_month.csv",
  mlBuildingMetrics: "../outputs/ml_baseline_metrics_by_building.csv",
  fairnessModelComparison: "../outputs/ml_baseline_model_comparison.csv",
  fairnessCoverage: "../outputs/ml_baseline_coverage.csv",
  fairnessPinball: "../outputs/ml_baseline_pinball_loss.csv",
  fairnessResidualCorrelations: "../outputs/ml_baseline_residual_correlations.csv",
  fairnessGroupBias: "../outputs/ml_baseline_group_bias_by_building.csv",
  champion: "../outputs/ml_baseline_champion.json",
};

const textFields = new Set([
  "date",
  "report_month",
  "selection_tier",
  "building_name_recent",
  "building_name_long_term",
  "cleaning_action",
  "replacement_source_day_count",
  "mapping_confidence",
  "review_required",
  "duplicate_group",
  "source_file",
  "source_section",
  "profile_source_months",
  "validation_period",
  "weather_feature_source",
  "weather_issued_at",
  "model",
  "prediction_method",
  "prediction_column",
  "interval",
  "feature",
]);

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindControls();

  try {
    const [
      predictions,
      mlPredictions,
      monthMetrics,
      buildingMetrics,
      mlMonthMetrics,
      mlBuildingMetrics,
      fairnessModelComparison,
      fairnessCoverage,
      fairnessPinball,
      fairnessResidualCorrelations,
      fairnessGroupBias,
      champion,
    ] = await Promise.all([
      loadCsv(files.predictions),
      loadOptionalCsv(files.mlPredictions),
      loadCsv(files.monthMetrics),
      loadCsv(files.buildingMetrics),
      loadOptionalCsv(files.mlMonthMetrics),
      loadOptionalCsv(files.mlBuildingMetrics),
      loadOptionalCsv(files.fairnessModelComparison),
      loadOptionalCsv(files.fairnessCoverage),
      loadOptionalCsv(files.fairnessPinball),
      loadOptionalCsv(files.fairnessResidualCorrelations),
      loadOptionalCsv(files.fairnessGroupBias),
      loadOptionalJson(files.champion),
    ]);

    state.mlPredictions = mlPredictions.map(coerceRow);
    state.predictions = mergeMlPredictions(predictions.map(coerceRow), state.mlPredictions);
    state.monthMetrics = [
      ...monthMetrics.map(coerceRow),
      ...mlMonthMetrics.map(coerceRow),
    ];
    state.buildingMetrics = [
      ...buildingMetrics.map(coerceRow),
      ...mlBuildingMetrics.map(coerceRow),
    ];
    state.fairnessModelComparison = fairnessModelComparison.map(coerceRow);
    state.fairnessCoverage = fairnessCoverage.map(coerceRow);
    state.fairnessPinball = fairnessPinball.map(coerceRow);
    state.fairnessResidualCorrelations = fairnessResidualCorrelations.map(coerceRow);
    state.fairnessGroupBias = fairnessGroupBias.map(coerceRow);
    state.champion = champion && champion.prediction_column ? champion : null;
    state.selectedMethod = determineDefaultMethod();

    populateFilters();
    render();
    setStatus("준비됨", "ready");
  } catch (error) {
    setStatus("CSV 로드 실패", "error");
    document.querySelector("#dailyChart").innerHTML = `<div class="empty">${escapeHtml(
      error.message,
    )}</div>`;
  }
}

function bindControls() {
  document.querySelector("#methodSelect").addEventListener("change", (event) => {
    state.selectedMethod = event.target.value;
    render();
  });

  document.querySelector("#buildingSelect").addEventListener("change", (event) => {
    state.selectedBuilding = event.target.value;
    render();
  });

  document.querySelector("#monthSelect").addEventListener("change", (event) => {
    state.selectedMonth = event.target.value;
    render();
  });

  document.querySelector("#resetButton").addEventListener("click", () => {
    state.selectedMethod = determineDefaultMethod();
    state.selectedBuilding = "all";
    state.selectedMonth = "all";
    document.querySelector("#methodSelect").value = state.selectedMethod;
    document.querySelector("#buildingSelect").value = state.selectedBuilding;
    document.querySelector("#monthSelect").value = state.selectedMonth;
    render();
  });
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} 읽기 실패 (${response.status})`);
  }

  return parseCsv(await response.text());
}

async function loadOptionalCsv(path) {
  try {
    return await loadCsv(path);
  } catch (_error) {
    return [];
  }
}

async function loadOptionalJson(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (_error) {
    return null;
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;
  const cleanText = text.replace(/^\uFEFF/, "");

  for (let index = 0; index < cleanText.length; index += 1) {
    const char = cleanText[index];
    const next = cleanText[index + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(value);
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }

  const [headers, ...body] = rows;
  if (!headers) {
    return [];
  }

  return body.map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])),
  );
}

function coerceRow(row) {
  const result = {};
  for (const [key, rawValue] of Object.entries(row)) {
    const value = typeof rawValue === "string" ? rawValue.trim() : rawValue;
    if (value === "") {
      result[key] = null;
      continue;
    }
    if (!textFields.has(key)) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        result[key] = parsed;
        continue;
      }
    }
    result[key] = value;
  }
  return result;
}

function mergeMlPredictions(baseRows, mlRows) {
  if (!mlRows.length) {
    return baseRows;
  }

  const mlByKey = new Map(
    mlRows.map((row) => [`${row.date}::${row.building_name_recent}`, row]),
  );

  return baseRows.map((row) => {
    const mlRow = mlByKey.get(`${row.date}::${row.building_name_recent}`);
    return mlRow ? { ...row, ...mlRow } : row;
  });
}

function determineDefaultMethod() {
  const candidates = [
    state.champion?.prediction_column,
    DEFAULT_METHOD,
    "pred_lightgbm_weather_q50_kwh",
    "pred_lightgbm_q50_kwh",
    "pred_weekday_recent_kwh",
    "pred_realtime_uniform_daily_kwh",
  ].filter(Boolean);

  return candidates.find((method) => hasPredictionColumn(method)) || DEFAULT_METHOD;
}

function hasPredictionColumn(method) {
  return state.predictions.some((row) => Object.prototype.hasOwnProperty.call(row, method));
}

function populateFilters() {
  const buildings = uniqueSorted(state.predictions.map((row) => row.building_name_recent));
  const months = uniqueSorted(state.predictions.map((row) => row.report_month));
  const methods = Object.entries(methodLabels).filter(([method]) => hasPredictionColumn(method));

  if (!methods.some(([method]) => method === state.selectedMethod)) {
    state.selectedMethod = methods[0]?.[0] || DEFAULT_METHOD;
  }

  fillSelect("#methodSelect", methods);
  fillSelect("#buildingSelect", [
    ["all", "전체 건물"],
    ...buildings.map((building) => [building, building]),
  ]);
  fillSelect("#monthSelect", [["all", "전체 월"], ...months.map((month) => [month, month])]);
  document.querySelector("#methodSelect").value = state.selectedMethod;
}

function fillSelect(selector, options) {
  const element = document.querySelector(selector);
  element.innerHTML = options
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`)
    .join("");
}

function render() {
  const rows = getFilteredValidationRows();
  const metrics = calculateMetrics(rows, state.selectedMethod);

  renderKpis(metrics);
  renderChart(rows, state.selectedMethod);
  renderDailyPredictionRows(rows, state.selectedMethod);
  renderFairnessDiagnostics();
  renderMonthComparison();
  renderBuildingMetrics();
  renderErrorRows();
}

function getFilteredValidationRows() {
  return state.predictions
    .filter((row) => isTrue(row.is_validation_target_clean))
    .filter(
      (row) => state.selectedBuilding === "all" || row.building_name_recent === state.selectedBuilding,
    )
    .filter((row) => state.selectedMonth === "all" || row.report_month === state.selectedMonth);
}

function calculateMetrics(rows, predColumn) {
  const validRows = rows.filter(
    (row) => Number.isFinite(row.usage_kwh_clean) && Number.isFinite(row[predColumn]),
  );
  const errors = validRows.map((row) => row[predColumn] - row.usage_kwh_clean);
  const absErrors = errors.map(Math.abs);
  const squaredErrors = errors.map((error) => error * error);
  const actualSum = sum(validRows.map((row) => row.usage_kwh_clean));
  const predSum = sum(validRows.map((row) => row[predColumn]));
  const mse = average(squaredErrors);

  return {
    rowCount: validRows.length,
    actualSum,
    predSum,
    mae: average(absErrors),
    rmse: mse === null ? null : Math.sqrt(mse),
    wape: actualSum ? sum(absErrors) / actualSum : null,
    biasPct: actualSum ? sum(errors) / actualSum : null,
  };
}

function renderKpis(metrics) {
  const items = [
    ["검증 행", formatInteger(metrics.rowCount)],
    ["실제 합계", `${formatNumber(metrics.actualSum)} kWh`],
    ["예측 합계", `${formatNumber(metrics.predSum)} kWh`],
    ["MAE", `${formatNumber(metrics.mae)} kWh`],
    ["RMSE", `${formatNumber(metrics.rmse)} kWh`],
    ["WAPE", formatPercent(metrics.wape)],
  ];

  document.querySelector("#kpiGrid").innerHTML = items
    .map(([label, value]) => `<article class="kpi"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
}

function renderChart(rows, predColumn) {
  const chart = document.querySelector("#dailyChart");
  document.querySelector("#chartSubtitle").textContent = `${methodLabels[predColumn] || predColumn} - ${filterLabel()}`;

  if (!rows.length) {
    chart.innerHTML = `<div class="empty">표시할 검증 행이 없습니다</div>`;
    return;
  }

  const grouped = aggregateByDate(rows, predColumn);
  if (!grouped.length) {
    chart.innerHTML = `<div class="empty">선택한 모델의 예측 행이 없습니다</div>`;
    return;
  }

  const values = grouped.flatMap((row) => [row.actual, row.predicted]).filter(Number.isFinite);
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values, 0);
  const width = 760;
  const height = 280;
  const pad = { top: 24, right: 22, bottom: 34, left: 58 };
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;
  const denominator = Math.max(maxValue - minValue, 1);
  const pointX = (index) =>
    pad.left + (grouped.length === 1 ? innerWidth / 2 : (index / (grouped.length - 1)) * innerWidth);
  const pointY = (value) => pad.top + innerHeight - ((value - minValue) / denominator) * innerHeight;
  const actualLine = grouped.map((row, index) => `${pointX(index)},${pointY(row.actual)}`).join(" ");
  const predLine = grouped.map((row, index) => `${pointX(index)},${pointY(row.predicted)}`).join(" ");
  const yTicks = [0, 0.5, 1].map((ratio) => {
    const value = minValue + denominator * ratio;
    const y = pointY(value);
    return `
      <line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#d9e0da" />
      <text x="12" y="${y + 4}" class="axis-label">${formatCompact(value)}</text>
    `;
  });

  const firstDate = grouped[0].date;
  const lastDate = grouped[grouped.length - 1].date;

  chart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="일별 실제값과 예측값 차트">
      ${yTicks.join("")}
      <polyline points="${actualLine}" fill="none" stroke="#117c78" stroke-width="3" stroke-linecap="round" />
      <polyline points="${predLine}" fill="none" stroke="#b06c00" stroke-width="5" stroke-linecap="round" />
      <text x="${pad.left}" y="${height - 10}" class="axis-label">${firstDate}</text>
      <text x="${width - pad.right - 74}" y="${height - 10}" class="axis-label">${lastDate}</text>
    </svg>
  `;
}

function renderDailyPredictionRows(rows, predColumn) {
  const grouped = aggregateByDate(rows, predColumn);
  const body = document.querySelector("#dailyPredictionBody");

  body.innerHTML =
    grouped
      .map((row) => {
        const error = row.predicted - row.actual;
        const errorClass = error >= 0 ? "positive" : "negative";
        return `
          <tr>
            <td>${escapeHtml(row.date)}</td>
            <td>${formatNumber(row.actual)}</td>
            <td>${formatNumber(row.predicted)}</td>
            <td class="${errorClass}">${formatSignedNumber(error)}</td>
          </tr>
        `;
      })
      .join("") || `<tr><td colspan="4">표시할 예측값이 없습니다</td></tr>`;
}

function aggregateByDate(rows, predColumn) {
  const grouped = new Map();
  for (const row of rows) {
    if (!Number.isFinite(row.usage_kwh_clean) || !Number.isFinite(row[predColumn])) {
      continue;
    }
    if (!grouped.has(row.date)) {
      grouped.set(row.date, { date: row.date, actual: 0, predicted: 0 });
    }
    const target = grouped.get(row.date);
    target.actual += row.usage_kwh_clean || 0;
    target.predicted += row[predColumn] || 0;
  }
  return [...grouped.values()].sort((left, right) => left.date.localeCompare(right.date));
}

function renderMonthComparison() {
  const rows = state.monthMetrics
    .filter((row) => row.prediction_method === state.selectedMethod)
    .sort((left, right) => String(left.report_month).localeCompare(String(right.report_month)));

  if (!rows.length) {
    document.querySelector("#monthComparison").innerHTML = `<div class="empty">월별 지표가 없습니다</div>`;
    return;
  }

  const maxMae = Math.max(...rows.map((row) => row.mae_kwh).filter(Number.isFinite), 1);

  document.querySelector("#monthComparison").innerHTML = rows
    .map((row) => {
      const width = Math.max((row.mae_kwh / maxMae) * 100, 2);
      return `
        <div class="month-row">
          <div class="month-title">
            <span>${escapeHtml(row.report_month)}</span>
            <span>MAE ${formatNumber(row.mae_kwh)} - ${ratioMetricLabel(row)} ${formatPercent(ratioMetricValue(row))}</span>
          </div>
          <div class="bar"><span style="width:${width}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderBuildingMetrics() {
  const rows = state.buildingMetrics
    .filter((row) => row.prediction_method === state.selectedMethod)
    .sort((left, right) => right.mae_kwh - left.mae_kwh);

  document.querySelector("#buildingMetricsBody").innerHTML =
    rows
      .map(
        (row) => `
          <tr data-building="${escapeHtml(row.building_name_recent)}">
            <td>${escapeHtml(row.building_name_recent)}</td>
            <td>${formatNumber(row.mae_kwh)}</td>
            <td>${formatNumber(row.rmse_kwh)}</td>
            <td>${formatPercent(ratioMetricValue(row))}</td>
            <td>${formatNumber(metricBiasMean(row))}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="5">건물별 지표가 없습니다</td></tr>`;

  document.querySelectorAll("#buildingMetricsBody tr[data-building]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedBuilding = row.dataset.building;
      document.querySelector("#buildingSelect").value = state.selectedBuilding;
      render();
    });
  });
}

function renderErrorRows() {
  document.querySelector("#errorSubtitle").textContent = `${methodLabels[state.selectedMethod] || state.selectedMethod} 기준`;

  const rows = getFilteredValidationRows()
    .map((row) => {
      const predicted = row[state.selectedMethod];
      return {
        ...row,
        selected_pred_kwh: predicted,
        selected_abs_error_kwh:
          Number.isFinite(predicted) && Number.isFinite(row.usage_kwh_clean)
            ? Math.abs(predicted - row.usage_kwh_clean)
            : null,
      };
    })
    .filter((row) => Number.isFinite(row.selected_abs_error_kwh))
    .sort((left, right) => right.selected_abs_error_kwh - left.selected_abs_error_kwh)
    .slice(0, 18);

  document.querySelector("#errorRowsBody").innerHTML =
    rows
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(row.date)}</td>
            <td>${escapeHtml(row.building_name_recent)}</td>
            <td>${formatNumber(row.usage_kwh_clean)}</td>
            <td>${formatNumber(row.selected_pred_kwh)}</td>
            <td>${formatNumber(row.selected_abs_error_kwh)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="5">표시할 행이 없습니다</td></tr>`;
}

function renderFairnessDiagnostics() {
  renderFairnessKpis();
  renderFairnessModelComparison();
  renderCoverage();
  renderPinball();
  renderResidualCorrelations();
  renderGroupBias();
}

function renderFairnessKpis() {
  const championRow = findComparisonRow(state.champion?.prediction_column) || findComparisonRow(state.selectedMethod);
  const naive = state.fairnessModelComparison.find(
    (row) => row.model === "naive_last_week_same_weekday",
  );
  const coverage90 = findCoverageRow(state.champion?.prediction_column || state.selectedMethod, 90);
  const championLabel = championRow ? modelShortLabel(championRow.model) : "walk-forward";

  const cards = [
    {
      label: "검증 방식",
      value: championRow ? `${formatInteger(championRow.n_rows)}행` : "walk-forward",
      text: `${championLabel}을 과거 데이터로 순차 학습해 다음 날짜를 예측했습니다. random shuffle은 쓰지 않습니다.`,
    },
    {
      label: "naive baseline 대비",
      value: championRow && naive ? `${formatPercent(naive.wape)} -> ${formatPercent(championRow.wape)}` : "비교 필요",
      text: "전주 같은 요일 naive보다 WAPE가 낮아야 실제 운영 모델 후보로 볼 수 있습니다.",
    },
    {
      label: "90% 예측구간",
      value: coverage90 ? formatPercent(coverage90.actual_coverage) : "확인 필요",
      text: "목표 90%보다 낮으면 예측구간이 좁아 사용자에게 불리하게 평가될 수 있습니다.",
    },
    {
      label: "공정성 baseline 판정",
      value: "아직 보류",
      text: "오차는 개선됐지만 coverage 부족과 건물별 bias 점검이 더 필요합니다.",
    },
  ];

  document.querySelector("#fairnessKpis").innerHTML = cards
    .map(
      (card) => `
        <article>
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <p>${escapeHtml(card.text)}</p>
        </article>
      `,
    )
    .join("");
}

function renderFairnessModelComparison() {
  const body = document.querySelector("#fairnessModelBody");
  body.innerHTML =
    [...state.fairnessModelComparison]
      .sort((left, right) => left.wape - right.wape)
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(modelShortLabel(row.model))}</td>
            <td>${formatPercent(row.wape)}</td>
            <td>${formatPercent(row.bias_pct_sum_pred_minus_actual)}</td>
            <td>${formatNumber(row.mae_kwh)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="4">공정성 진단 CSV가 없습니다</td></tr>`;
}

function renderCoverage() {
  document.querySelector("#coverageBody").innerHTML =
    state.fairnessCoverage
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(coverageLabel(row.interval))}</td>
            <td>${formatPercent(row.nominal_coverage)}</td>
            <td>${formatPercent(row.actual_coverage)}</td>
            <td>${formatSignedPercent(row.coverage_gap_actual_minus_nominal)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="4">coverage 결과가 없습니다</td></tr>`;
}

function renderPinball() {
  document.querySelector("#pinballBody").innerHTML =
    state.fairnessPinball
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(modelShortLabel(row.model))}</td>
            <td>q${Math.round(row.quantile * 100)}</td>
            <td>${formatNumber(row.pinball_loss_kwh)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="3">pinball loss 결과가 없습니다</td></tr>`;
}

function renderResidualCorrelations() {
  const rows = [...state.fairnessResidualCorrelations]
    .sort((left, right) => Math.abs(right.spearman_r || 0) - Math.abs(left.spearman_r || 0))
    .slice(0, 6);

  document.querySelector("#residualCorrelationBody").innerHTML =
    rows
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(featureLabel(row.feature))}</td>
            <td>${formatSignedNumber(row.spearman_r)}</td>
            <td>${formatPValue(row.spearman_pvalue)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="3">오차 상관 결과가 없습니다</td></tr>`;
}

function renderGroupBias() {
  const championColumn = state.champion?.prediction_column;
  const rows = [...state.fairnessGroupBias]
    .filter((row) => !championColumn || row.prediction_column === championColumn)
    .sort((left, right) => left.bias_pct_sum_pred_minus_actual - right.bias_pct_sum_pred_minus_actual);
  renderBiasList("#underBiasList", rows.slice(0, 4));
  renderBiasList("#overBiasList", rows.slice(-4).reverse());
}

function renderBiasList(selector, rows) {
  document.querySelector(selector).innerHTML =
    rows
      .map(
        (row) => `
          <div class="bias-item">
            <span>${escapeHtml(row.building_name_recent)}</span>
            <strong>${formatSignedPercent(row.bias_pct_sum_pred_minus_actual)}</strong>
          </div>
        `,
      )
      .join("") || `<div class="empty-list">건물 영향 결과가 없습니다</div>`;
}

function findComparisonRow(predictionColumn) {
  return state.fairnessModelComparison.find((row) => row.prediction_column === predictionColumn);
}

function findCoverageRow(predictionColumn, nominalPct) {
  const prefix = coveragePrefixFromPrediction(predictionColumn);
  return state.fairnessCoverage.find((row) => row.interval === `${prefix}_${nominalPct}`);
}

function coveragePrefixFromPrediction(predictionColumn) {
  if (predictionColumn?.includes("weather_area_academic")) {
    return "lightgbm_weather_area_academic";
  }
  if (predictionColumn?.includes("weather_academic")) {
    return "lightgbm_weather_academic";
  }
  if (predictionColumn?.includes("weather_area")) {
    return "lightgbm_weather_area";
  }
  if (predictionColumn?.includes("weather")) {
    return "lightgbm_weather";
  }
  return "lightgbm";
}

function filterLabel() {
  const building = state.selectedBuilding === "all" ? "전체 건물" : state.selectedBuilding;
  const month = state.selectedMonth === "all" ? "전체 월" : state.selectedMonth;
  return `${building} / ${month}`;
}

function setStatus(text, type) {
  const element = document.querySelector("#loadStatus");
  element.textContent = text;
  element.className = `status ${type || ""}`.trim();
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right, "ko"));
}

function isTrue(value) {
  return String(value).trim().toLowerCase() === "true";
}

function sum(values) {
  return values.reduce((total, value) => total + (Number.isFinite(value) ? value : 0), 0);
}

function average(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? sum(clean) / clean.length : null;
}

function formatInteger(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString("ko-KR") : "-";
}

function formatNumber(value) {
  return Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { maximumFractionDigits: 1 })
    : "-";
}

function formatCompact(value) {
  return Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { notation: "compact", maximumFractionDigits: 1 })
    : "-";
}

function formatSignedNumber(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

function formatPercent(value) {
  return Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { style: "percent", maximumFractionDigits: 1 })
    : "-";
}

function formatSignedPercent(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatPercent(value)}`;
}

function ratioMetricValue(row) {
  return Number.isFinite(row.wape) ? row.wape : row.mape;
}

function ratioMetricLabel(row) {
  return Number.isFinite(row.wape) ? "WAPE" : "MAPE";
}

function metricBiasMean(row) {
  return Number.isFinite(row.bias_kwh_mean_pred_minus_actual)
    ? row.bias_kwh_mean_pred_minus_actual
    : row.bias_kwh_mean;
}

function formatPValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return value < 0.001 ? "<0.001" : value.toLocaleString("ko-KR", { maximumFractionDigits: 3 });
}

function modelShortLabel(model) {
  const labels = {
    naive_seasonal_month_profile: "계절 월평균 naive",
    naive_last_week_same_weekday: "지난주 같은 요일 naive",
    current_realtime_uniform_daily: "현재 월 균등 기준",
    current_weekday_recent_heuristic: "현재 휴리스틱",
    lightgbm_quantile_q50_walk_forward: "LightGBM q50",
    lightgbm_weather_quantile_q50_walk_forward: "LightGBM+weather q50",
    lightgbm_weather_area_quantile_q50_walk_forward: "LightGBM+weather+area q50",
    lightgbm_weather_academic_quantile_q50_walk_forward: "LightGBM+weather+academic q50",
    lightgbm_weather_area_academic_quantile_q50_walk_forward: "LightGBM+weather+area+academic q50",
    lightgbm_quantile_walk_forward: "LightGBM",
    lightgbm_weather_quantile_walk_forward: "LightGBM+weather",
    lightgbm_weather_area_quantile_walk_forward: "LightGBM+weather+area",
    lightgbm_weather_academic_quantile_walk_forward: "LightGBM+weather+academic",
    lightgbm_weather_area_academic_quantile_walk_forward: "LightGBM+weather+area+academic",
  };
  return labels[model] || model;
}

function coverageLabel(interval) {
  const labels = {
    lightgbm_90: "LightGBM 90%",
    lightgbm_80: "LightGBM 80%",
    lightgbm_weather_90: "weather 90%",
    lightgbm_weather_80: "weather 80%",
    lightgbm_weather_area_90: "weather+area 90%",
    lightgbm_weather_area_80: "weather+area 80%",
    lightgbm_weather_academic_90: "weather+academic 90%",
    lightgbm_weather_academic_80: "weather+academic 80%",
    lightgbm_weather_area_academic_90: "weather+area+academic 90%",
    lightgbm_weather_area_academic_80: "weather+area+academic 80%",
  };
  return labels[interval] || interval;
}

function featureLabel(feature) {
  const labels = {
    profile_monthly_kwh_mean: "월 프로파일",
    baseline_uniform_daily_kwh: "월 균등 기준값",
    pred_realtime_uniform_daily_kwh: "실시간 월 균등 예측",
    pred_weekday_recent_kwh: "요일+최근사용량 예측",
    is_weekend: "주말 여부",
    ml_day_of_week: "요일",
    ml_is_weekend: "주말 여부",
    ml_day_of_year: "연중 일자",
    ml_lag_1d_kwh: "전일 사용량",
    ml_lag_3d_mean_kwh: "최근 3일 평균",
    ml_lag_7d_mean_kwh: "최근 7일 평균",
    ml_lag_7d_same_weekday_kwh: "전주 같은 요일",
    weather_cdd_18: "냉방도일",
    weather_hdd_18: "난방도일",
    weather_rainfall_mm: "강수량",
    area_total_sqm: "건물 총면적",
    profile_kwh_per_sqm_month: "면적당 월 사용량",
  };
  return labels[feature] || feature;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
