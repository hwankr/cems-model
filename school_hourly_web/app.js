const DEFAULT_METHOD = "pred_lightgbm_school_hourly_day_ahead_kwh";

const files = {
  predictions: "../outputs/school_hourly_predictions.csv",
  comparison: "../outputs/school_hourly_model_comparison.csv",
  byHour: "../outputs/school_hourly_metrics_by_hour.csv",
  byDay: "../outputs/school_hourly_metrics_by_day.csv",
  topErrors: "../outputs/school_hourly_top_errors.csv",
  summary: "../outputs/school_hourly_run_summary.json",
};

const methodLabels = {
  pred_lightgbm_school_hourly_day_ahead_kwh: "LightGBM school hourly day-ahead",
  pred_lightgbm_school_hourly_day_ahead_weather_kwh: "LightGBM day-ahead + weather",
  pred_lightgbm_school_hourly_day_ahead_weather_academic_kwh:
    "LightGBM day-ahead + weather + academic",
  pred_lightgbm_school_hourly_operational_kwh: "LightGBM school hourly operational",
  pred_lightgbm_school_hourly_kwh: "LightGBM school hourly operational alias",
  pred_last_day_same_hour_kwh: "last day same hour",
  pred_last_week_same_hour_kwh: "last week same hour",
  pred_same_hour_7d_mean_kwh: "same hour 7d mean",
};

const textFields = new Set([
  "timestamp",
  "period_end",
  "measurement_date",
  "report_date",
  "report_month",
  "period_label",
  "source_type",
  "source_table",
  "source_file",
  "model",
  "prediction_method",
  "prediction_column",
]);

const state = {
  predictions: [],
  comparison: [],
  byHour: [],
  byDay: [],
  topErrors: [],
  summary: null,
  selectedMethod: DEFAULT_METHOD,
  selectedDay: "all",
  selectedHour: "all",
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindControls();
  try {
    const [predictions, comparison, byHour, byDay, topErrors, summary] = await Promise.all([
      loadCsv(files.predictions),
      loadCsv(files.comparison),
      loadCsv(files.byHour),
      loadCsv(files.byDay),
      loadCsv(files.topErrors),
      loadJson(files.summary),
    ]);

    state.predictions = predictions.map(coerceRow);
    state.comparison = comparison.map(coerceRow);
    state.byHour = byHour.map(coerceRow);
    state.byDay = byDay.map(coerceRow);
    state.topErrors = topErrors.map(coerceRow);
    state.summary = summary;
    state.selectedMethod = summary?.prediction_column || DEFAULT_METHOD;

    populateFilters();
    render();
    setStatus("준비됨", "ready");
  } catch (error) {
    setStatus("로드 실패", "error");
    document.querySelector("#hourlyChart").innerHTML = `<div class="empty">${escapeHtml(
      error.message,
    )}</div>`;
  }
}

function bindControls() {
  document.querySelector("#methodSelect").addEventListener("change", (event) => {
    state.selectedMethod = event.target.value;
    render();
  });
  document.querySelector("#daySelect").addEventListener("change", (event) => {
    state.selectedDay = event.target.value;
    render();
  });
  document.querySelector("#hourSelect").addEventListener("change", (event) => {
    state.selectedHour = event.target.value;
    render();
  });
  document.querySelector("#resetButton").addEventListener("click", () => {
    state.selectedMethod = state.summary?.prediction_column || DEFAULT_METHOD;
    state.selectedDay = "all";
    state.selectedHour = "all";
    document.querySelector("#methodSelect").value = state.selectedMethod;
    document.querySelector("#daySelect").value = state.selectedDay;
    document.querySelector("#hourSelect").value = state.selectedHour;
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

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} 읽기 실패 (${response.status})`);
  }
  return response.json();
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

function populateFilters() {
  const availableMethods = Object.entries(methodLabels).filter(([method]) =>
    state.predictions.some((row) => Object.prototype.hasOwnProperty.call(row, method)),
  );
  fillSelect("#methodSelect", availableMethods);
  fillSelect("#daySelect", [
    ["all", "전체 날짜"],
    ...uniqueSorted(state.predictions.map((row) => row.report_date)).map((day) => [day, day]),
  ]);
  fillSelect("#hourSelect", [
    ["all", "전체 시간"],
    ...Array.from({ length: 24 }, (_, hour) => [String(hour), `${String(hour).padStart(2, "0")}:00`]),
  ]);
  document.querySelector("#methodSelect").value = state.selectedMethod;
}

function fillSelect(selector, options) {
  document.querySelector(selector).innerHTML = options
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`)
    .join("");
}

function render() {
  const rows = getFilteredRows();
  const metrics = calculateMetrics(rows, state.selectedMethod);
  renderKpis(metrics);
  renderHourlyChart(rows, state.selectedMethod);
  renderModelComparison();
  renderHourPerformance();
  renderDayPerformance();
  renderTopErrors();
}

function getFilteredRows() {
  return state.predictions
    .filter((row) => state.selectedDay === "all" || row.report_date === state.selectedDay)
    .filter((row) => state.selectedHour === "all" || String(row.hour) === state.selectedHour)
    .filter((row) => Number.isFinite(row.usage_kwh) && Number.isFinite(row[state.selectedMethod]))
    .sort((left, right) => String(left.timestamp).localeCompare(String(right.timestamp)));
}

function calculateMetrics(rows, predictionColumn) {
  const errors = rows.map((row) => row[predictionColumn] - row.usage_kwh);
  const absErrors = errors.map(Math.abs);
  const squaredErrors = errors.map((error) => error * error);
  const actualSum = sum(rows.map((row) => row.usage_kwh));
  const predSum = sum(rows.map((row) => row[predictionColumn]));
  const mse = average(squaredErrors);
  return {
    rowCount: rows.length,
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
    ["Bias", formatSignedPercent(metrics.biasPct)],
  ];

  document.querySelector("#kpiGrid").innerHTML = items
    .map(([label, value]) => `<article class="kpi"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
}

function renderHourlyChart(rows, predictionColumn) {
  const chart = document.querySelector("#hourlyChart");
  document.querySelector("#chartSubtitle").textContent = `${
    methodLabels[predictionColumn] || predictionColumn
  } / ${filterLabel()}`;
  if (!rows.length) {
    chart.innerHTML = `<div class="empty">표시할 행이 없습니다</div>`;
    return;
  }

  const sampled = sampleRows(rows, 360);
  const values = sampled.flatMap((row) => [row.usage_kwh, row[predictionColumn]]).filter(Number.isFinite);
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values, 0);
  const width = 920;
  const height = 320;
  const pad = { top: 24, right: 24, bottom: 44, left: 64 };
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;
  const denominator = Math.max(maxValue - minValue, 1);
  const x = (index) =>
    pad.left + (sampled.length === 1 ? innerWidth / 2 : (index / (sampled.length - 1)) * innerWidth);
  const y = (value) => pad.top + innerHeight - ((value - minValue) / denominator) * innerHeight;
  const actualLine = sampled.map((row, index) => `${x(index)},${y(row.usage_kwh)}`).join(" ");
  const predictedLine = sampled.map((row, index) => `${x(index)},${y(row[predictionColumn])}`).join(" ");
  const yTicks = [0, 0.5, 1].map((ratio) => {
    const value = minValue + denominator * ratio;
    const tickY = y(value);
    return `
      <line x1="${pad.left}" y1="${tickY}" x2="${width - pad.right}" y2="${tickY}" stroke="#dde4e1" />
      <text x="14" y="${tickY + 4}" class="axis-label">${formatCompact(value)}</text>
    `;
  });
  const first = sampled[0].timestamp.slice(5, 16);
  const last = sampled[sampled.length - 1].timestamp.slice(5, 16);

  chart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="시간별 실제 사용량과 예측 사용량">
      ${yTicks.join("")}
      <polyline points="${actualLine}" fill="none" stroke="#0f766e" stroke-width="2.8" stroke-linecap="round" />
      <polyline points="${predictedLine}" fill="none" stroke="#c47a1c" stroke-width="2.8" stroke-linecap="round" />
      <text x="${pad.left}" y="${height - 14}" class="axis-label">${escapeHtml(first)}</text>
      <text x="${width - pad.right - 80}" y="${height - 14}" class="axis-label">${escapeHtml(last)}</text>
    </svg>
  `;
}

function renderModelComparison() {
  const rows = [...state.comparison].sort((left, right) => left.wape - right.wape);
  document.querySelector("#modelComparisonBody").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(methodLabels[row.prediction_column] || row.model)}</td>
          <td>${formatPercent(row.wape)}</td>
          <td>${formatNumber(row.mae_kwh)}</td>
          <td>${formatNumber(row.rmse_kwh)}</td>
          <td>${formatSignedPercent(row.bias_pct_sum_pred_minus_actual)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderHourPerformance() {
  const rows = state.byHour
    .filter((row) => row.prediction_column === state.selectedMethod)
    .sort((left, right) => left.hour - right.hour);
  const maxWape = Math.max(...rows.map((row) => row.wape).filter(Number.isFinite), 0.01);

  document.querySelector("#hourBars").innerHTML = rows
    .map((row) => {
      const width = Math.max((row.wape / maxWape) * 100, 2);
      return `
        <button class="hour-bar" type="button" data-hour="${row.hour}">
          <span>${String(row.hour).padStart(2, "0")}:00</span>
          <i><b style="width:${width}%"></b></i>
          <strong>${formatPercent(row.wape)}</strong>
        </button>
      `;
    })
    .join("");

  document.querySelectorAll(".hour-bar").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedHour = button.dataset.hour;
      document.querySelector("#hourSelect").value = state.selectedHour;
      render();
    });
  });
}

function renderDayPerformance() {
  const rows = state.byDay
    .filter((row) => row.prediction_column === state.selectedMethod)
    .sort((left, right) => String(left.report_date).localeCompare(String(right.report_date)));
  document.querySelector("#dayMetricsBody").innerHTML = rows
    .map(
      (row) => `
        <tr data-day="${escapeHtml(row.report_date)}">
          <td>${escapeHtml(row.report_date)}</td>
          <td>${formatPercent(row.wape)}</td>
          <td>${formatNumber(row.mae_kwh)}</td>
          <td>${formatSignedPercent(row.bias_pct_sum_pred_minus_actual)}</td>
        </tr>
      `,
    )
    .join("");

  document.querySelectorAll("#dayMetricsBody tr[data-day]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedDay = row.dataset.day;
      document.querySelector("#daySelect").value = state.selectedDay;
      render();
    });
  });
}

function renderTopErrors() {
  const rows = getFilteredRows()
    .map((row) => {
      const predicted = row[state.selectedMethod];
      const error = predicted - row.usage_kwh;
      return { ...row, predicted, error, absError: Math.abs(error) };
    })
    .sort((left, right) => right.absError - left.absError)
    .slice(0, 36);

  document.querySelector("#topErrorsBody").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.timestamp)}</td>
          <td>${formatNumber(row.usage_kwh)}</td>
          <td>${formatNumber(row.predicted)}</td>
          <td>${formatSignedNumber(row.error)}</td>
          <td>${formatNumber(row.absError)}</td>
        </tr>
      `,
    )
    .join("");
}

function sampleRows(rows, limit) {
  if (rows.length <= limit) {
    return rows;
  }
  const step = rows.length / limit;
  return Array.from({ length: limit }, (_, index) => rows[Math.floor(index * step)]);
}

function filterLabel() {
  const day = state.selectedDay === "all" ? "전체 날짜" : state.selectedDay;
  const hour = state.selectedHour === "all" ? "전체 시간" : `${String(state.selectedHour).padStart(2, "0")}:00`;
  return `${day} / ${hour}`;
}

function setStatus(text, type) {
  const element = document.querySelector("#loadStatus");
  element.textContent = text;
  element.className = `status ${type || ""}`.trim();
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort();
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

function formatPercent(value) {
  return Number.isFinite(value)
    ? value.toLocaleString("ko-KR", { style: "percent", maximumFractionDigits: 2 })
    : "-";
}

function formatSignedPercent(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatPercent(value)}`;
}

function formatSignedNumber(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
