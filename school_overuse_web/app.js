/**
 * 학교 전력 과다사용 모니터 — app.js
 * Reads ONLY data/monitor.json via fetch('data/monitor.json').
 * All rendering is done in vanilla JS with inline SVG charts.
 * No external libraries or CDN dependencies.
 */

'use strict';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  data: null,          // full monitor.json
  selectedDate: null,  // currently selected report_date string
  selectedOveruse: 0,  // index into data.overuse
};

// ─── Utility helpers ─────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

function fmt1(n) { return (Math.round(n * 10) / 10).toLocaleString('ko-KR'); }
function fmt0(n) { return Math.round(n).toLocaleString('ko-KR'); }
function fmtPct(n) { return (n * 100).toFixed(1) + '%'; }
function fmtPct2(n) { return (n * 100).toFixed(2) + '%'; }

// ─── Tab navigation ──────────────────────────────────────────────────────────
function initTabs() {
  $$('nav.tabs button').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('nav.tabs button').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      const tabId = btn.dataset.tab;
      $$('.panel').forEach(p => p.classList.remove('active'));
      $(`#panel-${tabId}`).classList.add('active');

      // Lazy render charts when tab first opened
      if (tabId === 'timeseries' && state.data) renderTimeseriesChart();
    });
  });
}

// ─── Fetch data ───────────────────────────────────────────────────────────────
async function loadData() {
  try {
    const res = await fetch('data/monitor.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    state.data = json;
    onDataReady(json);
  } catch (err) {
    document.body.insertAdjacentHTML('afterbegin',
      `<div style="background:#fff5f5;color:#c53030;padding:14px 24px;font-size:13px;border-bottom:2px solid #fc8181;">
         데이터 로딩 실패: ${err.message} — data/monitor.json 경로를 확인하세요.
       </div>`);
  }
}

// ─── Main entry after data loaded ────────────────────────────────────────────
function onDataReady(d) {
  // Fallback/shap banner
  if (d.meta.used_fallback || !d.meta.shap_available) {
    $('#fallback-banner').classList.add('visible');
  }

  renderOverview(d);
  renderDateSelect(d);
  renderOveruseTable(d);
  renderModelAccuracy(d);
  renderFeatureImportance(d);
  renderGlossaryExtra(d);

  // Auto-select first overuse row
  selectOveruseRow(0);
}

// ─── Panel 1: 개요 ───────────────────────────────────────────────────────────
function renderOverview(d) {
  const m = d.meta;
  const mt = d.metrics;

  // Period
  const startDate = m.validation_start.slice(0, 10);
  const endDate   = m.validation_end.slice(0, 10);
  $('#kpi-period').textContent = `${startDate} ~ ${endDate}`;
  $('#kpi-days').textContent   = `${m.validation_days}일`;

  // Rows
  $('#kpi-rows').textContent = fmt0(mt.n_rows);

  // Coverage
  const covPct = (mt.coverage * 100).toFixed(1);
  const rawPct = (mt.coverage_raw * 100).toFixed(1);
  $('#kpi-coverage').textContent     = covPct + '%';
  $('#kpi-coverage-sub').textContent = `보정 전 ${rawPct}% / 목표 80%`;
  // Color card by whether coverage >= target
  const covCard = $('#kpi-coverage-card');
  if (mt.coverage >= mt.coverage_target) {
    covCard.classList.add('ok');
  } else {
    covCard.classList.add('warn');
  }

  // Overuse hours
  const overusePct = ((mt.overuse_hours / mt.n_rows) * 100).toFixed(1);
  $('#kpi-overuse-hrs').textContent = fmt0(mt.overuse_hours);
  $('#kpi-overuse-pct').textContent = `전체의 ${overusePct}% / ${mt.n_rows}시간 중`;

  // Total excess
  $('#kpi-excess').textContent = fmt1(mt.overuse_total_exceedance_kwh);

  // WAPE
  $('#kpi-wape').textContent = fmtPct2(mt.p50_wape);

  // Calibration note
  $('#note-coverage').textContent = covPct;
}

// ─── Panel 2: 시계열 모니터 ──────────────────────────────────────────────────
function renderDateSelect(d) {
  const dates = [...new Set(d.series.map(r => r.report_date))].sort();
  const sel = $('#date-select');
  dates.forEach(dt => {
    const opt = document.createElement('option');
    opt.value = dt;
    opt.textContent = dt;
    sel.appendChild(opt);
  });
  state.selectedDate = dates[0];
  sel.value = dates[0];

  sel.addEventListener('change', () => {
    state.selectedDate = sel.value;
    renderTimeseriesChart();
  });

  // Render immediately if timeseries panel is active (it's not by default, but safe)
  renderTimeseriesChart();
}

function renderTimeseriesChart() {
  const d = state.data;
  if (!d || !state.selectedDate) return;

  const dayRows = d.series.filter(r => r.report_date === state.selectedDate);
  if (dayRows.length === 0) return;

  const container = $('#timeseries-chart');

  // SVG dimensions
  const W = 860, H = 360;
  const padL = 68, padR = 20, padT = 20, padB = 44;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  // Y domain
  const allVals = dayRows.flatMap(r => [r.actual, r.p10, r.p90]);
  const yMin = Math.max(0, Math.min(...allVals) * 0.92);
  const yMax = Math.max(...allVals) * 1.06;

  const xScale = i => padL + (i / 23) * chartW;
  const yScale = v => padT + chartH - ((v - yMin) / (yMax - yMin)) * chartH;

  // Axis helpers
  const nGridLines = 4;
  const yTicks = Array.from({ length: nGridLines + 1 }, (_, i) =>
    yMin + (i / nGridLines) * (yMax - yMin)
  );

  // Build SVG
  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-label="${state.selectedDate} 시간별 전력 사용량">`;

  // Grid lines
  yTicks.forEach(yv => {
    const cy = yScale(yv);
    svg += `<line x1="${padL}" y1="${cy}" x2="${W - padR}" y2="${cy}" stroke="#e2e8f0" stroke-width="1"/>`;
    svg += `<text x="${padL - 6}" y="${cy + 4}" text-anchor="end" font-size="11" fill="#a0aec0">${fmt0(yv)}</text>`;
  });

  // Band area path (p90 left→right, then p10 right→left)
  let bandPath = `M ${xScale(0)} ${yScale(dayRows[0].p90)}`;
  for (let i = 1; i < dayRows.length; i++) bandPath += ` L ${xScale(i)} ${yScale(dayRows[i].p90)}`;
  for (let i = dayRows.length - 1; i >= 0; i--) bandPath += ` L ${xScale(i)} ${yScale(dayRows[i].p10)}`;
  bandPath += ' Z';
  svg += `<path d="${bandPath}" fill="rgba(66,153,225,0.15)" stroke="none"/>`;

  // Band border lines
  let p10Path = `M ${xScale(0)} ${yScale(dayRows[0].p10)}`;
  let p90Path = `M ${xScale(0)} ${yScale(dayRows[0].p90)}`;
  for (let i = 1; i < dayRows.length; i++) {
    p10Path += ` L ${xScale(i)} ${yScale(dayRows[i].p10)}`;
    p90Path += ` L ${xScale(i)} ${yScale(dayRows[i].p90)}`;
  }
  svg += `<path d="${p10Path}" fill="none" stroke="#4299e1" stroke-width="1.2" stroke-dasharray="4,3"/>`;
  svg += `<path d="${p90Path}" fill="none" stroke="#4299e1" stroke-width="1.2" stroke-dasharray="4,3"/>`;

  // P50 dashed line
  let p50Path = `M ${xScale(0)} ${yScale(dayRows[0].p50)}`;
  for (let i = 1; i < dayRows.length; i++) p50Path += ` L ${xScale(i)} ${yScale(dayRows[i].p50)}`;
  svg += `<path d="${p50Path}" fill="none" stroke="#2b6cb0" stroke-width="1.8" stroke-dasharray="7,4"/>`;

  // Actual line
  let actualPath = `M ${xScale(0)} ${yScale(dayRows[0].actual)}`;
  for (let i = 1; i < dayRows.length; i++) actualPath += ` L ${xScale(i)} ${yScale(dayRows[i].actual)}`;
  svg += `<path d="${actualPath}" fill="none" stroke="#1a202c" stroke-width="2.5"/>`;

  // Hover hit areas + dots for overuse/underuse
  dayRows.forEach((row, i) => {
    const cx = xScale(i);
    const cy = yScale(row.actual);

    if (row.is_overuse) {
      svg += `<circle cx="${cx}" cy="${cy}" r="5" fill="#e53e3e" stroke="#fff" stroke-width="1.5"/>`;
    } else if (row.band_position < 0) {
      svg += `<circle cx="${cx}" cy="${cy}" r="4" fill="#a0aec0" stroke="#fff" stroke-width="1"/>`;
    }

    // Invisible wide hit area for tooltip
    const tipData = JSON.stringify({
      ts: row.timestamp,
      actual: row.actual,
      p10: row.p10,
      p50: row.p50,
      p90: row.p90,
      exc: row.exceedance_kwh,
      isOveruse: row.is_overuse,
    }).replace(/"/g, '&quot;');

    svg += `<rect x="${cx - 18}" y="${padT}" width="36" height="${chartH}"
      fill="transparent" data-tip="${tipData}"
      class="chart-hit" style="cursor:crosshair;"/>`;
  });

  // X axis labels
  [0, 3, 6, 9, 12, 15, 18, 21, 23].forEach(h => {
    const cx = xScale(h);
    svg += `<text x="${cx}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#718096">${h}시</text>`;
    svg += `<line x1="${cx}" y1="${padT + chartH}" x2="${cx}" y2="${padT + chartH + 4}" stroke="#cbd5e0" stroke-width="1"/>`;
  });

  // Y axis line
  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + chartH}" stroke="#cbd5e0" stroke-width="1"/>`;
  // X axis line
  svg += `<line x1="${padL}" y1="${padT + chartH}" x2="${W - padR}" y2="${padT + chartH}" stroke="#cbd5e0" stroke-width="1"/>`;

  svg += '</svg>';
  container.innerHTML = svg;

  // Tooltip logic
  const tooltip = $('#chart-tooltip');
  $$('.chart-hit', container).forEach(el => {
    el.addEventListener('mousemove', e => {
      const tip = JSON.parse(el.dataset.tip);
      tooltip.innerHTML = `
        <strong>${tip.ts.slice(11, 16)} (${tip.ts.slice(5, 10)})</strong>
        실제: ${fmt1(tip.actual)} kWh<br>
        P10: ${fmt1(tip.p10)} / P50: ${fmt1(tip.p50)} / P90: ${fmt1(tip.p90)}<br>
        ${tip.isOveruse ? `<span style="color:#fc8181;">초과량: ${fmt1(tip.exc)} kWh</span>` : '정상 범위'}
      `;
      tooltip.classList.add('visible');
      posTooltip(e);
    });
    el.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
  });
}

function posTooltip(e) {
  const t = $('#chart-tooltip');
  const vw = window.innerWidth, vh = window.innerHeight;
  let x = e.clientX + 14, y = e.clientY - 10;
  if (x + 180 > vw) x = e.clientX - 190;
  if (y + 120 > vh) y = e.clientY - 120;
  t.style.left = x + 'px';
  t.style.top  = y + 'px';
}

// ─── Panel 3: 과다사용 분석 ───────────────────────────────────────────────────
function renderOveruseTable(d) {
  const tbody = $('#overuse-tbody');
  if (!d.overuse || d.overuse.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#a0aec0;padding:20px;">과다사용 없음</td></tr>';
    return;
  }

  // overuse is sorted by exceedance_kwh desc (from JSON)
  tbody.innerHTML = d.overuse.map((row, idx) => `
    <tr data-idx="${idx}" class="${idx === 0 ? 'selected' : ''}">
      <td>${row.timestamp.replace('T', ' ').slice(0, 16)}</td>
      <td>${fmt1(row.actual)}</td>
      <td>${fmt1(row.p90)}</td>
      <td><span class="overuse-badge">+${fmt1(row.exceedance_kwh)}</span></td>
      <td>${(row.exceedance_pct * 100).toFixed(1)}%</td>
    </tr>
  `).join('');

  $$('#overuse-tbody tr').forEach(tr => {
    tr.addEventListener('click', () => {
      $$('#overuse-tbody tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      selectOveruseRow(parseInt(tr.dataset.idx, 10));
    });
  });
}

function selectOveruseRow(idx) {
  const d = state.data;
  if (!d || !d.overuse[idx]) return;
  state.selectedOveruse = idx;

  const row = d.overuse[idx];
  const detail = $('#overuse-detail');
  detail.style.display = 'block';

  // Summary line
  $('#overuse-summary-line').innerHTML =
    `정상 기대 P50 <strong>${fmt1(row.p50)}</strong> kWh &nbsp;·&nbsp; ` +
    `정상상한 P90 <strong>${fmt1(row.p90)}</strong> kWh &nbsp;·&nbsp; ` +
    `실제 <strong>${fmt1(row.actual)}</strong> kWh &nbsp;→&nbsp; ` +
    `상한보다 <strong style="color:var(--color-overuse);">${fmt1(row.exceedance_kwh)} kWh` +
    ` (${(row.exceedance_pct * 100).toFixed(1)}%)</strong> 초과`;

  // SHAP bar chart
  renderShapBars(row.explanations);
}

function renderShapBars(explanations) {
  const wrap = $('#shap-bar-chart');
  if (!explanations || explanations.length === 0) {
    wrap.innerHTML = '<p style="color:#a0aec0;font-size:13px;">SHAP 정보 없음</p>';
    return;
  }

  const maxAbs = Math.max(...explanations.map(e => Math.abs(e.shap_kwh)));

  wrap.innerHTML = explanations.map(e => {
    const pct = (Math.abs(e.shap_kwh) / maxAbs * 100).toFixed(1);
    const dir = e.shap_kwh >= 0 ? 'up' : 'down';
    const sign = e.shap_kwh >= 0 ? '+' : '';
    return `
      <div class="shap-bar-row">
        <div class="shap-bar-label" title="${e.feature}">${e.label_ko}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${dir}" style="width:${pct}%;"></div>
        </div>
        <div class="shap-bar-value">${sign}${fmt1(e.shap_kwh)} kWh</div>
      </div>
    `;
  }).join('');
}

// ─── Panel 4: 모델 정확도 ────────────────────────────────────────────────────
function renderModelAccuracy(d) {
  const tbody = $('#baseline-tbody');
  tbody.innerHTML = d.baselines.map((b) => {
    const isP50 = b.model.toLowerCase().includes('p50');
    return `
      <tr class="${isP50 ? 'highlight-row' : ''}">
        <td>${escHtml(b.model)}</td>
        <td>${fmtPct2(b.wape)}</td>
        <td>${fmt1(b.mae_kwh)}</td>
        <td>${fmt1(b.rmse_kwh)}</td>
      </tr>
    `;
  }).join('');

  const mt = d.metrics;
  $('#acc-coverage').textContent     = (mt.coverage * 100).toFixed(1);
  $('#acc-coverage-raw').textContent = (mt.coverage_raw * 100).toFixed(1);
  $('#acc-pinball-p10').textContent  = mt.pinball_p10.toFixed(2);
  $('#acc-pinball-p50').textContent  = mt.pinball_p50.toFixed(2);
  $('#acc-pinball-p90').textContent  = mt.pinball_p90.toFixed(2);
}

// ─── Feature importance (accuracy panel) ─────────────────────────────────────
function renderFeatureImportance(d) {
  const container = $('#feature-importance-list');
  if (!container || !d.feature_importance || d.feature_importance.length === 0) return;

  const top10 = d.feature_importance.slice(0, 10);
  const maxVal = top10[0].mean_abs_shap; // already sorted desc

  container.innerHTML = top10.map(fi => {
    const pct = (fi.mean_abs_shap / maxVal * 100).toFixed(1);
    return `
      <div class="shap-bar-row">
        <div class="shap-bar-label" title="${escHtml(fi.feature)}">${escHtml(fi.label_ko)}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill up" style="width:${pct}%;"></div>
        </div>
        <div class="shap-bar-value">${fmt1(fi.mean_abs_shap)}</div>
      </div>
    `;
  }).join('');
}

// ─── Panel 5: 용어 설명 (dynamic extra) ──────────────────────────────────────
function renderGlossaryExtra(d) {
  // Only add terms not already statically in the HTML (to avoid duplicates)
  const staticTerms = new Set([
    '분위수(Quantile)', 'P10 / P50 / P90', '정상 밴드 / 과다사용',
    'day-ahead(하루 전 기준)', 'SHAP', 'WAPE', 'Pinball loss',
    'Coverage(적중률)', 'CDD/HDD', 'CQR(Conformal Quantile Regression)',
  ]);

  const extra = d.glossary.filter(g => !staticTerms.has(g.term));
  if (extra.length === 0) return;

  const container = $('#glossary-extra');
  container.innerHTML = extra.map(g => `
    <div class="glossary-card">
      <h3>${escHtml(g.term)}</h3>
      <p>${escHtml(g.desc)}</p>
    </div>
  `).join('');
}

// ─── Util ────────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
initTabs();
loadData();
