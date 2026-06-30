/* ============================================================
   CEMS 절감 데모 — app.js
   fetch('data/savings.json') → render all 6 tabs
   ============================================================ */

'use strict';

/* ---------- 유틸 ----------------------------------------- */
const fmt = {
  num:  (v, d=0)       => v == null ? '—' : Number(v).toLocaleString('ko-KR', {minimumFractionDigits:d, maximumFractionDigits:d}),
  pct:  (v, d=2)       => v == null ? '—' : (v*100).toFixed(d)+'%',
  pctDisplay: (v, d=2) => v == null ? '—' : (v>=0?'+':'')+v.toFixed(d)+'%',
  kwh:  (v)            => fmt.num(v,1)+' kWh',
  sign: (v)            => v > 0 ? '절감' : v < 0 ? '초과' : '균형',
};

function signClass(v) {
  if (v > 0.001) return 'saving';
  if (v < -0.001) return 'overuse';
  return 'neutral';
}

/* ---------- 탭 라우터 ------------------------------------- */
let activeTab = 'overview';
function initTabs() {
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}
function switchTab(id) {
  activeTab = id;
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-'+id));
}

/* ---------- 메인 ----------------------------------------- */
async function main() {
  try {
    const res = await fetch('data/savings.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // fallback 배너
    if (data.meta && data.meta.used_fallback) {
      document.querySelectorAll('.fallback-banner').forEach(b => b.classList.add('visible'));
    }

    initTabs();
    renderOverview(data);
    renderAccuracy(data);
    renderChart(data);
    renderScorecard(data);
    renderLeaderboard(data);
    renderGlossary(data);

    // 로딩 상태 숨기기
    document.querySelectorAll('.loading-state').forEach(el => el.remove());
    // 첫 탭 활성화
    switchTab('overview');

  } catch(e) {
    document.body.innerHTML = `<div class="error-state">데이터 로딩 실패: ${e.message}</div>`;
  }
}

/* ============================================================
   1. 주제(개요)
   ============================================================ */
function renderOverview(data) {
  const { meta, scorecard } = data;

  // 폴백 배너 텍스트
  document.querySelectorAll('.fallback-banner').forEach(el => {
    el.textContent = '경고: 이 데이터는 폴백 모드로 생성되었습니다. 결과 해석에 주의하세요.';
  });

  // KPI 스트립
  const kpiData = [
    { label: '검증 기간', value: `${meta.reporting_start.slice(0,10)} ~ ${meta.reporting_end.slice(0,10)}`, unit: '', cls: '' },
    { label: '검증 시간 수', value: fmt.num(meta.reporting_rows), unit: '시간', cls: '' },
    { label: '총 연면적', value: fmt.num(meta.total_area_sqm,0), unit: '㎡', cls: '' },
    { label: '참여 학교 수', value: '447,761㎡ 규모', unit: '다수', cls: '' },
  ];

  const strip = document.getElementById('overview-kpi');
  if (strip) {
    strip.innerHTML = kpiData.map(k => `
      <div class="kpi-card kpi-card--${k.cls || 'accent'}">
        <div class="kpi-card__label">${k.label}</div>
        <div class="kpi-card__value">${k.value}<span class="kpi-card__unit">${k.unit}</span></div>
      </div>
    `).join('');
  }

  // meta.note 표시
  const noteEl = document.getElementById('overview-note');
  if (noteEl) {
    noteEl.innerHTML = `
      <strong>동결(frozen) 베이스라인이란?</strong><br>
      ${meta.note || '베이스라인은 대회 전 데이터로 동결 → 절감해도 베이스라인이 따라 내려가지 않음.'}<br><br>
      대회 참여 전 <strong>${meta.reference_start.slice(0,10)} ~ ${meta.reference_end.slice(0,10)}</strong> 기간의 패턴으로
      베이스라인을 고정합니다. 절감을 실행해도 베이스라인이 낮아지지 않아
      <strong>baseline chasing</strong>(절감할수록 기준이 내려가는 문제)을 원천 차단합니다.
    `;
  }
}

/* ============================================================
   2. 정확도 증명
   ============================================================ */
function renderAccuracy(data) {
  const acc = data.accuracy;
  const sc  = data.scorecard;

  const wapePct  = (acc.wape  * 100).toFixed(2);
  const coverPct = (acc.coverage * 100).toFixed(1);

  const frozen = acc.baselines.find(b => b.model === 'frozen P50');
  const naive  = acc.baselines.find(b => b.model === 'naive_same_dow_hour_profile');
  const frozenWape = frozen ? (frozen.wape*100).toFixed(2) : '—';
  const naiveWape  = naive  ? (naive.wape*100).toFixed(2)  : '—';
  const gainPct    = frozen && naive ? (((naive.wape - frozen.wape)/naive.wape)*100).toFixed(0) : '—';

  const el = document.getElementById('accuracy-content');
  if (!el) return;

  el.innerHTML = `
    <div class="accuracy-grid">
      <div class="acc-card">
        <div class="acc-card__icon">📐</div>
        <div class="acc-card__label">Held-out WAPE</div>
        <div class="acc-card__value">${wapePct}%</div>
        <div class="acc-card__desc">
          보류 검증 세트 ${data.meta.heldout_days}일 기준.
          예측 오차율 ${wapePct}% — 낮을수록 정확.
        </div>
      </div>
      <div class="acc-card">
        <div class="acc-card__icon">🎯</div>
        <div class="acc-card__label">적중률 (coverage)</div>
        <div class="acc-card__value">${coverPct}%</div>
        <div class="acc-card__desc">
          실제 값이 P10~P90 구간 안에 든 비율.
          80% 구간의 이론적 목표 대비 ${coverPct}% 달성.
        </div>
      </div>
      <div class="acc-card">
        <div class="acc-card__icon">⚡</div>
        <div class="acc-card__label">MAE</div>
        <div class="acc-card__value">${fmt.num(acc.mae_kwh,0)}</div>
        <div class="acc-card__desc">
          평균 절대 오차 ${fmt.num(acc.mae_kwh,1)} kWh/시간.
          전체 총 연면적 ${fmt.num(data.meta.total_area_sqm/10000,1)}만 ㎡ 대비 수준.
        </div>
      </div>
    </div>

    <div class="vs-bar">
      <div class="vs-bar__title">frozen P50 vs 단순 프로파일 (WAPE — 낮을수록 좋음)</div>
      <div class="vs-bar__row">
        <span class="vs-bar__name">frozen P50 (이 모델)</span>
        <div class="vs-bar__track">
          <div class="vs-bar__fill vs-bar__fill--frozen"
               style="width:${Math.min(parseFloat(frozenWape)*3,100)}%"></div>
        </div>
        <span class="vs-bar__pct vs-bar__pct--frozen">${frozenWape}%</span>
      </div>
      <div class="vs-bar__row">
        <span class="vs-bar__name">naive 같은 요일·시간 프로파일</span>
        <div class="vs-bar__track">
          <div class="vs-bar__fill vs-bar__fill--naive"
               style="width:${Math.min(parseFloat(naiveWape)*3,100)}%"></div>
        </div>
        <span class="vs-bar__pct vs-bar__pct--naive">${naiveWape}%</span>
      </div>
    </div>

    <div class="info-box">
      <strong>결론:</strong> frozen P50 <strong>${frozenWape}%</strong> vs naive <strong>${naiveWape}%</strong>
      → 약 <strong>${gainPct}%</strong> 더 정확합니다.<br>
      이 데이터만(시간별 검침 + 공공 날씨 + 학사일정)으로 베이스라인이 이만큼 정확합니다.
    </div>
  `;
}

/* ============================================================
   3. 베이스라인 대비 절감 차트
   ============================================================ */
function renderChart(data) {
  const container = document.getElementById('chart-content');
  if (!container) return;

  // 날짜 목록
  const dates = [...new Set(data.series.map(r => r.report_date))].sort();
  const dailyMap = {};
  data.daily.forEach(d => { dailyMap[d.report_date] = d; });

  // 날짜 선택 select
  const select = document.getElementById('chart-date-select');
  if (!select) return;
  select.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join('');
  select.value = dates[0];

  function drawForDate(date) {
    const rows = data.series.filter(r => r.report_date === date);
    const day  = dailyMap[date] || {};

    // 일 KPI 필
    const kpiEl = document.getElementById('chart-day-kpi');
    if (kpiEl) {
      const avoided  = day.avoided_kwh ?? 0;
      const avoidPct = day.avoided_pct ?? 0;
      const cls = signClass(avoided);
      kpiEl.innerHTML = `
        <span class="day-kpi-pill day-kpi-pill--${cls === 'saving' ? 'saving' : cls === 'overuse' ? 'overuse' : 'neutral'}">
          일 절감: ${avoided >= 0 ? '+' : ''}${fmt.num(avoided,1)} kWh
        </span>
        <span class="day-kpi-pill day-kpi-pill--${cls === 'saving' ? 'saving' : cls === 'overuse' ? 'overuse' : 'neutral'}">
          절감률: ${(avoidPct >= 0 ? '+' : '')+avoidPct.toFixed(2)}%
        </span>
        <span class="day-kpi-pill day-kpi-pill--neutral">
          베이스라인: ${fmt.kwh(day.baseline_kwh)}
        </span>
        <span class="day-kpi-pill day-kpi-pill--neutral">
          실제: ${fmt.kwh(day.actual_kwh)}
        </span>
      `;
    }

    // SVG 그리기
    const svgEl = document.getElementById('chart-svg');
    if (!svgEl) return;

    const W = 800, H = 320;
    const PAD = { top:20, right:20, bottom:40, left:58 };
    const chartW = W - PAD.left - PAD.right;
    const chartH = H - PAD.top  - PAD.bottom;

    const allVals = rows.flatMap(r => [r.p10, r.p90, r.actual]).filter(v => v != null);
    const minV = Math.min(...allVals) * 0.95;
    const maxV = Math.max(...allVals) * 1.05;

    const xScale = i  => PAD.left + (i / (rows.length - 1 || 1)) * chartW;
    const yScale = v  => PAD.top  + chartH - ((v - minV) / (maxV - minV)) * chartH;

    // --- 밴드 (p10~p90) ---
    const bandPath = rows.map((r,i) => (i===0?'M':'L')+xScale(i)+','+yScale(r.p10)).join(' ')
      + ' ' + [...rows].reverse().map((r,i,arr) => 'L'+xScale(arr.length-1-i)+','+yScale(r.p90)).join(' ')
      + ' Z';

    // 절감 밴드: actual < p50 구간 (초록)
    // SVG y축은 아래로 커지므로 actual < p50 → yActual > yP50 → 음수 height 방지
    let savingFills = [];
    rows.forEach((r,i) => {
      if (r.actual < r.p50) {
        const x = xScale(i);
        const yActual = yScale(r.actual);
        const yP50   = yScale(r.p50);
        const rectY  = Math.min(yActual, yP50);
        const rectH  = Math.abs(yP50 - yActual);
        if (rectH > 0) {
          savingFills.push(`<rect x="${x-3}" y="${rectY}" width="6" height="${rectH}" fill="rgba(34,197,94,.18)" rx="1"/>`);
        }
      }
    });

    // --- p50 점선 ---
    const p50Path = rows.map((r,i) => (i===0?'M':'L')+xScale(i)+','+yScale(r.p50)).join(' ');
    // --- actual 실선 ---
    const actualPath = rows.map((r,i) => (i===0?'M':'L')+xScale(i)+','+yScale(r.actual)).join(' ');

    // X축 틱 (매 4시간)
    const xTicks = rows.filter(r => r.hour % 4 === 0).map(r => {
      const i = rows.findIndex(rr => rr.hour === r.hour);
      return { x: xScale(i), label: `${String(r.hour).padStart(2,'0')}:00` };
    });
    // Y축 틱 (5개)
    const yStep = (maxV - minV) / 4;
    const yTicks = Array.from({length:5}, (_,i) => {
      const v = minV + yStep * i;
      return { y: yScale(v), label: fmt.num(v, 0) };
    });

    // 점 (확실한 절감/초과)
    const dots = rows.map((r,i) => {
      if (r.is_confirmed_saving) {
        return `<circle cx="${xScale(i)}" cy="${yScale(r.actual)}" r="4" fill="var(--color-saving)" opacity=".85"/>`;
      }
      if (r.is_overuse) {
        return `<circle cx="${xScale(i)}" cy="${yScale(r.actual)}" r="4" fill="var(--color-overuse)" opacity=".85"/>`;
      }
      return '';
    }).join('');

    // 히트 영역 (툴팁)
    const hitRects = rows.map((r,i) => {
      const x = xScale(i);
      const cls = r.is_confirmed_saving ? 'saving' : r.is_overuse ? 'overuse' : '';
      const avoTxt = r.avoided_kwh >= 0
        ? `+${fmt.num(r.avoided_kwh,1)} kWh (절감)`
        : `${fmt.num(r.avoided_kwh,1)} kWh (초과)`;
      return `<rect x="${x-8}" y="${PAD.top}" width="16" height="${chartH}"
        fill="transparent" data-hour="${r.hour}" data-actual="${r.actual}"
        data-p50="${r.p50}" data-p10="${r.p10}" data-p90="${r.p90}"
        data-avoided="${avoTxt}" data-cls="${cls}"
        class="chart-hit" style="cursor:crosshair"/>`;
    }).join('');

    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgEl.innerHTML = `
      <!-- 배경 그리드 -->
      ${yTicks.map(t => `<line x1="${PAD.left}" y1="${t.y}" x2="${W-PAD.right}" y2="${t.y}" stroke="var(--color-border)" stroke-width="1"/>`).join('')}

      <!-- 밴드 (p10~p90) -->
      <path d="${bandPath}" fill="rgba(96,165,250,.10)" stroke="none"/>

      <!-- 절감 구간 강조 -->
      ${savingFills.join('')}

      <!-- p50 점선 -->
      <path d="${p50Path}" fill="none" stroke="var(--color-p50)" stroke-width="1.5"
            stroke-dasharray="5,4" opacity=".8"/>

      <!-- actual 실선 -->
      <path d="${actualPath}" fill="none" stroke="var(--color-actual)" stroke-width="2"/>

      <!-- 확실한 절감·초과 점 -->
      ${dots}

      <!-- 히트 영역 -->
      ${hitRects}

      <!-- Y축 레이블 -->
      ${yTicks.map(t => `<text x="${PAD.left-6}" y="${t.y+4}" text-anchor="end" font-size="10" fill="var(--color-muted)">${t.label}</text>`).join('')}

      <!-- X축 레이블 -->
      ${xTicks.map(t => `<text x="${t.x}" y="${H-8}" text-anchor="middle" font-size="10" fill="var(--color-muted)">${t.label}</text>`).join('')}

      <!-- 축 선 -->
      <line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${H-PAD.bottom}" stroke="var(--color-border)" stroke-width="1"/>
      <line x1="${PAD.left}" y1="${H-PAD.bottom}" x2="${W-PAD.right}" y2="${H-PAD.bottom}" stroke="var(--color-border)" stroke-width="1"/>

      <!-- Y축 단위 -->
      <text x="12" y="${H/2}" text-anchor="middle" font-size="10" fill="var(--color-muted)"
            transform="rotate(-90,12,${H/2})">kWh</text>
    `;

    // 툴팁
    const tooltip = document.getElementById('chart-tooltip');
    svgEl.querySelectorAll('.chart-hit').forEach(rect => {
      rect.addEventListener('mousemove', e => {
        if (!tooltip) return;
        const hour    = rect.dataset.hour;
        const actual  = parseFloat(rect.dataset.actual);
        const p50     = parseFloat(rect.dataset.p50);
        const p10     = parseFloat(rect.dataset.p10);
        const p90     = parseFloat(rect.dataset.p90);
        const avoided = rect.dataset.avoided;
        const cls     = rect.dataset.cls;
        const valCls  = cls === 'saving' ? 'tooltip-saving' : cls === 'overuse' ? 'tooltip-overuse' : '';
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top  = (e.clientY - 10) + 'px';
        tooltip.innerHTML = `
          <div class="tooltip-row"><span class="tooltip-label">시각</span><span class="tooltip-val">${date} ${String(hour).padStart(2,'0')}:00</span></div>
          <div class="tooltip-row"><span class="tooltip-label">실제</span><span class="tooltip-val">${fmt.num(actual,1)} kWh</span></div>
          <div class="tooltip-row"><span class="tooltip-label">P50 (베이스라인)</span><span class="tooltip-val">${fmt.num(p50,1)} kWh</span></div>
          <div class="tooltip-row"><span class="tooltip-label">P10~P90</span><span class="tooltip-val">${fmt.num(p10,0)}~${fmt.num(p90,0)} kWh</span></div>
          <div class="tooltip-row"><span class="tooltip-label">절감/초과</span><span class="tooltip-val ${valCls}">${avoided}</span></div>
        `;
      });
      rect.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.display = 'none';
      });
    });
  }

  // 초기 렌더
  drawForDate(select.value);
  select.addEventListener('change', () => drawForDate(select.value));
}

/* ============================================================
   4. 스코어카드
   ============================================================ */
function renderScorecard(data) {
  const sc = data.scorecard;
  const el = document.getElementById('scorecard-content');
  if (!el) return;

  const avoided = sc.avoided_sum_kwh;
  const avoidPct = sc.avoided_pct_display;
  const avoidPerSqm = sc.avoided_per_sqm_kwh;
  const cls = signClass(avoided);
  const label = cls === 'saving' ? '절감' : cls === 'overuse' ? '초과' : '균형';

  const cards = [
    {
      label: '총 베이스라인',
      value: fmt.num(sc.baseline_sum_kwh, 0),
      unit: 'kWh',
      badge: `${sc.n_rows}시간 합계`,
      cls: 'neutral',
    },
    {
      label: '총 실제 사용량',
      value: fmt.num(sc.actual_sum_kwh, 0),
      unit: 'kWh',
      badge: `${sc.n_rows}시간 합계`,
      cls: 'neutral',
    },
    {
      label: '순 절감량',
      value: (avoided >= 0 ? '+' : '') + fmt.num(avoided, 1),
      unit: 'kWh',
      badge: label,
      cls: cls === 'saving' ? 'saving' : cls === 'overuse' ? 'overuse' : 'neutral',
    },
    {
      label: '절감률',
      value: (avoidPct >= 0 ? '+' : '') + avoidPct.toFixed(2),
      unit: '%',
      badge: label,
      cls: cls === 'saving' ? 'saving' : cls === 'overuse' ? 'overuse' : 'neutral',
    },
    {
      label: '면적당 절감',
      value: (avoidPerSqm >= 0 ? '+' : '') + fmt.num(avoidPerSqm, 4),
      unit: 'kWh/㎡',
      badge: '총 연면적 기준',
      cls: signClass(avoidPerSqm) === 'saving' ? 'saving' : signClass(avoidPerSqm) === 'overuse' ? 'overuse' : 'neutral',
    },
    {
      label: '확실한 절감 시간',
      value: fmt.num(sc.confirmed_saving_hours),
      unit: '시간',
      badge: '실제 < P10',
      cls: 'saving',
    },
    {
      label: '초과 시간',
      value: fmt.num(sc.overuse_hours),
      unit: '시간',
      badge: '실제 > P90',
      cls: 'overuse',
    },
    {
      label: '검증 시간 수',
      value: fmt.num(sc.n_rows),
      unit: '시간',
      badge: `${sc.n_rows / 24}일 × 24h`,
      cls: 'neutral',
    },
  ];

  el.innerHTML = `
    <div class="scorecard-grid">
      ${cards.map(c => `
        <div class="sc-card sc-card--${c.cls}">
          <div class="sc-card__label">${c.label}</div>
          <div class="sc-card__value">${c.value}<span class="sc-card__unit">${c.unit}</span></div>
          <span class="sc-card__badge">${c.badge}</span>
        </div>
      `).join('')}
    </div>
    <div class="info-box">
      <strong>해석:</strong> 6월 전체 순 절감률은 약 <strong>${avoidPct.toFixed(2)}%</strong>로
      베이스라인 대비 거의 비슷하거나 약간 초과합니다.
      하지만 <strong>확실한 절감 시간(actual &lt; P10)이 ${sc.confirmed_saving_hours}시간</strong>으로
      초과 시간(${sc.overuse_hours}시간)보다 약 ${sc.overuse_hours > 0 ? Math.round(sc.confirmed_saving_hours / sc.overuse_hours * 10)/10 + '배' : '—'} 많습니다.
      동결 베이스라인 덕분에 절감 여부를 공정하게 판단할 수 있습니다.
    </div>
  `;
}

/* ============================================================
   5. 리더보드
   ============================================================ */
function renderLeaderboard(data) {
  const el = document.getElementById('leaderboard-content');
  if (!el) return;

  // rank 기준 정렬
  const lb = [...data.leaderboard].sort((a, b) => a.rank - b.rank);

  // 절감률 절댓값 최대
  const maxAbsPct = Math.max(...lb.map(r => Math.abs(r.avoided_pct)));

  const rankBadge = r => `<span class="lb-rank lb-rank--${r}">${r}</span>`;
  const pctCls    = v => v >= 0 ? 'saving' : 'overuse';

  el.innerHTML = `
    <div class="lb-table-wrap">
      <table class="lb-table">
        <thead>
          <tr>
            <th>순위</th>
            <th>라운드</th>
            <th>절감률 %</th>
            <th>면적당 (kWh/㎡)</th>
            <th>순 절감 kWh</th>
            <th>기간(일)</th>
          </tr>
        </thead>
        <tbody>
          ${lb.map(r => `
            <tr>
              <td>${rankBadge(r.rank)}</td>
              <td>${r.round}</td>
              <td style="color:var(--color-${pctCls(r.avoided_pct)}); font-weight:700;">
                ${(r.avoided_pct >= 0 ? '+' : '')}${(r.avoided_pct*100).toFixed(2)}%
              </td>
              <td style="color:var(--color-${pctCls(r.avoided_per_sqm_kwh)});">
                ${(r.avoided_per_sqm_kwh >= 0 ? '+' : '')}${r.avoided_per_sqm_kwh.toFixed(4)}
              </td>
              <td style="color:var(--color-${pctCls(r.avoided_kwh)});">
                ${(r.avoided_kwh >= 0 ? '+' : '')}${fmt.num(r.avoided_kwh, 1)}
              </td>
              <td>${r.days}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>

    <div class="lb-bar-wrap">
      <div class="lb-bar-title">주차별 절감률 비교</div>
      ${lb.map(r => {
        const cls = pctCls(r.avoided_pct);
        const barW = maxAbsPct > 0 ? (Math.abs(r.avoided_pct) / maxAbsPct * 100) : 0;
        return `
          <div class="lb-bar-row">
            <span class="lb-bar-name">${rankBadge(r.rank)} ${r.round}</span>
            <div class="lb-bar-track">
              <div class="lb-bar-fill lb-bar-fill--${cls}" style="width:${barW}%"></div>
            </div>
            <span class="lb-bar-pct lb-bar-pct--${cls}">
              ${(r.avoided_pct >= 0 ? '+' : '')}${(r.avoided_pct*100).toFixed(2)}%
            </span>
          </div>
        `;
      }).join('')}
    </div>
    <p class="lb-note">
      ※ 여기선 같은 학교의 주차를 참가자로 시연했습니다 — 학교 A·B·C로 바꿔도 동일하게 동작합니다.
      면적당(kWh/㎡) 기준으로 채점하면 학교 규모가 달라도 공정한 비교가 됩니다.
    </p>
  `;
}

/* ============================================================
   6. 공정성·용어
   ============================================================ */
function renderGlossary(data) {
  const el = document.getElementById('glossary-content');
  if (!el) return;

  el.innerHTML = `
    <div class="fairness-box">
      <strong>왜 절대 kWh가 아닌 베이스라인 대비 %·면적당으로 채점해야 할까요?</strong><br><br>
      학교마다 규모(연면적)가 다르고, 기준 소비량도 다릅니다.
      <strong>절대 kWh로 비교하면 큰 학교가 항상 유리</strong>합니다.
      베이스라인 대비 절감률(%)은 각 학교의 자기 기준 대비 성과를 측정하고,
      면적당(kWh/㎡)은 규모 차이를 정규화합니다.
      <br><br>
      이 두 지표를 결합하면 <strong>1천 ㎡짜리 소규모 학교와 10만 ㎡ 대형 학교가 같은 선에서 경쟁</strong>할 수 있습니다.
      그리고 이 비교를 가능하게 하는 베이스라인을 <strong>큰 설비 없이</strong>
      시간별 검침 + 공공 날씨 + 학사일정만으로 만든 것이 이 프로젝트의 핵심입니다.
    </div>

    <div class="glossary-grid">
      ${data.glossary.map(g => `
        <div class="glossary-card">
          <div class="glossary-card__term">${g.term}</div>
          <div class="glossary-card__desc">${g.desc}</div>
        </div>
      `).join('')}
    </div>
  `;
}

/* ---------- 진입점 --------------------------------------- */
document.addEventListener('DOMContentLoaded', main);
