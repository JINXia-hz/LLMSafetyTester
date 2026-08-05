/* LLMSEC 安全评估工作台前端逻辑 */

// ---------- 全局状态与常量 ----------
// 绢本金碧 · 唐化配色：石青主色、土红撞色、描金点缀
const C = {
  primary: '#46586B', accent: '#A85B43', warn: '#A85B43',
  safe: '#75876B', ochre: '#B98A44', deep: '#7A4A35', gold: '#BFA03C',
  muted: '#8A8571', text: '#2F343B',
};
const PLOT_CFG = { responsive: true, displayModeBar: false };
const PLOT_FONT = { family: 'ui-sans-serif, system-ui, sans-serif', color: C.text };

// ---------- 主题（绢本纸日 / 石窟夜色） ----------
const THEME_CHART = {
  light: { text: '#2F343B', grid: '#E3D8B8', primary: '#46586B', muted: '#8A8571' },
  dark:  { text: '#E4D8BE', grid: '#454D55', primary: '#7A96AF', muted: '#9A917B' },
};
let theme = localStorage.getItem('llmsec-theme') || 'light';

function tangLayout() {
  const t = THEME_CHART[theme];
  const axis = { gridcolor: t.grid, zerolinecolor: t.grid };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: PLOT_FONT.family, color: t.text },
    xaxis: axis, yaxis: { ...axis },
    polar: { bgcolor: 'rgba(0,0,0,0)', radialaxis: axis, angularaxis: axis },
  };
}

// Plotly 统一走绢本金碧主题模板（深合并，业务 layout 的 xaxis/yaxis 不会被底色覆盖）
const _newPlot = Plotly.newPlot.bind(Plotly);
Plotly.newPlot = (id, traces, layout = {}, cfg) => {
  const base = tangLayout();
  const merged = {
    ...base, ...layout,
    font: { ...base.font, ...(layout.font || {}) },
    xaxis: { ...base.xaxis, ...(layout.xaxis || {}) },
    yaxis: { ...base.yaxis, ...(layout.yaxis || {}) },
    polar: layout.polar ? { ...base.polar, ...layout.polar } : undefined,
  };
  if (!layout.polar) delete merged.polar;
  return _newPlot(id, traces, merged, cfg);
};

// 匾额等级印章：等级 → 印字
const SEAL_CHARS = { safe: '安', allergic: '警', vulnerable: '伤', broken: '破', inconclusive: '?' };
function setBanner(level) {
  $('ov_banner').className = 'banner plaque mb-2 level-' + level;
  const seal = $('ov_seal');
  seal.className = 'seal level-' + level;
  seal.textContent = SEAL_CHARS[level] || '?';
  seal.classList.remove('seal-anim'); void seal.offsetWidth; seal.classList.add('seal-anim'); // 重触发盖印
}

let currentRun = '';           // '' = 最新
let activeSection = 'overview';
let lastOverview = null;       // 最近一次总览数据（批次对比用）
const loaded = {};             // section -> 已加载的 run

const $ = id => document.getElementById(id);
const fmtPct = v => (v == null ? 'N/A' : (v * 100).toFixed(1) + '%');
const fmtNum = (v, d = 1) => (v == null ? 'N/A' : Number(v).toFixed(d));

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}
function runQuery() { return currentRun ? `?run=${encodeURIComponent(currentRun)}` : ''; }
function setStatus(msg) { $('status').textContent = msg || ''; }

// HTML 转义：服务器字符串插入 innerHTML 前统一过这道
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
// markdown → 安全 HTML（marked 不过滤原始 HTML，必须过 DOMPurify）
const mdSafe = md => DOMPurify.sanitize(marked.parse(md || ''));
// 清空图表：切到无数据批次时避免上一批次的图残留
function clearCharts(ids) {
  ids.forEach(id => { const el = $(id); if (el) { Plotly.purge(el); el.innerHTML = ''; } });
}

// ---------- 导航 ----------
const SECTIONS = ['overview', 'threats', 'report', 'clusters', 'model', 'run'];
document.querySelectorAll('#nav .nav-item').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('#nav .nav-item').forEach(n => n.classList.remove('active'));
    el.classList.add('active');
    activeSection = el.dataset.section;
    history.replaceState(null, '', '#' + activeSection);  // 板块可直达/可收藏
    document.querySelectorAll('.section').forEach(s => s.classList.remove('visible'));
    $('sec-' + activeSection).classList.add('visible');
    loadSection(activeSection);
  });
});

function loadSection(name) {
  if (loaded[name] === currentRun) return;
  loaded[name] = currentRun;
  ({ overview: loadOverview, threats: loadThreats, report: loadReport,
     clusters: loadClusters, model: loadModel, run: loadRunSection })[name]();
}
function invalidate() { for (const k in loaded) delete loaded[k]; loadSection(activeSection); }

// ---------- 主题切换（依赖 $ 与 invalidate，定义于此） ----------
function applyTheme(t, rerender = true) {
  theme = t;
  localStorage.setItem('llmsec-theme', t);
  document.documentElement.dataset.theme = t === 'dark' ? 'dark' : '';
  const tc = THEME_CHART[t];
  PLOT_FONT.color = tc.text;
  C.primary = tc.primary; C.muted = tc.muted;   // 图表系列色随主题微调
  $('themeBtn').textContent = t === 'dark' ? '☀️ 纸日' : '🌙 夜色';
  if (rerender) invalidate();                    // 重绘当前板块图表
}
$('themeBtn').addEventListener('click', () => applyTheme(theme === 'dark' ? 'light' : 'dark'));

// ---------- 运行批次 ----------
async function loadRuns() {
  const data = await api('/api/runs');
  const sel = $('runSelect');
  sel.innerHTML = '<option value="">最新</option>';
  const cmp = $('cmpSelect');
  // 占位空项：默认不选中任何批次，避免"未选择"分支成死代码
  cmp.innerHTML = '<option value="">选择对比批次…</option>';
  data.runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.name;
    // 富化条目：印字等级 + 目标模型 + ASR，一眼分辨批次质量（option 纯文本，无法上色）
    opt.textContent = r.has_report
      ? `${SEAL_CHARS[r.security_level] || '▫'} ${r.name} · ${r.target_model || '?'} · ASR ${fmtPct(r.asr)}`
      : `${r.name} (无报告)`;
    sel.appendChild(opt);
    cmp.appendChild(opt.cloneNode(true));   // 批次对比下拉共用同一清单
  });
  // 任务完成后列表会重建：保留用户已选批次，避免被弹回"最新"
  if (currentRun && [...sel.options].some(o => o.value === currentRun)) sel.value = currentRun;
}
$('runSelect').addEventListener('change', e => { currentRun = e.target.value; invalidate(); });
$('refreshBtn').addEventListener('click', async () => { await loadRuns(); invalidate(); });

// ---------- 总览 ----------
async function loadOverview() {
  loadTrend();   // 趋势是跨批次全局数据，与当前批次无关，独立加载独立失败
  try {
    const d = await api('/api/overview' + runQuery());
    if (!d.available) {
      lastOverview = null;
      setBanner('inconclusive');
      $('ov_target').textContent = '目标模型: -';
      $('ov_verdict').textContent = d.message || '暂无运行数据';
      $('ov_recommendation').textContent = '';
      ['ov_asr', 'ov_fpr', 'ov_boundary', 'ov_conf', 'ov_tested', 'ov_above', 'ov_tax']
        .forEach(id => { $(id).textContent = '-'; });
      $('ov_tax_sub').textContent = '';
      clearCharts(['chart_radar', 'chart_harm']);
      return;
    }
    const level = d.security_level || 'inconclusive';
    lastOverview = d;
    if (cmpActive) renderCompare();   // 切批次后对比面板"当前"列同步重渲染
    setBanner(level);
    $('ov_target').textContent = `目标模型: ${d.target_model || '-'}  ·  批次 ${d.run}`;
    $('ov_verdict').textContent = d.overall_verdict || level.toUpperCase();
    $('ov_recommendation').textContent = d.recommendation ? '💡 ' + d.recommendation : '';
    // stale_report 提示：存在更新批次时展示；切回最新批次（无 message）时清除旧文案
    setStatus(d.message || '');

    $('ov_asr').textContent = fmtPct(d.asr);
    $('ov_fpr').textContent = fmtPct(d.fpr);
    $('ov_boundary').textContent = fmtNum(d.boundary_elo, 0);
    $('ov_conf').textContent = fmtPct(d.boundary_confidence);
    $('ov_tested').textContent = `${d.total_tested}/${d.total_methods}`;
    $('ov_above').textContent = d.predicted_above_boundary != null
      ? `${d.methods_above_boundary} (实测${d.tested_above_boundary}/预测${d.predicted_above_boundary})`
      : d.methods_above_boundary;

    // 越狱税：优先基线对比呈现；null = 该批未测（攻击集无数学探针）
    if (d.jailbreak_tax_baseline_accuracy != null && d.jailbreak_tax_attack_accuracy != null) {
      $('ov_tax').textContent = `${fmtPct(d.jailbreak_tax_baseline_accuracy)} → ${fmtPct(d.jailbreak_tax_attack_accuracy)}`;
      const drop = d.jailbreak_tax_drop != null ? `退化 ${fmtPct(d.jailbreak_tax_drop)}` : '';
      const probed = d.jailbreak_tax_probed != null ? `探针 ${d.jailbreak_tax_probed} 条` : '';
      $('ov_tax_sub').textContent = [drop, probed].filter(Boolean).join(' · ');
    } else if (d.jailbreak_tax_mean != null) {
      $('ov_tax').textContent = fmtNum(d.jailbreak_tax_mean, 2);
      const hi = d.jailbreak_tax_high_ratio != null ? `高税占比 ${fmtPct(d.jailbreak_tax_high_ratio)}` : '';
      const probed = d.jailbreak_tax_probed != null ? `探针 ${d.jailbreak_tax_probed} 条` : '';
      $('ov_tax_sub').textContent = ['无基线对照', hi, probed].filter(Boolean).join(' · ');
    } else {
      $('ov_tax').textContent = '未测试';
      $('ov_tax_sub').textContent = '攻击集无数学探针';
    }

    // 雷达图（闭合）
    const r = d.radar;
    Plotly.newPlot('chart_radar', [{
      type: 'scatterpolar',
      r: [...r.values, r.values[0]],
      theta: [...r.labels, r.labels[0]],
      fill: 'toself',
      fillcolor: 'rgba(70,88,107,0.18)',
      line: { color: C.primary, width: 2 },
      marker: { size: 6, color: C.primary },
    }], {
      polar: { radialaxis: { range: [0, 1], tickformat: '.0%', tickfont: { size: 10 } } },
      margin: { t: 24, b: 24, l: 62, r: 62 }, font: PLOT_FONT, showlegend: false,
    }, PLOT_CFG);

    const harm = Object.entries(d.harm_type_asr || {}).sort((a, b) => b[1] - a[1]);
    if (!harm.length) {
      clearCharts(['chart_harm']);
      $('chart_harm').innerHTML = '<div class="text-xs py-8 text-center" style="color: var(--c-muted);">该批次无类别统计数据</div>';
    } else {
      Plotly.newPlot('chart_harm', [{
        x: harm.map(i => i[0]), y: harm.map(i => i[1]), type: 'bar',
        text: harm.map(i => fmtPct(i[1])), textposition: 'auto',
        marker: { color: C.accent },
      }], { yaxis: { tickformat: '.0%', range: [0, 1] }, margin: { t: 10 }, font: PLOT_FONT }, PLOT_CFG);
    }
  } catch (e) { setStatus('总览加载失败: ' + e.message); }
}

// ---------- 安全趋势（跨批次，/api/trend） ----------
let trendTarget = '';          // '' = 全部目标
let trendTargetCounts = {};    // {target: 批次数}，仅在"全部"视图时从完整数据统计
async function loadTrend() {
  const el = $('chart_trend');
  if (!el) return;
  try {
    const d = await api('/api/trend' + (trendTarget ? '?target=' + encodeURIComponent(trendTarget) : ''));
    // "全部"视图时统计各目标批次数（供 chip 标签显示）
    if (!trendTarget) {
      trendTargetCounts = {};
      (d.trend || []).forEach(p => {
        trendTargetCounts[p.target] = (trendTargetCounts[p.target] || 0) + 1;
      });
    }
    // 目标过滤 chips：有目标时始终渲染（含"全部"按钮），不再因选中后 targets 缩为 1 而消失
    const chips = $('trendTargets');
    chips.innerHTML = '';
    const allTargets = d.targets || [];
    if (allTargets.length > 0) {
      const mk = (val, label) => {
        const b = document.createElement('button');
        b.className = 'btn text-xs ' + (trendTarget === val ? 'btn-primary' : 'btn-plain');
        b.textContent = label;
        b.onclick = () => { trendTarget = val; loadTrend(); };
        chips.appendChild(b);
      };
      mk('', '全部');
      allTargets.forEach(t => {
        const n = trendTargetCounts[t] || 0;
        mk(t, n > 0 ? `${t} (${n})` : t);
      });
    }
    const pts = d.trend || [];
    if (!pts.length) {
      $('trendMeta').textContent = '';
      el.innerHTML = '<div class="text-xs py-8 text-center" style="color: var(--c-muted);">暂无带报告的批次，趋势无从谈起</div>';
      return;
    }
    $('trendMeta').textContent = `（${pts.length} 个批次 · 全局数据，不随批次切换）`;
    // 按目标分组：每目标两条 trace——ELO 实线（左轴）+ ASR 点线（右轴）
    // target 为 null 的批次归入"未知目标"，避免分组键/trace 名出现 'null'
    const byTarget = {};
    pts.forEach(p => { const k = p.target || '未知目标'; (byTarget[k] = byTarget[k] || []).push(p); });
    const traces = [];
    Object.entries(byTarget).forEach(([t, arr], i) => {
      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
      arr.sort((a, b) => (a.run < b.run ? -1 : 1));
      const xs = arr.map(p => p.run);
      traces.push({
        x: xs, y: arr.map(p => p.elo), type: 'scatter', mode: 'lines+markers',
        name: `${t} · ELO`, line: { color, width: 2 }, marker: { size: 6, color },
        hovertemplate: '%{x}<br>ELO %{y:.0f}<extra>' + esc(t) + '</extra>',
      });
      traces.push({
        x: xs, y: arr.map(p => p.asr), type: 'scatter', mode: 'lines+markers',
        name: `${t} · ASR`, yaxis: 'y2',
        line: { color, width: 1.5, dash: 'dot' }, marker: { size: 5, symbol: 'circle-open', color },
        hovertemplate: '%{x}<br>ASR %{y:.1%}<extra>' + esc(t) + '</extra>',
      });
    });
    Plotly.newPlot('chart_trend', traces, {
      margin: { t: 10 }, height: 320, font: PLOT_FONT,
      xaxis: { tickangle: -30, tickfont: { size: 10 } },
      yaxis: { title: 'ELO' },
      yaxis2: { overlaying: 'y', side: 'right', tickformat: '.0%', title: 'ASR', range: [0, 1] },
      legend: { orientation: 'h', y: -0.32, font: { size: 10 } },
    }, PLOT_CFG);
  } catch (e) { $('trendMeta').textContent = '（趋势加载失败）'; }
}

// ---------- 批次对比 ----------
let cmpActive = false;
$('cmpBtn').addEventListener('click', () => {
  cmpActive = !cmpActive;
  $('cmpPanel').classList.toggle('hidden', !cmpActive);
  $('cmpSelect').classList.toggle('hidden', !cmpActive);
  $('cmpHint').classList.toggle('hidden', !cmpActive);
  $('cmpBtn').className = 'btn text-xs ' + (cmpActive ? 'btn-accent' : 'btn-plain');
  if (cmpActive) renderCompare();
});
$('cmpSelect').addEventListener('change', renderCompare);

function cmpRow(name, cur, oth, delta, goodWhenDown, suffix = '') {
  let cls = 'cmp-delta-flat', txt = '—';
  if (delta != null && Math.abs(delta) > 1e-9) {
    const good = goodWhenDown ? delta < 0 : delta > 0;
    cls = good ? 'cmp-delta-up-good' : 'cmp-delta-up-bad';
    txt = (delta > 0 ? '+' : '') + delta + suffix;
  }
  return `<tr><td>${name}</td><td>${cur}</td><td>${oth}</td><td class="${cls}">${txt}</td></tr>`;
}

async function renderCompare() {
  const other = $('cmpSelect').value;
  if (!other || !lastOverview || !lastOverview.available) {
    if (!other) { $('cmpTable').innerHTML = '<span class="text-xs" style="color: var(--c-muted);">选择一个对比批次</span>'; clearCharts(['chart_cmp_radar']); }
    return;
  }
  if (lastOverview.run === other) {
    $('cmpTable').innerHTML = '<span class="text-xs" style="color: var(--c-muted);">与当前批次相同，换个批次试试</span>';
    clearCharts(['chart_cmp_radar']);
    return;
  }
  try {
    const o = await api('/api/overview?run=' + encodeURIComponent(other));
    if (!o.available) { $('cmpTable').innerHTML = '<span class="text-xs" style="color: var(--c-muted);">该批次无总览数据</span>'; return; }
    const a = lastOverview;

    // 雷达叠加：当前=石青，对比=土红
    const r = a.radar;
    Plotly.newPlot('chart_cmp_radar', [
      { type: 'scatterpolar', r: [...r.values, r.values[0]], theta: [...r.labels, r.labels[0]],
        fill: 'toself', fillcolor: 'rgba(70,88,107,0.15)', name: a.run,
        line: { color: C.primary, width: 2 }, marker: { size: 5, color: C.primary } },
      { type: 'scatterpolar', r: [...o.radar.values, o.radar.values[0]], theta: [...o.radar.labels, o.radar.labels[0]],
        fill: 'toself', fillcolor: 'rgba(168,91,67,0.12)', name: o.run,
        line: { color: C.accent, width: 2, dash: 'dot' }, marker: { size: 5, color: C.accent } },
    ], {
      polar: { radialaxis: { range: [0, 1], tickformat: '.0%', tickfont: { size: 10 } } },
      margin: { t: 24, b: 24, l: 62, r: 62 }, font: PLOT_FONT,
      showlegend: true, legend: { orientation: 'h', y: -0.12, font: { size: 10 } },
    }, PLOT_CFG);

    // 增量表（红=变差，绿=变好）
    const pctD = (x, y) => (x == null || y == null) ? null : +(((y - x) * 100).toFixed(1));
    const numD = (x, y) => (x == null || y == null) ? null : +(y - x).toFixed(1);
    $('cmpTable').innerHTML = `<table>
      <tr><th>指标</th><th>当前 ${esc(a.run)}</th><th>对比 ${esc(o.run)}</th><th>Δ</th></tr>
      ${cmpRow('ASR 攻击成功率', fmtPct(a.asr), fmtPct(o.asr), pctD(a.asr, o.asr), true, '%')}
      ${cmpRow('FPR 误杀率', fmtPct(a.fpr), fmtPct(o.fpr), pctD(a.fpr, o.fpr), true, '%')}
      ${cmpRow('ELO 安全边界', fmtNum(a.boundary_elo, 0), fmtNum(o.boundary_elo, 0), numD(a.boundary_elo, o.boundary_elo), false)}
      ${cmpRow('边界置信度', fmtPct(a.boundary_confidence), fmtPct(o.boundary_confidence), pctD(a.boundary_confidence, o.boundary_confidence), false, '%')}
      ${cmpRow('已测方法', `${a.total_tested}/${a.total_methods}`, `${o.total_tested}/${o.total_methods}`, numD(a.total_tested, o.total_tested), false)}
      ${cmpRow('边界以上威胁', a.methods_above_boundary, o.methods_above_boundary, numD(a.methods_above_boundary, o.methods_above_boundary), true)}
    </table>`;
  } catch (e) { $('cmpTable').innerHTML = '<span class="text-xs">对比加载失败: ' + esc(e.message) + '</span>'; }
}

// ---------- 威胁表格：点击表头排序 ----------
let threatRows = [];
let threatSort = { key: 'elo', dir: -1 };   // 默认 ELO 降序
const THREAT_KEY = { elo: t => t.elo ?? -Infinity, asr: t => t.asr ?? -Infinity, tax: t => t.mean_jailbreak_tax ?? -Infinity, tested: t => t.tested ? 1 : 0 };

function renderThreatTable() {
  const tbody = $('threatTable');
  tbody.innerHTML = '';
  const get = THREAT_KEY[threatSort.key];
  const rows = [...threatRows].sort((a, b) => (get(a) - get(b)) * threatSort.dir);
  rows.forEach(t => {
    const tr = document.createElement('tr');
    tr.style.borderTop = '1px solid #E3D8B8';
    const badge = t.tested
      ? '<span class="badge badge-gt">实测</span>'
      : '<span class="badge badge-pred">预测</span>';
    const ci = t.ci95 ? `[${fmtNum(t.ci95[0], 0)}, ${fmtNum(t.ci95[1], 0)}]` : '-';
    tr.innerHTML = `<td class="py-2 pr-4 font-mono text-xs">${esc(t.method)}</td>
      <td class="py-2 pr-4 font-semibold">${fmtNum(t.elo)}</td>
      <td class="py-2 pr-4">${t.asr != null ? fmtPct(t.asr) : '-'}</td>
      <td class="py-2 pr-4">${t.mean_jailbreak_tax != null ? fmtNum(t.mean_jailbreak_tax, 2) : '-'}</td>
      <td class="py-2 pr-4">${badge}</td><td class="py-2 text-xs">${ci}</td>`;
    tbody.appendChild(tr);
  });
  document.querySelectorAll('#sec-threats th[data-sort]').forEach(th => {
    th.classList.toggle('sort-asc', th.dataset.sort === threatSort.key && threatSort.dir === 1);
    th.classList.toggle('sort-desc', th.dataset.sort === threatSort.key && threatSort.dir === -1);
  });
}
document.querySelectorAll('#sec-threats th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const k = th.dataset.sort;
    threatSort = threatSort.key === k ? { key: k, dir: -threatSort.dir } : { key: k, dir: -1 };
    renderThreatTable();
  });
});

// ---------- 威胁看板 ----------
async function loadThreats() {
  try {
    const [d, elo] = await Promise.all([api('/api/threats' + runQuery()), api('/api/elo' + runQuery())]);
    if (!d.available) {
      clearCharts(['chart_top_threats', 'chart_convergence']);
      $('threatTable').innerHTML = '';
      $('defenseList').innerHTML = '<span style="color: var(--c-muted);">无数据</span>';
      $('upsetList').innerHTML = '<span style="color: var(--c-muted);">无数据</span>';
      return;
    }

    // 防御方当前 ELO 描金参考线：威胁条与防御边界的相对位置一目了然
    const defEntries = Object.entries(elo.round_defender_elos || {});
    let defShapes = [], defAnno = [];
    if (defEntries.length && defEntries[0][1].length) {
      const dv = defEntries[0][1];
      const curElo = dv[dv.length - 1];
      defShapes = [{ type: 'line', x0: curElo, x1: curElo, yref: 'paper', y0: 0, y1: 1,
                     line: { color: C.gold, width: 1.5, dash: 'dash' } }];
      defAnno = [{ x: curElo, y: 1, yref: 'paper', text: '防御方 ELO', showarrow: false,
                   xanchor: 'left', yanchor: 'bottom', font: { size: 10, color: C.gold } }];
    }

    const top = (d.top_threats || []).slice(0, 10);
    Plotly.newPlot('chart_top_threats', [{
      y: top.map(t => t.method).reverse(),
      x: top.map(t => t.elo).reverse(),
      type: 'bar', orientation: 'h',
      text: top.map(t => fmtNum(t.elo, 0)).reverse(), textposition: 'auto',
      marker: { color: top.map(t => t.tested ? C.warn : C.muted).reverse() },
    }], { margin: { t: 10 }, height: 380, font: PLOT_FONT,
          xaxis: { title: 'ELO（红=实测，灰=预测）' },
          shapes: defShapes, annotations: defAnno }, PLOT_CFG);

    // 收敛曲线
    const series = Object.entries(elo.round_defender_elos || {});
    if (series.length) {
      const [name, vals] = series[0];
      Plotly.newPlot('chart_convergence', [{
        x: vals.map((_, i) => i + 1), y: vals, type: 'scatter', mode: 'lines+markers',
        line: { color: C.primary, width: 2 }, marker: { size: 7 },
        name,
      }], { margin: { t: 10 }, height: 380, font: PLOT_FONT,
            xaxis: { title: '轮次' }, yaxis: { title: `防御方 ELO（${name}）` } }, PLOT_CFG);
    }

    // 威胁表格（数据入缓存，交给可排序渲染器）
    threatRows = d.top_threats || [];
    renderThreatTable();

    // 防御强项
    const dl = $('defenseList');
    dl.innerHTML = '';
    (d.strong_defenses || []).slice(0, 8).forEach(t => {
      dl.innerHTML += `<div class="flex justify-between"><span class="font-mono text-xs">${esc(t.method)}</span>
        <span style="color: var(--c-safe); font-weight:600;">ELO ${fmtNum(t.elo, 0)}</span></div>`;
    });
    if (!dl.innerHTML) dl.innerHTML = '<span style="color: var(--c-muted);">无数据</span>';

    // 意外盲区（兼容 list / {weakness:[...]} 两种结构）
    const ul = $('upsetList');
    ul.innerHTML = '';
    let upsets = d.upsets || [];
    if (!Array.isArray(upsets)) upsets = upsets.weakness || [];
    upsets.slice(0, 8).forEach(u => {
      ul.innerHTML += `<div class="flex justify-between">
        <span class="font-mono text-xs">${esc(u.attacker || u.method || '')}</span>
        <span style="color: var(--c-warn); font-weight:600;">gap ${fmtNum(u.elo_gap ?? u.surprise, 0)}</span></div>`;
    });
    if (!ul.innerHTML) ul.innerHTML = '<span style="color: var(--c-muted);">无数据</span>';
  } catch (e) { setStatus('威胁看板加载失败: ' + e.message); }
}

// ---------- 报告 ----------
async function loadReport() {
  try {
    const d = await api('/api/report-md' + runQuery());
    const nav = $('reportNav'), body = $('reportBody');
    nav.innerHTML = ''; body.innerHTML = '';
    if (!d.available) {
      body.innerHTML = '<div class="card text-sm" style="color: var(--c-muted);">该批次没有 security_report.md</div>';
      return;
    }
    // 原始 .md 下载（打印美化交给浏览器打印）
    nav.innerHTML = `<a href="/api/report/download${runQuery()}" class="block px-2 py-1 mb-2 rounded text-xs font-semibold text-center"
      style="border: 1px solid var(--c-gold); color: var(--c-gold);">⬇ 下载报告 (.md)</a>`;
    // 按 ## 分段
    const chunks = d.markdown.split(/^## /m);
    const head = chunks[0];
    const headTitle = (head.match(/^# (.+)$/m) || [])[1] || '安全评估报告';
    body.innerHTML += `<div class="card report-body"><h1>${esc(headTitle)}</h1>${mdSafe(head.replace(/^# .+$/m, ''))}</div>`;
    chunks.slice(1).forEach((chunk, i) => {
      const nl = chunk.indexOf('\n');
      const title = nl > 0 ? chunk.slice(0, nl).trim() : chunk.trim();
      const content = nl > 0 ? chunk.slice(nl + 1) : '';
      const anchor = `rep-${i}`;
      nav.innerHTML += `<a href="#${anchor}" class="rep-link block px-2 py-1 rounded hover:bg-stone-100" style="color: var(--c-primary);">${esc(title)}</a>`;
      const div = document.createElement('div');
      div.className = 'card report-body';
      div.id = anchor;
      div.innerHTML = `<h2>${esc(title)}</h2>${mdSafe(content)}`;
      body.appendChild(div);
    });
  } catch (e) { setStatus('报告加载失败: ' + e.message); }
}

// ---------- 聚类分析 ----------
const CLUSTER_COLORS = [
  '#46586B', '#A85B43', '#75876B', '#B98A44', '#7A4A35', '#8a7ba8', '#BFA03C',
  '#5f8d8b', '#a87b7b', '#96896f', '#845f8d', '#63855a', '#c07a5a', '#a8948a',
];
let projMethod = 'pca';

async function loadProjection(method) {
  projMethod = method;
  $('projPcaBtn').className = 'btn text-xs ' + (method === 'pca' ? 'btn-primary' : 'btn-plain');
  $('projTsneBtn').className = 'btn text-xs ' + (method === 'tsne' ? 'btn-primary' : 'btn-plain');
  try {
    const d = await api('/api/cluster-projection?method=' + method);
    if (!d.available) { $('projMeta').textContent = '（无聚类 artifacts）'; return; }

    const byCluster = {};
    d.points.forEach(p => { (byCluster[p.cluster] = byCluster[p.cluster] || []).push(p); });
    const traces = Object.entries(byCluster).map(([cid, pts], i) => ({
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      type: 'scatter', mode: 'markers',
      name: cid === '-1' ? '噪声' : pts[0].cluster_name,
      customdata: pts.map(p => p.cluster),   // 点选联动簇卡片用
      marker: {
        size: 9,
        color: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
        opacity: pts.map(p => p.tested ? 0.95 : 0.4),
      },
      text: pts.map(p =>
        `${p.method}<br>${p.cluster_name}` +
        (p.elo != null ? `<br>ELO ${p.elo}` : '') +
        (p.tested ? '<br>实测' : '<br>预测')),
      hovertemplate: '%{text}<extra></extra>',
    }));

    const meta = method === 'pca' && d.explained_variance
      ? `两维解释方差 ${(d.explained_variance[0] * 100).toFixed(1)}% + ${(d.explained_variance[1] * 100).toFixed(1)}%`
      : method === 'tsne' ? `perplexity=${d.perplexity}` : '';
    $('projMeta').textContent = `（${d.n} 种方法 · ${meta} · 实心=实测 空心=预测 · 点选可定位簇卡片 · 数据来自最近一次聚类，全局）`;

    Plotly.newPlot('chart_projection', traces, {
      margin: { t: 10 }, height: 520, font: PLOT_FONT,
      xaxis: { title: method === 'pca' ? 'PC1' : 't-SNE 1' },
      yaxis: { title: method === 'pca' ? 'PC2' : 't-SNE 2' },
      legend: { orientation: 'h', y: -0.18, font: { size: 10 } },
    }, PLOT_CFG);

    // 点选投影点 → 滚动到对应簇卡片并描金闪高（k 切割视图的簇 id 语义不同，不联动）
    const gd = $('chart_projection');
    if (gd.removeAllListeners) gd.removeAllListeners('plotly_click');
    gd.on('plotly_click', ev => {
      const cid = ev.points && ev.points[0] && ev.points[0].customdata;
      if (cid == null) return;
      const card = document.getElementById('clcard-' + cid);
      if (!card) return;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.remove('clcard-flash'); void card.offsetWidth;   // 重触发
      card.classList.add('clcard-flash');
      setTimeout(() => card.classList.remove('clcard-flash'), 1600);
    });
  } catch (e) { $('projMeta').textContent = '（投影加载失败: ' + e.message + '）'; }
}

async function loadClusters() {
  try {
    loadProjection(projMethod);
    loadClusterTree();
    const d = await api('/api/clusters' + runQuery());
    if (!d.available) {
      ['cl_methods', 'cl_n', 'cl_sil', 'cl_db'].forEach(id => { $(id).textContent = '-'; });
      $('rvBanner').className = 'banner level-inconclusive mb-3';
      $('rvBanner').style.padding = '12px 16px';
      $('rvVerdict').textContent = '暂无聚类数据';
      $('rvStats').textContent = d.reason === 'no_cluster' ? '该批次无聚类分析结果，可先在"运行控制"跑聚类安全分析' : '';
      $('clusterCards').innerHTML = '';
      clearCharts(['chart_cluster_cover', 'chart_rv']);
      return;
    }
    $('cl_methods').textContent = d.n_methods ?? '-';
    $('cl_n').textContent = d.n_clusters ?? '-';
    $('cl_sil').textContent = fmtNum(d.validation?.silhouette, 4);
    $('cl_db').textContent = fmtNum(d.validation?.davies_bouldin, 4);

    const cl = (d.clusters || []).slice(0, 20);
    Plotly.newPlot('chart_cluster_cover', [
      {
        x: cl.map(c => c.name), y: cl.map(c => c.size), type: 'bar', name: '簇规模',
        marker: { color: C.primary },
      },
      {
        x: cl.map(c => c.name), y: cl.map(c => c.test_coverage), type: 'scatter',
        mode: 'lines+markers', name: '测试覆盖率', yaxis: 'y2',
        line: { color: C.accent, width: 2 },
      },
    ], {
      margin: { t: 10 }, font: PLOT_FONT,
      yaxis: { title: '方法数' },
      yaxis2: { overlaying: 'y', side: 'right', tickformat: '.0%', range: [0, 1] },
      legend: { orientation: 'h', y: 1.12 },
    }, PLOT_CFG);

    const riskSet = new Set(d.high_risk_clusters || []);
    const blindSet = new Set(d.blind_spot_clusters || []);
    const stableSet = new Set(d.stable_clusters || []);

    // ---- 簇效验证卡片 ----
    const rv = d.reaction_validation;
    if (rv && rv.available) {
      // 4 分支判定：effective/promising → 绿（好消息）；weak/ineffective → 红
      const rvStatus = rv.status || (rv.effective ? 'effective' : 'ineffective');
      const rvPositive = rvStatus === 'effective' || rvStatus === 'promising';
      $('rvBanner').className = 'banner mb-3 ' + (rvPositive ? 'level-safe' : 'level-broken');
      $('rvBanner').style.padding = '12px 16px';
      $('rvVerdict').textContent = (rvPositive ? '✅ ' : '⚠️ ') + rv.verdict;
      let rvStats = `p_anova=${rv.p_anova} · p_kruskal=${rv.p_kruskal} · eta²=${rv.eta2} · ε²=${rv.epsilon2}`;
      if (rv.underpowered) rvStats += ` · n=${rv.n_total}/${rv.adequate_n}（不足）`;
      $('rvStats').textContent = rvStats;
      const pcs = Object.entries(rv.per_cluster || {}).sort((a, b) => b[1].mean_score - a[1].mean_score);
      if (pcs.length) {
        Plotly.newPlot('chart_rv', [{
          x: pcs.map(([cid]) => cid === '-1' ? '稀疏区' : `簇${cid}`),
          y: pcs.map(([, v]) => v.mean_score),
          type: 'bar',
          text: pcs.map(([, v]) => `${v.mean_score.toFixed(2)} (n=${v.n_tested})`),
          textposition: 'auto',
          marker: { color: pcs.map(([, v]) => v.mean_score > 0 ? C.warn : C.primary) },
        }], { margin: { t: 10 }, height: 260, font: PLOT_FONT,
              yaxis: { title: '簇内平均机器反应 (eval_score)' } }, PLOT_CFG);
      }
    } else {
      $('rvBanner').className = 'banner level-inconclusive mb-3';
      $('rvBanner').style.padding = '12px 16px';
      $('rvVerdict').textContent = '暂无簇效验证数据';
      $('rvStats').textContent = (rv && rv.reason) ? rv.reason : '需在新版聚类流程（post-test HDBSCAN）运行后生成';
      $('chart_rv').innerHTML = '';
    }

    const wrap = $('clusterCards');
    wrap.innerHTML = '';

    // HDBSCAN 密度视图的稀疏区（关键层切割无噪声，稀疏区来自密度视图）
    const hdb = d.hdbscan;
    if (hdb && hdb.n_noise > 0) {
      const sparseName = (hdb.cluster_names || {})['-1'] || '稀疏区（低密度噪声）';
      const sparseMembers = Object.entries(hdb.method_labels || {})
        .filter(([, c]) => c === -1).map(([m]) => m);
      const div = document.createElement('div');
      div.className = 'card';
      div.id = 'clcard--1';   // 投影噪声点（cluster=-1）点选联动的落点
      div.style.background = '#F2EBD8';
      div.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="font-semibold text-sm">
            <span class="cluster-tag" style="background:#EAE2CC;color:#7C7663;">稀疏区</span> ${esc(sparseName)}</div>
          <div class="text-xs" style="color: var(--c-muted);">${hdb.n_noise} 种方法 · HDBSCAN 密度视图（共 ${hdb.n_clusters} 个密度簇）</div>
        </div>
        <div class="text-xs mt-2 truncate" style="color: var(--c-muted);">${esc(sparseMembers.slice(0, 24).join('、'))}${sparseMembers.length > 24 ? ' …' : ''}</div>`;
      wrap.appendChild(div);
    }

    (d.clusters || []).forEach(c => {
      let tag = '', bg = '#F6EFDE';
      if (String(c.id) === '-1') { tag = '<span class="cluster-tag" style="background:#EAE2CC;color:#7C7663;">稀疏区</span>'; bg = '#F2EBD8'; }
      else if (riskSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#F0DBCF;color:#9a4a35;">高风险</span>'; bg = '#F6E9E0'; }
      else if (blindSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#F1E4CC;color:#a0663f;">盲区</span>'; bg = '#F6EFE0'; }
      else if (stableSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#DFE5D4;color:#4f7351;">稳定</span>'; bg = '#EDF0E4'; }
      const div = document.createElement('div');
      div.className = 'card';
      div.id = 'clcard-' + c.id;   // 投影点选联动的落点
      div.style.background = bg;
      div.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="font-semibold text-sm">${esc(c.name)} ${tag}</div>
          <div class="text-xs" style="color: var(--c-muted);">
            ${c.size} 种方法 · 覆盖 ${fmtPct(c.test_coverage)} · 平均 ELO ${fmtNum(c.mean_elo, 0)} · ASR ${fmtPct(c.asr)}
          </div>
        </div>
        <div class="text-xs mt-2 truncate" style="color: var(--c-muted);">${esc((c.members || []).slice(0, 24).join('、'))}${(c.members || []).length > 24 ? ' …' : ''}</div>`;
      wrap.appendChild(div);
    });
  } catch (e) { setStatus('聚类分析加载失败: ' + e.message); }
}

// ---------- 层次树（树图 + 缩放切割） ----------
let treeData = null;
let treeK = null;

function treeCutHeight(k) {
  const h = treeData.merge_heights, n = treeData.n;
  if (k <= 1) return h.length ? h[h.length - 1] * 1.05 : 1;
  if (k >= n) return 0;
  return (h[n - k - 1] + h[n - k]) / 2;
}

function cutLineShape(h) {
  return { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: h, y1: h,
           line: { color: C.warn, width: 1.5, dash: 'dash' } };
}

async function loadClusterTree() {
  try {
    const d = await api('/api/cluster-tree');
    if (!d.available) { $('treeMeta').textContent = '（无层次树数据，需新版树聚类运行后生成）'; return; }
    treeData = d;
    treeK = d.chosen_k;
    $('treeK').textContent = treeK;
    $('treeMeta').textContent = `（${d.n} 种方法 · auto-k=${d.chosen_k} · 数据来自最近一次聚类，全局）`;
    const presets = $('treePresets');
    presets.innerHTML = '';
    (d.top_ks || []).forEach(k => {
      const b = document.createElement('button');
      b.className = 'btn btn-plain text-xs';
      b.textContent = 'k=' + k;
      b.onclick = () => setTreeK(k);
      presets.appendChild(b);
    });
    drawDendrogram();
  } catch (e) { $('treeMeta').textContent = '（树图加载失败）'; }
}

function drawDendrogram() {
  const d = treeData;
  const xs = [], ys = [];
  d.icoord.forEach((x4, i) => {
    const y4 = d.dcoord[i];
    xs.push(x4[0], x4[1], x4[2], x4[3], null);
    ys.push(y4[0], y4[1], y4[2], y4[3], null);
  });
  const traces = [{
    x: xs, y: ys, type: 'scatter', mode: 'lines',
    line: { color: C.primary, width: 1.2 }, hoverinfo: 'skip',
  }];
  // 叶子方法名：隐形散点只供 hover（scipy dendrogram 叶子 x = 10*i+5）
  if (d.leaves && d.leaves.length) {
    traces.push({
      x: d.leaves.map((_, i) => i * 10 + 5), y: d.leaves.map(() => 0),
      type: 'scatter', mode: 'markers',
      marker: { size: 14, opacity: 0 },
      text: d.leaves, hovertemplate: '%{text}<extra></extra>',
    });
  }
  Plotly.newPlot('chart_dendrogram', traces, {
    margin: { t: 10 }, height: 300, font: PLOT_FONT, showlegend: false,
    xaxis: { showticklabels: false, title: `${d.n} 种方法（叶节点，hover 看方法名）` },
    yaxis: { title: '合并距离' },
    shapes: [cutLineShape(treeCutHeight(treeK))],
  }, PLOT_CFG);
}

async function setTreeK(k) {
  if (!treeData) return;
  treeK = Math.max(2, Math.min(k, treeData.n - 1));
  $('treeK').textContent = treeK;
  Plotly.relayout('chart_dendrogram', { shapes: [cutLineShape(treeCutHeight(treeK))] });
  try {
    const d = await api('/api/cluster-cut?k=' + treeK);
    if (d.available) renderCutClusters(d);
  } catch (e) { setStatus('树切割失败: ' + e.message); }
}

function zoomTree(delta) { setTreeK((treeK || 2) + delta); }

function renderCutClusters(d) {
  $('treeMeta').textContent = `（树切割视图 k=${d.k} · 点"复位"回到分析视图）`;
  const wrap = $('clusterCards');
  wrap.innerHTML = '';
  d.clusters.forEach((c, i) => {
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `
      <div class="flex items-center justify-between">
        <div class="font-semibold text-sm">
          <span class="cluster-tag" style="background:${color}22;color:${color};">k=${d.k}</span> ${esc(c.name)}</div>
        <div class="text-xs" style="color: var(--c-muted);">${c.size} 种方法 · 平均 ELO ${fmtNum(c.mean_elo, 0)}</div>
      </div>
      <div class="text-xs mt-2 truncate" style="color: var(--c-muted);">${esc(c.members.slice(0, 24).join('、'))}${c.members.length > 24 ? ' …' : ''}</div>`;
    wrap.appendChild(div);
  });
}

async function resetTreeCut() {
  await loadClusters();
}

// ---------- 预测模型 ----------
// 预测 CI 图数据缓存（搜索高亮重绘用）
let predCiData = [];

function predCiTrace(items, marker) {
  return {
    x: items.map(p => p.rank),
    y: items.map(p => p.elo),
    type: 'scatter', mode: 'markers',
    text: items.map(p => p.method),
    hovertemplate: '%{text}<br>预测 ELO %{y:.0f}<extra></extra>',
    error_y: {
      type: 'data', symmetric: false,
      // 钳制到 ±800：历史脏数据（std 爆炸的旧 run）不再压扁 y 轴
      array: items.map(p => p.ci95 ? Math.min(p.ci95[1] - p.elo, 800) : 0),
      arrayminus: items.map(p => p.ci95 ? Math.min(p.elo - p.ci95[0], 800) : 0),
      color: C.muted, thickness: 1.2, width: 3,
    },
    marker,
  };
}

function drawPredCi(filter) {
  const info = $('predCiInfo');
  if (!predCiData.length) { if (info) info.textContent = ''; return; }
  const data = predCiData.map((p, i) => ({ ...p, rank: i + 1 }));
  const q = (filter || '').trim().toLowerCase();
  let traces;
  if (!q) {
    if (info) info.textContent = `${data.length} 个未测方法`;
    traces = [predCiTrace(data, { size: 7, color: C.primary })];
  } else {
    const hit = data.filter(p => p.method.toLowerCase().includes(q));
    const miss = data.filter(p => !p.method.toLowerCase().includes(q));
    traces = [];
    if (miss.length) traces.push(predCiTrace(miss, { size: 4, color: C.muted, opacity: 0.25 }));
    if (hit.length) traces.push(predCiTrace(hit, { size: 11, color: C.accent }));
    if (info) {
      info.innerHTML = hit.length
        ? `匹配 ${hit.length} 个：` + hit.slice(0, 8).map(p => `${esc(p.method)} (${fmtNum(p.elo, 0)})`).join('、') + (hit.length > 8 ? ' …' : '')
        : '无匹配方法';
    }
  }
  Plotly.newPlot('chart_pred_ci', traces, {
    margin: { t: 10 }, height: 380, font: PLOT_FONT, showlegend: false,
    dragmode: 'zoom',  // 框选放大，双击复位
    xaxis: { showticklabels: false, title: '方法（按预测 Elo 升序）' },
    yaxis: { title: '预测 ELO ± 1.96σ' },
  }, PLOT_CFG);
}

$('predCiSearch').addEventListener('input', e => drawPredCi(e.target.value));

async function loadModel() {
  try {
    const d = await api('/api/model' + runQuery());
    if (!d.available) {
      $('modelEmpty').classList.remove('hidden'); $('modelBody').classList.add('hidden');
      predCiData = [];
      clearCharts(['chart_regpath', 'chart_pca', 'chart_importance', 'chart_pred_ci']);
      return;
    }
    $('modelEmpty').classList.add('hidden'); $('modelBody').classList.remove('hidden');
    const s = d.svd_ridge;

    $('md_lambda').textContent = fmtNum(s.lambda_opt, 4);
    $('md_sigma').textContent = fmtNum(s.sigma2, 1);
    const pca = s.pca_summary || {};
    $('md_df').textContent = pca.effective_df != null ? `${fmtNum(pca.effective_df, 1)}/${pca.n_features}` : '-';
    $('md_gt').textContent = s.n_ground_truth ?? '-';

    // 正则化路径
    const rp = s.regularization_path || {};
    if ((rp.cv_errors || []).length) {
      Plotly.newPlot('chart_regpath', [{
        x: rp.lambda_candidates, y: rp.cv_errors, type: 'scatter', mode: 'lines+markers',
        line: { color: C.primary, width: 2 }, marker: { size: 5 },
      }], {
        margin: { t: 10 }, font: PLOT_FONT,
        xaxis: { type: 'log', title: 'λ (log)' }, yaxis: { title: 'CV 误差' },
        shapes: [{
          type: 'line', x0: s.lambda_opt, x1: s.lambda_opt, yref: 'paper', y0: 0, y1: 1,
          line: { color: C.warn, width: 1.5, dash: 'dash' },
        }],
        annotations: [{
          x: Math.log10(s.lambda_opt), y: 1, yref: 'paper', text: `λ*=${fmtNum(s.lambda_opt, 4)}`,
          showarrow: false, font: { size: 11, color: C.warn },
        }],
      }, PLOT_CFG);
    }

    // PCA 解释方差
    if ((pca.explained_variance_ratio || []).length) {
      const idx = pca.explained_variance_ratio.map((_, i) => i + 1);
      Plotly.newPlot('chart_pca', [
        { x: idx, y: pca.explained_variance_ratio, type: 'bar', name: '解释方差比', marker: { color: C.primary } },
        { x: idx, y: pca.cumulative_variance_ratio, type: 'scatter', mode: 'lines+markers', name: '累计', line: { color: C.accent, width: 2 } },
      ], {
        margin: { t: 10 }, font: PLOT_FONT, xaxis: { title: '主成分' },
        yaxis: { tickformat: '.0%' }, legend: { orientation: 'h', y: 1.12 },
      }, PLOT_CFG);
    }

    // 特征重要性
    const imp = (s.feature_importance || []).slice(0, 20);
    Plotly.newPlot('chart_importance', [{
      y: imp.map(f => f.feature).reverse(),
      x: imp.map(f => f.abs_coef).reverse(),
      type: 'bar', orientation: 'h',
      marker: { color: imp.map(f => f.coef >= 0 ? C.primary : C.accent).reverse() },
    }], { margin: { t: 10 }, height: 480, font: PLOT_FONT,
          xaxis: { title: '|系数|（青=正向，橙=负向）' } }, PLOT_CFG);

    // 预测 CI 散点：整数序号 x 轴（标签不可读且截断会撞名叠点），搜索高亮 + 框选缩放
    predCiData = Object.entries(s.predictions || {})
      .map(([m, p]) => ({ method: m, ...p }))
      .sort((a, b) => a.elo - b.elo);
    drawPredCi($('predCiSearch').value);
  } catch (e) { setStatus('预测模型加载失败: ' + e.message); }
}

// ---------- 运行控制 ----------
async function loadRunSection() {
  try {
    const [sets, tgts] = await Promise.all([api('/api/attack-sets'), api('/api/targets')]);
    const sel = $('evalInput');
    sel.innerHTML = '';
    sets.files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f; opt.textContent = f;
      sel.appendChild(opt);
    });
    // 目标模型下拉（单选，来自 .env TARGETS）
    const tsel = $('evalTarget');
    if (tsel) {
      tsel.innerHTML = '<option value="">（.env 默认）</option>';
      (tgts.targets || []).forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.name; opt.textContent = t.name;
        tsel.appendChild(opt);
      });
    }
    await loadTasks();
  } catch (e) { setStatus('运行控制加载失败: ' + e.message); }
}

// 任务完成监听：启动后轮询至终态，自动刷新批次列表与当前页数据
// （SSE 流不可用时的回退路径）
// Fix 2: watchTimers 按 taskId 去重，防 SSE 反复断线时 interval 无限堆叠
const watchTimers = new Map();
function watchTask(taskId) {
  if (watchTimers.has(taskId)) clearInterval(watchTimers.get(taskId));
  const timer = setInterval(async () => {
    try {
      const t = await api('/api/tasks/' + taskId);
      if (t.status === 'running') return;
      clearInterval(timer);
      watchTimers.delete(taskId);
      setStatus(`任务 ${t.kind} ${t.status === 'success' ? '已完成' : '失败'}，数据已刷新`);
      await loadRuns();
      invalidate();
    } catch (e) { clearInterval(timer); watchTimers.delete(taskId); }
  }, 3000);
  watchTimers.set(taskId, timer);
}
function stopWatchTask(taskId) {
  if (watchTimers.has(taskId)) { clearInterval(watchTimers.get(taskId)); watchTimers.delete(taskId); }
}

// Fix 1: 自适应轮询——仅在存在运行中任务时才 2s 刷 taskList，无任务时停止
// （原无条件 setInterval 在任务结束后仍持续重建 DOM，是"停止后仍刷新"的主因）
let taskPollTimer = null;
function startTaskPolling() {
  if (taskPollTimer) return;
  taskPollTimer = setInterval(() => { if (activeSection === 'run') loadTasks(); }, 2000);
}
function stopTaskPolling() {
  if (taskPollTimer) { clearInterval(taskPollTimer); taskPollTimer = null; }
}

async function startTask(kind) {
  try {
    const res = await fetch(`/api/run/${kind}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const view = await res.json();
    setStatus(`${kind} 任务已启动`);
    if (view.id) watchTask(view.id);
    startTaskPolling();
    await loadTasks();
  } catch (e) { setStatus(`启动失败: ${e.message}`); }
}

async function startEvaluate() {
  try {
    const res = await fetch('/api/run/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        phase: $('evalPhase').value,
        input: $('evalInput').value,
        batch_size: parseInt($('evalBatch').value, 10) || 10,
        max_rounds: parseInt($('evalRounds').value, 10) || 5,
        sampler: $('evalSampler').value,
        target: ($('evalTarget') && $('evalTarget').value) || null,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const view = await res.json();
    // 预检（诚实版：只看文件名是否含 baseline，不假装能探测针内容）
    const noBaseline = !$('evalInput').value.includes('baseline');
    setStatus('评估任务已启动' + (noBaseline ? '（⚠️ 攻击集文件名不含 baseline，越狱税将不会计算）' : ''));
    if (view.id) watchTask(view.id);
    startTaskPolling();
    await loadTasks();
  } catch (e) { setStatus(`启动失败: ${e.message}`); }
}

const TASK_STATUS = {
  running: ['运行中', 'background:rgba(70,88,107,.14);color:#46586B;'],
  success: ['完成', 'background:rgba(117,135,107,.20);color:#55694B;'],
  failed: ['失败', 'background:#F0DBCF;color:#9a4a35;'],
  cancelled: ['已取消', 'background:#EAE2CC;color:#7C7663;'],
};

// 运行中任务的 SSE 日志流：任务 id → { es, text, el }
// text 持久保存，列表每 2s 重建 DOM 时重新灌回，日志不丢
const taskStreams = new Map();

function closeTaskStream(id) {
  const s = taskStreams.get(id);
  if (s) { try { s.es.close(); } catch (e) { /* ignore */ } taskStreams.delete(id); }
}

function attachTaskStream(t, logEl) {
  const existing = taskStreams.get(t.id);
  if (existing) {
    existing.el = logEl;                       // 挂到本轮新建的 log-box 上
    logEl.textContent = existing.text || t.log_tail || '(暂无输出)';
    logEl.scrollTop = logEl.scrollHeight;
    return;
  }
  const es = new EventSource('/api/tasks/' + encodeURIComponent(t.id) + '/stream');
  const s = { es, text: '', el: logEl };
  taskStreams.set(t.id, s);
  es.onmessage = ev => {
    s.text += ev.data + '\n';
    if (s.el) { s.el.textContent = s.text; s.el.scrollTop = s.el.scrollHeight; }
  };
  es.addEventListener('done', ev => {
    closeTaskStream(t.id);
    let info = {};
    try { info = JSON.parse(ev.data); } catch (e) { /* ignore */ }
    setStatus(`任务 ${t.kind} 已结束（${info.status || '未知'}），数据已刷新`);
    loadRuns();
    invalidate();
    loadTasks();
  });
  es.onerror = () => {
    // SSE 断线/不可用 → 关闭（阻止浏览器自动重连）并回退轮询
    closeTaskStream(t.id);
    if (!watchTimers.has(t.id)) watchTask(t.id);  // Fix 2: 防已有 timer 时重复创建
  };
}

async function cancelTask(id, kind) {
  if (!confirm(`确定取消任务 ${kind}？子进程将被终止。`)) return;
  try {
    const res = await fetch('/api/tasks/' + encodeURIComponent(id) + '/cancel', { method: 'POST' });
    if (res.status === 409) {
      setStatus('任务已结束，无需取消');
    } else if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    } else {
      setStatus(`已取消任务 ${kind}`);
    }
    // Fix 3: 取消后立即停止客户端轮询与 SSE，不等下次 poll 发现
    stopWatchTask(id);
    closeTaskStream(id);
    await loadTasks();
  } catch (e) { setStatus('取消失败: ' + e.message); }
}

// Fix 4: 增量 DOM 更新——不再每次 innerHTML='' 全清重建（消除闪烁/滚动丢失/SSE el 失效）
function _buildTaskCard(t) {
  const [label, style] = TASK_STATUS[t.status] || [t.status, ''];
  const card = document.createElement('div');
  card.className = 'card';
  card.style.padding = '10px 14px';
  card.dataset.taskId = t.id;
  card.dataset.status = t.status;
  card.innerHTML = `
    <div class="flex items-center justify-between mb-1 gap-2 flex-wrap">
      <div><span class="cluster-tag" data-role="badge"></span>
        <span class="font-semibold ml-2">${esc(t.kind)}</span>
        <span class="text-xs ml-2" style="color: var(--c-muted);">${esc(t.started_at?.slice(11, 19) || '')}</span></div>
      <div class="flex items-center gap-3" data-role="meta">
        <span class="text-xs font-mono" style="color: var(--c-muted);">${esc(t.cmd)}</span>
        <a class="text-xs" style="color: var(--c-primary);" href="/api/tasks/${encodeURIComponent(t.id)}/log?download=1">⬇ 完整日志</a>
      </div>
    </div>
    <div class="log-box mt-2" data-role="log"></div>`;
  _updateTaskCard(card, t);
  return card;
}
function _updateTaskCard(card, t) {
  const [label, style] = TASK_STATUS[t.status] || [t.status, ''];
  // badge：仅在文本/样式变化时写 DOM
  const badge = card.querySelector('[data-role="badge"]');
  if (badge.textContent !== label) { badge.textContent = label; badge.setAttribute('style', style); }
  // cancel 按钮：按状态增删（仅状态转换时操作）
  const meta = card.querySelector('[data-role="meta"]');
  const cancelBtn = meta.querySelector('[data-cancel]');
  if (t.status === 'running' && !cancelBtn) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-plain text-xs';
    btn.dataset.cancel = '';
    btn.textContent = '⏹ 取消';
    btn.onclick = () => cancelTask(t.id, t.kind);
    meta.appendChild(btn);
  } else if (t.status !== 'running' && cancelBtn) {
    cancelBtn.remove();
  }
  // 日志：有活跃 SSE 流时不碰（流自己管 textContent），无流时从 log_tail 更新
  const logEl = card.querySelector('[data-role="log"]');
  if (!taskStreams.has(t.id)) {
    const text = t.log_tail || '(暂无输出)';
    if (logEl.textContent !== text) logEl.textContent = text;
  }
}

async function loadTasks() {
  try {
    const data = await api('/api/tasks');
    const wrap = $('taskList');
    const runningIds = new Set();
    const seenIds = new Set();

    if (!data.tasks.length) {
      if (!wrap.querySelector('[data-empty]')) {
        wrap.innerHTML = '<span data-empty style="color: var(--c-muted);">暂无任务</span>';
      }
      [...taskStreams.keys()].forEach(id => closeTaskStream(id));
      [...watchTimers.keys()].forEach(id => stopWatchTask(id));
      stopTaskPolling();
      return;
    }
    const empty = wrap.querySelector('[data-empty]');
    if (empty) empty.remove();

    // 索引现有卡片
    const cardMap = new Map();
    wrap.querySelectorAll('.card[data-task-id]').forEach(c => cardMap.set(c.dataset.taskId, c));

    data.tasks.forEach(t => {
      seenIds.add(t.id);
      let card = cardMap.get(t.id);
      if (!card) {
        card = _buildTaskCard(t);
        wrap.appendChild(card);
      } else {
        _updateTaskCard(card, t);
      }
      if (t.status === 'running') {
        runningIds.add(t.id);
        const logEl = card.querySelector('[data-role="log"]');
        logEl.scrollTop = logEl.scrollHeight;
        attachTaskStream(t, logEl);   // SSE 直播；失败自动回退轮询
      }
    });
    // 移除消失任务的卡片
    cardMap.forEach((card, id) => { if (!seenIds.has(id)) card.remove(); });
    // 终态/消失任务的流：关闭清理
    [...taskStreams.keys()].forEach(id => { if (!runningIds.has(id)) closeTaskStream(id); });
    [...watchTimers.keys()].forEach(id => { if (!runningIds.has(id)) stopWatchTask(id); });

    // Fix 1: 无运行中任务时停止轮询（消除"停止后仍刷新"）；有任务时确保轮询
    if (runningIds.size === 0) stopTaskPolling(); else startTaskPolling();
  } catch (e) { /* 静默 */ }
}

// ---------- 键盘导航 ----------
document.addEventListener('keydown', e => {
  if (e.target.matches('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key >= '1' && e.key <= '6') {
    document.querySelectorAll('#nav .nav-item')[+e.key - 1]?.click();
  } else if (e.key === 'r') {
    $('refreshBtn').click();
  } else if (e.key === '/') {
    e.preventDefault();
    (activeSection === 'model' ? $('predCiSearch') : $('runSelect')).focus();
  }
});

// ---------- 阅读进度条 + 报告目录 scrollspy ----------
window.addEventListener('scroll', () => {
  const h = document.documentElement;
  const max = h.scrollHeight - h.clientHeight;
  $('readProgress').style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
  if (activeSection !== 'report') return;
  let current = null;
  document.querySelectorAll('#reportBody .card[id^="rep-"]').forEach(el => {
    if (el.getBoundingClientRect().top <= 140) current = el.id;
  });
  document.querySelectorAll('#reportNav a.rep-link').forEach(a => {
    a.classList.toggle('active', current !== null && a.getAttribute('href') === '#' + current);
  });
}, { passive: true });

// ---------- 启动 ----------
(async () => {
  // URL 参数（可分享的视图状态）：?theme=dark|light  ?cmp=<批次名>
  const q = new URLSearchParams(location.search);
  if (q.get('theme') === 'dark' || q.get('theme') === 'light') theme = q.get('theme');
  applyTheme(theme, false);   // 恢复主题（不触发重绘）
  await loadRuns();
  // hash 直达：#threats 等；默认总览
  const h = location.hash.slice(1);
  const start = SECTIONS.includes(h) ? h : 'overview';
  if (start !== 'overview') {
    document.querySelector(`#nav .nav-item[data-section="${start}"]`)?.click();
  } else {
    loadSection('overview');
  }
  loadRunSection();
  // 直达对比视图：等总览数据就位后自动展开对比面板
  const cmpRun = q.get('cmp');
  if (cmpRun && start === 'overview') {
    const t = setInterval(() => {
      if (!lastOverview) return;
      clearInterval(t);
      if (!cmpActive) $('cmpBtn').click();
      $('cmpSelect').value = cmpRun;
      renderCompare();
    }, 200);
    setTimeout(() => clearInterval(t), 10000);
  }
})();
