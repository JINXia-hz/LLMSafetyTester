/* LLMSEC 安全评估工作台前端逻辑 */

// ---------- 全局状态与常量 ----------
const C = {
  primary: '#5f8d8b', accent: '#c07a5a', warn: '#b0563f',
  safe: '#6d8f6e', muted: '#8a8f93', text: '#3c4447',
};
const PLOT_CFG = { responsive: true, displayModeBar: false };
const PLOT_FONT = { family: 'ui-sans-serif, system-ui, sans-serif', color: C.text };

let currentRun = '';           // '' = 最新
let activeSection = 'overview';
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

// ---------- 导航 ----------
document.querySelectorAll('#nav .nav-item').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('#nav .nav-item').forEach(n => n.classList.remove('active'));
    el.classList.add('active');
    activeSection = el.dataset.section;
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

// ---------- 运行批次 ----------
async function loadRuns() {
  const data = await api('/api/runs');
  const sel = $('runSelect');
  sel.innerHTML = '<option value="">最新</option>';
  data.runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.name;
    opt.textContent = r.name + (r.has_report ? '' : ' (无报告)');
    sel.appendChild(opt);
  });
}
$('runSelect').addEventListener('change', e => { currentRun = e.target.value; invalidate(); });
$('refreshBtn').addEventListener('click', async () => { await loadRuns(); invalidate(); });

// ---------- 总览 ----------
async function loadOverview() {
  try {
    const d = await api('/api/overview' + runQuery());
    if (!d.available) { $('ov_verdict').textContent = '暂无运行数据'; return; }
    const level = d.security_level || 'inconclusive';
    $('ov_banner').className = 'banner mb-5 level-' + level;
    $('ov_target').textContent = `目标模型: ${d.target_model || '-'}  ·  批次 ${d.run}`;
    $('ov_verdict').textContent = d.overall_verdict || level.toUpperCase();
    $('ov_recommendation').textContent = d.recommendation ? '💡 ' + d.recommendation : '';

    $('ov_asr').textContent = fmtPct(d.asr);
    $('ov_fpr').textContent = fmtPct(d.fpr);
    $('ov_boundary').textContent = fmtNum(d.boundary_elo, 0);
    $('ov_conf').textContent = fmtPct(d.boundary_confidence);
    $('ov_tested').textContent = `${d.total_tested}/${d.total_methods}`;
    $('ov_above').textContent = d.methods_above_boundary;

    // 雷达图（闭合）
    const r = d.radar;
    Plotly.newPlot('chart_radar', [{
      type: 'scatterpolar',
      r: [...r.values, r.values[0]],
      theta: [...r.labels, r.labels[0]],
      fill: 'toself',
      fillcolor: 'rgba(95,141,139,0.18)',
      line: { color: C.primary, width: 2 },
      marker: { size: 6, color: C.primary },
    }], {
      polar: { radialaxis: { range: [0, 1], tickformat: '.0%', tickfont: { size: 10 } } },
      margin: { t: 20, b: 20, l: 40, r: 40 }, font: PLOT_FONT, showlegend: false,
    }, PLOT_CFG);

    const harm = Object.entries(d.harm_type_asr || {}).sort((a, b) => b[1] - a[1]);
    Plotly.newPlot('chart_harm', [{
      x: harm.map(i => i[0]), y: harm.map(i => i[1]), type: 'bar',
      text: harm.map(i => fmtPct(i[1])), textposition: 'auto',
      marker: { color: C.accent },
    }], { yaxis: { tickformat: '.0%', range: [0, 1] }, margin: { t: 10 }, font: PLOT_FONT }, PLOT_CFG);
  } catch (e) { setStatus('总览加载失败: ' + e.message); }
}

// ---------- 威胁看板 ----------
async function loadThreats() {
  try {
    const [d, elo] = await Promise.all([api('/api/threats' + runQuery()), api('/api/elo')]);
    if (!d.available) return;

    const top = (d.top_threats || []).slice(0, 10);
    Plotly.newPlot('chart_top_threats', [{
      y: top.map(t => t.method).reverse(),
      x: top.map(t => t.elo).reverse(),
      type: 'bar', orientation: 'h',
      text: top.map(t => fmtNum(t.elo, 0)).reverse(), textposition: 'auto',
      marker: { color: top.map(t => t.tested ? C.warn : C.muted).reverse() },
    }], { margin: { t: 10, l: 10 }, height: 380, font: PLOT_FONT,
          xaxis: { title: 'ELO（红=实测，灰=预测）' } }, PLOT_CFG);

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

    // 威胁表格
    const tbody = $('threatTable');
    tbody.innerHTML = '';
    (d.top_threats || []).forEach(t => {
      const tr = document.createElement('tr');
      tr.style.borderTop = '1px solid #efede8';
      const badge = t.tested
        ? '<span class="badge badge-gt">实测</span>'
        : '<span class="badge badge-pred">预测</span>';
      const ci = t.ci95 ? `[${fmtNum(t.ci95[0], 0)}, ${fmtNum(t.ci95[1], 0)}]` : '-';
      tr.innerHTML = `<td class="py-2 pr-4 font-mono text-xs">${t.method}</td>
        <td class="py-2 pr-4 font-semibold">${fmtNum(t.elo)}</td>
        <td class="py-2 pr-4">${t.asr != null ? fmtPct(t.asr) : '-'}</td>
        <td class="py-2 pr-4">${badge}</td><td class="py-2 text-xs">${ci}</td>`;
      tbody.appendChild(tr);
    });

    // 防御强项
    const dl = $('defenseList');
    dl.innerHTML = '';
    (d.strong_defenses || []).slice(0, 8).forEach(t => {
      dl.innerHTML += `<div class="flex justify-between"><span class="font-mono text-xs">${t.method}</span>
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
        <span class="font-mono text-xs">${u.attacker || u.method || ''}</span>
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
    // 按 ## 分段
    const chunks = d.markdown.split(/^## /m);
    const head = chunks[0];
    const headTitle = (head.match(/^# (.+)$/m) || [])[1] || '安全评估报告';
    body.innerHTML += `<div class="card report-body"><h1>${headTitle}</h1>${marked.parse(head.replace(/^# .+$/m, ''))}</div>`;
    chunks.slice(1).forEach((chunk, i) => {
      const nl = chunk.indexOf('\n');
      const title = nl > 0 ? chunk.slice(0, nl).trim() : chunk.trim();
      const content = nl > 0 ? chunk.slice(nl + 1) : '';
      const anchor = `rep-${i}`;
      nav.innerHTML += `<a href="#${anchor}" class="block px-2 py-1 rounded hover:bg-stone-100" style="color: var(--c-primary);">${title}</a>`;
      const div = document.createElement('div');
      div.className = 'card report-body';
      div.id = anchor;
      div.innerHTML = `<h2>${title}</h2>${marked.parse(content)}`;
      body.appendChild(div);
    });
  } catch (e) { setStatus('报告加载失败: ' + e.message); }
}

// ---------- 聚类分析 ----------
async function loadClusters() {
  try {
    const d = await api('/api/clusters' + runQuery());
    if (!d.available) return;
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
    const wrap = $('clusterCards');
    wrap.innerHTML = '';
    (d.clusters || []).forEach(c => {
      let tag = '', bg = '#fff';
      if (riskSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#f3ded8;color:#9a4a35;">高风险</span>'; bg = '#faf3f1'; }
      else if (blindSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#f5e8dc;color:#a0663f;">盲区</span>'; bg = '#faf6f1'; }
      else if (stableSet.has(c.id)) { tag = '<span class="cluster-tag" style="background:#e3ede4;color:#4f7351;">稳定</span>'; bg = '#f4f7f4'; }
      const div = document.createElement('div');
      div.className = 'card';
      div.style.background = bg;
      div.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="font-semibold text-sm">${c.name} ${tag}</div>
          <div class="text-xs" style="color: var(--c-muted);">
            ${c.size} 种方法 · 覆盖 ${fmtPct(c.test_coverage)} · 平均 ELO ${fmtNum(c.mean_elo, 0)} · ASR ${fmtPct(c.asr)}
          </div>
        </div>
        <div class="text-xs mt-2 truncate" style="color: var(--c-muted);">${(c.members || []).slice(0, 24).join('、')}${(c.members || []).length > 24 ? ' …' : ''}</div>`;
      wrap.appendChild(div);
    });
  } catch (e) { setStatus('聚类分析加载失败: ' + e.message); }
}

// ---------- 预测模型 ----------
async function loadModel() {
  try {
    const d = await api('/api/model' + runQuery());
    if (!d.available) { $('modelEmpty').classList.remove('hidden'); $('modelBody').classList.add('hidden'); return; }
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
    }], { margin: { t: 10, l: 10 }, height: 480, font: PLOT_FONT,
          xaxis: { title: '|系数|（青=正向，橙=负向）' } }, PLOT_CFG);

    // 预测 CI 散点
    const preds = Object.entries(s.predictions || {})
      .map(([m, p]) => ({ method: m, ...p }))
      .sort((a, b) => a.elo - b.elo);
    if (preds.length) {
      Plotly.newPlot('chart_pred_ci', [{
        x: preds.map(p => p.method),
        y: preds.map(p => p.elo),
        type: 'scatter', mode: 'markers',
        error_y: {
          type: 'data', symmetric: false,
          array: preds.map(p => p.ci95 ? p.ci95[1] - p.elo : 0),
          arrayminus: preds.map(p => p.ci95 ? p.elo - p.ci95[0] : 0),
          color: C.muted, thickness: 1.2, width: 3,
        },
        marker: { size: 7, color: C.primary },
      }], { margin: { t: 10 }, height: 380, font: PLOT_FONT,
            xaxis: { tickfont: { size: 9 } }, yaxis: { title: '预测 ELO ± 1.96σ' } }, PLOT_CFG);
    }
  } catch (e) { setStatus('预测模型加载失败: ' + e.message); }
}

// ---------- 运行控制 ----------
async function loadRunSection() {
  const sets = await api('/api/attack-sets');
  const sel = $('evalInput');
  sel.innerHTML = '';
  sets.files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    sel.appendChild(opt);
  });
  await loadTasks();
}

async function startTask(kind) {
  try {
    await api(`/api/run/${kind}`);
    setStatus(`${kind} 任务已启动`);
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
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    setStatus('评估任务已启动');
    await loadTasks();
  } catch (e) { setStatus(`启动失败: ${e.message}`); }
}

const TASK_STATUS = {
  running: ['运行中', 'background:#e4e9ee;color:#5a7186;'],
  success: ['完成', 'background:#e3ede4;color:#4f7351;'],
  failed: ['失败', 'background:#f3ded8;color:#9a4a35;'],
};

async function loadTasks() {
  try {
    const data = await api('/api/tasks');
    const wrap = $('taskList');
    wrap.innerHTML = '';
    if (!data.tasks.length) {
      wrap.innerHTML = '<span style="color: var(--c-muted);">暂无任务</span>';
      return;
    }
    data.tasks.forEach(t => {
      const [label, style] = TASK_STATUS[t.status] || [t.status, ''];
      const div = document.createElement('div');
      div.className = 'card';
      div.style.padding = '10px 14px';
      div.innerHTML = `
        <div class="flex items-center justify-between mb-1">
          <div><span class="cluster-tag" style="${style}">${label}</span>
            <span class="font-semibold ml-2">${t.kind}</span>
            <span class="text-xs ml-2" style="color: var(--c-muted);">${t.started_at?.slice(11, 19) || ''}</span></div>
          <div class="text-xs font-mono" style="color: var(--c-muted);">${t.cmd}</div>
        </div>
        <div class="log-box mt-2">${(t.log_tail || '(暂无输出)').replace(/</g, '&lt;')}</div>`;
      wrap.appendChild(div);
    });
  } catch (e) { /* 静默 */ }
}
setInterval(() => { if (activeSection === 'run') loadTasks(); }, 2000);

// ---------- 启动 ----------
(async () => { await loadRuns(); loadSection('overview'); loadRunSection(); })();
