/* sections.js — 只读看板板块：运行批次/总览/趋势/批次对比/威胁（依赖 core.js） */

// ---------- 运行批次 ----------
function _runLabel(r) {
  // 进行中批次：⏳ 占位，无印章/ASR（报告还可能只覆盖部分目标）
  if (r.active) return `⏳ ${r.name} · 进行中…`;
  // 标签去重：run 名（ts/target）里已含目标，不再重复拼 target；
  // 时间截短成 MM-DD HH:MM，完整 run 名放 title。
  const m = /^(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})/.exec(r.batch || '');
  const shortTs = m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : (r.batch || r.name);
  if (r.has_report) {
    const seal = SEAL_CHARS[r.security_level] || '▫';
    const tg = r.target_model || r.target || '?';
    return `${seal} ${shortTs} · ${tg} · ASR ${fmtPct(r.asr)}`;
  }
  return `${r.name} (无报告)`;
}

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
    opt.title = r.name;   // 完整批次名（含目录时间戳）悬浮可见
    opt.textContent = _runLabel(r);
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
        .forEach(id => { $(id).textContent = '-'; $(id)._v = null; });
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
    $('ov_recommendation').textContent = d.recommendation ? '按：' + d.recommendation : '';
    // stale_report 提示：存在更新批次时展示；切回最新批次（无 message）时清除旧文案
    setStatus(d.message || '');

    setMetric('ov_asr', d.asr, fmtPct);
    setMetric('ov_fpr', d.fpr, fmtPct);
    setMetric('ov_boundary', d.boundary_elo, v => fmtNum(v, 0));
    setMetric('ov_conf', d.boundary_confidence, fmtPct);
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
