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

// Plotly 默认白底与册页米白卡片冲突：统一透明底（一层包装，业务代码无需逐个改）
const _newPlot = Plotly.newPlot.bind(Plotly);
Plotly.newPlot = (id, traces, layout = {}, cfg) =>
  _newPlot(id, traces, { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', ...layout }, cfg);

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
    if (!d.available) {
      setBanner('inconclusive');
      $('ov_target').textContent = '目标模型: -';
      $('ov_verdict').textContent = '暂无运行数据';
      $('ov_recommendation').textContent = '';
      ['ov_asr', 'ov_fpr', 'ov_boundary', 'ov_conf', 'ov_tested', 'ov_above', 'ov_tax']
        .forEach(id => { $(id).textContent = '-'; });
      $('ov_tax_sub').textContent = '';
      clearCharts(['chart_radar', 'chart_harm']);
      return;
    }
    const level = d.security_level || 'inconclusive';
    setBanner(level);
    $('ov_target').textContent = `目标模型: ${d.target_model || '-'}  ·  批次 ${d.run}`;
    $('ov_verdict').textContent = d.overall_verdict || level.toUpperCase();
    $('ov_recommendation').textContent = d.recommendation ? '💡 ' + d.recommendation : '';

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

    const top = (d.top_threats || []).slice(0, 10);
    Plotly.newPlot('chart_top_threats', [{
      y: top.map(t => t.method).reverse(),
      x: top.map(t => t.elo).reverse(),
      type: 'bar', orientation: 'h',
      text: top.map(t => fmtNum(t.elo, 0)).reverse(), textposition: 'auto',
      marker: { color: top.map(t => t.tested ? C.warn : C.muted).reverse() },
    }], { margin: { t: 10 }, height: 380, font: PLOT_FONT,
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
      nav.innerHTML += `<a href="#${anchor}" class="block px-2 py-1 rounded hover:bg-stone-100" style="color: var(--c-primary);">${esc(title)}</a>`;
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
    $('projMeta').textContent = `（${d.n} 种方法 · ${meta} · 实心=实测 空心=预测 · 数据来自最近一次聚类，全局）`;

    Plotly.newPlot('chart_projection', traces, {
      margin: { t: 10 }, height: 520, font: PLOT_FONT,
      xaxis: { title: method === 'pca' ? 'PC1' : 't-SNE 1' },
      yaxis: { title: method === 'pca' ? 'PC2' : 't-SNE 2' },
      legend: { orientation: 'h', y: -0.18, font: { size: 10 } },
    }, PLOT_CFG);
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
      $('rvStats').textContent = '';
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
      $('rvBanner').className = 'banner mb-3 ' + (rv.effective ? 'level-safe' : 'level-broken');
      $('rvBanner').style.padding = '12px 16px';
      $('rvVerdict').textContent = (rv.effective ? '✅ ' : '⚠️ ') + rv.verdict;
      $('rvStats').textContent = `p_anova=${rv.p_anova} · p_kruskal=${rv.p_kruskal} · eta²=${rv.eta2} · ε²=${rv.epsilon2}`;
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
  Plotly.newPlot('chart_dendrogram', [{
    x: xs, y: ys, type: 'scatter', mode: 'lines',
    line: { color: C.primary, width: 1.2 }, hoverinfo: 'skip',
  }], {
    margin: { t: 10 }, height: 300, font: PLOT_FONT, showlegend: false,
    xaxis: { showticklabels: false, title: `${d.n} 种方法（叶节点）` },
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
function watchTask(taskId) {
  const timer = setInterval(async () => {
    try {
      const t = await api('/api/tasks/' + taskId);
      if (t.status === 'running') return;
      clearInterval(timer);
      setStatus(`任务 ${t.kind} ${t.status === 'success' ? '已完成' : '失败'}，数据已刷新`);
      await loadRuns();
      invalidate();
    } catch (e) { clearInterval(timer); }
  }, 3000);
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
    setStatus('评估任务已启动');
    if (view.id) watchTask(view.id);
    await loadTasks();
  } catch (e) { setStatus(`启动失败: ${e.message}`); }
}

const TASK_STATUS = {
  running: ['运行中', 'background:rgba(70,88,107,.14);color:#46586B;'],
  success: ['完成', 'background:rgba(117,135,107,.20);color:#55694B;'],
  failed: ['失败', 'background:#F0DBCF;color:#9a4a35;'],
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
            <span class="font-semibold ml-2">${esc(t.kind)}</span>
            <span class="text-xs ml-2" style="color: var(--c-muted);">${t.started_at?.slice(11, 19) || ''}</span></div>
          <div class="text-xs font-mono" style="color: var(--c-muted);">${esc(t.cmd)}</div>
        </div>
        <div class="log-box mt-2">${esc(t.log_tail || '(暂无输出)')}</div>`;
      wrap.appendChild(div);
    });
  } catch (e) { /* 静默 */ }
}
setInterval(() => { if (activeSection === 'run') loadTasks(); }, 2000);

// ---------- 启动 ----------
(async () => {
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
})();
