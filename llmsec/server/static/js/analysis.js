/* analysis.js — 聚类分析/层次树/预测模型（依赖 core.js） */

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
      ['cl_methods', 'cl_n', 'cl_sil', 'cl_db'].forEach(id => { $(id).textContent = '-'; $(id)._v = null; });
      $('rvBanner').className = 'banner level-inconclusive mb-3';
      $('rvBanner').style.padding = '12px 16px';
      $('rvVerdict').textContent = '暂无聚类数据';
      $('rvStats').textContent = d.reason === 'no_cluster' ? '该批次无聚类分析结果，可先在"运行控制"跑聚类安全分析' : '';
      $('clusterCards').innerHTML = '';
      clearCharts(['chart_cluster_cover', 'chart_rv']);
      return;
    }
    setMetric('cl_methods', d.n_methods ?? null, v => (v == null ? '-' : fmtNum(v, 0)));
    setMetric('cl_n', d.n_clusters ?? null, v => (v == null ? '-' : fmtNum(v, 0)));
    setMetric('cl_sil', d.validation?.silhouette ?? null, v => fmtNum(v, 4));
    setMetric('cl_db', d.validation?.davies_bouldin ?? null, v => fmtNum(v, 4));

    // 聚类退化提示：HDBSCAN 密度视图在小样本/特征区分度低时会把全部方法判为噪声（n_noise≈n_methods），
    // silhouette 归零。原看板默默显示 0 值，用户无法察觉聚类无效。这里显式提示。
    const degEl = $('clDegenerate');
    if (degEl) {
      const sil = d.validation?.silhouette;
      const nNoise = d.n_noise ?? 0;
      const nMethods = d.n_methods ?? 0;
      const allNoise = nMethods > 0 && nNoise >= nMethods;
      const silZero = sil != null && Math.abs(sil) < 1e-6;
      if (allNoise || silZero) {
        degEl.classList.remove('hidden');
        degEl.textContent = allNoise
          ? `⚠ 聚类退化为全噪声（${nNoise}/${nMethods} 方法被判为噪声）：密度视图未能区分出簇，本次聚类结论不可用。通常因样本不足或特征区分度低，建议增加测试轮次。`
          : '⚠ 轮廓系数为 0：簇间无明确分离，聚类结果参考价值有限。';
      } else {
        degEl.classList.add('hidden');
      }
    }

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
      $('rvVerdict').innerHTML = `<span class="seal-chip">${rvPositive ? '显著' : '存疑'}</span>${esc(rv.verdict)}`;
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

function renderBlendPanel(b) {
  const simModels = b.unified_sim_weighted_models || [];
  const donorSims = b.donor_similarities || {};
  const lambdas = b.per_target_lambda || {};
  const fallbackLam = lambdas._fallback_uniform;
  const spm = b.samples_per_model || {};
  $('blendSub').innerHTML =
    `均匀 universal：${b.unified_fallback_trained ? '已训练' : '无'} · ` +
    `fallback λ*：${fallbackLam != null ? fmtNum(fallbackLam, 4) : '—'} · ` +
    `启用 sim-加权：${simModels.length} 个目标 · ` +
    `已建模目标：${(b.models_trained || []).length}`;
  const rows = (b.models_trained || []).map(t => {
    const active = simModels.includes(t);
    const sims = donorSims[t] || {};
    const simStr = Object.keys(sims).length
      ? Object.entries(sims).sort((a, c) => c[1] - a[1]).map(([dn, sv]) => `${esc(dn)}:${sv.toFixed(2)}`).join('、')
      : '—';
    const lam = lambdas[t];
    return `<tr style="border-top:1px solid var(--c-border);">
      <td class="py-1 pr-2 font-mono">${esc(t)}</td>
      <td class="py-1 px-2 text-center">${active ? '<span style="color:var(--c-good)">✓ 启用</span>' : '<span style="color:var(--c-muted)">回退均匀</span>'}</td>
      <td class="py-1 px-2" style="color: var(--c-muted);">${simStr}</td>
      <td class="py-1 px-2 text-center">${lam != null ? fmtNum(lam, 4) : '—'}</td>
      <td class="py-1 pl-2 text-center">${spm[t] ?? '—'}</td>
    </tr>`;
  }).join('');
  $('blendBody').innerHTML = `<table class="w-full text-xs">
    <thead><tr style="color: var(--c-muted);">
      <th class="text-left py-1 pr-2">目标</th><th>sim-加权</th><th class="text-left py-1 px-2">donor 相似度</th><th>λ*</th><th>GT</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

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

    // ---- 多模型层（BlendPredictor · 发现层 sim-加权迁移）----
    const blend = d.blend_predictor;
    if (blend && !blend.error && (blend.unified_fallback_trained || (blend.unified_sim_weighted_models || []).length)) {
      $('blendPanel').style.display = '';
      renderBlendPanel(blend);
    } else {
      $('blendPanel').style.display = 'none';
    }

    // ---- 单模型层（ColdStartPredictor · SVD-Ridge）----
    const s = d.svd_ridge;
    if (!s) {
      // 多目标 run 可能只有多模型层，无单模型 SVD-Ridge 诊断
      ['md_lambda', 'md_sigma', 'md_df', 'md_gt'].forEach(id => { const e = $(id); if (e) e.textContent = '-'; });
      clearCharts(['chart_regpath', 'chart_pca', 'chart_importance', 'chart_pred_ci']);
      predCiData = [];
      return;
    }

    setMetric('md_lambda', s.lambda_opt ?? null, v => fmtNum(v, 4));
    setMetric('md_sigma', s.sigma2 ?? null, v => fmtNum(v, 1));
    const pca = s.pca_summary || {};
    $('md_df').textContent = pca.effective_df != null ? `${fmtNum(pca.effective_df, 1)}/${pca.n_features}` : '-';
    setMetric('md_gt', s.n_ground_truth ?? null, v => (v == null ? '-' : fmtNum(v, 0)));

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
