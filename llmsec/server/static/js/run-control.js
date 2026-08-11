/* run-control.js — 运行控制：HPO/目标/env/任务进度/评估启动（依赖 core.js） */

// ---------- 运行控制 ----------
// 任务 kind → 友好标签（HPO 等新 kind 在任务列表里显示中文）
const TASK_LABELS = {
  evaluate: '自适应评估', hpo: 'HPO 搜索',
};
function taskLabel(kind) { return TASK_LABELS[kind] || kind; }

// ---- 运行控制：任务选择器（先选任务，再展开对应配置） ----
function selectTask(task) {
  document.querySelectorAll('.task-card').forEach(c => c.classList.toggle('active', c.dataset.task === task));
  ['evaluate', 'hpo', 'env'].forEach(t => {
    const f = $('form-' + t);
    if (!f) return;
    const show = t === task;
    if (show && f.classList.contains('hidden')) {
      f.classList.remove('hidden');
      f.classList.remove('task-form'); void f.offsetWidth; f.classList.add('task-form');  // 重触发入场
    } else if (!show) {
      f.classList.add('hidden');
    }
  });
}

// ---- HPO 配置台 ----
let _hpoParamsLoaded = false;
let hpoParamsCache = [];

async function loadHpoParams() {
  try {
    const d = await api('/api/hpo/params');
    hpoParamsCache = d.params || [];
    renderHpoFactors();
  } catch (e) { setStatus('HPO 因子加载失败: ' + e.message); }
}

function renderHpoFactors() {
  const box = $('hpoFactors');
  if (!box) return;
  box.innerHTML = hpoParamsCache.map(p => {
    const isCat = p.type === 'categorical' || (p.choices && p.choices.length);
    const cur = p.current != null ? `当前 ${p.current}` : 'CLI 参数';
    const range = isCat
      ? `<input type="text" data-f="${p.name}-choices" value="${(p.choices || []).join(',')}" style="width:11em;" title="逗号分隔候选">`
      : `<input type="number" data-f="${p.name}-low" value="${p.low ?? ''}" title="下限">
         <input type="number" data-f="${p.name}-high" value="${p.high ?? ''}" title="上限">
         <input type="number" data-f="${p.name}-step" value="${p.step ?? ''}" title="步长">`;
    return `<label class="factor-row" data-row="${p.name}">
      <input type="checkbox" data-f="${p.name}-chk">
      <span class="factor-name" title="${esc(p.group)} · ${esc(cur)}">${p.name}</span>
      <span class="factor-cur">${esc(cur)}</span>
      <span class="factor-range">${range}</span>
    </label>`;
  }).join('');
  box.querySelectorAll('input[type=checkbox]').forEach(chk => {
    chk.addEventListener('change', () => {
      chk.closest('.factor-row').classList.toggle('on', chk.checked);
      updateFactorCount();
    });
  });
  updateFactorCount();
}

function updateFactorCount() {
  const n = document.querySelectorAll('#hpoFactors input[type=checkbox]:checked').length;
  const el = $('factorCount');
  if (el) el.textContent = `已选 ${n}`;
}

function factorSelectAll(on) {
  document.querySelectorAll('#hpoFactors input[type=checkbox]').forEach(chk => {
    chk.checked = on;
    chk.closest('.factor-row').classList.toggle('on', on);
  });
  updateFactorCount();
}

function toggleFactors() {
  const d = $('factorDrawer'), t = $('factorToggle');
  d.classList.toggle('open');
  t.classList.toggle('open');
}

// 优化方向分段开关 + 时间预算快捷选择（一次性绑定）
(function wireRunControls() {
  document.querySelectorAll('#hpoDirSeg button').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#hpoDirSeg button').forEach(x => x.classList.toggle('active', x === b));
      $('hpoDir').value = b.dataset.v;
    });
  });
  const clock = $('hpoWallClock'), pop = $('hpoWallPop');
  if (clock && pop) {
    clock.addEventListener('click', e => { e.stopPropagation(); pop.classList.toggle('hidden'); });
    pop.querySelectorAll('button').forEach(b => {
      b.addEventListener('click', () => { $('hpoWall').value = b.dataset.m; pop.classList.add('hidden'); });
    });
    document.addEventListener('click', e => { if (!pop.contains(e.target)) pop.classList.add('hidden'); });
  }
})();

function collectHpoSpace() {
  const space = {};
  document.querySelectorAll('#hpoFactors input[type=checkbox]').forEach(chk => {
    if (!chk.checked) return;
    const name = chk.dataset.f.replace(/-chk$/, '');
    const p = hpoParamsCache.find(x => x.name === name) || {};
    const isCat = p.type === 'categorical' || (p.choices && p.choices.length);
    if (isCat) {
      const el = document.querySelector(`[data-f="${name}-choices"]`);
      const choices = (el && el.value || '').split(',').map(s => s.trim()).filter(Boolean);
      space[name] = { type: 'categorical', choices };
    } else {
      const lo = parseFloat(document.querySelector(`[data-f="${name}-low"]`).value);
      const hi = parseFloat(document.querySelector(`[data-f="${name}-high"]`).value);
      const st = parseFloat(document.querySelector(`[data-f="${name}-step"]`).value);
      if (!isFinite(lo) || !isFinite(hi)) { throw new Error(`${name}: low/high 必填`); }
      const spec = { type: p.type || 'float', low: lo, high: hi };
      if (isFinite(st)) spec.step = st;
      if (p.log) spec.log = true;
      space[name] = spec;
    }
  });
  return space;
}

function collectHpoConfig() {
  return {
    name: ($('hpoName').value || '').trim() || `dash-${Date.now()}`,
    objective: { metric: $('hpoMetric').value, direction: $('hpoDir').value, aggregate: 'mean' },
    strategy: $('hpoStrategy').value,
    max_trials: intVal('hpoTrials', 12),
    max_wall_minutes: intVal('hpoWall', 0),
    repeats: intVal('hpoRepeats', 1),
    fixed: { input: $('hpoInput').value },
    space: collectHpoSpace(),
    est_methods_per_trial: 50,
  };
}
function intVal(id, dflt) { const v = parseInt($(id).value, 10); return isFinite(v) ? v : dflt; }

async function hpoPreview() {
  try {
    const cfg = collectHpoConfig();
    const r = await fetch('/api/hpo/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    const warn = (d.warnings || []).length ? ` · ⚠ ${d.warnings.join('; ')}` : '';
    $('hpoPreview').innerHTML =
      `<span style="color:var(--c-text);">${d.n_configs} configs × ${cfg.repeats} = <b>${d.n_trials}</b> trials</span> · ` +
      `约 <b>${d.est_method_calls}</b> 次方法调用${warn}`;
  } catch (e) { $('hpoPreview').innerHTML = `<span style="color:var(--c-warn);">预览失败: ${e.message}</span>`; }
}

async function startHpo() {
  try {
    const cfg = collectHpoConfig();
    const res = await fetch('/api/run/hpo', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || res.statusText);
    setStatus(`HPO study '${cfg.name}' 已启动（任务 ${d.id}）`);
    startTaskPolling();
    loadTasks();
  } catch (e) { setStatus('HPO 启动失败: ' + e.message); }
}

// ---- 目标「+」弹窗 ----
function openAddTarget() {
  ['atName', 'atModel', 'atUrl', 'atKey'].forEach(id => { const e = $(id); if (e) e.value = ''; });
  $('addTargetModal').classList.remove('hidden');
}
function closeAddTarget() { $('addTargetModal').classList.add('hidden'); }

// ---- 目标探活缓存 ----
let probeCache = {};  // {name: {reachable, latency_ms, error}}

async function refreshProbeCache(targetName) {
  try {
    const url = targetName ? `/api/targets/probe?name=${encodeURIComponent(targetName)}` : '/api/targets/probe';
    const d = await api(url);
    (d.targets || []).forEach(t => { probeCache[t.name] = t; });
    updateProbeUI();
  } catch { /* 静默：探活失败不阻塞 */ }
}

function updateProbeUI() {
  const tsel = $('evalTarget');
  const hint = $('probeHint');
  if (!tsel) return;
  // 更新下拉项灰显
  [...tsel.options].forEach(opt => {
    if (!opt.value) return; // 跳过"全部目标"
    const info = probeCache[opt.value];
    if (info && !info.reachable) {
      opt.textContent = `⚠ ${opt.value}（不可通）`;
      opt.disabled = true;
      opt.title = info.error || '连接失败';
    } else {
      opt.textContent = opt.value;
      opt.disabled = false;
      opt.title = '';
    }
  });
  // 当前选中被禁用时回退
  if (tsel.selectedOptions[0] && tsel.selectedOptions[0].disabled) tsel.value = '';
  // 统计可达数 + 提示
  if (hint) {
    const all = Object.keys(probeCache);
    const ok = all.filter(n => probeCache[n].reachable);
    if (all.length === 0) { hint.textContent = ''; return; }
    if (tsel.value === '') {
      // "全部目标" 模式
      if (ok.length === 0) {
        hint.textContent = '❌ 无可达目标'; hint.style.color = 'var(--c-warn)';
      } else if (ok.length < all.length) {
        hint.textContent = `✅ ${ok.length}/${all.length} 可达：${ok.join('、')}`;
        hint.style.color = 'var(--c-muted)';
      } else {
        hint.textContent = `✅ 全部 ${ok.length} 个目标可达`;
        hint.style.color = 'var(--c-muted)';
      }
    } else {
      hint.textContent = '';
    }
  }
}

async function submitAddTarget() {
  const name = $('atName').value.trim();
  const url = $('atUrl').value.trim();
  if (!name || !url) { setStatus('目标名与 base_url 必填'); return; }
  try {
    const res = await fetch('/api/targets/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name, model: $('atModel').value.trim() || name, base_url: url, api_key: $('atKey').value.trim() || 'none',
      }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || res.statusText);
    setStatus(`目标 ${name} 已写入 .env（${d.prefix}），刷新下拉…`);
    closeAddTarget();
    await loadRunSection();  // 刷新下拉
    refreshProbeCache(name);  // 探单个新目标通性
  } catch (e) { setStatus('添加目标失败: ' + e.message); }
}

// ---- 连接配置（.env）----
async function loadEnv() {
  try {
    const d = await api('/api/env');
    const t = d.target || {}, g = d.generator || {};
    if ($('envTUrl')) $('envTUrl').value = t.base_url || '';
    if ($('envTModel')) $('envTModel').value = t.model || '';
    if ($('envTKey')) $('envTKey').value = '';
    if ($('envTKeyHint')) $('envTKeyHint').textContent = `当前：${t.api_key_masked || '未配置'}`;
    if ($('envGUrl')) $('envGUrl').value = g.base_url || '';
    if ($('envGModel')) $('envGModel').value = g.model || '';
    if ($('envGKey')) $('envGKey').value = '';
    if ($('envGKeyHint')) $('envGKeyHint').textContent = `当前：${g.api_key_masked || '未配置'}`;
    if ($('envJudge')) $('envJudge').value = d.judge_model || '';
  } catch (e) { /* 静默：未配置时不阻塞运行控制页 */ }
}

async function saveEnv() {
  const body = {};
  const collect = (id, key) => { const v = $(id).value.trim(); if (v) body[key] = v; };
  collect('envTUrl', 'target_base_url'); collect('envTModel', 'target_model'); collect('envTKey', 'target_api_key');
  collect('envGUrl', 'generator_base_url'); collect('envGModel', 'generator_model'); collect('envGKey', 'generator_api_key');
  collect('envJudge', 'judge_model');
  if (Object.keys(body).length === 0) { setStatus('未填写任何字段'); return; }
  try {
    const res = await fetch('/api/env', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || res.statusText);
    setStatus(`环境配置已保存（${(d.updated || []).join(', ')}）`);
    loadEnv();  // 刷新掩码
    // 若改了 TARGET_* → 重新探活（legacy 单目标可能变了）
    if ((d.updated || []).some(k => k.startsWith('TARGET_'))) {
      refreshProbeCache();
    }
  } catch (e) { setStatus('环境配置保存失败: ' + e.message); }
}

async function loadRunSection() {
  try {
    const [sets, tgts] = await Promise.all([api('/api/attack-sets'), api('/api/targets')]);
    const sel = $('evalInput');
    sel.innerHTML = '';
    (sets.files || []).forEach(f => {
      const opt = document.createElement('option');
      // f 可能是字符串（旧格式）或对象 {name, size_kb, mtime, n_records}
      const name = typeof f === 'string' ? f : f.name;
      const label = typeof f === 'string' ? f : `${f.name} (${f.n_records}条 ${f.size_kb}KB)`;
      opt.value = name; opt.textContent = label;
      sel.appendChild(opt);
    });
    // HPO 配置台的攻击集下拉同步填充
    const hpoSel = $('hpoInput');
    if (hpoSel) {
      hpoSel.innerHTML = '';
      (sets.files || []).forEach(f => {
        const name = typeof f === 'string' ? f : f.name;
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        hpoSel.appendChild(opt);
      });
    }
    // HPO 因子清单（仅加载一次）
    if (!_hpoParamsLoaded) { _hpoParamsLoaded = true; loadHpoParams(); }
    // 环境参数配置（右卡）每次刷新（值可能被保存更新）
    loadEnv();
    // 目标模型下拉（单选，来自 .env TARGETS）
    const tsel = $('evalTarget');
    if (tsel) {
      tsel.innerHTML = '<option value="">全部目标（多模型扫描）</option>';
      (tgts.targets || []).forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.name; opt.textContent = t.name;
        tsel.appendChild(opt);
      });
      tsel.addEventListener('change', updateProbeUI);
      // 后台探查可通性（不阻塞 UI）
      refreshProbeCache();
    }
    await loadTasks();
  } catch (e) { setStatus('运行控制加载失败: ' + e.message); }
  setupDropZone();
}

// 攻击集拖拽上传
function setupDropZone() {
  const dz = $('dropZone');
  if (!dz) return;
  if (dz.dataset.bound) return;        // 幂等：loadRunSection 每次切页都会重跑，避免重复绑定 click 监听器（弹窗弹多次的根因）
  dz.dataset.bound = '1';
  const fileInput = $('dropFile');
  const browse = $('dropBrowse');

  function uploadFile(file) {
    if (!file.name.endsWith('.jsonl')) { setStatus('只支持 .jsonl 文件'); return; }
    const fd = new FormData();
    fd.append('file', file);
    setStatus(`上传中: ${file.name}...`);
    fetch('/api/attack-sets/upload', { method: 'POST', body: fd })
      .then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)))
      .then(d => {
        setStatus(`已导入 ${d.name}（${d.n_records}条 ${d.size_kb}KB）`);
        loadRunSection(); // 刷新下拉
      })
      .catch(e => setStatus(`上传失败: ${e.detail || e.message || e}`));
  }

  // 拖拽
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.style.borderColor = 'var(--c-accent)'; });
  dz.addEventListener('dragleave', () => { dz.style.borderColor = 'var(--c-border)'; });
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.style.borderColor = 'var(--c-border)';
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });
  // 点击选择
  dz.addEventListener('click', () => fileInput.click());
  if (browse) browse.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); fileInput.click(); });
  // 阻止 input 自身合成 click 冒泡到 dz，否则 dz 的 click handler 会再次调 fileInput.click() → 选择框弹两次
  fileInput.addEventListener('click', e => e.stopPropagation());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    fileInput.value = '';
  });
}

// 任务完成监听：启动后轮询至终态，自动刷新批次列表与当前页数据
// （SSE 流不可用时的回退路径）
// Fix 2: watchTimers 按 taskId 去重，防 SSE 反复断线时 interval 无限堆叠
const watchTimers = new Map();
function watchTask(taskId) {
  if (watchTimers.has(taskId)) clearInterval(watchTimers.get(taskId));
  const timer = setInterval(async () => {
    try {
      await fetchAndApplyProgress(taskId);   // SSE 断线时的进度兜底
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

async function startEvaluate() {
  try {
    const target = ($('evalTarget') && $('evalTarget').value) || null;
    const body = {
      phase: $('evalPhase').value,
      input: $('evalInput').value,
      batch_size: parseInt($('evalBatch').value, 10) || 10,
      max_rounds: parseInt($('evalRounds').value, 10) || 5,
      sampler: $('evalSampler').value,
    };
    if (target) {
      // 单目标：直接传
      body.target = target;
    } else {
      // "全部目标" → 只传探活可达的
      const reachable = Object.keys(probeCache).filter(n => probeCache[n].reachable);
      if (reachable.length === 0) {
        setStatus('❌ 无可达目标，请检查 API 配置或等待探活完成');
        return;
      }
      if (reachable.length === 1) {
        body.target = reachable[0];  // 单个可达 → 走单目标路径（更高效）
      } else {
        body.targets = reachable.join(',');
        body.target_concurrency = reachable.length;  // 多目标全并发（各目标独立端点，无共享限速）
      }
    }
    const res = await fetch('/api/run/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const view = await res.json();
    setStatus('评估任务已启动' + (view.has_tax_probe === false ? '（该攻击集无数学探针，越狱税将不计算）' : ''));
    if (view.id) watchTask(view.id);
    startTaskPolling();
    await loadTasks();
  } catch (e) { setStatus(`启动失败: ${e.message}`); }
}

const TASK_STATUS = {
  running: ['运行中', 'background:rgba(70,88,107,.14);color:#46586B;'],
  queued: ['排队中', 'background:rgba(200,170,60,.18);color:#8a7218;'],
  success: ['完成', 'background:rgba(117,135,107,.20);color:#55694B;'],
  failed: ['失败', 'background:#F0DBCF;color:#9a4a35;'],
  cancelled: ['已取消', 'background:#EAE2CC;color:#7C7663;'],
};

// 运行中任务的 SSE 进度流：任务 id → { es }
// 服务端推 event:progress（每轮/每 trial 一条 JSON），前端据此更新进度表
const taskStreams = new Map();

function closeTaskStream(id) {
  const s = taskStreams.get(id);
  if (s) { try { s.es.close(); } catch (e) { /* ignore */ } taskStreams.delete(id); }
}

function attachTaskStream(t) {
  if (taskStreams.has(t.id)) return;   // 已连接（列表轮询重建卡片时复用）
  const es = new EventSource('/api/tasks/' + encodeURIComponent(t.id) + '/stream');
  taskStreams.set(t.id, { es });
  es.addEventListener('progress', ev => {
    let rec; try { rec = JSON.parse(ev.data); } catch (e) { return; }
    applyProgress(t.id, rec);
  });
  es.addEventListener('done', ev => {
    closeTaskStream(t.id);
    let info = {};
    try { info = JSON.parse(ev.data); } catch (e) { /* ignore */ }
    // 任务结束：无 active 目标，重渲染为终态灰表
    if (progressState[t.id]) { progressState[t.id].running = false; recomputeEvalState(progressState[t.id]); renderProgressBox(t.id); }
    setStatus(`任务 ${t.kind} 已结束（${info.status || '未知'}），数据已刷新`);
    loadRuns();
    invalidate();
    loadTasks();
  });
  es.onerror = () => {
    // SSE 断线/不可用 → 关闭（阻止浏览器自动重连）并回退轮询（watchTask 内兼刷进度）
    closeTaskStream(t.id);
    if (!watchTimers.has(t.id)) watchTask(t.id);
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
  card.className = 'card task-card';
  card.style.padding = '10px 14px';
  card.dataset.taskId = t.id;
  card.dataset.status = '';   // 留空，让 _updateTaskCard 首次按状态自动折叠/展开
  card.innerHTML = `
    <div class="flex items-center justify-between mb-1 gap-2 flex-wrap" data-role="header" style="cursor:pointer;">
      <div class="flex items-center gap-2"><span class="task-chevron" data-role="chevron">▶</span><span class="cluster-tag" data-role="badge"></span>
        <span class="font-semibold">${esc(taskLabel(t.kind))}</span>
        <span class="text-xs" style="color: var(--c-muted);">${esc(t.started_at?.slice(11, 19) || '')}</span></div>
      <div class="flex items-center gap-3" data-role="meta">
        <div class="mini-bar" data-role="minibar"><i></i></div>
        <a class="text-xs" style="color: var(--c-primary);" href="/api/tasks/${encodeURIComponent(t.id)}/log?download=1">⬇ 完整日志</a>
      </div>
    </div>
    <div class="progress-box" data-role="progress"></div>`;
  // 点击头部切换展开/收起（取消按钮、下载链接不触发）
  card.querySelector('[data-role="header"]').addEventListener('click', e => {
    if (e.target.closest('button, a')) return;
    card.dataset.collapsed = (card.dataset.collapsed === '1') ? '0' : '1';
  });
  _updateTaskCard(card, t);
  return card;
}
function _updateTaskCard(card, t) {
  // 仅在状态变化时自动折叠/展开（running 展开；queued/终态收起）。轮询不干预手动操作
  if (card.dataset.status !== t.status) {
    card.dataset.collapsed = (t.status === 'running') ? '0' : '1';
    card.dataset.status = t.status;
  }
  const [label, style] = TASK_STATUS[t.status] || [t.status, ''];
  // badge：仅在文本/样式变化时写 DOM
  const badge = card.querySelector('[data-role="badge"]');
  if (badge.textContent !== label) { badge.textContent = label; badge.setAttribute('style', style); }
  // cancel 按钮：按状态增删（仅状态转换时操作）
  const meta = card.querySelector('[data-role="meta"]');
  const cancelBtn = meta.querySelector('[data-cancel]');
  if ((t.status === 'running' || t.status === 'queued') && !cancelBtn) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-plain text-xs';
    btn.dataset.cancel = '';
    btn.textContent = '⏹ 取消';
    btn.onclick = () => cancelTask(t.id, t.kind);
    meta.appendChild(btn);
  } else if (t.status !== 'running' && t.status !== 'queued' && cancelBtn) {
    cancelBtn.remove();
  }
  // 进度：有状态即按需拉快照建表；SSE 流与轮询都会调 renderProgressBox 刷新
  if (t.status === 'running' || t.status === 'queued') {
    seedProgress(t.id, t.kind, t.status);
  }
  renderProgressBox(t.id);
}

// ---------- 任务进度状态与渲染 ----------
// taskId -> { kind, running, seeded, maxRounds, order:[targets], targets:{name:rec}, done:Set, activeTarget, hpo:rec }
const progressState = {};

function _newProgressState(kind) {
  return { kind, running: false, seeded: false, maxRounds: null,
           order: [], targets: {}, done: new Set(), activeTarget: null, hpo: null,
           hist: {}, dispPct: {} };
}

function _mergeSnapshot(st, d) {
  st.kind = d.kind;
  st.maxRounds = d.max_rounds || st.maxRounds;
  if (d.kind === 'hpo') {
    st.hpo = d.progress || null;
  } else {
    (d.targets || []).forEach(tg => { if (!st.order.includes(tg)) st.order.push(tg); });
    Object.entries(d.progress || {}).forEach(([tg, rec]) => {
      if (rec && rec.target) { st.targets[tg] = rec; if (!st.order.includes(tg)) st.order.push(tg); _recomputeDisp(st, tg); }
    });
    recomputeEvalState(st);
  }
}

// 初次渲染时拉一次快照建表（含未启动目标的占位行）；已 seed 则只同步 running 标志
async function seedProgress(taskId, kind, status) {
  let st = progressState[taskId];
  if (st && st.seeded) {
    // 已建表：仅同步 running 标志并重算 active（queued→running 晋升时高亮当前目标）
    st.running = (status === 'running');
    recomputeEvalState(st);
    renderProgressBox(taskId);
    return;
  }
  st = progressState[taskId] = _newProgressState(kind);
  st.seeded = true;
  st.running = (status === 'running');
  try {
    const d = await api('/api/tasks/' + encodeURIComponent(taskId) + '/progress');
    _mergeSnapshot(st, d);
    renderProgressBox(taskId);
  } catch (e) { /* 静默 */ }
}

// SSE 断线兜底轮询：每次都拉快照刷新（不受 seeded 限制）
async function fetchAndApplyProgress(taskId) {
  try {
    const d = await api('/api/tasks/' + encodeURIComponent(taskId) + '/progress');
    const st = progressState[taskId] || (progressState[taskId] = _newProgressState(d.kind));
    st.seeded = true;
    _mergeSnapshot(st, d);
    renderProgressBox(taskId);
  } catch (e) { /* 静默 */ }
}

// SSE 单条进度记录 → 更新对应目标行 / HPO 单行
function applyProgress(taskId, rec) {
  const st = progressState[taskId] || (progressState[taskId] = _newProgressState('evaluate'));
  st.seeded = true;
  if (rec.phase === 'hpo') { st.kind = 'hpo'; st.hpo = rec; renderProgressBox(taskId); return; }
  const tg = rec.target;
  if (tg) {
    if (!st.order.includes(tg)) st.order.push(tg);
    st.targets[tg] = rec;
    _recomputeDisp(st, tg);
  }
  if (st.maxRounds == null && rec.max_rounds) st.maxRounds = rec.max_rounds;
  st.running = true;
  recomputeEvalState(st);
  renderProgressBox(taskId);
}

// 重算每目标的 active/done 状态（接收 state 对象，便于 seed 与 apply 复用）
function recomputeEvalState(st) {
  if (!st || st.kind === 'hpo') return;
  st.done = new Set();
  Object.entries(st.targets).forEach(([tg, rec]) => {
    if (rec.phase === 'attack_done' || rec.converged ||
        (st.maxRounds && rec.round != null && rec.round >= st.maxRounds)) {
      st.done.add(tg);
    }
  });
  // active = 非 done 中 ts 最新者；任务结束（running=false）时无 active → 全灰
  st.activeTarget = null;
  if (st.running) {
    let best = null, bestTs = '';
    Object.entries(st.targets).forEach(([tg, rec]) => {
      if (st.done.has(tg)) return;
      const ts = rec.ts || '';
      if (ts >= bestTs) { bestTs = ts; best = tg; }
    });
    st.activeTarget = best;
  }
}

function _progSig(st) {
  // 进度表内容指纹：未变则跳过重建（防 2s 轮询反复重建导致进度条从 0 重绘闪烁）
  if (st.kind === 'hpo') return 'hpo:' + JSON.stringify(st.hpo || {});
  return 'ev:' + (st.maxRounds ?? '') + '|' + st.order.map(tg => {
    const r = st.targets[tg] || {};
    return tg + ':' + [r.round ?? '', r.elo ?? '', r.delta ?? '', r.ci_half ?? '',
      r.progress_pct ?? '', r.phase ?? '',
      st.activeTarget === tg ? 'A' : (st.done.has(tg) ? 'D' : '')].join(',');
  }).join(';');
}

function _textBar(pct, width = 14) {
  // 盲文进度条：[⣿⣿⣿⣿⣿⣦⣀⣀] —— 盲文字符等高（不像 █/░ 高低不齐）。
  // 已填 ⣿ + 过渡 ⣦ 走金色 pg-bf，空槽 ⣀ 走暗色 pg-em。
  const total = (Math.max(0, Math.min(100, pct || 0)) / 100) * width;
  const full = Math.min(width, Math.floor(total));
  const frac = total - full;
  let filled = '⣿'.repeat(full);
  let nEmpty = width - full;
  if (full < width && frac >= 0.5) { filled += '⣦'; nEmpty -= 1; }
  return '[<span class="pg-bf">' + filled + '</span><span class="pg-em">' + '⣀'.repeat(Math.max(0, nEmpty)) + '</span>]';
}

function renderProgressBox(taskId) {
  const card = document.querySelector('.card[data-task-id="' + taskId + '"]');
  if (!card) return;
  const box = card.querySelector('[data-role="progress"]');
  if (!box) return;
  _renderMiniBar(taskId, card);   // 头部细条始终同步（始终可见）
  const st = progressState[taskId];
  if (!st) { if (box.innerHTML) { box.innerHTML = ''; box.dataset.sig = ''; } return; }
  const sig = _progSig(st);
  if (box.dataset.sig === sig) return;   // 内容未变，跳过重建（防轮询闪烁 + 保光标动画连续）
  box.dataset.sig = sig;
  // 真·终端窗口：标题栏（主题色圆点 + 任务标题）+ 主体（每目标一行）
  const title = esc(taskLabel(st.kind)) + ' · ' + esc(taskId.split('-').pop());
  const bodyHtml = st.kind === 'hpo'
    ? renderHpoLine(st.hpo)
    : st.order.map(tg => renderTargetRow(tg, st.targets[tg], st)).join('');
  box.innerHTML =
    '<div class="term-header"><span class="term-dots"><i></i><i></i><i></i></span>' +
    '<span class="term-title">' + title + '</span></div>' +
    '<div class="term-body">' + bodyHtml + '</div>';
}

// 头部极简进度条（始终可见）：按 progressState + 任务状态算填充/动画类。
// running+有 pct → 金色填充；running+无 pct → 描金流光（"模拟"）；终态/排队 → 固定色。
function _renderMiniBar(taskId, card) {
  card = card || document.querySelector('.card[data-task-id="' + taskId + '"]');
  if (!card) return;
  const bar = card.querySelector('[data-role="minibar"]');
  if (!bar) return;
  const status = card.dataset.status || '';
  const st = progressState[taskId];
  let cls = '', pct = null;
  if (status === 'success') cls = 'done';
  else if (status === 'failed') cls = 'failed';
  else if (status === 'cancelled') cls = 'cancelled';
  else if (status === 'queued') cls = 'queued';
  else { pct = _overallPct(st); cls = pct != null ? '' : 'sim'; }
  bar.className = 'mini-bar' + (cls ? ' ' + cls : '');
  const fill = bar.querySelector('i');
  if (!fill) return;
  fill.style.width = (pct != null) ? Math.max(0, Math.min(100, pct)) + '%' : '';
}

// 单目标展示进度：对 (round, progress_pct) 历史做 OLS 线性回归，把受 ci_half 噪声
// （前中期常为 0、非单调）的收敛进度"拉成"近线性上升；叠加 round/max 线性地板与
// 单调高水位 → 条整体线性上升、永不归零/倒退。纯展示用，不影响后端数据。
function _recomputeDisp(st, tg) {
  const rec = st.targets[tg];
  if (!rec) return;
  st.hist || (st.hist = {}); st.dispPct || (st.dispPct = {});
  // 1) 入历史（按 round 去重 upsert；progress_pct==null 的不进回归）
  if (rec.round != null && rec.progress_pct != null) {
    const hist = st.hist[tg] || (st.hist[tg] = []);
    const ex = hist.find(p => p.x === rec.round);
    if (ex) ex.y = rec.progress_pct; else hist.push({ x: rec.round, y: rec.progress_pct });
  }
  // 2) OLS 拟合 progress_pct ~ round
  const hist = st.hist[tg] || [];
  let ols = null;
  if (hist.length >= 2) {
    const n = hist.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const p of hist) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
    const den = n * sxx - sx * sx;
    ols = Math.abs(den) > 1e-9
      ? (() => { const b = (n * sxy - sx * sy) / den, a = (sy - b * sx) / n; return a + b * (rec.round != null ? rec.round : sx / n); })()
      : sy / n;
  } else if (hist.length === 1) {
    ols = hist[0].y;
  }
  // 3) 线性地板 round/max_rounds（保证从第 1 轮就上升，不被早期全 0 困住）
  const floor = (st.maxRounds && rec.round != null) ? (rec.round / st.maxRounds) * 100 : 0;
  // 4) 取较大者 → 终态封顶 100 → clamp
  let est = Math.max(ols != null ? ols : 0, floor);
  if (rec.phase === 'attack_done' || rec.converged) est = 100;
  est = Math.max(0, Math.min(100, est));
  // 5) 单调高水位（永不倒退/归零）
  st.dispPct[tg] = Math.max(st.dispPct[tg] || 0, Math.round(est));
}

// 汇总进度（%）：evaluate 取各目标回归平滑值 dispPct 均值（无则 round/max 兜底）；HPO 取 config 进度
function _overallPct(st) {
  if (!st) return null;
  if (st.kind === 'hpo') {
    const c = st.hpo, tot = c && c.configs_total, done = c && c.configs_done;
    return (typeof tot === 'number' && tot > 0 && typeof done === 'number') ? Math.round(done / tot * 100) : null;
  }
  // 用回归平滑后的 dispPct 求均值（无 dispPct 时退回 round 兜底）
  const dps = Object.keys(st.targets || {}).filter(tg => st.dispPct && st.dispPct[tg] != null);
  if (dps.length) return Math.round(dps.reduce((s, tg) => s + st.dispPct[tg], 0) / dps.length);
  const withRound = Object.values(st.targets || {}).filter(r => r && r.round != null);
  if (withRound.length && st.maxRounds) {
    return Math.round(withRound.reduce((s, r) => s + r.round, 0) / (withRound.length * st.maxRounds) * 100);
  }
  return null;
}

function renderTargetRow(tg, rec, st) {
  const has = !!rec && rec.round != null;
  const isDone = st.done.has(tg);
  const isActive = st.activeTarget === tg && !isDone;
  const cls = isActive ? 'prog-line pg-active' : 'prog-line pg-idle';
  const mark = isActive ? '<span class="pg-mark">▶</span> ' : '  ';
  const name = esc(tg).padEnd(14);
  if (!has) {
    return `<div class="${cls}">${mark}${name}等待中</div>`;
  }
  const roundTxt = 'R' + rec.round + '/' + (st.maxRounds || '?');
  let deltaTxt = '';
  if (rec.delta != null && rec.delta !== 0) {
    const up = rec.delta > 0;
    deltaTxt = ` <span class="pg-delta ${up ? 'up' : 'down'}">${up ? '↑' : '↓'}${fmtNum(Math.abs(rec.delta), 0)}</span>`;
  }
  const ciTxt = rec.ci_half != null ? '±' + fmtNum(rec.ci_half, 0) : '±—';
  // 展示进度取回归平滑值 dispPct（_recomputeDisp）：线性上升、永不归零/倒退。
  const pct = (st.dispPct && st.dispPct[tg] != null) ? st.dispPct[tg] : 0;
  let tail;
  if (isDone) {
    tail = `<span class="pg-status">${rec.converged ? '已收敛' : '完成'}</span>`;
  } else {
    tail = `<span class="pg-bar-txt">${_textBar(pct)}</span> ${pct}%`;
  }
  const statusTxt = isActive ? ' <span class="pg-status">运行中</span>' : '';
  return `<div class="${cls}">${mark}${name}${roundTxt}  ELO ${fmtNum(rec.elo, 0)}${deltaTxt}  CI${ciTxt}  ${tail}${statusTxt}</div>`;
}

function renderHpoLine(rec) {
  if (!rec) return '<div class="prog-line pg-idle">HPO 搜索准备中</div>';
  const tDone = rec.trial_done ?? 0;
  const tTot = rec.trial_total_est ?? '?';
  const cDone = rec.configs_done ?? 0;
  const cTot = rec.configs_total ?? '?';
  const pct = (typeof cTot === 'number' && cTot > 0) ? Math.round(cDone / cTot * 100) : 0;
  let best = '';
  if (rec.best_metric != null) best = `  best ${esc(rec.metric_name || '')}=${fmtNum(rec.best_metric, 3)}`;
  return `<div class="prog-line pg-active"><span class="pg-mark">▶</span> HPO  trial ${tDone}/${tTot}  config ${cDone}/${cTot}${best}</div>
<div class="prog-line pg-active">    <span class="pg-bar-txt">${_textBar(pct)}</span> ${pct}%</div>`;
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
      Object.keys(progressState).forEach(id => delete progressState[id]);
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
        attachTaskStream(t);   // SSE 进度直播；失败自动回退轮询
      }
    });
    // 移除消失任务的卡片 + 进度状态
    cardMap.forEach((card, id) => { if (!seenIds.has(id)) { card.remove(); delete progressState[id]; } });
    // 终态/消失任务的流：关闭清理
    [...taskStreams.keys()].forEach(id => { if (!runningIds.has(id)) closeTaskStream(id); });
    [...watchTimers.keys()].forEach(id => { if (!runningIds.has(id)) stopWatchTask(id); });

    // Fix 1: 无运行中任务时停止轮询（消除"停止后仍刷新"）；有任务时确保轮询
    // 仅当有排队中任务时才轮询（探测 queued→running 晋级）；单任务运行靠 SSE 直播 + done 事件刷新
    const hasQueued = data.tasks.some(t => t.status === 'queued');
    if (hasQueued) startTaskPolling(); else stopTaskPolling();
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
