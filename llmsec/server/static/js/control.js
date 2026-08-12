/* control.js — 控制台「中书台」：对话 + 坊·工作区 + 衡·对比合并（依赖 core.js 全局）
 *
 * 对话经 /api/control/chat（LLM tool-calling，未配置 LLM 时规则兜底）。
 * 工作区经 /api/control/workspaces、/api/control/fork、/api/control/fork-and-run。
 * 对比/合并经 /api/control/compare、/api/control/merge。
 */

// ---------- 控制台 ----------
let _chatBusy = false;
let _ctrlBound = false;
// session_id + pendingConfirm 持久化到 sessionStorage（刷新不丢，关标签页才丢）
let _sessionId = sessionStorage.getItem('ctrl_session_id') || null;
let _pendingConfirm = null;   // 门下省待确认的 ticket（blocked 时填）
let _forkOptsLoaded = false;   // fork+run 抽屉的目标/攻击集下拉是否已填充
let _greeted = false;          // 开场白只发一次

async function loadControlSection() {
  // LLM 状态小印章
  try {
    const st = await api('/api/control/llm-status');
    const badge = $('ctrl-llm-badge');
    if (st.configured) {
      badge.textContent = 'LLM 已接入';
      badge.className = 'ctrl-badge on';
    } else {
      badge.textContent = '规则模式';
      badge.className = 'ctrl-badge off';
      badge.title = '未配置 GENERATOR_API_KEY，对话走规则兜底';
    }
  } catch { /* 忽略 */ }
  if (!_greeted) {
    _greeted = true;
    appendChat('assistant', mdSafe('中书省候旨。陛下有何吩咐？臣可：**查批次**、**对比 run**、**fork 工作区**、**审查报告**、**合并 R 矩阵**、**清缓存**。'));
  }
  loadWorkspaces();
  loadPickLists();
  if (!_ctrlBound) bindControl();
}

function bindControl() {
  _ctrlBound = true;
  const input = $('ctrl-chat-input');
  $('ctrl-chat-send').onclick = sendChat;
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $('ctrl-ws-refresh').onclick = loadWorkspaces;
  $('ctrl-fork-btn').onclick = () => doFork(false);
  $('ctrl-fork-run-btn').onclick = onForkRun;
  // 快捷指令 chips：点击即发送
  $('ctrl-chips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip || _chatBusy) return;
    input.value = chip.dataset.q;
    sendChat();
  });
  // 工作区删除：事件委托（名字含引号也不怕）
  $('ctrl-ws-list').addEventListener('click', e => {
    const btn = e.target.closest('.ws-del');
    if (btn) deleteWs(btn.dataset.name);
  });
  $('ctrl-cmp-btn').onclick = doCompare;
  $('ctrl-mrg-btn').onclick = doMerge;
  // 重置对话（清上下文 + session）
  const resetBtn = $('ctrl-chat-reset');
  if (resetBtn) resetBtn.onclick = resetChat;
}

async function resetChat() {
  if (_chatBusy) return;
  // 通知后端清 session
  if (_sessionId) {
    try {
      await fetch('/api/control/chat/reset', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: '', session_id: _sessionId }),
      });
    } catch { /* 忽略 */ }
  }
  _sessionId = null;
  _pendingConfirm = null;
  sessionStorage.removeItem('ctrl_session_id');
  $('ctrl-chat-log').innerHTML = '';
  setStatus('对话已重置');
}

// ============================================================
// 中书对话
// ============================================================
async function sendChat() {
  if (_chatBusy) return;
  const input = $('ctrl-chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  _chatBusy = true;
  $('ctrl-chat-send').disabled = true;
  appendChat('user', text);
  appendChat('thinking', '中书拟票中 <span class="chat-cursor">▍</span>');
  try {
    const body = { text, session_id: _sessionId };
    // 若有待确认的封驳令牌，且用户输入了「确认」，带上 token
    if (_pendingConfirm && (text === '确认' || text === 'confirm' || text === '是')) {
      body.confirm_token = _pendingConfirm.token;
    }
    const res = await fetch('/api/control/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const data = await res.json();
    removeThinking();
    // 记住 session_id（上下文记忆 + sessionStorage 持久化）
    if (data.session_id) {
      _sessionId = data.session_id;
      sessionStorage.setItem('ctrl_session_id', _sessionId);
    }
    // 门下省封驳：展示劝谏卡片
    if (data.blocked) {
      _pendingConfirm = data.blocked;
      renderConfirmCard(data.blocked);
      appendChat('assistant', mdSafe(data.reply));
      setStatus('门下省封驳——等待确认');
    } else {
      _pendingConfirm = null;
      // 中书省计划卡片（先规划后执行）
      if (data.plan) renderPlanCard(data.plan);
      // 工具调用轨迹：折叠行（器字小印 + name(args) → result）
      for (const tc of (data.tool_calls || [])) appendToolCall(tc);
      // 回复 + 模式小印
      const modeTag = data.mode === 'llm' ? ''
        : `<span class="ws-tag pending" style="margin-left:6px;">${
            data.mode === 'fallback' ? 'LLM失败·规则兜底'
            : data.mode === 'confirmed' ? '已确认执行'
            : data.mode === 'cancelled' ? '已取消'
            : data.mode === 'error' ? '执行失败'
            : data.mode === 'rule' ? '规则模式'
            : data.mode
          }</span>`;
      appendChat('assistant', mdSafe(data.reply) + modeTag);
      setStatus('控制台对话完成');
    }
    // LLM 可能动了工作区（fork/delete/merge），静默刷新列表
    loadWorkspaces();
  } catch (e) {
    removeThinking();
    appendChat('error', '✕ ' + esc(e.message));
    setStatus('控制台对话失败: ' + e.message);
  } finally {
    _chatBusy = false;
    $('ctrl-chat-send').disabled = false;
  }
}

function renderPlanCard(plan) {
  // 渲染中书省执行计划卡片（先规划后执行）
  const log = $('ctrl-chat-log');
  const div = document.createElement('div');
  div.className = 'chat-msg chat-plan';
  div.innerHTML = `
    <div class="chat-role"><span class="seal-mini seal-accent">书</span> 中书省·拟票</div>
    <div class="rounded border border-[var(--accent)]/20 bg-[var(--accent)]/5 p-3 mt-1 text-sm">${mdSafe(plan)}</div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderConfirmCard(ticket) {
  // 渲染门下省劝谏卡片（确认/取消按钮）
  const log = $('ctrl-chat-log');
  const div = document.createElement('div');
  div.className = 'chat-msg chat-blocked';
  div.innerHTML = `
    <div class="chat-role"><span class="seal-mini" style="background:#c0392b;">门</span> 门下省·封驳</div>
    <div class="rounded border border-red-500/30 bg-red-500/5 p-3 mt-1">
      <div class="font-medium text-red-300 mb-1">⚠ ${esc(ticket.summary)}</div>
      <div class="text-xs text-white/60 whitespace-pre-line mb-2">${esc(ticket.detail)}</div>
      <div class="flex gap-2">
        <button class="confirm-yes px-3 py-1 rounded bg-red-600 text-white text-xs hover:bg-red-500">确认执行</button>
        <button class="confirm-no px-3 py-1 rounded border border-white/20 text-xs hover:bg-white/5">取消</button>
      </div>
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  div.querySelector('.confirm-yes').onclick = () => doConfirm(ticket.token);
  div.querySelector('.confirm-no').onclick = () => doReject();
}

async function doConfirm(token) {
  // 带确认令牌重发
  const input = $('ctrl-chat-input');
  input.value = '确认';
  await sendChat();
}

async function doReject() {
  _pendingConfirm = null;
  // 通知后端清除 pending
  try {
    await fetch('/api/control/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: '取消', session_id: _sessionId, confirm_token: 'REJECT' }),
    });
  } catch { /* 忽略 */ }
  appendChat('assistant', '已取消该操作。');
  setStatus('已取消');
}

function appendChat(role, html) {
  const log = $('ctrl-chat-log');
  const div = document.createElement('div');
  if (role === 'user') {
    div.className = 'chat-msg chat-user';
    div.innerHTML = `<div class="chat-role" style="justify-content:flex-end;">你 <span class="seal-mini">你</span></div>${esc(html)}`;
  } else if (role === 'assistant') {
    div.className = 'chat-msg chat-assistant';
    div.innerHTML = `<div class="chat-role"><span class="seal-mini seal-accent">中</span> 中书</div>${html}`;
  } else if (role === 'thinking') {
    div.className = 'chat-msg chat-thinking';
    div.innerHTML = html;
  } else if (role === 'error') {
    div.className = 'chat-msg chat-error';
    div.innerHTML = html;
  }
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function appendToolCall(tc) {
  const log = $('ctrl-chat-log');
  const d = document.createElement('details');
  d.className = 'chat-tool';
  const args = JSON.stringify(tc.args || {});
  const brief = args.length > 46 ? args.slice(0, 46) + '…' : args;
  const result = String(tc.result ?? '');
  d.innerHTML = `<summary><span class="seal-mini seal-gold" style="width:18px;height:18px;font-size:10px;">器</span>`
    + `<span style="color:var(--c-text);">${esc(tc.name)}</span><span>${esc(brief)}</span>`
    + `<span class="tool-caret" style="margin-left:auto;">▸</span></summary>`
    + `<div class="tool-detail">入参 ${esc(args)}\n→ ${esc(result.length > 800 ? result.slice(0, 800) + '…（截断）' : result)}</div>`;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function removeThinking() {
  const log = $('ctrl-chat-log');
  const last = log.lastElementChild;
  if (last && last.classList.contains('chat-thinking')) last.remove();
}

// ============================================================
// 坊 · 工作区
// ============================================================
async function loadWorkspaces() {
  try {
    const data = await api('/api/control/workspaces');
    renderWorkspaces(data.workspaces || []);
    refreshForkSources();
  } catch (e) {
    $('ctrl-ws-list').innerHTML = `<div class="text-xs" style="color: var(--c-accent);">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderWorkspaces(list) {
  const el = $('ctrl-ws-list');
  if (!list.length) {
    el.innerHTML = '<div class="ws-meta" style="padding:6px 2px;">（暂无工作区——fork 一个隔离副本做实验）</div>';
    return;
  }
  el.innerHTML = list.map(w => {
    const tag = w.merged
      ? `<span class="ws-tag merged" title="已合并→${esc(w.merged_to || '')}">已合并</span>`
      : `<span class="ws-tag pending">未合并</span>`;
    const size = w.size ? `${(w.size / 1024).toFixed(0)}KB` : '';
    return `<div class="ws-row">
      <span class="seal-mini">坊</span>
      <div style="min-width:0;flex:1;">
        <div class="flex items-center gap-2"><span class="ws-name">${esc(w.name)}</span>${tag}</div>
        <div class="ws-meta">源 ${esc(w.source)} · ${w.records || 0} 条${size ? ' · ' + size : ''}</div>
      </div>
      <button class="ws-del" data-name="${esc(w.name)}" title="删除该工作区（不影响全局）">删</button>
    </div>`;
  }).join('');
}

async function refreshForkSources() {
  // fork 来源：global + 最近的历史 run
  const sel = $('ctrl-fork-source');
  const cur = sel.value;
  let opts = '<option value="global">global（当前全局 R）</option>';
  try {
    const data = await api('/api/runs');
    for (const r of (data.runs || []).filter(r => r.has_report).slice(0, 15)) {
      opts += `<option value="run:${esc(r.name)}">run: ${esc(r.name)}</option>`;
    }
  } catch { /* 忽略 */ }
  sel.innerHTML = opts;
  sel.value = cur;
}

// 「建立并运行」：第一次点击展开参数抽屉，第二次点击才执行
async function onForkRun() {
  const drawer = $('ctrl-fork-opts');
  const btn = $('ctrl-fork-run-btn');
  if (!drawer.classList.contains('open')) {
    drawer.classList.add('open');
    btn.textContent = '确认运行 ▸';
    if (!_forkOptsLoaded) { _forkOptsLoaded = true; loadForkOpts(); }
    return;
  }
  await doFork(true);
}

async function loadForkOpts() {
  try {
    const [sets, tgts] = await Promise.all([api('/api/attack-sets'), api('/api/targets')]);
    const isel = $('ctrl-fork-input');
    isel.innerHTML = '';
    (sets.files || []).forEach(f => {
      const name = typeof f === 'string' ? f : f.name;
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      isel.appendChild(opt);
    });
    const tsel = $('ctrl-fork-target');
    tsel.innerHTML = '<option value="">全部目标</option>';
    (tgts.targets || []).forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.name; opt.textContent = t.name;
      tsel.appendChild(opt);
    });
  } catch { /* 静默：下拉留默认 */ }
}

function closeForkOpts() {
  $('ctrl-fork-opts').classList.remove('open');
  $('ctrl-fork-run-btn').textContent = '建立并运行';
}

async function doFork(andRun) {
  const name = $('ctrl-fork-name').value.trim();
  const source = $('ctrl-fork-source').value;
  if (!name) { setStatus('请输入工作区名'); return; }
  setStatus(`Fork ${name}…`);
  try {
    const body = { name, source, note: andRun ? '看板 fork+run' : '看板 fork' };
    const url = andRun ? '/api/control/fork-and-run' : '/api/control/fork';
    if (andRun) {
      body.target = $('ctrl-fork-target').value || null;
      body.input_file = $('ctrl-fork-input').value || 'attacks/l1.jsonl';
      body.max_rounds = parseInt($('ctrl-fork-rounds').value, 10) || 5;
    }
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const data = await res.json();
    if (andRun && data.task) {
      setStatus(`✓ ${name} fork 成功，runner 任务已启动（任务ID: ${data.task.id}）。进度见「运行控制」`);
      setTimeout(() => document.querySelector('[data-section="run"]')?.click(), 1500);
    } else {
      setStatus(`✓ 工作区 ${name} 创建成功`);
    }
    $('ctrl-fork-name').value = '';
    closeForkOpts();
    loadWorkspaces();
    loadPickLists();
  } catch (e) {
    setStatus('Fork 失败: ' + e.message);
  }
}

async function deleteWs(name) {
  if (!confirm(`确认删除工作区 ${name}？（仅删隔离副本，不影响全局）`)) return;
  try {
    const res = await fetch(`/api/control/workspaces/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    setStatus(`✓ 工作区 ${name} 已删除`);
    loadWorkspaces();
    loadPickLists();
  } catch (e) {
    setStatus('删除失败: ' + e.message);
  }
}

// ============================================================
// 衡 · 对比与合并
// ============================================================
// 共享数据：历史 run + 工作区，供对比/合并两个多选列表
let _pickRuns = [];        // [{value, label, tag}]
let _pickWs = [];          // 工作区名列表

async function loadPickLists() {
  try {
    const [runsData, wsData] = await Promise.all([api('/api/runs'), api('/api/control/workspaces')]);
    _pickWs = (wsData.workspaces || []).map(w => w.name);
    _pickRuns = (runsData.runs || []).filter(r => r.has_report)
      .map(r => ({ value: r.name, label: r.name, tag: 'batch' }));
    // 工作区 run（ws:<name>，compare 会解析其下第一份报告）
    const wsRuns = _pickWs.map(n => ({ value: 'ws:' + n, label: n, tag: 'ws' }));
    renderPickList($('ctrl-cmp-list'), [..._pickRuns, ...wsRuns], 'cmp');
    // 合并源：global + 各工作区
    const mrgSources = [{ value: 'global', label: 'global（全局 R）', tag: 'batch' }, ...wsRuns];
    renderPickList($('ctrl-mrg-list'), mrgSources, 'mrg');
    // 合并目标：global + 各工作区
    const tsel = $('ctrl-mrg-target');
    const cur = tsel.value;
    tsel.innerHTML = '<option value="global">global（全局 R）</option>'
      + _pickWs.map(n => `<option value="ws:${esc(n)}">ws: ${esc(n)}</option>`).join('');
    tsel.value = cur;
  } catch { /* 静默 */ }
}

function renderPickList(el, items, ns) {
  if (!items.length) {
    el.innerHTML = '<div class="ws-meta" style="padding:6px;">（暂无可选项）</div>';
    return;
  }
  el.innerHTML = items.map(it => `<label class="pick-row">
    <input type="checkbox" data-ns="${ns}" value="${esc(it.value)}">
    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(it.label)}</span>
    ${it.tag === 'ws' ? '<span class="ws-tag pending" style="margin-left:auto;">坊</span>' : ''}
  </label>`).join('');
}

function pickedValues(ns) {
  return [...document.querySelectorAll(`input[type="checkbox"][data-ns="${ns}"]:checked`)].map(c => c.value);
}

// ---- 对比 ----
const CMP_METRICS = [
  ['security_level', '安全等级', v => v || '-'],
  ['asr', 'ASR 攻击成功率', fmtPct],
  ['fpr', 'FPR 误杀率', fmtPct],
  ['boundary_elo', '边界 ELO', v => fmtNum(v, 0)],
  ['boundary_confidence', '边界置信度', fmtPct],
  ['coverage', '覆盖率', fmtPct],
  ['conv_rounds', '收敛轮次', v => v ?? '-'],
  ['ci_half', 'CI 半宽', v => fmtNum(v, 1)],
  ['total_methods', '方法数', v => v ?? '-'],
  ['methods_above_boundary', '越界威胁', v => v ?? '-'],
];

async function doCompare() {
  const runs = pickedValues('cmp');
  if (runs.length < 2) { setStatus('对比至少勾选 2 个 run'); return; }
  const out = $('ctrl-heng-result');
  out.innerHTML = '<div class="ws-meta">对比中…</div>';
  try {
    const res = await fetch('/api/control/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ runs }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const d = await res.json();
    renderCompare(d, out);
    setStatus('✓ 对比完成');
  } catch (e) {
    out.innerHTML = `<div class="text-xs" style="color: var(--c-accent);">对比失败: ${esc(e.message)}</div>`;
  }
}

function renderCompare(d, out) {
  const rows = d.runs || [];
  if (!rows.length) { out.innerHTML = '<div class="ws-meta">所选 run 均无报告</div>'; return; }
  const short = n => n.length > 26 ? '…' + n.slice(-25) : n;
  let html = '<div class="overflow-x-auto"><table class="cmp-table"><thead><tr><th>指标</th>'
    + rows.map(r => `<th title="${esc(r.run)}">${esc(short(r.run))}</th>`).join('') + '</tr></thead><tbody>';
  for (const [key, label, fmt] of CMP_METRICS) {
    html += `<tr><td style="color: var(--c-muted);">${label}</td>`
      + rows.map(r => `<td>${esc(fmt(r[key]))}</td>`).join('') + '</tr>';
  }
  html += '</tbody></table></div>';
  if (d.missing && d.missing.length) {
    html += `<div class="ws-meta mt-1">⚠ 无报告被跳过：${d.missing.map(esc).join('、')}</div>`;
  }
  out.innerHTML = html;
}

// ---- 合并 ----
async function doMerge() {
  const sources = pickedValues('mrg');
  const target = $('ctrl-mrg-target').value;
  const confirm = $('ctrl-mrg-confirm').checked;
  if (!sources.length) { setStatus('合并至少勾选 1 个源'); return; }
  if (confirm && !window.confirm(`确认把 ${sources.join('、')} 合并进 ${target}？该操作会写目标 R 矩阵。`)) return;
  const out = $('ctrl-heng-result');
  out.innerHTML = `<div class="ws-meta">${confirm ? '合并执行中…' : '合并预览（dry-run）…'}</div>`;
  try {
    const res = await fetch('/api/control/merge', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sources, target, models: null, confirm }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const d = await res.json();
    renderMerge(d, out);
    setStatus(confirm ? '✓ 合并已执行' : '✓ 合并预览完成（未落盘）');
    if (confirm) { $('ctrl-mrg-confirm').checked = false; loadWorkspaces(); }
  } catch (e) {
    out.innerHTML = `<div class="text-xs" style="color: var(--c-accent);">合并失败: ${esc(e.message)}</div>`;
  }
}

function renderMerge(d, out) {
  const extra = d.extra || {};
  const dry = d.dry_run !== false;
  const badge = dry
    ? '<span class="ws-tag pending">dry-run 预览</span>'
    : '<span class="ws-tag merged">已执行</span>';
  let html = `<div class="flex items-center gap-2 mb-1">${badge}<span class="text-xs" style="color: var(--c-muted);">汇入 ${esc(extra.target || '')}</span></div>`;
  if (dry) {
    const pm = extra.per_model || {};
    const entries = Object.entries(pm);
    html += `<div class="text-xs mb-1">将新增 <b>${extra.total_new ?? 0}</b> 条记录：</div>`;
    if (entries.length) {
      html += '<table class="cmp-table"><thead><tr><th>模型</th><th>新增条数</th></tr></thead><tbody>'
        + entries.map(([m, p]) => `<tr><td>${esc(m)}</td><td>${p.new_to_target ?? 0}</td></tr>`).join('')
        + '</tbody></table>';
    }
    html += '<div class="ws-meta mt-1">勾选「确认执行」后再点合并即真正落盘。</div>';
  } else {
    const mc = extra.merged_counts || {};
    html += `<div class="text-xs mb-1">已合并 <b>${extra.total_merged ?? 0}</b> 条记录。</div>`;
    const entries = Object.entries(mc);
    if (entries.length) {
      html += '<table class="cmp-table"><thead><tr><th>模型</th><th>合并条数</th></tr></thead><tbody>'
        + entries.map(([m, n]) => `<tr><td>${esc(m)}</td><td>${n}</td></tr>`).join('')
        + '</tbody></table>';
    }
  }
  out.innerHTML = html;
}
