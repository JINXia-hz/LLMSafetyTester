/* control.js — 宣政殿「中书省」面板：对话 + 意图理解 + Plan 准奏（依赖 core.js 全局）
 *
 * 三省同殿：三衙同列并览，无 tab 切换。
 *   中书省（本文件）：对话主入口。
 *     简单查询 → 自己处理（list_runs/compare/list_workspaces/review_run）
 *     复杂指令 → 转交尚书省拟案 → 收到 plan_pending → 展示方案 + 准奏/驳回按钮
 *     用户准奏 → 调 /api/control/plan/approve → 尚书省执行（进度在尚书省衙署）
 *   流程条节点实时显三省职守状态（setProvStatus），点击节点跳转对应衙署。
 *
 * 封驳不再在此处理——门下省经总线监听尚书省每一步，封驳卡片在门下省衙署。
 * 坊·工作区 / 衡·对比合并 的手动看板已撤，相关操作由中书对话（工具调用）承接。
 */

// ---------- 宣政殿 ----------
let _chatBusy = false;
let _ctrlBound = false;
// session_id 持久化到 sessionStorage（刷新不丢，关标签页才丢）
let _sessionId = sessionStorage.getItem('ctrl_session_id') || null;
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
    appendChat('assistant', mdSafe('臣中书令见驾，恭请圣安。臣可：**查批次**、**对比 run**、**fork 工作区**、**审查报告**、**合并 R 矩阵**、**清缓存**。'));
  }
  if (!_ctrlBound) bindControl();
  // 初始化尚书省 + 门下省衙署
  if (window.loadShangshuSection) loadShangshuSection();
  if (window.loadMenxiaSection) loadMenxiaSection();
}

function bindControl() {
  _ctrlBound = true;
  const input = $('ctrl-chat-input');
  $('ctrl-chat-send').onclick = sendChat;
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  // 快捷指令 chips：点击即发送
  $('ctrl-chips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip || _chatBusy) return;
    input.value = chip.dataset.q;
    sendChat();
  });
  // 重置对话（清上下文 + session）
  const resetBtn = $('ctrl-chat-reset');
  if (resetBtn) resetBtn.onclick = resetChat;
  // 流程条节点：点击跳转对应衙署（窄屏纵列时定位用）+ 描金闪高
  document.querySelectorAll('.court-node').forEach(node => {
    node.onclick = () => flashPanel(node.dataset.goto);
  });
}

// ---------- 三省职守状态（流程条节点） ----------
function setProvStatus(prov, text, busy) {
  const el = $('st-' + prov);
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('busy', !!busy);
}

// 跳转并描金闪高某衙署（准奏后引至尚书省 / 流程条点击共用）
function flashPanel(id) {
  const p = $(id);
  if (!p) return;
  p.scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'nearest' });
  p.classList.remove('clcard-flash'); void p.offsetWidth; p.classList.add('clcard-flash');
  setTimeout(() => p.classList.remove('clcard-flash'), 1200);
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
  sessionStorage.removeItem('ctrl_session_id');
  $('ctrl-chat-log').innerHTML = '';
  setProvStatus('zhongshu', '候旨');
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
  setProvStatus('zhongshu', '拟票中…', true);
  appendChat('user', text);
  appendChat('thinking', '中书拟票中 <span class="chat-cursor">▍</span>');
  try {
    const res = await fetch('/api/control/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: _sessionId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const data = await res.json();
    removeThinking();
    if (data.session_id) {
      _sessionId = data.session_id;
      sessionStorage.setItem('ctrl_session_id', _sessionId);
    }
    // 复杂指令：尚书省已拟案
    if (data.plan_pending) {
      if (data.plan_pending.auto_executed) {
        // 便宜行事：全部 low 级，自动执行，不展示准奏卡片
        appendChat('assistant', mdSafe(data.plan_pending.rendered_plan || '尚书省已便宜行事。') +
          '<br><span class="ws-tag merged" style="margin-left:6px;">便宜行事·已自动执行</span>');
        if (window.shangshuTrackPlan) window.shangshuTrackPlan(data.plan_pending.plan_id);
        setStatus('便宜行事——已提交执行');
      } else {
        // 正常流程：展示方案 + 准奏/驳回/改拟按钮
        renderPlanPendingCard(data.plan_pending);
        setStatus('尚书省已拟案——待天子圣裁');
      }
    } else {
      // 简单查询/回复：工具调用轨迹 + 回复
      for (const tc of (data.tool_calls || [])) appendToolCall(tc);
      const modeTag = data.mode === 'llm' ? ''
        : `<span class="mini-tag warn" style="margin-left:6px;">${
            data.mode === 'fallback' ? 'LLM失败·规则兜底'
            : data.mode === 'rule' ? '规则模式'
            : data.mode
          }</span>`;
      appendChat('assistant', mdSafe(data.reply) + modeTag);
      setProvStatus('zhongshu', '候旨');
      setStatus('对话完成');
    }
  } catch (e) {
    removeThinking();
    appendChat('error', '✕ ' + esc(e.message));
    setProvStatus('zhongshu', '候旨');
    setStatus('对话失败: ' + e.message);
  } finally {
    _chatBusy = false;
    $('ctrl-chat-send').disabled = false;
  }
}

// ============================================================
// Plan 准奏（复杂指令 → 尚书省拟案 → 用户裁决）
// ============================================================
function renderPlanPendingCard(pp) {
  // pp = {plan_id, steps, rendered_plan}
  const log = $('ctrl-chat-log');
  const div = document.createElement('div');
  div.className = 'chat-msg chat-plan';
  // 步骤摘要
  const stepsHtml = (pp.steps || []).map((s, i) => {
    const deps = s.depends_on && s.depends_on.length ? ` ← 依赖 ${s.depends_on.join(',')}` : '';
    return `<div style="padding:2px 0; font-size:0.8rem;">${i+1}. ${esc(s.description || s.capability)}${deps}</div>`;
  }).join('');
  div.innerHTML = `
    <div class="chat-role"><span class="seal-mini" style="background:var(--c-safe);">尚</span> 尚书省·拟票（经中书省润色）</div>
    <div class="chat-plan-card">
      <div style="font-size:0.875rem; line-height:1.6; margin-bottom:8px;">${mdSafe(pp.rendered_plan || '')}</div>
      <details style="font-size:0.75rem; color:var(--c-muted); margin-bottom:8px;">
        <summary style="cursor:pointer;">技术步骤（${(pp.steps||[]).length} 步）</summary>
        <div style="padding-top:4px;">${stepsHtml}</div>
      </details>
      <div style="display:flex; gap:8px;">
        <button class="btn-approve plan-approve">准奏</button>
        <button class="btn-reject plan-reject">驳回</button>
        <button class="btn-rewrite plan-rewrite">改拟</button>
      </div>
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  div.querySelector('.plan-approve').onclick = () => approvePlan(pp.plan_id, div);
  div.querySelector('.plan-reject').onclick = () => rejectPlan(pp.plan_id, div);
  div.querySelector('.plan-rewrite').onclick = () => rewritePlan(pp, div);
}

function rewritePlan(pp, cardDiv) {
  // 改拟：把当前方案摘要回填到输入框，用户编辑后发送→重新走拟案流程
  cardDiv.querySelector('.plan-approve').disabled = true;
  cardDiv.querySelector('.plan-reject').disabled = true;
  cardDiv.querySelector('.plan-rewrite').disabled = true;
  // 驳回旧 Plan
  try { fetch('/api/control/plan/reject', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({plan_id: pp.plan_id}),
  }); } catch { /* 忽略 */ }
  // 回填输入框：列出步骤让用户修改
  const stepsBrief = (pp.steps || []).map((s, i) =>
    `${i+1}. ${s.description || s.capability}`).join('；');
  const input = $('ctrl-chat-input');
  input.value = `修改方案，当前步骤：${stepsBrief}。请调整：`;
  input.focus();
  appendChat('assistant', '陛下要修改方案？请在输入框编辑修改意见后发送，臣将转交尚书省重新拟案。');
  setStatus('等待修改意见');
}

async function approvePlan(planId, cardDiv) {
  cardDiv.querySelector('.plan-approve').disabled = true;
  cardDiv.querySelector('.plan-reject').disabled = true;
  // 通知尚书省面板开始跟踪此 plan
  if (window.shangshuTrackPlan) window.shangshuTrackPlan(planId);
  // 提交到执行队列（异步，不阻塞——approve 端点立即返回 queue_status）
  try {
    const res = await fetch('/api/control/plan/approve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planId, session_id: _sessionId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendChat('error', '✕ 提交失败: ' + esc(err.detail || res.status));
      return;
    }
    const data = await res.json();
    if (data.queue_status === 'queued') {
      appendChat('assistant', '陛下已准奏。尚书省领旨，已排入执行队列——进度见**尚书省**面板。');
    } else if (data.queue_status === 'duplicate') {
      appendChat('assistant', '此计划已在执行队列中，无需重复提交。');
    } else {
      appendChat('assistant', '陛下已准奏，尚书省正在执行——进度见**尚书省**面板。');
    }
    setStatus('已提交执行队列');
  } catch (e) {
    appendChat('error', '✕ 网络错误: ' + esc(e.message));
  }
}

async function rejectPlan(planId, cardDiv) {
  cardDiv.querySelector('.plan-approve').disabled = true;
  cardDiv.querySelector('.plan-reject').disabled = true;
  try {
    await fetch('/api/control/plan/reject', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planId }),
    });
  } catch { /* 忽略 */ }
  appendChat('assistant', '陛下已驳回此案。');
  setProvStatus('zhongshu', '候旨');
  setProvStatus('shangshu', '待命');
  setStatus('已驳回');
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
