/* control.js — 控制台：对话中间者 + Fork 工作区管理（依赖 core.js）
 *
 * 对话经 /api/control/chat（LLM tool-calling，未配置 LLM 时规则兜底）。
 * 工作区管理经 /api/control/workspaces、/api/control/fork、/api/control/fork-and-run。
 */

// ---------- 控制台 ----------
let _chatBusy = false;

async function loadControlSection() {
  // 检测 LLM 状态
  try {
    const st = await api('/api/control/llm-status');
    const badge = $('ctrl-llm-badge');
    if (st.configured) {
      badge.textContent = 'LLM 已接入';
      badge.className = 'text-xs px-2 py-1 rounded bg-green-500/10 border border-green-500/30 text-green-400';
    } else {
      badge.textContent = 'LLM 未配置（规则模式）';
      badge.className = 'text-xs px-2 py-1 rounded bg-yellow-500/10 border border-yellow-500/30 text-yellow-400';
    }
  } catch { /* 忽略 */ }
  // 加载工作区列表
  loadWorkspaces();
  // 绑定事件（只绑一次）
  if (!_ctrlBound) bindControl();
}

let _ctrlBound = false;
function bindControl() {
  _ctrlBound = true;
  const input = $('ctrl-chat-input');
  $('ctrl-chat-send').onclick = sendChat;
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $('ctrl-ws-refresh').onclick = loadWorkspaces;
  $('ctrl-fork-btn').onclick = () => doFork(false);
  $('ctrl-fork-run-btn').onclick = () => doFork(true);
}

// ---- 对话 ----
async function sendChat() {
  if (_chatBusy) return;
  const input = $('ctrl-chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  _chatBusy = true;
  $('ctrl-chat-send').disabled = true;
  // 渲染用户消息
  appendChat('user', text);
  appendChat('thinking', '思考中…');
  try {
    const res = await fetch('/api/control/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.status);
    }
    const data = await res.json();
    removeThinking();
    // 渲染工具调用轨迹（如果有）
    if (data.tool_calls && data.tool_calls.length) {
      for (const tc of data.tool_calls) {
        appendChat('tool', `🔧 ${tc.name}(${JSON.stringify(tc.args)}) → ${tc.result}`);
      }
    }
    // 渲染回复（markdown 安全渲染）
    const modeTag = data.mode === 'llm' ? '' : (data.mode === 'fallback' ? ' ⚠LLM失败,规则兜底' : ' [规则模式]');
    appendChat('assistant', mdSafe(data.reply) + `<span class="text-xs text-white/30 ml-2">${modeTag}</span>`);
    setStatus('控制台对话完成');
  } catch (e) {
    removeThinking();
    appendChat('error', '❌ ' + e.message);
    setStatus('控制台对话失败: ' + e.message);
  } finally {
    _chatBusy = false;
    $('ctrl-chat-send').disabled = false;
  }
}

function appendChat(role, html) {
  const log = $('ctrl-chat-log');
  const div = document.createElement('div');
  const styles = {
    user: 'bg-[var(--accent)]/10 border border-[var(--accent)]/20 ml-8',
    assistant: 'bg-white/5 border border-white/10 mr-8',
    tool: 'bg-white/[0.02] border border-white/5 text-xs text-white/60 mr-8 font-mono',
    thinking: 'bg-white/5 border border-white/10 text-white/40 italic mr-8',
    error: 'bg-red-500/10 border border-red-500/30 text-red-400 mr-8',
  };
  div.className = `rounded px-3 py-2 text-sm ${styles[role] || ''}`;
  if (role === 'user') div.innerHTML = `<span class="text-white/40 text-xs mr-2">你</span>${esc(html)}`;
  else if (role === 'assistant') div.innerHTML = `<span class="text-white/40 text-xs mr-2 block mb-1">助手</span>${html}`;
  else div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function removeThinking() {
  const log = $('ctrl-chat-log');
  const last = log.lastElementChild;
  if (last && last.classList.contains('italic')) last.remove();
}

// ---- 工作区管理 ----
async function loadWorkspaces() {
  try {
    const data = await api('/api/control/workspaces');
    renderWorkspaces(data.workspaces || []);
    // 顺便刷新 fork 来源下拉（含历史 run）
    refreshForkSources();
  } catch (e) {
    $('ctrl-ws-list').innerHTML = `<div class="text-red-400 text-xs">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderWorkspaces(list) {
  const el = $('ctrl-ws-list');
  if (!list.length) {
    el.innerHTML = '<div class="text-white/30 text-xs">（暂无 fork 工作区）</div>';
    return;
  }
  el.innerHTML = list.map(w => {
    const mergedTag = w.merged
      ? `<span class="text-xs text-green-400">✓已合并→${esc(w.merged_to||'')}</span>`
      : `<span class="text-xs text-yellow-400">未合并</span>`;
    const size = w.size ? `${(w.size/1024).toFixed(0)}KB` : '';
    return `<div class="flex items-center justify-between rounded px-2 py-1.5 bg-white/[0.02]">
      <div class="flex items-center gap-2">
        <span class="font-medium">${esc(w.name)}</span>
        <span class="text-xs text-white/40">源:${esc(w.source)}</span>
        <span class="text-xs text-white/40">${w.records||0}条</span>
        ${size ? `<span class="text-xs text-white/40">${size}</span>` : ''}
        ${mergedTag}
      </div>
      <button class="text-xs text-red-400 hover:text-red-300 px-2" onclick="deleteWs('${esc(w.name)}')">删</button>
    </div>`;
  }).join('');
}

async function refreshForkSources() {
  // fork 来源：global + 最近的历史 run
  const sel = $('ctrl-fork-source');
  const cur = sel.value;
  let opts = '<option value="global">来源: global</option>';
  try {
    const data = await api('/api/runs');
    const runs = (data.runs || []).slice(0, 15);
    for (const r of runs) {
      opts += `<option value="run:${esc(r.name)}">run: ${esc(r.name)}</option>`;
    }
  } catch { /* 忽略 */ }
  sel.innerHTML = opts;
  sel.value = cur;
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
      // fork-and-run 需要运行参数
      const target = prompt('目标模型名（留空跑全部 .env TARGETS）:', '');
      if (target === null) { setStatus('已取消'); return; }
      body.target = target || null;
      body.max_rounds = 5;
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
      // 跳到运行控制看任务进度
      setTimeout(() => document.querySelector('[data-section="run"]')?.click(), 1500);
    } else {
      setStatus(`✓ Fork 工作区 ${name} 创建成功`);
    }
    $('ctrl-fork-name').value = '';
    loadWorkspaces();
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
  } catch (e) {
    setStatus('删除失败: ' + e.message);
  }
}

// 暴露给 inline onclick
window.deleteWs = deleteWs;
