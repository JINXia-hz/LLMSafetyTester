/* shangshu.js — 宣政殿「尚书省」面板：Plan 执行进度（依赖 core.js 全局）
 *
 * 跟踪正在执行的 Plan，轮询 /api/control/bus/feed 获取步骤进度。
 * 每步显示状态（pending/running/done/blocked/failed/skipped）+ 结果摘要。
 *
 * 被 control.js 调用：window.shangshuTrackPlan(planId) 开始跟踪某 plan。
 */

let _ssBound = false;
let _trackedPlanId = null;      // 当前跟踪的 plan id
let _planStatus = {};           // plan_id → {steps: [...], status}
let _pollTimer = null;          // 轮询定时器
let _busLatestTs = 0;           // 已处理的最新总线消息 ts

function loadShangshuSection() {
  if (!_ssBound) bindShangshu();
  // 启动轮询（即使没跟踪 plan，也看总线消息更新状态）
  if (!_pollTimer) _pollTimer = setInterval(pollBus, 2000);
}

function bindShangshu() {
  _ssBound = true;
  const refresh = $('ss-refresh');
  if (refresh) refresh.onclick = refreshPlanDisplay;
}

// control.js 的 approvePlan 调此函数开始跟踪
window.shangshuTrackPlan = function(planId) {
  _trackedPlanId = planId;
  refreshPlanDisplay();
};

async function refreshPlanDisplay() {
  if (!_trackedPlanId) return;
  try {
    const data = await api(`/api/control/plan/${_trackedPlanId}/status`);
    _planStatus[_trackedPlanId] = data;
    renderPlan(data);
  } catch (e) {
    $('ss-plan-area').innerHTML = `<div class="text-xs" style="color:var(--c-accent);">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderPlan(plan) {
  const area = $('ss-plan-area');
  if (!plan) {
    area.innerHTML = '<div class="text-xs text-center py-8" style="color:var(--c-muted);">尚无执行计划。</div>';
    return;
  }
  const statusBadge = {
    drafted: '<span class="ws-tag pending">待准奏</span>',
    approved: '<span class="ws-tag pending">已准奏</span>',
    executing: '<span class="ws-tag pending">执行中</span>',
    done: '<span class="ws-tag merged">已完成</span>',
    rejected: '<span class="ws-tag" style="background:#555;">已驳回</span>',
  }[plan.status] || `<span class="ws-tag">${esc(plan.status)}</span>`;

  let html = `<div class="flex items-center gap-2 mb-2">
    <span class="seal-mini" style="background:#2d6a4f;">尚</span>
    <span style="font-weight:600; font-size:0.9rem;">${esc(plan.intent || '执行计划')}</span>
    ${statusBadge}
  </div>`;

  // 步骤列表
  for (const s of (plan.steps || [])) {
    html += renderStep(s);
  }

  // 完成摘要
  if (plan.status === 'done' && plan.summary) {
    html += `<div style="margin-top:8px; padding:8px; border-radius:4px; background:rgba(45,106,79,0.1); font-size:0.8rem;">${mdSafe(plan.summary)}</div>`;
  }

  area.innerHTML = html;
}

function renderStep(s) {
  const statusIcon = {
    pending: '⏳',
    running: '🔄',
    done: '✓',
    blocked: '🛡️',
    failed: '✕',
    skipped: '⊘',
  }[s.status] || '?';
  const statusColor = {
    pending: 'var(--c-muted)',
    running: '#fbbf24',
    done: '#2d6a4f',
    blocked: '#b91c1c',
    failed: '#b91c1c',
    skipped: 'var(--c-muted)',
  }[s.status] || 'var(--c-text)';

  let detail = '';
  if (s.status === 'done' && s.result) {
    const r = typeof s.result === 'string' ? s.result : JSON.stringify(s.result);
    const brief = r.length > 200 ? r.slice(0, 200) + '…' : r;
    detail = `<div style="font-size:0.75rem; color:var(--c-text); opacity:0.6; padding-top:2px;">${esc(brief)}</div>`;
  } else if (s.status === 'failed' && s.error) {
    detail = `<div style="font-size:0.75rem; color:#fca5a5; padding-top:2px;">${esc(s.error)}</div>`;
  } else if (s.status === 'blocked' && s.ticket) {
    detail = `<div style="font-size:0.75rem; color:#fca5a5; padding-top:2px;">🛡️ ${esc(s.ticket.summary || '')}（门下省面板可准奏）</div>`;
  } else if (s.status === 'skipped' && s.error) {
    detail = `<div style="font-size:0.75rem; color:var(--c-muted); padding-top:2px;">${esc(s.error)}</div>`;
  }

  const deps = s.depends_on && s.depends_on.length ? ` <span style="color:var(--c-muted);">←${s.depends_on.join(',')}</span>` : '';
  return `<div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
    <div style="display:flex; align-items:center; gap:6px;">
      <span style="color:${statusColor};">${statusIcon}</span>
      <span style="font-size:0.8125rem;">${esc(s.description || s.capability)}</span>
      <span style="font-size:0.7rem; color:var(--c-muted); margin-left:auto;">${esc(s.capability)}${deps}</span>
    </div>
    ${detail}
  </div>`;
}

// 轮询总线消息，更新 plan 状态
async function pollBus() {
  try {
    const data = await api(`/api/control/bus/feed?since=${_busLatestTs}`);
    const msgs = data.messages || [];
    if (msgs.length) {
      _busLatestTs = data.latest_ts || _busLatestTs;
      // 如果有 plan 进度更新，刷新显示
      const hasProgress = msgs.some(m =>
        m.kind === 'plan_progress' || m.kind === 'plan_done' ||
        m.kind === 'step_done' || m.kind === 'step_blocked' ||
        m.kind === 'step_failed' || m.kind === 'block'
      );
      if (hasProgress && _trackedPlanId) {
        refreshPlanDisplay();
      }
    }
  } catch { /* 静默 */ }
}
