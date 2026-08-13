/* shangshu.js — 宣政殿「尚书省」衙署：Plan 执行进度（依赖 core.js / control.js 全局）
 *
 * 跟踪正在执行的 Plan，轮询 /api/control/bus/feed 获取步骤进度。
 * 每步显示状态（pending/running/done/blocked/failed/skipped）+ 结果摘要，
 * 同步更新议政流程条「尚书省」节点的职守状态。
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
    $('ss-plan-area').innerHTML = `<div class="text-xs" style="color: var(--c-accent);">加载失败: ${esc(e.message)}</div>`;
  }
}

// 流程条职守状态：plan 状态 + 步骤完成度
function updateShangshuStatus(plan) {
  const steps = plan.steps || [];
  const done = steps.filter(s => s.status === 'done' || s.status === 'skipped').length;
  const map = {
    drafted:   ['待准奏', true],
    approved:  ['已准奏 · 待发', true],
    executing: [`执行中 · ${done}/${steps.length}`, true],
    done:      ['已完成'],
    rejected:  ['已驳回'],
  };
  const [text, busy] = map[plan.status] || [plan.status || '待命'];
  setProvStatus('shangshu', text, busy);
}

function renderPlan(plan) {
  const area = $('ss-plan-area');
  if (!plan) {
    area.innerHTML = `<div class="court-empty">
      <div class="court-empty-seal">尚</div>
      <div class="e-title">尚无执行计划</div>
      <div class="e-sub">复杂旨意经中书拟票、陛下准奏后，此处呈现步骤进度</div>
    </div>`;
    return;
  }
  updateShangshuStatus(plan);
  const statusTag = {
    drafted: '<span class="mini-tag warn">待准奏</span>',
    approved: '<span class="mini-tag warn">已准奏</span>',
    executing: '<span class="mini-tag warn">执行中</span>',
    done: '<span class="mini-tag ok">已完成</span>',
    rejected: '<span class="mini-tag mut">已驳回</span>',
  }[plan.status] || `<span class="mini-tag mut">${esc(plan.status)}</span>`;

  let html = `<div class="flex items-center gap-2 mb-2">
    <span style="font-weight:700; font-size:0.9rem; font-family:var(--serif);">${esc(plan.intent || '执行计划')}</span>
    ${statusTag}
  </div>`;

  // 步骤列表
  for (const s of (plan.steps || [])) {
    html += renderStep(s);
  }

  // 完成摘要
  if (plan.status === 'done' && plan.summary) {
    html += `<div class="ss-summary">${mdSafe(plan.summary)}</div>`;
  }

  area.innerHTML = html;
}

function renderStep(s) {
  // 状态环字符：running 由 CSS 画旋转金环，无字符
  const statusMark = {
    pending: '○', running: '', done: '✓', blocked: '驳', failed: '✕', skipped: '⊘',
  }[s.status] ?? '?';

  let detail = '';
  if (s.status === 'done' && s.result) {
    const r = typeof s.result === 'string' ? s.result : JSON.stringify(s.result);
    const brief = r.length > 200 ? r.slice(0, 200) + '…' : r;
    detail = `<div class="ss-detail">${esc(brief)}</div>`;
  } else if (s.status === 'failed' && s.error) {
    detail = `<div class="ss-detail">${esc(s.error)}</div>`;
  } else if (s.status === 'blocked' && s.ticket) {
    detail = `<div class="ss-detail">${esc(s.ticket.summary || '')}（门下省衙署可准奏放行）</div>`;
  } else if (s.status === 'skipped' && s.error) {
    detail = `<div class="ss-detail">${esc(s.error)}</div>`;
  }

  const deps = s.depends_on && s.depends_on.length ? ` ←${s.depends_on.join(',')}` : '';
  return `<div class="ss-step" data-st="${esc(s.status)}">
    <div class="ss-row">
      <span class="ss-ico">${statusMark}</span>
      <span class="ss-title">${esc(s.description || s.capability)}</span>
      <span class="ss-cap">${esc(s.capability)}${esc(deps)}</span>
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
