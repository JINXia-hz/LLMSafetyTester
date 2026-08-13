/* menxia.js — 宣政殿「门下省」面板：封驳记录 + 审查简报（依赖 core.js 全局）
 *
 * 轮询总线消息，展示：
 *   - 封驳令（block）+ 准奏按钮（放行该步）
 *   - 审查简报（review）
 *   - 异常呈递（step_failed）
 *   - Plan 完成通知（plan_done）
 */

let _mxBound = false;
let _mxPollTimer = null;
let _mxLatestTs = 0;
let _mxEntries = [];   // 门下省日志条目 [{type, ts, html}]

function loadMenxiaSection() {
  if (!_mxBound) bindMenxia();
  if (!_mxPollTimer) _mxPollTimer = setInterval(pollMenxiaBus, 2000);
  // 首次加载也拉一次封驳列表
  refreshBlocks();
}

function bindMenxia() {
  _mxBound = true;
  const refresh = $('mx-refresh');
  if (refresh) refresh.onclick = () => { refreshBlocks(); renderMenxiaLog(); };
}

async function pollMenxiaBus() {
  try {
    const data = await api(`/api/control/bus/feed?since=${_mxLatestTs}`);
    const msgs = data.messages || [];
    if (!msgs.length) return;
    _mxLatestTs = data.latest_ts || _mxLatestTs;
    for (const m of msgs) {
      handleBusMessage(m);
    }
    renderMenxiaLog();
  } catch { /* 静默 */ }
}

function handleBusMessage(m) {
  // 只关心门下省相关消息
  if (m.kind === 'block') {
    const t = m.payload.ticket;
    if (!t) return;
    _mxEntries.push({
      type: 'block', ts: m.ts,
      html: `<div style="border-left:3px solid #b91c1c; padding:8px 12px; background:rgba(185,28,28,0.08); border-radius:0 4px 4px 0;">
        <div style="font-weight:600; color:#fca5a5; font-size:0.8125rem;">🛡️ ${esc(t.summary)}</div>
        <div style="font-size:0.75rem; color:var(--c-text); opacity:0.7; white-space:pre-line; margin:4px 0;">${esc(t.detail)}</div>
        <div style="display:flex; gap:6px; align-items:center;">
          <span style="font-size:0.7rem; color:var(--c-muted);">步骤 ${esc(m.payload.step_id)} · ${esc(t.capability)} · ${esc(t.risk_level)}级</span>
          <button class="mx-unblock" data-plan="${esc(m.payload.plan_id)}" data-step="${esc(m.payload.step_id)}"
            style="margin-left:auto; padding:3px 10px; border-radius:3px; background:#b91c1c; color:#fff; font-size:0.7rem; border:none; cursor:pointer;">准奏放行</button>
        </div>
      </div>`,
    });
  } else if (m.kind === 'review') {
    const p = m.payload;
    if (p.type === 'failure_report') {
      _mxEntries.push({
        type: 'failure', ts: m.ts,
        html: `<div style="border-left:3px solid #f59e0b; padding:8px 12px; background:rgba(245,158,11,0.08); border-radius:0 4px 4px 0;">
          <div style="color:#fbbf24; font-size:0.8125rem;">⚠ 步骤 ${esc(p.step_id)} 执行失败</div>
          <div style="font-size:0.75rem; color:var(--c-text); opacity:0.7; margin-top:2px;">${esc(p.error || '')}</div>
        </div>`,
      });
    } else if (p.reviews && p.reviews.length) {
      for (const rv of p.reviews) {
        const digest = rv.review?.digest || rv.review?.summary || JSON.stringify(rv.review || {}).slice(0, 300);
        _mxEntries.push({
          type: 'review', ts: m.ts,
          html: `<div style="border-left:3px solid #2d6a4f; padding:8px 12px; background:rgba(45,106,79,0.08); border-radius:0 4px 4px 0;">
            <div style="color:#52b788; font-size:0.8125rem;">📋 审查简报（步骤 ${esc(rv.step_id)}）</div>
            <div style="font-size:0.75rem; color:var(--c-text); opacity:0.8; white-space:pre-line; margin-top:4px;">${mdSafe(digest)}</div>
          </div>`,
        });
      }
    }
  } else if (m.kind === 'plan_done') {
    _mxEntries.push({
      type: 'plan_done', ts: m.ts,
      html: `<div style="text-align:center; font-size:0.75rem; color:var(--c-muted); padding:4px;">— 执行完毕 —</div>`,
    });
  }
  // 只保留最近 30 条
  if (_mxEntries.length > 30) _mxEntries = _mxEntries.slice(-30);
}

async function refreshBlocks() {
  try {
    const data = await api('/api/control/blocks');
    // blocks 已通过总线消息展示，这里不重复渲染
  } catch { /* 静默 */ }
}

function renderMenxiaLog() {
  const log = $('mx-log');
  if (!_mxEntries.length) {
    log.innerHTML = '<div class="text-xs text-center py-8" style="color:var(--c-muted);">门下省候旨。</div>';
    return;
  }
  log.innerHTML = _mxEntries.map(e => e.html).join('');
  // 绑定准奏放行按钮
  log.querySelectorAll('.mx-unblock').forEach(btn => {
    btn.onclick = () => unblockStep(btn.dataset.plan, btn.dataset.step, btn);
  });
  log.scrollTop = log.scrollHeight;
}

async function unblockStep(planId, stepId, btn) {
  btn.disabled = true;
  try {
    const res = await fetch('/api/control/plan/block/approve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planId, step_id: stepId }),
    });
    if (res.ok) {
      btn.textContent = '已放行';
      btn.style.background = '#2d6a4f';
    }
  } catch { btn.disabled = false; }
}
