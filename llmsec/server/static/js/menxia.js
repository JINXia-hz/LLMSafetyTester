/* menxia.js — 宣政殿「门下省」衙署：封驳记录 + 审查简报（依赖 core.js / control.js 全局）
 *
 * 轮询总线消息，展示：
 *   - 封驳令（block）+ 准奏按钮（放行该步）
 *   - 审查简报（review）
 *   - 异常呈递（step_failed）
 *   - Plan 完成通知（plan_done）
 * 待裁封驳计数同步到议政流程条「门下省」节点。
 */

let _mxBound = false;
let _mxPollTimer = null;
let _mxLatestTs = 0;
let _mxEntries = [];        // 门下省日志条目 [{type, ts, html, blockKey?}]
let _mxPendingBlocks = 0;   // 待圣裁的封驳数

function loadMenxiaSection() {
  if (!_mxBound) bindMenxia();
  if (!_mxPollTimer) _mxPollTimer = setInterval(pollMenxiaBus, 2000);
}

// 离开宣政殿时停止轮询（core.js 的 loadSection 调用）——
// 此前 setInterval 永不清除，2s 轮询持续到关页
function unloadMenxiaSection() {
  if (_mxPollTimer) { clearInterval(_mxPollTimer); _mxPollTimer = null; }
}

function bindMenxia() {
  _mxBound = true;
  const refresh = $('mx-refresh');
  if (refresh) refresh.onclick = () => { renderMenxiaLog(); };
}

// 流程条职守状态：有待裁封驳时显计数，否则监察中
function updateMenxiaStatus() {
  if (_mxPendingBlocks > 0) {
    setProvStatus('menxia', `封驳 ${_mxPendingBlocks} 起 · 待圣裁`, true);
  } else {
    setProvStatus('menxia', '监察中');
  }
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
    _mxPendingBlocks++;
    updateMenxiaStatus();
    _mxEntries.push({
      type: 'block', ts: m.ts, blockKey: `${m.payload.plan_id}:${m.payload.step_id}`,
      html: `<div class="mx-entry mx-block">
        <div class="mx-title"><span class="seal-mini" style="width:18px;height:18px;font-size:10px;background:var(--c-warn);">驳</span> ${esc(t.summary)}</div>
        <div class="mx-body">${esc(t.detail)}</div>
        <div style="display:flex; gap:6px; align-items:center; margin-top:6px;">
          <span style="font-size:0.68rem; color:var(--c-muted);">步骤 ${esc(m.payload.step_id)} · ${esc(t.capability)} · ${esc(t.risk_level)}级</span>
          <button class="mx-unblock" data-plan="${esc(m.payload.plan_id)}" data-step="${esc(m.payload.step_id)}">准奏放行</button>
        </div>
      </div>`,
    });
  } else if (m.kind === 'review') {
    const p = m.payload;
    if (p.type === 'failure_report') {
      _mxEntries.push({
        type: 'failure', ts: m.ts,
        html: `<div class="mx-entry mx-fail">
          <div class="mx-title">⚠ 步骤 ${esc(p.step_id)} 执行失败</div>
          <div class="mx-body">${esc(p.error || '')}</div>
        </div>`,
      });
    } else if (p.reviews && p.reviews.length) {
      for (const rv of p.reviews) {
        const digest = rv.review?.digest || rv.review?.summary || JSON.stringify(rv.review || {}).slice(0, 300);
        _mxEntries.push({
          type: 'review', ts: m.ts,
          html: `<div class="mx-entry mx-review">
            <div class="mx-title">📋 审查简报（步骤 ${esc(rv.step_id)}）</div>
            <div class="mx-body">${mdSafe(digest)}</div>
          </div>`,
        });
      }
    }
  } else if (m.kind === 'plan_done') {
    _mxEntries.push({
      type: 'plan_done', ts: m.ts,
      html: `<div class="mx-entry mx-fin">— 执行完毕 —</div>`,
    });
  }
  // 只保留最近 30 条
  if (_mxEntries.length > 30) _mxEntries = _mxEntries.slice(-30);
}

function renderMenxiaLog() {
  const log = $('mx-log');
  if (!_mxEntries.length) {
    log.innerHTML = `<div class="court-empty">
      <div class="court-empty-seal">门</div>
      <div class="e-title">门下省候旨</div>
      <div class="e-sub">执行中的危险步骤将在此封驳，任务完成后呈递简报</div>
    </div>`;
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
      btn.style.background = 'var(--c-safe)';
      _mxPendingBlocks = Math.max(0, _mxPendingBlocks - 1);
      updateMenxiaStatus();
    } else {
      btn.disabled = false;
    }
  } catch { btn.disabled = false; }
}
