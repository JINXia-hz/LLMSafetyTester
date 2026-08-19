/* menxia.js — 宣政殿「门下省」衙署：封驳记录 + 审查简报（依赖 core.js / control.js 全局）
 *
 * 轮询总线消息，展示：
 *   - 封驳令（block）+ 准奏按钮（放行该步）
 *   - 封驳解除（step_unblocked：用户放行 reason=approve / 随 Plan 驳回撤销 reason=reject）
 *     → 递减待裁计数、按钮翻成已放行印。幂等去重：本地按钮与总线消息双路径
 *       谁先到谁生效（applyUnblock），跨标签页放行、刷新重放均能配平。
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
let _mxUnblocked = new Set();  // 已解除的 blockKey（plan:step）——计数/翻按钮幂等

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
    // E-7：待裁计数以 ctl_tickets 权威清单为基准（服务重启/总线缓冲挤出/重放
    // 都不再影响计数），总线消息只负责卡片渲染与乐观更新
    if (typeof data.pending_count === 'number') {
      _mxPendingBlocks = data.pending_count;
    }
    const activeKeys = new Set(
      (data.active_blocks || []).map(b => `${b.plan_id}:${b.step_id}`));
    const msgs = data.messages || [];
    if (msgs.length) {
      _mxLatestTs = data.latest_ts || _mxLatestTs;
      for (const m of msgs) {
        handleBusMessage(m);
      }
    }
    reconcileBlocks(activeKeys);
    updateMenxiaStatus();
    renderMenxiaLog();
  } catch { /* 静默 */ }
}

// 对账：已不在权威清单里的封驳卡翻成"已放行"印（他页放行/驳回撤销/刷新重放
// 的兜底；applyUnblock 已处理过的由 Set 幂等跳过）。计数不做本地增减——下轮
// 轮询的 pending_count 即真相。
function reconcileBlocks(activeKeys) {
  for (const e of _mxEntries) {
    if (!e.blockKey || _mxUnblocked.has(e.blockKey)) continue;
    if (!activeKeys.has(e.blockKey)) {
      _mxUnblocked.add(e.blockKey);
      e.html = e.html.replace(
        /<button class="mx-unblock"[^>]*>准奏放行<\/button>/,
        '<span class="mx-unblock-done">已放行</span>');
    }
  }
}

function handleBusMessage(m) {
  // 只关心门下省相关消息
  if (m.kind === 'block') {
    const t = m.payload.ticket;
    if (!t) return;
    // plan_id 在消息信封顶层（notify_routed 的信封字段），step_id 在 payload
    const key = `${m.plan_id}:${m.payload.step_id}`;
    const done = _mxUnblocked.has(key);   // 重放乱序兜底：已解除的不再渲染按钮
    const actionHtml = done
      ? '<span class="mx-unblock-done">已放行</span>'
      : `<button class="mx-unblock" data-plan="${esc(m.plan_id)}" data-step="${esc(m.payload.step_id)}">准奏放行</button>`;
    _mxEntries.push({
      type: 'block', ts: m.ts, blockKey: key,
      html: `<div class="mx-entry mx-block">
        <div class="mx-title"><span class="seal-mini" style="width:18px;height:18px;font-size:10px;background:var(--c-warn);">驳</span> ${esc(t.summary)}</div>
        <div class="mx-body">${esc(t.detail)}</div>
        <div style="display:flex; gap:6px; align-items:center; margin-top:6px;">
          <span style="font-size:0.68rem; color:var(--c-muted);">步骤 ${esc(m.payload.step_id)} · ${esc(t.capability)} · ${esc(t.risk_level)}级</span>
          ${actionHtml}
        </div>
      </div>`,
    });
  } else if (m.kind === 'step_unblocked') {
    // 封驳解除：放行（reason=approve）或随 Plan 驳回撤销（reason=reject）
    if (m.plan_id && m.payload.step_id) {
      applyUnblock(m.plan_id, m.payload.step_id,
        m.payload.reason === 'reject' ? '已随驳回撤销' : '已放行');
    }
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

// 幂等解除一个封驳：乐观递减计数 + 卡片按钮翻成已放行印（下轮轮询的
// pending_count 权威值会覆写计数，乐观值只为即时反馈）。
// 双路径调用：① 本页按钮点击成功后（即时反馈）② 总线 step_unblocked 消息
// （他页放行 / Plan 驳回撤销）——Set 去重，谁先到谁生效。
function applyUnblock(planId, stepId, label) {
  const key = `${planId}:${stepId}`;
  if (_mxUnblocked.has(key)) return;
  _mxUnblocked.add(key);
  _mxPendingBlocks = Math.max(0, _mxPendingBlocks - 1);
  const e = _mxEntries.find(x => x.blockKey === key);
  if (e) {
    e.html = e.html.replace(
      /<button class="mx-unblock"[^>]*>准奏放行<\/button>/,
      `<span class="mx-unblock-done">${esc(label)}</span>`);
  }
  updateMenxiaStatus();
  renderMenxiaLog();
}

async function unblockStep(planId, stepId, btn) {
  btn.disabled = true;
  try {
    const res = await fetch('/api/control/plan/block/approve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planId, step_id: stepId }),
    });
    if (res.ok) {
      applyUnblock(planId, stepId, '已放行');   // 总线消息稍后到达时幂等跳过
    } else if (res.status === 404) {
      // 令已被他处清除（他页放行 / Plan 已驳回）——同样按已处理收场
      applyUnblock(planId, stepId, '已放行（令已失效）');
    } else {
      btn.disabled = false;
    }
  } catch { btn.disabled = false; }
}
