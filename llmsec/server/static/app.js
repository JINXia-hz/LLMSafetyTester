/* app.js — 入口：报告渲染 + 打印 PDF（绢本存档）+ 阅读进度 + 启动；依赖 core/sections/analysis/run-control（按序先加载） */

// ---------- 报告 ----------
// 最近一次报告的结构化缓存（PDF 打印用）：{run, headTitle, head, sections:[{title, content}]}
let lastReport = null;

async function loadReport() {
  try {
    const d = await api('/api/report-md' + runQuery());
    const nav = $('reportNav'), body = $('reportBody');
    nav.innerHTML = ''; body.innerHTML = '';
    lastReport = null;
    if (!d.available) {
      body.innerHTML = '<div class="card text-sm" style="color: var(--c-muted);">该批次没有 security_report.md</div>';
      return;
    }
    // .md 原始下载；PDF 走浏览器打印（美化文档见 buildPrintDoc / @media print）
    nav.innerHTML = `<div class="report-nav-title">目录</div>
      <a href="/api/report/download${runQuery()}" class="block px-2 py-1 mb-1 rounded text-xs font-semibold text-center"
      style="border: 1px solid var(--c-gold); color: var(--c-gold);">下载 .md</a>
      <button id="btnPdf" class="block w-full px-2 py-1 mb-2 rounded text-xs font-semibold text-center"
      title="在打印对话框中选择「另存为 PDF」；建议取消勾选「页眉和页脚」"
      style="border: 1px solid var(--c-primary); color: var(--c-primary);">下载 PDF</button>`;
    // 按 ## 分段
    const chunks = d.markdown.split(/^## /m);
    const head = chunks[0];
    const headTitle = (head.match(/^# (.+)$/m) || [])[1] || '安全评估报告';
    const sections = chunks.slice(1).map(chunk => {
      const nl = chunk.indexOf('\n');
      return {
        title: (nl > 0 ? chunk.slice(0, nl) : chunk).trim(),
        content: nl > 0 ? chunk.slice(nl + 1) : '',
      };
    });
    lastReport = { run: d.run, headTitle, head: head.replace(/^# .+$/m, '').trim(), sections };
    body.innerHTML += `<div class="card report-body report-head"><h1>${esc(headTitle)}</h1>${mdSafe(head.replace(/^# .+$/m, ''))}</div>`;
    sections.forEach((sec, i) => {
      const anchor = `rep-${i}`;
      nav.innerHTML += `<a href="#${anchor}" class="rep-link block px-2 py-1 rounded hover:bg-stone-100" style="color: var(--c-primary);">${esc(sec.title)}</a>`;
      const div = document.createElement('div');
      div.className = 'card report-body';
      div.id = anchor;
      div.innerHTML = `<h2>${esc(sec.title)}</h2>${mdSafe(sec.content)}`;
      body.appendChild(div);
    });
    // 注意：必须在 innerHTML 变更全部结束后再挂监听——上面的 += 会重建 nav 子树、丢弃已挂监听器
    $('btnPdf').addEventListener('click', printReport);
    if (PRINT_PREVIEW) enterPrintPreview();   // ?printdoc=1：屏幕预览打印版
  } catch (e) { setStatus('报告加载失败: ' + e.message); }
}

// ---------- 打印 PDF「绢本存档」：封面 + 目录 + 五个板块 live 排印 ----------
// printAll 流程：加 body.printing-all 平铺所有板块（plotly 真实渲染）→ 加载五板块数据 →
// 图表重排 → window.print()；afterprint 卸 class 恢复屏幕。浏览器页眉页脚无法由 CSS 去除，
// 需用户在打印对话框取消勾选（按钮 title 有提示）。
const PRINT_PREVIEW = new URLSearchParams(location.search).has('printdoc');
const PRINT_PARTS = [
  ['sec-overview', '壹 · 总览'],
  ['sec-threats', '贰 · 威胁看板'],
  ['sec-report', '叁 · 安全评估报告'],
  ['sec-clusters', '肆 · 聚类分析'],
  ['sec-model', '伍 · 预测模型'],
];
let _printPrevTheme = null;   // 印前若为夜色则临时切绢本浅色，印完恢复
let _printPrep = false;       // 整备中标志：防 loadReport → enterPrintPreview 递归

// 封面元数据（目标模型/等级/时间）取总览缓存；直达报告页时缓存为空，补拉一次
async function ensureOverview() {
  if (lastOverview) return;
  try {
    const d = await api('/api/overview' + runQuery());
    if (d && d.available) lastOverview = d;
  } catch (e) { /* 拉不到则封面省略对应字段 */ }
}

async function printReport() {
  if (_printPrep) return;
  setStatus('正在整备打印文档…');
  await preparePrintDoc();
  setStatus('');
  window.print();
}

// 打印整备：浅色 + 封面/目录 + 部件题签 + 五板块加载渲染 + 图表重排
async function preparePrintDoc() {
  _printPrep = true;
  try {
    _printPrevTheme = theme;
    if (theme === 'dark') applyTheme('light');          // 纸质走绢本浅色
    document.body.classList.add('printing-all');
    await ensureOverview();
    buildPrintDoc();
    addPrintPartTitles();
    // 各 loader 内部已 try/catch 兜底，五板块并行拉取渲染
    await Promise.all([loadOverview(), loadThreats(), loadReport(), loadClusters(), loadModel()]);
    // 隐藏期渲染的 plotly 图表尺寸为 0，板块可见后统一重排到可打印幅面
    if (window.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(gd => {
      try { Plotly.Plots.resize(gd); } catch (e) { /* 单图失败不阻塞 */ }
    });
    await new Promise(r => setTimeout(r, 400));          // 等重排与楷体字形落位
    await _freezeChartsAsImages();                       // 图表转 PNG：根治跨页（见函数注）
  } finally { _printPrep = false; }
}

// 图表转 PNG：Plotly 内部 position:absolute 的 SVG 会被浏览器忽略 break-inside:avoid，
// 落到页底的图会被从中间切开。转成 <img>（单一内联元素）后 break-inside:avoid 100% 生效。
// 逐图 try/catch——无数据/未初始化/转图失败的图跳过（保留实时图，最多仍可能跨页，不报错）。
async function _freezeChartsAsImages() {
  if (!window.Plotly || typeof Plotly.toImage !== 'function') return;   // partial build 无 toImage → 跳过
  const tasks = [];
  document.querySelectorAll('.js-plotly-plot').forEach(gd => {
    const fl = gd._fullLayout;
    if (!fl) return;   // 未初始化（无数据）图跳过
    try {
      const w = Math.round(gd.clientWidth || fl.width || 680);
      const h = Math.round(fl.height || 360);
      tasks.push(
        Plotly.toImage(gd, { format: 'png', width: w, height: h, scale: 2 })
          .then(url => {
            const img = document.createElement('img');
            img.className = 'print-chart';
            img.src = url;
            gd.parentNode.insertBefore(img, gd.nextSibling);
            gd.classList.add('print-hidden-plot');
          })
          .catch(() => { /* 单图失败不阻塞 */ })
      );
    } catch (e) { /* 同步异常（如尺寸读取失败）跳过该图 */ }
  });
  await Promise.all(tasks);
}

// 封面 + 目录 → #printRoot（正文五个板块直接排印 live DOM，不在此复制）
function buildPrintDoc() {
  const root = $('printRoot');
  if (!root) return false;
  const ov = lastOverview || {};
  const sealChar = SEAL_CHARS[ov.security_level] || '录';   // 无等级数据时用「录」（存档之意）
  const runName = (lastReport && lastReport.run) || ov.run || currentRun || '最新批次';
  const meta = [['批次', runName]];
  if (ov.target_model) meta.push(['目标模型', ov.target_model]);
  if (ov.generated_at) meta.push(['评估时间', String(ov.generated_at).replace('T', ' ').slice(0, 19)]);
  const title = (lastReport && lastReport.headTitle) || '目标模型安全评估报告';
  root.innerHTML = `
    <div class="pdoc-cover"><div class="pdoc-frame"><div class="pdoc-frame-inner">
      <div class="pdoc-cover-main">
        <div class="pdoc-vtitle">${esc(title)}</div>
        <div class="pdoc-cover-side">
          <div class="pdoc-seal">${esc(sealChar)}</div>
          <div class="pdoc-meta">${meta.map(([k, v]) =>
            `<div class="pdoc-meta-row"><span class="pdoc-meta-k">${esc(k)}</span><span class="pdoc-meta-v">${esc(v)}</span></div>`).join('')}</div>
        </div>
      </div>
    </div></div></div>
    <div class="pdoc-toc">
      <div class="pdoc-toc-title">目　录</div>
      ${PRINT_PARTS.map(([, label]) => `<div class="pdoc-toc-row">${esc(label)}</div>`).join('')}
    </div>`;
  return true;
}

function addPrintPartTitles() {
  document.querySelectorAll('.print-part').forEach(el => el.remove());
  PRINT_PARTS.forEach(([id, label]) => {
    const sec = $(id);
    if (!sec) return;
    const div = document.createElement('div');
    div.className = 'print-part';
    div.textContent = label;
    sec.insertBefore(div, sec.firstChild);
  });
}

// 印完（含取消）恢复屏幕现场；预览模式保留，由「退出预览」按钮清理
function cleanupPrintDoc() {
  document.body.classList.remove('printing-all', 'preview-print');
  document.querySelectorAll('.print-part').forEach(el => el.remove());
  // 撤销图表转图：移除打印图片、恢复实时 Plotly，重绘回屏幕尺寸
  document.querySelectorAll('.print-chart').forEach(el => el.remove());
  document.querySelectorAll('.print-hidden-plot').forEach(el => el.classList.remove('print-hidden-plot'));
  const root = $('printRoot');
  if (root) root.innerHTML = '';
  if (_printPrevTheme === 'dark') applyTheme('dark');
  _printPrevTheme = null;
  const btn = $('pdocExit');
  if (btn) btn.remove();
  invalidate();   // 清 loaded 缓存重绘，防打印态 resize 尺寸残留
}
window.addEventListener('afterprint', () => {
  if (!document.body.classList.contains('preview-print')) cleanupPrintDoc();
});

// ?printdoc=1 屏幕预览：与打印同款的平铺排布，便于查验样式
async function enterPrintPreview() {
  if (_printPrep) return;
  await preparePrintDoc();
  document.body.classList.add('preview-print');
  if (!$('pdocExit')) {
    const btn = document.createElement('button');
    btn.id = 'pdocExit'; btn.className = 'pdoc-exit'; btn.textContent = '退出预览';
    btn.addEventListener('click', cleanupPrintDoc);
    document.body.appendChild(btn);
  }
}

// ---------- 阅读进度条 + 报告目录 scrollspy ----------
window.addEventListener('scroll', () => {
  const h = document.documentElement;
  const max = h.scrollHeight - h.clientHeight;
  $('readProgress').style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
  if (activeSection !== 'report') return;
  let current = null;
  document.querySelectorAll('#reportBody .card[id^="rep-"]').forEach(el => {
    if (el.getBoundingClientRect().top <= 140) current = el.id;
  });
  document.querySelectorAll('#reportNav a.rep-link').forEach(a => {
    a.classList.toggle('active', current !== null && a.getAttribute('href') === '#' + current);
  });
}, { passive: true });

// ---------- 启动 ----------
(async () => {
  const bootT0 = performance.now();   // splash 最小展示计时
  // 全部资源（含 defer 的 CDN 脚本）就绪信号
  const winLoad = document.readyState === 'complete'
    ? Promise.resolve()
    : new Promise(r => window.addEventListener('load', r, { once: true }));
  // URL 参数（可分享的视图状态）：?theme=dark|light  ?cmp=<批次名>
  const q = new URLSearchParams(location.search);
  if (q.get('theme') === 'dark' || q.get('theme') === 'light') theme = q.get('theme');
  applyTheme(theme, false);   // 恢复主题（不触发重绘）
  splashStage('览批次…');
  await loadRuns();
  // hash 直达：#threats 等；默认总览
  const h = location.hash.slice(1);
  const start = SECTIONS.includes(h) ? h : 'overview';
  if (start !== 'overview') {
    document.querySelector(`#nav .nav-item[data-section="${start}"]`)?.click();
  } else {
    loadSection('overview');
  }
  loadRunSection();
  splashStage('绘丹青…');
  // splash 收尾：全部资源就绪 + 数据首轮加载 + 至少展示 1s，避免一闪而过
  await winLoad;
  setTimeout(dismissSplash, Math.max(0, 1000 - (performance.now() - bootT0)));
  // 直达对比视图：等总览数据就位后自动展开对比面板
  const cmpRun = q.get('cmp');
  if (cmpRun && start === 'overview') {
    const t = setInterval(() => {
      if (!lastOverview) return;
      clearInterval(t);
      if (!cmpActive) $('cmpBtn').click();
      $('cmpSelect').value = cmpRun;
      renderCompare();
    }, 200);
    setTimeout(() => clearInterval(t), 10000);
  }
})();
