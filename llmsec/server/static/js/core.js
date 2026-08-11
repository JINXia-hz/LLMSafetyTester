/* core.js — 全局常量/工具/主题/导航/骨架（基础层，须最先加载；各分块脚本共享其全局） */

/* LLMSEC 安全评估工作台前端逻辑 */

// ---------- 全局状态与常量 ----------
// 绢本金碧 · 唐化配色：石青主色、土红撞色、描金点缀
const C = {
  primary: '#46586B', accent: '#A85B43', warn: '#A85B43',
  safe: '#75876B', ochre: '#B98A44', deep: '#7A4A35', gold: '#BFA03C',
  muted: '#8A8571', text: '#2F343B',
};
const PLOT_CFG = { responsive: true, displayModeBar: false };
const PLOT_FONT = { family: 'ui-sans-serif, system-ui, sans-serif', color: C.text };

// ---------- 主题（绢本纸日 / 石窟夜色） ----------
const THEME_CHART = {
  light: { text: '#2F343B', grid: '#E3D8B8', primary: '#46586B', muted: '#8A8571' },
  dark:  { text: '#EFE3C6', grid: '#4B4136', primary: '#7E9AB4', muted: '#AC9F83' },  // 漆夜玄朱联动
};
let theme = localStorage.getItem('llmsec-theme') || 'light';

function tangLayout() {
  const t = THEME_CHART[theme];
  const axis = { gridcolor: t.grid, zerolinecolor: t.grid };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: PLOT_FONT.family, color: t.text },
    xaxis: axis, yaxis: { ...axis },
    polar: { bgcolor: 'rgba(0,0,0,0)', radialaxis: axis, angularaxis: axis },
  };
}

// Plotly 统一走绢本金碧主题模板（深合并，业务 layout 的 xaxis/yaxis 不会被底色覆盖）
const _newPlot = Plotly.newPlot.bind(Plotly);
Plotly.newPlot = (id, traces, layout = {}, cfg) => {
  const base = tangLayout();
  const merged = {
    ...base, ...layout,
    font: { ...base.font, ...(layout.font || {}) },
    xaxis: { ...base.xaxis, ...(layout.xaxis || {}) },
    yaxis: { ...base.yaxis, ...(layout.yaxis || {}) },
    polar: layout.polar ? { ...base.polar, ...layout.polar } : undefined,
  };
  if (!layout.polar) delete merged.polar;
  return _newPlot(id, traces, merged, cfg);
};

// 匾额等级印章：等级 → 印字
const SEAL_CHARS = { safe: '安', allergic: '警', vulnerable: '伤', broken: '破', inconclusive: '?' };
function setBanner(level) {
  $('ov_banner').className = 'banner plaque mb-2 level-' + level;
  const seal = $('ov_seal');
  seal.className = 'seal level-' + level;
  seal.textContent = SEAL_CHARS[level] || '?';
  seal.classList.remove('seal-anim'); void seal.offsetWidth; seal.classList.add('seal-anim'); // 重触发盖印
}

let currentRun = '';           // '' = 最新
let activeSection = 'overview';
let lastOverview = null;       // 最近一次总览数据（批次对比用）
const loaded = {};             // section -> 已加载的 run

const $ = id => document.getElementById(id);
const fmtPct = v => (v == null ? 'N/A' : (v * 100).toFixed(1) + '%');
const fmtNum = (v, d = 1) => (v == null ? 'N/A' : Number(v).toFixed(d));

// ---------- 动效基础设施 ----------
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;

// 数字滚动：值变化时 500ms rAF 插值（ease-out cubic）；null/初设/未变化/reduced-motion 直接写终值
function setMetric(id, num, fmt) {
  const el = $(id);
  if (!el) return;
  const prev = el._v;
  el._v = num;
  if (num == null || prev == null || prev === num || REDUCED_MOTION) {
    el.textContent = fmt(num);
    return;
  }
  cancelAnimationFrame(el._raf);
  const t0 = performance.now(), dur = 500;
  const step = t => {
    if (el._v !== num) return;                     // 已被更新的值取代，旧动画静默终止
    const k = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(prev + (num - prev) * e);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

// 盖印开卷 splash 收尾：淡出后移除节点；每会话标记在播完时才写入
function dismissSplash() {
  const sp = $('splash');
  if (!sp) return;
  try { sessionStorage.setItem('llmsec-splashed', '1'); } catch (e) { /* 隐私模式 */ }
  sp.classList.add('splash-out');
  setTimeout(() => sp.remove(), 350);
}
// 阶段字幕（展卷…→览批次…→绘丹青…）
function splashStage(t) { const el = $('splashStage'); if (el) el.textContent = t; }
window.addEventListener('load', () => setTimeout(dismissSplash, 4000));  // 兜底：防 API 挂起死白屏

// 骨架屏：section 数据拉取期间给指标卡数值位/图表容器铺描金呼吸块
const SECTION_SKELETONS = {
  overview: { metrics: ['ov_asr', 'ov_fpr', 'ov_boundary', 'ov_conf', 'ov_tested', 'ov_above', 'ov_tax'], charts: ['chart_radar', 'chart_harm'] },
  threats: { metrics: [], charts: ['chart_top_threats', 'chart_convergence'] },
  clusters: { metrics: ['cl_methods', 'cl_n', 'cl_sil', 'cl_db'], charts: ['chart_projection', 'chart_dendrogram', 'chart_rv', 'chart_cluster_cover'] },
  model: { metrics: ['md_lambda', 'md_sigma', 'md_df', 'md_gt'], charts: ['chart_regpath', 'chart_pca', 'chart_importance', 'chart_pred_ci'] },
};
function toggleSkeletons(sec, on) {
  const cfg = SECTION_SKELETONS[sec];
  if (!cfg) return;
  cfg.metrics.forEach(id => $(id)?.classList.toggle('loading', on));
  cfg.charts.forEach(id => {
    const el = $(id);
    if (!el) return;
    el.classList.toggle('chart-loading', on);
    if (on) el.style.height = '';   // 清掉上次钉住的高度，让本次渲染重新测量
    // 修复"图表被下方内容覆盖"：chart-loading 的 min-height 会让 Plotly 把容器当作
    // 已定高（不设内联高度），撤骨架后容器坍缩（Plotly 的 svg 是绝对定位）。
    // 撤骨架时若已渲染，按 Plotly 计算高度把容器钉住。
    if (!on && el.classList.contains('js-plotly-plot') && !el.style.height) {
      el.style.height = (el._fullLayout?.height || 450) + 'px';
    }
  });
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}
function runQuery() { return currentRun ? `?run=${encodeURIComponent(currentRun)}` : ''; }
function setStatus(msg) { $('status').textContent = msg || ''; }

// HTML 转义：服务器字符串插入 innerHTML 前统一过这道
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
// markdown → 安全 HTML（marked 不过滤原始 HTML，必须过 DOMPurify）
const mdSafe = md => DOMPurify.sanitize(marked.parse(md || ''));
// 清空图表：切到无数据批次时避免上一批次的图残留
function clearCharts(ids) {
  ids.forEach(id => { const el = $(id); if (el) { Plotly.purge(el); el.innerHTML = ''; } });
}

// ---------- 导航 ----------
const SECTIONS = ['overview', 'threats', 'report', 'clusters', 'model', 'run'];
let _secToken = 0;   // 翻页过渡令牌：快速连点时作废旧过渡
document.querySelectorAll('#nav .nav-item').forEach(el => {
  el.addEventListener('click', () => {
    const target = el.dataset.section;
    document.querySelectorAll('#nav .nav-item').forEach(n => n.classList.remove('active'));
    el.classList.add('active');
    history.replaceState(null, '', '#' + target);  // 板块可直达/可收藏
    const next = $('sec-' + target);
    if (target === activeSection && next.classList.contains('visible')) return;
    activeSection = target;
    const cur = document.querySelector('.section.visible');
    const tok = ++_secToken;
    const show = () => {
      if (tok !== _secToken) return;               // 已被更新的点击作废
      document.querySelectorAll('.section').forEach(s => s.classList.remove('visible', 'section-enter', 'section-exit'));
      next.classList.add('visible', 'section-enter');
      next.addEventListener('animationend', () => next.classList.remove('section-enter'), { once: true });
      loadSection(target);
    };
    if (REDUCED_MOTION || !cur || cur === next) {
      show();
    } else {
      cur.classList.add('section-exit');           // 先退出（.14s），再进场
      setTimeout(show, 140);
    }
  });
});

function loadSection(name) {
  if (loaded[name] === currentRun) return;
  loaded[name] = currentRun;
  toggleSkeletons(name, true);
  Promise.resolve(({ overview: loadOverview, threats: loadThreats, report: loadReport,
     clusters: loadClusters, model: loadModel, run: loadRunSection })[name]())
    .finally(() => toggleSkeletons(name, false));   // 渲染（含空数据分支）完成后撤骨架
}
function invalidate() { for (const k in loaded) delete loaded[k]; loadSection(activeSection); }

// ---------- 主题切换（依赖 $ 与 invalidate，定义于此） ----------
function applyTheme(t, rerender = true) {
  theme = t;
  localStorage.setItem('llmsec-theme', t);
  document.documentElement.dataset.theme = t === 'dark' ? 'dark' : '';
  const tc = THEME_CHART[t];
  PLOT_FONT.color = tc.text;
  C.primary = tc.primary; C.muted = tc.muted;   // 图表系列色随主题微调
  $('themeBtn').textContent = t === 'dark' ? '昼' : '夜';   // 楷体单字（印章语言），title 已说明含义
  if (rerender) invalidate();                    // 重绘当前板块图表
}
$('themeBtn').addEventListener('click', () => applyTheme(theme === 'dark' ? 'light' : 'dark'));
