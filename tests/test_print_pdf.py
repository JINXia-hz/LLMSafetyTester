"""打印 PDF「绢本存档」的静态契约测试。

架构：封面 + 目录由 JS 构建进 #printRoot；总览/威胁看板/报告/聚类/预测五个板块
直接以 live DOM 平铺排印（body.printing-all），window.print 输出，afterprint 恢复。
服务端无 PDF 依赖。这里锁定该管线在 index.html / app.js 中的关键挂载点：
- printRoot 容器（DOM 序在最前，封面先印）、@page、print-color-adjust（底色输出开关）
- printing-all 平铺规则、部件题签 .print-part、封面/印章/目录样式
- app.js 的 PRINT_PARTS 与五个 section id 对应、buildPrintDoc、afterprint 清理、
  ?printdoc=1 预览钩子、下载按钮
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_INDEX = _ROOT / "llmsec" / "server" / "templates" / "index.html"
_APPJS = _ROOT / "llmsec" / "server" / "static" / "app.js"


def _index() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _appjs() -> str:
    return _APPJS.read_text(encoding="utf-8")


def test_print_root_and_page_rules():
    html = _index()
    assert 'id="printRoot"' in html
    # printRoot 必须在 <body> 最前（封面 DOM 序先于各板块，否则印出来封面在最后）
    assert html.index('id="printRoot"') < html.index('id="sec-overview"')
    assert "@page" in html
    # 浏览器默认省墨丢背景色，缺了它封面宣纸底/朱砂印全白
    assert "print-color-adjust" in html
    # 平铺排布开关
    assert "body.printing-all" in html
    assert "body.preview-print" in html


def test_print_styles_cover_key_elements():
    html = _index()
    for cls in ["pdoc-cover", "pdoc-frame", "pdoc-vtitle", "pdoc-seal",
                "pdoc-toc", "pdoc-exit", "print-part", "print-chart", "print-hidden-plot"]:
        assert f".{cls}" in html, f"缺少打印样式 .{cls}"
    # 封面满铺整页：@page margin:0 + 封面 210×297mm 无负边距（旧负边距方案会溢出页盒产生空白页）
    assert "@page" in html and "margin: 0;" in html
    assert "margin: -12mm -10mm" not in html, "负边距满铺已废弃（@page margin:0 取代）"
    # 唐式函套纹样：双线描金框 + 联珠虚线 + 回纹菱花带
    assert "double #A98A4B" in html, "封面框应为双线描金"
    assert "pdoc-frame-inner::before" in html and "pdoc-frame-inner::after" in html
    # 打印态隐藏屏幕交互杂物（报告侧栏目录等）
    assert "#reportNav" in html and "#cmpPanel" in html


def test_print_parts_match_sections():
    """PRINT_PARTS 引用的 section id 必须真实存在于 index.html。"""
    js, html = _appjs(), _index()
    m = re.search(r"PRINT_PARTS\s*=\s*\[(.*?)\];", js, re.S)
    assert m, "app.js 缺少 PRINT_PARTS 定义"
    ids = re.findall(r"\['(sec-[\w-]+)'", m.group(1))
    assert len(ids) == 5, f"应排印五个板块，实得 {ids}"
    for sid in ids:
        assert f'id="{sid}"' in html, f"index.html 缺少板块 #{sid}"
    # 运行控制不进入打印
    assert "sec-run" not in ids


def test_appjs_print_pipeline():
    js = _appjs()
    assert "function buildPrintDoc" in js
    assert "function addPrintPartTitles" in js
    assert "window.print()" in js
    # 隐藏期渲染的图表尺寸为 0，平铺后必须重排
    assert "Plotly.Plots.resize" in js
    # 图表转 PNG：根治 Plotly 绝对定位 SVG 跨页（单一 <img> 才能 break-inside:avoid）
    assert "Plotly.toImage" in js and "_freezeChartsAsImages" in js
    # 印后恢复屏幕现场（含暗夜主题还原 + 撤销转图）
    assert "afterprint" in js
    assert "cleanupPrintDoc" in js
    # ?printdoc=1 屏幕预览钩子
    assert "printdoc" in js
    assert "preview-print" in js
    # 报告目录里的下载按钮（监听器须在 nav innerHTML 变更结束后挂载，见 loadReport 注释）
    assert "btnPdf" in js
    assert js.index("addEventListener('click', printReport)") > js.index("sections.forEach")
