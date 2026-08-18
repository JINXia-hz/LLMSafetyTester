"""env 模板防漂移：.env.example 与代码实际读取的环境变量双向对齐。

正向：代码里字面量读取的每个 env 键（os.getenv / os.environ.get / os.environ[k]）
     必须出现在 .env.example（含注释行），否则新用户照模板配不全。
反向：模板里的每个键必须被代码（或显式豁免的第三方库/机制）消费，
     否则是误导用户的死配置。
动态键（f-string/前缀扫描，无法静态提取字面量）按模式规则豁免：
  TARGET_<N>_<FIELD>（多目标编号方案）、LLMSEC_PARAM_<NAME>（params 覆盖机制）。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 只扫源码目录：tests/ 里的 monkeypatch.setenv 是测试注入，不是产品读取
SCAN_DIRS = ("llmsec", "control", "docker")

# 代码读取、但模板刻意不含的键（新增豁免必须写明原因）
CODE_ONLY_ALLOWLIST = {
    # control 开发/嵌套部署专用（仓库根/python 解释器定位），与 .env 业务语义无关
    "LLMSEC_REPO_ROOT",
    "PYTHON",
}

# 模板包含、但不被仓库代码直接读取的键（第三方库 / 运行环境消费）
TEMPLATE_ONLY_ALLOWLIST = {
    # sentence-transformers 库的模型缓存目录（库自身读取）
    "SENTENCE_TRANSFORMERS_HOME",
}

# 动态前缀模式：模板中匹配这些模式的键无需字面量读取证据
TEMPLATE_PATTERN_EXEMPT = (
    re.compile(r"^TARGET_\d+_(NAME|TYPE|MODEL|BASE_URL|API_KEY)$"),  # 多目标编号方案
    re.compile(r"^LLMSEC_PARAM_[A-Z0-9_]+$"),                        # params.py 常量覆盖
)

_READ_PATTERNS = (
    # os.getenv("NAME" / os.environ.get("NAME" / os.environ["NAME"]（后随非 =，排除赋值）
    re.compile(r"""os\.getenv\(\s*["']([A-Z][A-Z0-9_]+)["']"""),
    re.compile(r"""os\.environ\.get\(\s*["']([A-Z][A-Z0-9_]+)["']"""),
    re.compile(r"""os\.environ\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\](?!\s*=)"""),
)
_TEMPLATE_KEY = re.compile(r"^[#\s]*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def _code_read_keys() -> set[str]:
    keys: set[str] = set()
    for d in SCAN_DIRS:
        for f in (ROOT / d).rglob("*"):
            if f.suffix not in (".py", ".sh") or not f.is_file():
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            for pat in _READ_PATTERNS:
                keys.update(pat.findall(text))
    return keys


def _template_keys() -> set[str]:
    return set(_TEMPLATE_KEY.findall((ROOT / ".env.example").read_text(encoding="utf-8")))


def test_env_example_covers_all_code_read_keys():
    """正向：代码字面量读取的键 ⊆ 模板 ∪ 豁免清单。"""
    missing = _code_read_keys() - _template_keys() - CODE_ONLY_ALLOWLIST
    assert not missing, (
        f"以下环境变量被代码读取但 .env.example 未收录（补模板或在 "
        f"CODE_ONLY_ALLOWLIST 写明豁免原因）：{sorted(missing)}"
    )


def test_env_example_has_no_dead_keys():
    """反向：模板键必须被代码消费（或命中动态前缀模式 / 库消费豁免）。"""
    dead = {
        k for k in _template_keys() - _code_read_keys() - TEMPLATE_ONLY_ALLOWLIST
        if not any(p.match(k) for p in TEMPLATE_PATTERN_EXEMPT)
    }
    assert not dead, (
        f"以下键出现在 .env.example 但代码从不读取（删除，或确认是库消费后加入 "
        f"TEMPLATE_ONLY_ALLOWLIST）：{sorted(dead)}"
    )
