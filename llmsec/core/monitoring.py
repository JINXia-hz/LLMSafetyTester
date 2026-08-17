"""core.monitoring — 轻量监控告警设施。

双通道告警：
  1. Webhook：POST JSON 到 LLMSEC_ALERT_WEBHOOK（飞书/钉钉/企业微信/Slack 入站 webhook 均可）。
     非阻塞（线程池提交），失败只 stderr，绝不影响主流程。
  2. 事件文件：append 写 output/alerts.jsonl（每行一个 JSON 事件，线程锁保护）。
     零外部依赖，离线可查，看板可消费。

去抖：同 title 哈希在 _DEDUP_WINDOW 秒内只发一次（防刷屏），内存 dict + 时间戳。

配置（.env）：
  LLMSEC_ALERT_WEBHOOK   webhook URL（空=不启用 webhook）
  LLMSEC_ALERT_LEVEL     最低告警级别（info/warning/error，默认 warning）

设计原则：
  - 告警设施本身绝不影响业务路径（所有调用方用 try/except 包裹）。
  - 无新依赖（仅用标准库 urllib + threading）。
  - 幂等：重复调用安全。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ============================================================
# 常量与配置
# ============================================================
_LEVEL_ORDER = {"info": 0, "warning": 1, "error": 2}

# 去抖窗口：同 title 在该秒数内只发一次（防刷屏）
_DEDUP_WINDOW = 15 * 60  # 15 分钟

# webhook 超时（秒）——短超时防阻塞
_WEBHOOK_TIMEOUT = 10

# 单例线程池（惰性创建，避免 import 期开线程）
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()

# 去抖状态：{title_hash: last_emit_ts}
_dedup: dict[str, float] = {}
_dedup_lock = threading.Lock()

# 事件文件写锁
_file_lock = threading.Lock()

# alerts.jsonl 轮转阈值（超过则轮转为 .1，与 llmsec.log 的 RotatingFileHandler
# 同策略）——原先 append-only 无轮转，长期运行只增不减
_ALERTS_MAX_BYTES = 10 * 1024 * 1024


def _min_level() -> int:
    """读取最低告警级别（LLMSEC_ALERT_LEVEL，默认 warning）。"""
    return _LEVEL_ORDER.get(os.getenv("LLMSEC_ALERT_LEVEL", "warning").lower(), 1)


def _webhook_url() -> str:
    """读取 webhook URL（LLMSEC_ALERT_WEBHOOK，空=不启用）。"""
    return os.getenv("LLMSEC_ALERT_WEBHOOK", "").strip()


def _alerts_file():
    """获取告警事件文件路径（惰性，避免 import 期依赖 config）。"""
    from llmsec.core.config import ALERTS_FILE

    return ALERTS_FILE


# ============================================================
# 内部
# ============================================================
def _title_hash(title: str, context: dict) -> str:
    """title + 关键 context 字段的哈希（用于去抖键）。

    纳入 context 中标识"哪个对象"的字段（如 task_id/run_name），
    使不同对象的同名告警不互相去抖。
    """
    # 提取常见标识字段
    identity_keys = ("task_id", "run_name", "plan_id", "study_name", "model", "name")
    identity = {k: context.get(k) for k in identity_keys if context.get(k) is not None}
    raw = f"{title}|{json.dumps(identity, sort_keys=True, ensure_ascii=False)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _should_emit(dedup_key: str) -> bool:
    """去抖判定：该 key 在 _DEDUP_WINDOW 内未发过则允许，并记录时间戳。"""
    now = time.time()
    with _dedup_lock:
        last = _dedup.get(dedup_key, 0)
        if now - last < _DEDUP_WINDOW:
            return False
        _dedup[dedup_key] = now
        # 顺手清理过期条目（防内存无限增长）
        if len(_dedup) > 200:
            cutoff = now - _DEDUP_WINDOW
            stale = [k for k, v in _dedup.items() if v < cutoff]
            for k in stale:
                del _dedup[k]
        return True


def _write_event_file(event: dict) -> None:
    """追加写告警事件到 output/alerts.jsonl（线程锁保护，失败静默）。

    超过 _ALERTS_MAX_BYTES 时轮转为 .1 后缀——轮转在锁内做，避免并发追加截断。
    """
    try:
        path = _alerts_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with _file_lock:
            try:
                if path.stat().st_size > _ALERTS_MAX_BYTES:
                    rotated = path.with_suffix(path.suffix + ".1")
                    rotated.unlink(missing_ok=True)
                    path.replace(rotated)
            except OSError:
                pass  # 不存在/被占用：跳过轮转直接追加
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"[monitoring] 写 alerts.jsonl 失败: {e}", file=sys.stderr)


def _post_webhook(url: str, payload: dict) -> None:
    """POST JSON 到 webhook URL（在工作线程中调用，失败只 stderr）。"""
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=_WEBHOOK_TIMEOUT) as resp:
            # 2xx 即成功，不读 body
            if resp.status >= 300:
                print(f"[monitoring] webhook 返回非 2xx: {resp.status}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"[monitoring] webhook 请求失败: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[monitoring] webhook 异常: {type(e).__name__}: {e}", file=sys.stderr)


def _get_executor():
    """惰性获取单例 ThreadPoolExecutor（避免 import 期开线程）。"""
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            from concurrent.futures import ThreadPoolExecutor

            _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alert-webhook")
    return _executor


def _build_payload(level: str, title: str, detail: str, context: dict) -> dict:
    """构造通用 webhook payload。

    结构兼容主流入站 webhook（飞书/钉钉/企业微信/Slack）：
    飞书/钉钉/企业微信通常包一层 {msg_type: "text", content: {text: "..."}}
    或 {text: {...}}，但各家不同。本函数输出通用结构，由用户侧适配器转换。
    文档中提供各平台的适配说明。
    """
    return {
        "source": "llmsec",
        "level": level,
        "title": title,
        "detail": detail,
        "context": context,
        "ts": datetime.now().isoformat(),
    }


# ============================================================
# 公开 API
# ============================================================
def emit_alert(
    level: str,
    title: str,
    detail: str = "",
    context: dict | None = None,
    *,
    force: bool = False,
) -> bool:
    """发出一条告警（双通道：webhook + 事件文件）。

    Args:
        level:   "info" / "warning" / "error"
        title:   告警标题（简短，用于去抖键）
        detail:  详细说明
        context: 上下文 dict（如 {task_id, cmd, log_path}）
        force:   True=跳过去抖（测试/紧急场景）

    Returns:
        是否实际发出（被去抖或级别过滤时返回 False）。

    安全保证：本函数绝不抛异常（所有路径 try/except），调用方无需包裹。
    """
    try:
        lvl = level.lower()
        if lvl not in _LEVEL_ORDER:
            lvl = "warning"
        # 级别过滤：低于阈值的丢弃
        if _LEVEL_ORDER[lvl] < _min_level():
            return False

        ctx = dict(context or {})
        dedup_key = _title_hash(title, ctx)

        # 去抖：force=True 时跳过查但仍记录（防 force 后紧接的正常调用重复发）
        if force:
            with _dedup_lock:
                _dedup[dedup_key] = time.time()
        elif not _should_emit(dedup_key):
            return False

        event = _build_payload(lvl, title, detail, ctx)

        # 通道 1：事件文件（同步写，快且可靠）
        _write_event_file(event)

        # 通道 2：webhook（非阻塞，线程池提交）
        url = _webhook_url()
        if url:
            try:
                _get_executor().submit(_post_webhook, url, event)
            except Exception as e:
                print(f"[monitoring] webhook 提交失败: {e}", file=sys.stderr)

        return True
    except Exception as e:
        # 最终兜底：告警设施本身绝不影响业务
        print(f"[monitoring] emit_alert 异常: {type(e).__name__}: {e}", file=sys.stderr)
        return False



def alert_task_failed(task_id: str, kind: str, cmd: str, log_path: str, returncode: int) -> None:
    """任务终态=failed 时的标准告警（task_manager 调用）。"""
    emit_alert(
        level="error",
        title=f"任务失败: {kind}",
        detail=f"子进程退出码 {returncode}。查看日志定位原因。",
        context={
            "task_id": task_id,
            "kind": kind,
            "cmd": cmd,
            "log_path": str(log_path),
            "returncode": returncode,
        },
    )


def alert_zombie_task(task_id: str, kind: str, cmd: str, running_minutes: float) -> None:
    """僵尸任务检测告警（running 超 N 分钟无产出，task_manager 调用）。"""
    emit_alert(
        level="warning",
        title=f"僵尸任务: {kind}",
        detail=f"任务已运行 {running_minutes:.0f} 分钟无产出，可能卡死。",
        context={
            "task_id": task_id,
            "kind": kind,
            "cmd": cmd,
            "running_minutes": round(running_minutes, 1),
        },
    )


def alert_study_aborted(study_name: str, consecutive_failures: int, detail: str = "") -> None:
    """HPO study 熔断告警（连续失败达到阈值，study.py 调用）。"""
    emit_alert(
        level="error",
        title=f"Study 熔断: {study_name}",
        detail=detail or f"连续 {consecutive_failures} 个 trial 失败/超时，study 已中止。",
        context={
            "study_name": study_name,
            "consecutive_failures": consecutive_failures,
        },
    )
