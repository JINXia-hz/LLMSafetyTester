"""llmsec.tui.task_store — 任务状态层（TUI 与后端的唯一接触面）。

TUI 是 task_manager 的第三个消费者（另两个：dashboard router / MCP server，
见 mcp/tools/tasks.py 模块注释——三者的 TASKS 注册表互相隔离）。本层合并两个来源：

  - 本进程任务：直读 task_manager.TASKS（可取消、状态权威）
  - 外部任务：扫描 TASK_LOG_DIR 的 .log/.progress.jsonl（看板/MCP 启动的、或 TUI
    重启前的），构造只读 detached 视图——进度照常直播（jsonl 落盘共享），但不可取消

progress.jsonl 增量 tail 回放（记住每个文件的字节 offset，只解析新到的完整行），
避免长任务每 2s 全量重读。任务 id 格式 "{kind}-{HHMMSS}-{hex6}"，kind 取前缀。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llmsec.core.config import TASK_LOG_DIR
from llmsec.tui.render import EvalProgressState

EXTERNAL = "external"  # TUI 特有状态：磁盘扫描发现的外部任务
TERMINAL_STATUSES = ("success", "failed", "cancelled")
_EXTERNAL_MAX = 20  # 外部任务最多显示条数（旧日志无限堆积，只取最近）


@dataclass
class TaskSnapshot:
    """一次 refresh 时的任务快照（视图 + 回放后的进度状态）。"""

    id: str
    kind: str
    status: str
    cmd: str = ""
    started_at: str = ""
    owned: bool = True
    log_tail: str = ""
    state: EvalProgressState | None = None
    meta: dict | None = None  # launch 层结构化摘要（targets/max_rounds/study）
    pid: int | None = None  # 外部任务的进程号（meta.json 提供，跨进程取消用）


def _tail_text(path: Path, limit: int = 4000) -> str:
    """读文件尾部 limit 字符（与 task_view 的 4KB tail 同口径）。

    r7/M-12：seek 距离用 limit 而非硬编码 4096——原先 full_log 传
    limit=2_000_000 也只能读到固定 4KB，"完整日志"对外部任务恒被截断。
    """
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _pid_alive(pid: int | None) -> bool:
    """进程存活探测（无 psutil 依赖：win32 用 OpenProcess，posix 用 kill 0 探测）。"""
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == 259  # STILL_ACTIVE
            return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _kill_pid(pid: int) -> bool:
    """跨进程强杀（win32 taskkill /T 连子进程树一起；posix SIGTERM→SIGKILL）。"""
    if sys.platform == "win32":
        import subprocess

        # 不做文本解码：taskkill 在中文 Windows 输出 GBK，UTF-8 环境下解码会抛错；
        # 这里只关心 returncode。
        r = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
        return r.returncode == 0
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        pass
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


class TaskStore:
    def __init__(self, log_dir: Path | str | None = None) -> None:
        # 可注入目录（测试用）；默认 TASK_LOG_DIR（output/tasks/）
        self._dir = Path(log_dir) if log_dir is not None else TASK_LOG_DIR
        self._states: dict[str, EvalProgressState] = {}
        self._offsets: dict[str, int] = {}
        self._prev_status: dict[str, str] = {}

    @property
    def log_dir(self) -> Path:
        return self._dir

    # ============================================================
    # 刷新
    # ============================================================
    def refresh(self) -> tuple[list[TaskSnapshot], bool]:
        """全量刷新：本进程任务 + 磁盘扫描外部任务，progress 增量回放。

        Returns:
            (snapshots 按 started_at 倒序, 是否有任务新进入终态)。
            终态转换信号供上层触发 runs 数据重载。
        """
        from llmsec.server import task_manager

        snaps: dict[str, TaskSnapshot] = {}
        for view in task_manager.list_tasks():
            snaps[view["id"]] = TaskSnapshot(
                id=view["id"],
                kind=view["kind"],
                status=view["status"],
                cmd=view.get("cmd", ""),
                started_at=view.get("started_at", ""),
                owned=True,
                log_tail=view.get("log_tail", ""),
                meta=view.get("meta"),
            )
        for snap in self._scan_detached(snaps):
            snaps[snap.id] = snap

        runs_dirty = False
        for s in snaps.values():
            s.state = self._replay(s.id, s.kind)
            s.state.set_running(s.status == "running" or s.status == EXTERNAL)
            # launch 层 meta：预声明目标占位行（progress 记录到达前渲染「等待中」）
            if s.meta and s.meta.get("targets"):
                s.state.declare_targets(list(s.meta["targets"]), s.meta.get("max_rounds"))
            if not s.owned:
                s.log_tail = _tail_text(self._dir / f"{s.id}.log")
            prev = self._prev_status.get(s.id)
            if prev is not None and prev != s.status and s.status in TERMINAL_STATUSES:
                runs_dirty = True
            self._prev_status[s.id] = s.status

        ordered = sorted(snaps.values(), key=lambda s: s.started_at or "", reverse=True)
        return ordered, runs_dirty

    def _scan_detached(self, owned: dict[str, TaskSnapshot]) -> list[TaskSnapshot]:
        """构造不在本进程 TASKS 里的任务的只读视图。

        storage 重构：外部任务发现走目录库（query_tasks 自带 meta.json 对账，
        免逐个读文件）；状态为 running/queued 时以 PID 存活为准（持有进程崩溃
        后无人回写终态）；PID 已死且无终态记录 → "ended"。无 meta.json 的裸
        文件残留（旧世代 .log/.progress.jsonl，目录扫兜底）保持 EXTERNAL 未知态。
        """
        if not self._dir.is_dir():
            return []
        from llmsec.storage import contract as _storage

        out: list[TaskSnapshot] = []
        seen: set[str] = set()
        try:
            rows = _storage.query_tasks(limit=_EXTERNAL_MAX, tasks_dir=self._dir)
        except Exception:
            rows = []
        for row in rows:
            tid = row.task_id
            seen.add(tid)
            if tid in owned:
                continue
            status = row.status or EXTERNAL
            if status in ("running", "queued") and not _pid_alive(row.pid):
                # 持有进程已退出且无人回写终态（如看板被关闭后任务自然结束）
                status = "ended"
            out.append(
                TaskSnapshot(
                    id=tid,
                    kind=row.kind,
                    status=status,
                    cmd=row.cmd or "",
                    started_at=row.started_at or "",
                    owned=False,
                    meta=row.meta,
                    pid=row.pid,
                )
            )
        # 裸文件残留兜底：只在目录里存在、目录库无行的旧任务（一次性 iterdir，零文件读）
        mtimes: dict[str, float] = {}
        for f in self._dir.iterdir():
            name = f.name
            if name.endswith(".meta.json"):
                tid = name[: -len(".meta.json")]
            elif name.endswith(".progress.jsonl"):
                tid = name[: -len(".progress.jsonl")]
            elif name.endswith(".log"):
                tid = name[: -len(".log")]
            else:
                continue
            if not tid or "-" not in tid or tid in owned or tid in seen:
                continue
            try:
                mtimes[tid] = max(mtimes.get(tid, 0.0), f.stat().st_mtime)
            except OSError:
                continue
        for tid, mtime in sorted(mtimes.items(), key=lambda kv: kv[1], reverse=True):
            if len(out) >= _EXTERNAL_MAX:
                break
            out.append(
                TaskSnapshot(
                    id=tid,
                    kind=tid.split("-", 1)[0],
                    status=EXTERNAL,
                    owned=False,
                    started_at=datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                )
            )
        return out[:_EXTERNAL_MAX]

    def _replay(self, task_id: str, kind: str) -> EvalProgressState:
        """增量回放 progress.jsonl：从上次 offset 读新完整行，逐条 apply_record。"""
        st = self._states.get(task_id)
        if st is None:
            st = EvalProgressState(task_id, kind if kind != "?" else "evaluate")
            self._states[task_id] = st
        path = self._dir / f"{task_id}.progress.jsonl"
        if not path.exists():
            return st
        try:
            with open(path, "rb") as f:
                off = self._offsets.get(task_id, 0)
                size = path.stat().st_size
                if size < off:
                    # 文件被截断（task id 唯一，正常不发生）：状态重置从头回放
                    st = EvalProgressState(st.task_id, st.kind)
                    self._states[task_id] = st
                    off = 0
                f.seek(off)
                chunk = f.read()
        except OSError:
            return st
        if not chunk:
            return st
        consumed = len(chunk)
        if not chunk.endswith(b"\n"):
            nl = chunk.rfind(b"\n")  # 尾部半行（并发写入中）留到下次
            if nl < 0:
                return st
            chunk = chunk[: nl + 1]
            consumed = nl + 1
        self._offsets[task_id] = self._offsets.get(task_id, 0) + consumed
        for line in chunk.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st.apply_record(rec)
        return st

    # ============================================================
    # 动作（统一走 launch 层——与 Web 看板 / MCP 同一条链路）
    # ============================================================
    def start_evaluation(self, **kwargs) -> dict:
        """提交评估。kwargs 为 LaunchSpec 字段（target/targets/input_file/max_rounds/...）。

        LaunchError 统一翻译成 {"error", "hint"}，面板层直接 notify。
        """
        from llmsec.server.launch import LaunchError, LaunchSpec, launch_evaluation

        spec = LaunchSpec(**kwargs)
        try:
            return launch_evaluation(spec)
        except LaunchError as e:
            return {"error": str(e), "hint": e.hint}

    def start_hpo(self, yaml_path: str) -> dict:
        """以 hpo 任务启动 study.yaml（launch 层校验路径并携带 meta）。"""
        from llmsec.server.launch import LaunchError, launch_hpo_study

        try:
            return launch_hpo_study(yaml_path)
        except LaunchError as e:
            return {"error": str(e), "hint": e.hint}

    def cancel(self, task_id: str) -> dict:
        """取消任务：本进程走 task_manager；外部任务经目录库行的 PID 跨进程强杀。"""
        from llmsec.server.task_manager import cancel_task
        from llmsec.storage import contract as _storage

        view = cancel_task(task_id)
        if view is not None:
            return view
        # 外部任务：库行带 PID 且进程仍活 → 跨进程终止（确认弹窗已在面板层）
        row = _storage.get_task(task_id, tasks_dir=self._dir)
        pid = row.pid if row is not None else None
        if isinstance(pid, int) and _pid_alive(pid):
            if _kill_pid(pid):
                try:
                    _storage.update_task(task_id, status="cancelled")
                except Exception:
                    pass  # P4：库行即唯一真相；写失败不阻断取消回报
                return {"id": task_id, "status": "cancelled", "killed_pid": pid}
            return {"error": f"强杀 PID {pid} 失败（taskkill 返回非零），请手动处理"}
        return {"error": "任务不存在或已结束（外部任务无存活 PID，无法跨进程取消）"}

    def full_log(self, task_id: str) -> str:
        """读任务完整日志：本进程走 task_manager，外部任务直接读文件。"""
        from llmsec.server.task_manager import read_full_log

        log = read_full_log(task_id)
        if not log:
            log = _tail_text(self._dir / f"{task_id}.log", limit=2_000_000)
        return log


# ============================================================
# 展示助手（console 的 ls tasks 与 top 视图共用）
# ============================================================
_KIND_LABEL = {"evaluate": "评估", "hpo": "HPO"}


def kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, kind)


def short_cmd(cmd: str) -> str:
    """从任务命令行提取有辨识度的短摘要（目标名 / yaml 名）。

    仅作 meta 缺席时的兜底（launch 层统一携带 meta 后，常规任务不走这里）。
    """
    if not cmd:
        return ""
    toks = cmd.split()
    for flag in ("--target", "--targets"):
        if flag in toks:
            i = toks.index(flag)
            if i + 1 < len(toks):
                v = toks[i + 1]
                return v.replace(",", "+") if flag == "--targets" else v
    if "llmsec.experiments" in cmd:
        last = toks[-1].replace("\\", "/").rsplit("/", 1)[-1]
        return last
    return cmd[:48]


def task_summary(snap: TaskSnapshot) -> str:
    """任务短摘要：优先 launch 层 meta（结构化，无反向解析），兜底 short_cmd。"""
    meta = snap.meta or {}
    if meta.get("targets"):
        return "+".join(meta["targets"])
    if meta.get("study"):
        return str(meta["study"])
    return short_cmd(snap.cmd)


# ============================================================
# 表单数据源
# ============================================================
def attack_files() -> list[str]:
    """attacks/ 目录下可选攻击集（文件名）。"""
    from llmsec.core.config import ATTACKS_DIR

    if not ATTACKS_DIR.is_dir():
        return []
    return sorted(f.name for f in ATTACKS_DIR.glob("*.jsonl"))


def study_yamls() -> list[str]:
    """可启动的 study 配置：仓库 experiments/ 与 output/experiments/（看板生成的）下的
    yaml，按 mtime 倒序，返回仓库根相对路径。"""
    from llmsec.core.config import OUTPUT_DIR, PROJECT_ROOT

    files: list[Path] = []
    for d in (PROJECT_ROOT / "experiments", OUTPUT_DIR / "experiments"):
        if d.is_dir():
            files.extend(d.glob("*.yaml"))
            files.extend(d.glob("*.yml"))
    files = [f for f in files if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        try:
            out.append(str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"))
        except ValueError:
            continue
    return out
