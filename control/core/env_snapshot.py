"""control.core.env_snapshot — .env 快照（独立资源，隔离连接配置）。

快照用途：让一次实验用隔离的模型列表 / judge / 参数，不碰全局 .env。
与 workspace 解耦——workspace 隔离 R 矩阵/产物，env_snapshot 隔离连接配置。
一个 run 可同时指定 workspace + env_snapshot。

存储：
  output/env_snapshots/
    _index.json               # [{name, created, source, keys, note}]
    <name>/
      .env                     # 完整 .env 副本

生命周期：create → edit → use(跑 run 时指定) → (可选) merge 回全局 / delete
merge 回全局 = 把快照里的 key 写回全局 .env（critical 级，门下省封驳）。
"""

from __future__ import annotations

import shutil

from control.config import LLMSEC_REPO
from control.core.store import AtomicIndexStore

ENV_SNAPSHOTS_DIR = LLMSEC_REPO / "output" / "env_snapshots"
_GLOBAL_ENV = LLMSEC_REPO / ".env"

# env 快照索引存储（原子读写 + Windows PermissionError 重试 + 并发锁）
# base_dir 传 lambda：测试期 monkeypatch ENV_SNAPSHOTS_DIR 后能动态生效
_store = AtomicIndexStore(lambda: ENV_SNAPSHOTS_DIR, "snapshots")

# .env 里受管理的 key 前缀（编辑/merge 时校验合法性，防乱写）
_ALLOWED_KEY_PREFIXES = (
    "TARGETS", "TARGET_",          # 目标模型配置
    "JUDGE_", "JUDGE_API_KEY",     # Judge 配置
    "GENERATOR_",                  # 生成器配置
    "CONTROL_",                    # 控制层配置
    "LLMSEC_PARAM_",               # params.py 运行时覆写
)
# 允许的裸 key（非前缀）
_ALLOWED_BARE_KEYS = {"TARGETS"}


def _is_allowed_key(key: str) -> bool:
    if key in _ALLOWED_BARE_KEYS:
        return True
    return any(key.startswith(p) for p in _ALLOWED_KEY_PREFIXES)


def _ensure_dir() -> None:
    _store.ensure_dir()


# ============================================================
# .env 解析（轻量，不依赖 python-dotenv 的 parse）
# ============================================================
def _parse_env(text: str) -> dict[str, str]:
    """解析 .env 文本为 dict（KEY=VALUE）。忽略注释和空行。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        # 去引号
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def _serialize_env(d: dict[str, str]) -> str:
    """dict → .env 文本（KEY=VALUE，保留引号规则）。"""
    lines = []
    for k, v in d.items():
        # 含空格或特殊字符的值加引号
        if any(c in v for c in (" ", "#", "\n")) or not v:
            lines.append(f'{k}="{v}"')
        else:
            lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def _read_global_env() -> dict[str, str]:
    """读全局 .env 为 dict。不存在返回空。"""
    if not _GLOBAL_ENV.exists():
        return {}
    return _parse_env(_GLOBAL_ENV.read_text(encoding="utf-8"))


# ============================================================
# CRUD
# ============================================================
def create(name: str, *, source: str = "global", note: str = "") -> dict:
    """创建 .env 快照。

    Args:
        name: 快照名（唯一）
        source: "global"=从全局 .env 复制；"blank"=空快照
        note: 备注
    """
    _ensure_dir()
    snap_dir = ENV_SNAPSHOTS_DIR / name
    if snap_dir.exists():
        raise FileExistsError(f"env 快照已存在: {name}")

    snap_dir.mkdir(parents=True)
    env_file = snap_dir / ".env"

    keys: dict[str, str] = {}
    if source == "global":
        keys = _read_global_env()
    elif source == "blank":
        keys = {}
    else:
        # source 可以是另一个快照名（基于它创建）
        src_dir = ENV_SNAPSHOTS_DIR / source
        src_env = src_dir / ".env"
        if not src_env.exists():
            raise FileNotFoundError(f"源快照不存在: {source}")
        keys = _parse_env(src_env.read_text(encoding="utf-8"))

    env_file.write_text(_serialize_env(keys), encoding="utf-8")

    info = {
        "name": name,
        "path": str(snap_dir.relative_to(LLMSEC_REPO)).replace("\\", "/"),
        "source": source,
        "note": note,
        "created": _store.now(),
        "keys": sorted(keys.keys()),
    }
    def _record(idx):
        idx["snapshots"][name] = info
        _store.save(idx)
    _store.update(_record)
    return info


def list_snapshots() -> list[dict]:
    """列出所有 .env 快照（按创建时间倒序）。"""
    idx = _store.load()
    snaps = list(idx.get("snapshots", {}).values())
    snaps.sort(key=lambda x: x.get("created", ""), reverse=True)
    return snaps


def get_snapshot(name: str) -> dict | None:
    idx = _store.load()
    return idx.get("snapshots", {}).get(name)


def edit_key(name: str, key: str, value: str) -> dict:
    """编辑快照内某个 key。

    Args:
        name: 快照名
        key: .env key（须在 _ALLOWED_KEY_PREFIXES 范围内）
        value: 新值
    """
    if not _is_allowed_key(key):
        raise ValueError(
            f"不允许的 key: {key}。受管理的 key 前缀: {_ALLOWED_KEY_PREFIXES}"
        )
    snap_dir = ENV_SNAPSHOTS_DIR / name
    env_file = snap_dir / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f"快照不存在: {name}")

    keys = _parse_env(env_file.read_text(encoding="utf-8"))
    keys[key] = value
    env_file.write_text(_serialize_env(keys), encoding="utf-8")

    # 更新索引的 keys 列表
    def _upd(idx):
        if name in idx.get("snapshots", {}):
            idx["snapshots"][name]["keys"] = sorted(keys.keys())
            _store.save(idx)
    _store.update(_upd)
    return {"name": name, "key": key, "value": value, "keys": sorted(keys.keys())}



def delete(name: str) -> dict:
    """删除快照（目录 + 索引项）。"""
    def _delete(idx):
        if name not in idx.get("snapshots", {}):
            raise KeyError(f"快照不存在: {name}")
        snap_dir = ENV_SNAPSHOTS_DIR / name
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
        info = idx["snapshots"].pop(name)
        _store.save(idx)
        return info
    info = _store.update(_delete)
    return {"deleted": name, "info": info}


def load_env_dict(name: str) -> dict[str, str]:
    """读快照的 .env 为 dict（供 invoker 注入 env_override）。"""
    snap_dir = ENV_SNAPSHOTS_DIR / name
    env_file = snap_dir / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f"快照不存在: {name}")
    return _parse_env(env_file.read_text(encoding="utf-8"))


def merge_to_global(name: str) -> dict:
    """把快照的 key 写回全局 .env（critical 级操作）。

    语义：快照里有的 key 覆盖全局同名 key；快照里没有的不动。
    全局 .env 会先备份到 .env.bak.<ts>。
    """
    snap_keys = load_env_dict(name)
    global_keys = _read_global_env()

    # 备份
    if _GLOBAL_ENV.exists():
        import time
        bak = _GLOBAL_ENV.with_name(f".env.bak.{int(time.time())}")
        shutil.copy2(_GLOBAL_ENV, bak)

    # 合并（快照覆盖全局）
    changed = []
    for k, v in snap_keys.items():
        if global_keys.get(k) != v:
            changed.append(k)
        global_keys[k] = v

    _GLOBAL_ENV.write_text(_serialize_env(global_keys), encoding="utf-8")

    # 更新索引
    def _upd(idx):
        if name in idx.get("snapshots", {}):
            idx["snapshots"][name]["merged_to_global"] = _store.now()
            _store.save(idx)
    _store.update(_upd)

    return {
        "merged": name,
        "changed_keys": changed,
        "total_keys_written": len(snap_keys),
    }
