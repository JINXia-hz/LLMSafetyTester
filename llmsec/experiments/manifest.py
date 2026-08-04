"""
experiments.manifest — per-trial 复现清单捕获。

每 trial 落盘 manifest.json：git 版本 / params 快照 / argv / 攻击集 sha1 /
seed / 库版本 / .env 脱敏键。确保"最佳 config"可精确复现。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path


def _git_info() -> dict:
    try:
        def g(*a):
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace").stdout.strip()
        return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(g("status", "--porcelain"))}
    except Exception as e:
        return {"error": str(e)}


def _attack_set_hash(path: str | Path) -> str | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        h = hashlib.sha1()
        h.update(p.read_bytes())
        return h.hexdigest()[:16]
    except Exception:
        return None


def _params_snapshot() -> dict:
    """抓 params.py 当前全部常量值（已被 LLMSEC_PARAM_* 覆盖后的生效值）。"""
    import llmsec.params as P
    return {k: getattr(P, k) for k in dir(P)
            if k.isupper() and not k.startswith("_") and not callable(getattr(P, k))}


def _env_redacted() -> dict:
    """记录与评估相关的 .env 键（脱敏：仅保留非密钥键的值，密钥只标存在。"""
    import os
    keys = ["TARGET_TYPE", "TARGET_BASE_URL", "TARGET_MODEL",
            "GENERATOR_BASE_URL", "GENERATOR_MODEL", "JUDGE_MODEL",
            "EMBEDDING_API_BASE", "EMBEDDING_API_MODEL", "HF_ENDPOINT",
            "TARGETS"]
    out = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    # 标注密钥存在性但不存值
    for k in list(os.environ):
        if "KEY" in k or "TOKEN" in k or "SECRET" in k:
            out[k + "__present"] = "yes"
    return out


def capture_manifest(
    work_dir: Path,
    argv: list[str],
    env_override: dict[str, str],
    seed: int,
    attack_set: str | None,
    config: dict,
) -> dict:
    """捕获并落盘 manifest.json 到 work_dir。"""
    manifest = {
        "captured_at": datetime.now().isoformat(),
        "git": _git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "config": config,
        "argv": argv,
        "env_override": env_override,
        "attack_set": attack_set,
        "attack_set_sha1": _attack_set_hash(attack_set) if attack_set else None,
        "params_snapshot": _params_snapshot(),
        "env_redacted": _env_redacted(),
        "library_versions": _lib_versions(),
    }
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _lib_versions() -> dict:
    vers = {}
    for mod in ("numpy", "sklearn", "optuna", "yaml", "openai"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return vers
