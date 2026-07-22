#!/usr/bin/env python3
"""
LLM 安全评估交互启动器（原根目录 launcher.py）

启动后交互式选择攻击集、模式、是否重置ELO等，
自动调用根目录 runner.py 薄壳执行评估。

用法：
    python launcher.py
"""

from pathlib import Path
import subprocess
import sys

# 优先使用项目根目录下的 .venv，避免系统 Python 缺少依赖
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    subprocess.run(
        [str(_VENV_PYTHON), "-m", "llmsec.pipeline.launcher"] + sys.argv[1:],
        cwd=_PROJECT_ROOT,
    )
    sys.exit(0)

import json
import os

from llmsec.core.config import ATTACKS_DIR, ELO_FILE, PROJECT_ROOT, load_env
from llmsec.core.logging import setup_console

setup_console()
load_env()


def list_attack_sets() -> dict[str, str]:
    """扫描 output/attacks/ 目录，返回 {显示名: 相对路径}。"""
    sets = {}
    if not os.path.exists(ATTACKS_DIR):
        return sets

    for fname in sorted(os.listdir(ATTACKS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(ATTACKS_DIR, fname)
        rel_path = os.path.join("attacks", fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first = json.loads(f.readline())
            if "prompt" in first and "method" in first:
                count = sum(1 for _ in open(fpath, "r", encoding="utf-8"))
                label = f"{fname} ({count} 条)"
                sets[label] = rel_path
        except Exception:
            pass
    return sets


def prompt_yn(msg: str, default: bool = True) -> bool:
    """询问 y/n。"""
    hint = "[Y/n]" if default else "[y/N]"
    ans = input(f"  {msg} {hint}: ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def prompt_choice(options: list[str], prompt_msg: str) -> int:
    """让用户从列表中选一项，返回索引。"""
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        try:
            choice = input(f"  {prompt_msg} [1-{len(options)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  ⚠  请输入 1-{len(options)}")


def prompt_int(msg: str, default: int) -> int:
    """询问整数。"""
    ans = input(f"  {msg} [{default}]: ").strip()
    if not ans:
        return default
    try:
        return int(ans)
    except ValueError:
        print(f"  ⚠  无效，使用默认值 {default}")
        return default


def main():
    print("=" * 60)
    print("  LLM 安全评估启动器")
    print("=" * 60)
    print()

    # ---- 显示当前配置（.env 已由 core.config.load_env 加载） ----
    target_type = os.getenv("TARGET_TYPE", "openai")
    target_model = os.getenv("TARGET_MODEL", "unknown")

    mode_desc = {
        "pcap_judge": f"PCAP Judge (模型: {os.getenv('PCAP_MODEL_VERSION', '?')})",
        "local_sim": "本地模拟模型",
        "openai": f"OpenAI API (模型: {target_model})",
    }.get(target_type, target_type)

    print(f"  当前模式: {mode_desc}")
    print(f"  TARGET_TYPE: {target_type}")
    print()

    # ---- 扫描攻击集 ----
    print("  可用的攻击集:")
    attack_sets = list_attack_sets()
    if not attack_sets:
        print("    未找到攻击集！请先运行 generate_attacks.py 或 generate_harmbench_attacks.py")
        sys.exit(1)

    keys = list(attack_sets.keys())
    idx = prompt_choice(keys, "选择攻击集")
    selected_file = attack_sets[keys[idx]]
    print(f"  已选择: {selected_file}")
    print()

    # ---- ELO 重置（路径统一为 output/state/elo.json） ----
    elo_reset = False
    if os.path.exists(ELO_FILE):
        elo_size = os.path.getsize(ELO_FILE)
        if elo_size > 100:
            print(f"  检测到已有 ELO 数据 ({elo_size} 字节)")
            elo_reset = prompt_yn("是否清空 ELO 重新开始？", default=True)
            if elo_reset:
                os.remove(ELO_FILE)
                print("  ELO 已重置")
            print()
    else:
        print("  无已有 ELO 数据，将从头开始")
        print()

    # ---- 参数 ----
    print("  运行参数:")
    batch = prompt_int("每轮攻击数 (batch size)", 10)
    rounds = prompt_int("最大轮次 (max rounds)", 5)
    print()

    # ---- 确认并运行 ----
    print("=" * 60)
    print("  运行确认")
    print(f"    攻击集 : {selected_file}")
    print(f"    模式   : {target_type}")
    print(f"    批次   : {batch} 条/轮")
    print(f"    轮次   : 最多 {rounds} 轮")
    print(f"    ELO重置: {'是' if elo_reset else '否'}")
    print("=" * 60)

    if not prompt_yn("确认启动？", default=True):
        print("  已取消")
        return

    print()
    print("  启动 runner ...")
    print()

    cmd = [
        sys.executable,
        "-m", "llmsec.pipeline.runner",
        "--input", selected_file,
        "--batch-size", str(batch),
        "--max-rounds", str(rounds),
    ]

    # cwd 设为仓库根目录（llmsec 的上一级），python -m llmsec.xxx 才能正确定位包
    result = subprocess.run(cmd, cwd=PROJECT_ROOT.parent)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
