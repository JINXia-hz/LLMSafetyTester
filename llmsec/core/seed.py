"""
core.seed — 全局随机种子（实验可复现性）

runner --seed 注入，供所有 RNG 站点统一读取（K-Fold/D-optimal/PCA），
确保同 seed 下评估流程确定论可复现。Elo 更新本身对给定输入序列确定论。

M-20 修复：set_global_seed 现在同时 seed 全局 random 和 numpy.random，
覆盖 text.gen_math（越狱税题目生成）等用全局 random 的站点，
兑现"同 seed 下确定论可复现"的设计承诺。
"""

import random

import numpy as np

_GLOBAL_SEED: int = 42


def set_global_seed(seed: int) -> int:
    """设置全局种子（runner 启动时调用）。

    同时 seed Python 全局 random 与 numpy.random，覆盖所有 RNG 站点。
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    random.seed(seed)       # M-20：覆盖 text.gen_math 等全局 random 站点
    np.random.seed(seed)    # 兜底覆盖 numpy 全局 RNG
    return _GLOBAL_SEED


def get_global_seed() -> int:
    """读取全局种子（各 RNG 站点调用）。"""
    return _GLOBAL_SEED
