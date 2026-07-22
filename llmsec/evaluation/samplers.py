#!/usr/bin/env python3
"""
攻击方法采样策略集合

为 runner Phase 1 提供可插拔的采样器，目标是在最少测试次数下收敛到
可靠的 Elo 安全边界。

策略：
- GapMinSampler：复刻原有逻辑，按 |att_elo - def_elo| 最小选择。
- InfoGainSampler：全局信息增益打分，兼顾分差、不确定性、簇覆盖、历史成功率。
- CoordinateDescentSampler：把"簇"视为坐标轴，外层按簇轮询，内层沿边界精细搜索。
- HybridSampler：前若干轮用 InfoGain 快速覆盖，之后切换到 CoordinateDescent 精细搜索。
"""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from llmsec.evaluation.elo import ELOTracker


class AttackSampler(ABC):
    """攻击方法采样器抽象基类。"""

    def __init__(self, **kwargs):
        pass

    def set_cluster_info(
        self,
        cluster_report: dict | None = None,
        cluster_artifacts: dict | None = None,
    ):
        """
        注入聚类信息。子类可选择是否使用。

        参数:
            cluster_report: run_clustering_pipeline 返回的报告 dict
            cluster_artifacts: joblib 加载的 cluster artifacts dict
        """
        self.cluster_report = cluster_report or {}
        self.cluster_artifacts = cluster_artifacts or {}

    @abstractmethod
    def select(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
        n: int,
        **kwargs,
    ) -> list[str]:
        """
        从候选方法中选择 n 个下一轮测试的方法。

        参数:
            candidates: 尚未测试的方法名列表
            tracker: ELOTracker 实例
            defender_name: 防御方名称
            n: 需要选择的方法数

        返回:
            选中的方法名列表（长度 ≤ n）
        """
        ...

    def _method_to_cluster(self, method: str) -> int:
        """辅助：从 cluster_report 中查方法的簇 ID。"""
        labels = (self.cluster_report or {}).get("method_labels", {})
        cid = labels.get(method, -1)
        try:
            return int(cid)
        except Exception:
            return -1

    def reset(self):
        """重置采样器内部状态（如轮询计数）。子类可覆盖。"""
        pass


# ============================================================
# 1. 分差最小采样器（原有逻辑）
# ============================================================
class GapMinSampler(AttackSampler):
    """按 |攻击Elo - 防御Elo| 最小选择。"""

    def select(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
        n: int,
        **kwargs,
    ) -> list[str]:
        if not candidates:
            return []
        pairs = tracker.suggest_next_pairing(candidates, [defender_name], n=n)
        seen = set()
        result = []
        for att, _ in pairs:
            if att not in seen:
                seen.add(att)
                result.append(att)
        return result[:n]


# ============================================================
# 2. 信息增益采样器
# ============================================================
class InfoGainSampler(AttackSampler):
    """
    全局信息增益采样器。

    对每个候选方法打分：
        score = gap
                + alpha * uncertainty
                + beta  * cluster_visit_count[cluster_id]
                - gamma * success_rate

    分值越低越优先。兼顾：
    - 接近 defender 边界（信息量大）
    - 测试次数少 / 结果方差大（需要探索）
    - 簇覆盖（避免扎堆同一类攻击）
    - 历史成功率高（优先可能突破的方向）
    """

    def __init__(
        self,
        alpha: float = 20.0,
        beta: float = 5.0,
        gamma: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 记录每个簇被选中的次数，用于簇覆盖惩罚
        self.cluster_visit_count: dict[int, int] = defaultdict(int)

    def reset(self):
        self.cluster_visit_count.clear()

    def select(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
        n: int,
        **kwargs,
    ) -> list[str]:
        if not candidates:
            return []

        def_elo = tracker.get_defender_elo(defender_name)
        scored = []

        for method in candidates:
            att_elo = tracker.get_attacker_elo(method)
            gap = abs(att_elo - def_elo)
            uncertainty = tracker.get_attacker_uncertainty(method)
            success_rate = tracker.get_attacker_success_rate(method)
            cid = self._method_to_cluster(method)
            visit_count = self.cluster_visit_count.get(cid, 0)

            score = (
                gap
                + self.alpha * uncertainty
                + self.beta * visit_count
                - self.gamma * success_rate
            )
            scored.append((score, method, cid))

        scored.sort(key=lambda x: x[0])
        selected = []
        for _, method, cid in scored[:n]:
            selected.append(method)
            self.cluster_visit_count[cid] += 1

        return selected


# ============================================================
# 3. 分层坐标下降采样器
# ============================================================
class CoordinateDescentSampler(AttackSampler):
    """
    把"簇"视为坐标轴的分层坐标下降采样器。

    外层：在所有簇之间轮询，每次聚焦一个坐标方向（攻击类别）。
    内层：在选定簇内，选择最接近 defender 边界且未充分测试的方法。

    这样可以结构化地遍历攻击类别，避免纯全局打分过早陷入局部最优，
    同时能在每个类别内部精细定位安全边界。
    """

    def __init__(
        self,
        min_tests_per_method: int = 1,
        min_tests_per_cluster: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_tests_per_method = min_tests_per_method
        self.min_tests_per_cluster = min_tests_per_cluster
        self._cluster_queue: list[int] = []
        self._current_cluster: int | None = None
        self._cluster_test_count: dict[int, int] = defaultdict(int)
        self._round_count = 0

    def reset(self):
        self._cluster_queue.clear()
        self._current_cluster = None
        self._cluster_test_count.clear()
        self._round_count = 0

    def _build_cluster_queue(self, candidates: list[str]) -> list[int]:
        """根据 cluster_report 构建待探索簇队列。"""
        labels = (self.cluster_report or {}).get("method_labels", {})
        clusters = set()
        for m in candidates:
            cid = labels.get(m, -1)
            try:
                cid = int(cid)
            except Exception:
                cid = -1
            if cid != -1:
                clusters.add(cid)
        if not clusters:
            return [-1]
        return sorted(clusters)

    def _pick_next_cluster(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
    ) -> int:
        """选择下一个要探索的簇。优先选测试少、邻近边界、高风险的簇。"""
        labels = (self.cluster_report or {}).get("method_labels", {})
        def_elo = tracker.get_defender_elo(defender_name)

        if not self._cluster_queue:
            self._cluster_queue = self._build_cluster_queue(candidates)

        # 给队列中每个簇打分，选最优
        best_cid = None
        best_score = None
        for cid in self._cluster_queue:
            cluster_methods = [
                m for m in candidates if self._method_to_cluster(m) == cid
            ]
            if not cluster_methods:
                continue

            # 该簇已测试次数
            tested_count = self._cluster_test_count.get(cid, 0)
            # 簇内方法到边界的平均距离
            avg_gap = sum(
                abs(tracker.get_attacker_elo(m) - def_elo) for m in cluster_methods
            ) / len(cluster_methods)
            # 簇内平均成功率
            avg_success = sum(
                tracker.get_attacker_success_rate(m) for m in cluster_methods
            ) / len(cluster_methods)

            # 分越低越优先：测试少、离边界近、成功率高
            score = tested_count + avg_gap - self.min_tests_per_cluster * avg_success
            if best_score is None or score < best_score:
                best_score = score
                best_cid = cid

        if best_cid is None:
            best_cid = self._cluster_queue[0] if self._cluster_queue else -1

        # 把选中的簇移到队尾，实现轮询
        if best_cid in self._cluster_queue:
            self._cluster_queue.remove(best_cid)
            self._cluster_queue.append(best_cid)

        return best_cid

    def select(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
        n: int,
        **kwargs,
    ) -> list[str]:
        if not candidates:
            return []

        self._round_count += 1
        def_elo = tracker.get_defender_elo(defender_name)

        # 选择当前聚焦的簇
        current_cid = self._pick_next_cluster(candidates, tracker, defender_name)
        self._current_cluster = current_cid

        # 在选定簇内选择最接近边界且未充分测试的方法
        cluster_methods = [
            m for m in candidates if self._method_to_cluster(m) == current_cid
        ]
        if not cluster_methods:
            # 退化：如果当前簇没有候选，从全局选最接近边界的
            cluster_methods = candidates

        def method_score(method: str) -> float:
            att_elo = tracker.get_attacker_elo(method)
            gap = abs(att_elo - def_elo)
            stats = tracker.attacker_stats.get(method, {})
            n_tests = stats.get("n_matches", 0)
            # 未充分测试的方法优先；同测试次数下按 gap 排序
            return n_tests + gap / 1000.0

        cluster_methods.sort(key=method_score)
        selected = cluster_methods[:n]

        for m in selected:
            self._cluster_test_count[self._method_to_cluster(m)] += 1

        return selected


# ============================================================
# 4. 混合采样器
# ============================================================
class HybridSampler(AttackSampler):
    """
    混合策略：前 explore_rounds 轮用 InfoGainSampler 快速建立跨簇覆盖，
    之后切换到 CoordinateDescentSampler 对重点簇精细搜索。
    """

    def __init__(
        self,
        explore_rounds: int = 2,
        info_gain_alpha: float = 20.0,
        info_gain_beta: float = 5.0,
        info_gain_gamma: float = 10.0,
        coordinate_min_tests_per_method: int = 1,
        coordinate_min_tests_per_cluster: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.explore_rounds = explore_rounds
        self._round_count = 0
        self._info_sampler = InfoGainSampler(
            alpha=info_gain_alpha,
            beta=info_gain_beta,
            gamma=info_gain_gamma,
        )
        self._coord_sampler = CoordinateDescentSampler(
            min_tests_per_method=coordinate_min_tests_per_method,
            min_tests_per_cluster=coordinate_min_tests_per_cluster,
        )
        # 记录上一轮实际使用的子策略
        self.last_sub_sampler: str | None = None

    def set_cluster_info(
        self,
        cluster_report: dict | None = None,
        cluster_artifacts: dict | None = None,
    ):
        super().set_cluster_info(cluster_report, cluster_artifacts)
        self._info_sampler.set_cluster_info(cluster_report, cluster_artifacts)
        self._coord_sampler.set_cluster_info(cluster_report, cluster_artifacts)

    def reset(self):
        self._round_count = 0
        self._info_sampler.reset()
        self._coord_sampler.reset()

    def select(
        self,
        candidates: list[str],
        tracker: ELOTracker,
        defender_name: str,
        n: int,
        **kwargs,
    ) -> list[str]:
        self._round_count += 1
        if self._round_count <= self.explore_rounds:
            self.last_sub_sampler = "infogain"
            return self._info_sampler.select(
                candidates, tracker, defender_name, n, **kwargs
            )
        self.last_sub_sampler = "coordinate"
        return self._coord_sampler.select(
            candidates, tracker, defender_name, n, **kwargs
        )


# ============================================================
# 工厂函数
# ============================================================
SAMPLER_REGISTRY = {
    "gap": GapMinSampler,
    "infogain": InfoGainSampler,
    "coordinate": CoordinateDescentSampler,
    "hybrid": HybridSampler,
}


def build_sampler(
    name: str,
    cluster_report: dict | None = None,
    cluster_artifacts: dict | None = None,
    **kwargs,
) -> AttackSampler:
    """
    根据名称构造采样器。

    参数:
        name: "gap" | "infogain" | "coordinate" | "hybrid"
        cluster_report: 聚类报告，供需要簇信息的采样器使用
        cluster_artifacts: 聚类 artifacts
        **kwargs: 采样器特定参数
    """
    name = name.lower()
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"未知采样器: {name}，可用: {list(SAMPLER_REGISTRY.keys())}")

    sampler = SAMPLER_REGISTRY[name](**kwargs)
    sampler.set_cluster_info(cluster_report, cluster_artifacts)
    return sampler
