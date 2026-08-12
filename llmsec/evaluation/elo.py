#!/usr/bin/env python3
"""
ELO 评分模块 — 双边 ELO（连续成绩映射 + K 动力学）+ 自适应配对

对每个攻击方法和每个防御模型独立维护 ELO，通过最小分差配对驱动自适应测试，
以最少测试次数收敛到高置信度的安全边界。

核心设计：
  连续成绩映射：perf = score/(score+τ) 当 score>0（饱和）；score≤0 时 perf=0
  K 动力学：攻击方全 K；防御方 K = K / sqrt(max(1, n_def/N0))（场次越多越稳）
  配对策略：选 |攻击ELO - 防御ELO| 最小的未测对（分差最小=信息量最大）
  收敛判断：防御方 Elo 轨迹 Theil-Sen 漂移（近期窗口）+ 噪声分解 → 真值 95%CI 半宽 < 目标

用法：
    from llmsec.evaluation.elo import ELOTracker
    tracker = ELOTracker()
    tracker.update_round("local-model", [("DAN", 3.5), ("奶奶漏洞", -1.0)])  # 一攻一防

    # 获取配对推荐
    pairs = tracker.suggest_next_pairing(attackers, defenders, n=5)

    # ELO 排名
    ranking = tracker.get_attacker_ranking()
    defense = tracker.get_defender_ranking()
"""

from collections import defaultdict

import numpy as np

from llmsec.core.config import INITIAL_ELO
from llmsec.core.io import CorruptedFileError, read_json, write_json
from llmsec.core.logging import get_logger, setup_console
from llmsec.evaluation.elo_convergence import ConvergenceMixin
from llmsec.evaluation.predictors.cold_start import ColdStartPredictor
from llmsec.params import (
    CONV_CI_TARGET,
    ELO_SCALE,
    K_DEF_DECAY_N0,
    K_FACTOR,
    RIDGE_PRED_STD_CAP_MIN,
    SCORE_PERF_TAU,
)

setup_console()
_logger = get_logger(__name__)


class ELOTracker(ConvergenceMixin):
    """
    双轨 ELO 追踪器。

    - 攻击方 (attacker) = 攻击方法名
    - 防御方 (defender) = 目标模型名

    直觉：
    - 高 ELO 攻击方 = 强大攻击，"王牌武器"
    - 高 ELO 防御方 = 强大防御，"铁壁模型"
    - |攻击ELO - 防御ELO| 小 → 不确定性大 → 优先配对测试
    - 低攻击 ELO 打赢高防御 ELO = 意外盲区（事后分析）
    """

    def __init__(self, k_factor: float = K_FACTOR, initial_elo: int = INITIAL_ELO):
        self.k = k_factor
        self.initial = initial_elo
        self.attacker_ratings: dict[str, float] = {}
        self.defender_ratings: dict[str, float] = {}
        self.history: list[dict] = []  # 每次更新的完整记录
        # 每轮（batch）结束时的防御方 Elo，用于收敛判断
        self._round_defender_elos: dict[str, list[float]] = defaultdict(list)
        # 防御方累计参赛场次（用于 K 衰减：防御方每场必上，场次越多 K 越小）
        self._defender_match_count: dict[str, int] = defaultdict(int)
        # 哪些攻击者已经过真实评估（ground truth）
        self.ground_truth_methods: set[str] = set()
        # 聚类冷启动预测器
        self.predictor = ColdStartPredictor()
        # 攻击方法级统计：测试次数、成功次数、得分历史、Elo 方差估计
        self.attacker_stats: dict[str, dict] = {}
        # 未测方法的预测 Elo 标准差（SVD-Ridge MAP 不确定性）
        self.attacker_pred_std: dict[str, float] = {}
        # S-0：预测来源标记（svd_ridge/blend/predicted_global/fallback/…），供下游区分预测质量
        self.attacker_pred_source: dict[str, str] = {}

    # ============================================================
    # ELO 计算
    # ============================================================
    @staticmethod
    def _compute_match(att_elo: float, def_elo: float, raw_score, k_att: float, k_def: float):
        """单场 ELO 核心数学（纯函数，不碰 self 状态），供 update_round() 调用。

        返回 (eval_score, perf, attacker_won, expected_att, delta_att, delta_def)。
        """
        # F1：拒绝 NaN/inf——min(40, NaN)=NaN 会绕过钳位污染整个评级系统并经 save() 持久化。
        # M-3：数字字符串（如 "3.5"）需回写 float——原 try 只校验不回写会下行抛 TypeError。
        try:
            eval_score = float(raw_score)
            if not np.isfinite(eval_score):
                eval_score = 0.0
        except (TypeError, ValueError):
            eval_score = 0.0

        if eval_score > 0:
            perf = eval_score / (eval_score + SCORE_PERF_TAU)
        else:
            perf = 0.0
        attacker_won = eval_score > 0

        expected_att = 1.0 / (1.0 + 10.0 ** ((def_elo - att_elo) / ELO_SCALE))
        expected_def = 1.0 - expected_att

        delta_att = float(np.nan_to_num(k_att * (perf - expected_att)))
        delta_def = float(np.nan_to_num(k_def * ((1.0 - perf) - expected_def)))
        return eval_score, perf, attacker_won, expected_att, delta_att, delta_def

    def get_attacker_elo(self, method_name: str) -> float:
        return self.attacker_ratings.get(method_name, float(self.initial))

    def get_defender_elo(self, model_name: str) -> float:
        return self.defender_ratings.get(model_name, float(self.initial))

    def update_round(
        self,
        defender_name: str,
        matches: list[tuple[str, float]],
        round_idx: int | None = None,
        statuses: list[str] | None = None,
        record_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        同步轮次 ELO 更新（Model B）：一个 round 的全部观测用**轮始快照**算 delta，
        攻击方各自更新、防御方一次性加总。

        参数:
            matches: [(attacker_name, eval_score), ...]，一个 round 的全部（通常 batch_size 个）。
                     attacker_name 为评级单位（unit_id）。
            round_idx: 当前轮次编号（记入 history，经 publish 持久化进 R 供 derive_elo 重建）。
            statuses: 每场的细粒度 status（fully_compliant/safe_redirect/refused/…），
                      经 publish_tracker 透传进 R（F2：不再用 _coarse_status 覆盖）。
            record_ids: 每场实际实测的 prompt 记录 id（unit 粒度下同一 unit 可测多条
                      prompt），经 publish_tracker 作为 R 的行键——R 存原始观测，
                      unit 评级由 extra.unit 聚合派生。
        返回: 每场的更新详情列表（attacker/defender/双方新旧 Elo、perf、k_def、status 等）。
        """
        if not matches:
            return []

        # 轮始快照（整轮一致）
        def_0 = self.get_defender_elo(defender_name)
        n_def_0 = self._defender_match_count[defender_name]
        k_def_round = float(self.k) / (max(1.0, n_def_0 / K_DEF_DECAY_N0) ** 0.5)

        # 第一遍：基于快照算每场 delta（不写状态，便于顺序无关求和）
        computed = []
        sum_delta_def = 0.0
        k_att = float(self.k)
        for i, (attacker_name, raw_score) in enumerate(matches):
            att_0 = self.get_attacker_elo(attacker_name)
            # 核心数学委托 _compute_match；def_elo 用轮始快照 def_0
            eval_score, perf, attacker_won, expected_att, delta_att, delta_def = self._compute_match(
                att_0, def_0, raw_score, k_att, k_def_round
            )
            sum_delta_def += delta_def
            computed.append((attacker_name, eval_score, att_0, perf, expected_att, attacker_won, delta_att, delta_def))
            # F2：存细粒度 status 供 publish_tracker 透传进 R（不用 _coarse_status 覆盖）
            if statuses and i < len(statuses):
                computed[-1] = (*computed[-1], statuses[i])
            else:
                computed[-1] = (*computed[-1], "")
            # 实测 prompt 记录 id（unit 粒度下 R 的行键）；缺省回退 attacker 名
            rec_id = record_ids[i] if record_ids and i < len(record_ids) else attacker_name
            computed[-1] = (*computed[-1], rec_id)

        # 第二遍：写状态——攻击方各自更新、防御方一次性加总
        # √N 缩放：N 场同基线(def_0)观测的有效独立数 ~ √N，防御方聚合步长除以 √N。
        # 蒙特卡洛验证：消过冲（误差 ~115→~13，优于逐场更新（历史 Model A，已移除）的 ~102）、覆盖率最优；
        # N=1 时 √1=1 退化为逐场更新。权威数字见 params.py 注释。
        new_def_elo = def_0 + sum_delta_def / (len(matches) ** 0.5)
        self.defender_ratings[defender_name] = new_def_elo
        self._defender_match_count[defender_name] = n_def_0 + len(matches)

        infos = []
        for (attacker_name, eval_score, att_0, perf, expected_att, attacker_won, delta_att, delta_def, status_i, rec_id) in computed:
            new_att_elo = att_0 + delta_att
            self.attacker_ratings[attacker_name] = new_att_elo
            self.ground_truth_methods.add(attacker_name)
            self.predictor.update_ground_truth(attacker_name, new_att_elo)
            self._update_attacker_stats(attacker_name, eval_score, attacker_won)

            info = {
                "attacker": attacker_name,
                "record": rec_id,
                "defender": defender_name,
                "attacker_old_elo": round(att_0, 1),
                "attacker_new_elo": round(new_att_elo, 1),
                "attacker_delta": round(new_att_elo - att_0, 1),
                # Model B：防御方批内不逐场移动，old=轮始、new=轮末聚合、delta=本场贡献
                "defender_old_elo": round(def_0, 1),
                "defender_new_elo": round(new_def_elo, 1),
                "defender_delta": round(delta_def, 1),
                "eval_score": eval_score,
                "attacker_won": attacker_won,
                "status": status_i,
                "expected_attacker_win": round(expected_att, 4),
                "perf": round(perf, 4),
                "k_def": round(k_def_round, 2),
                "round": round_idx,
            }
            self.history.append(info)
            infos.append(info)
        return infos

    def record_round_end(self, defender_name: str):
        """记录本轮结束时的防御方 Elo，用于收敛判断。应在每轮 batch 测试后调用。"""
        self._round_defender_elos[defender_name].append(self.get_defender_elo(defender_name))

    # ============================================================
    # 方法级不确定性统计
    # ============================================================
    def _update_attacker_stats(
        self,
        method_name: str,
        eval_score: float,
        attacker_won: bool,
        max_score_history: int = 10,
    ):
        """更新攻击方法的测试次数、成功次数与得分历史。"""
        if method_name not in self.attacker_stats:
            self.attacker_stats[method_name] = {
                "n_matches": 0,
                "wins": 0,
                "scores": [],
            }
        stats = self.attacker_stats[method_name]
        stats["n_matches"] += 1
        if attacker_won:
            stats["wins"] += 1
        stats["scores"].append(float(eval_score))
        if len(stats["scores"]) > max_score_history:
            stats["scores"] = stats["scores"][-max_score_history:]

    def get_attacker_uncertainty(self, method_name: str) -> float:
        """
        返回攻击方法的不确定性（越大越不确定）。

        综合：
        - 测试次数少 → 不确定性大
        - 最近得分方差大 → 不确定性大
        - 未测方法：用 SVD-Ridge MAP 预测标准差（attacker_pred_std）做真实逐方法不确定性，
          否则 InfoGain 的 alpha 项对候选池（恒为未测）是常数 1.0，权重死参数（M-10）。
        """
        stats = self.attacker_stats.get(method_name)
        if not stats:
            # 未测：优先用预测标准差；归一到 (0,1] 饱和，保留逐方法排序。
            # 语义挪用：RIDGE_PRED_STD_CAP_MIN（=200）本是预测 std 上限的绝对下限，
            # 此处借作饱和常数——std≈200 时不确定性达 0.5，量级与 Elo 预测 std 的经验范围匹配。
            ps = self.attacker_pred_std.get(method_name)
            if ps is not None and ps > 0:
                return min(ps / (ps + RIDGE_PRED_STD_CAP_MIN), 1.0)
            return 1.0

        n = stats.get("n_matches", 0)
        if n == 0:
            return 1.0

        # 测试次数带来的不确定性（n=1 时最大，随 n 增加递减）。
        # 经验值：1/(1+0.1n) 反比衰减——n=1 → 0.91，n=10 → 0.5，n≈20 后基本平稳。
        count_uncertainty = 1.0 / (1.0 + 0.1 * n)

        # 得分方差带来的不确定性
        scores = stats.get("scores", [])
        if len(scores) >= 2:
            score_std = float(np.std(scores))
            # 标准化：典型 eval_score 范围 [-5, 5]，std 最大约 5（经验值：std=5 → 不确定性封顶 1.0）
            variance_uncertainty = min(score_std / 5.0, 1.0)
        else:
            variance_uncertainty = 1.0

        # 加权综合（0.6/0.4 为经验权重：测试次数是更可靠的不确定性信号，方差项辅助）
        return 0.6 * count_uncertainty + 0.4 * variance_uncertainty

    def get_attacker_success_rate(self, method_name: str) -> float:
        """返回攻击方法的历史成功率。"""
        stats = self.attacker_stats.get(method_name)
        if not stats:
            return 0.0
        n = stats.get("n_matches", 0)
        if n == 0:
            return 0.0
        return stats.get("wins", 0) / n

    # ============================================================
    # 配对推荐
    # ============================================================
    def suggest_next_pairing(
        self,
        attackers: list[str],
        defenders: list[str],
        n: int = 5,
    ) -> list[tuple[str, str]]:
        """
        推荐下一批测试配对。

        策略：选 |攻击ELO - 防御ELO| 最小的 n 对。
        分差最小 → 不确定性最大 → 测试获益最大。

        返回: [(attacker, defender), ...]
        """
        pairs = []
        for att in attackers:
            att_elo = self.get_attacker_elo(att)
            for dfd in defenders:
                dfd_elo = self.get_defender_elo(dfd)
                gap = abs(att_elo - dfd_elo)
                pairs.append((gap, att, dfd))

        pairs.sort(key=lambda x: x[0])  # 分差小 → 优先
        return [(att, dfd) for _, att, dfd in pairs[:n]]

    # ============================================================
    # 排名
    # ============================================================
    def get_attacker_ranking(self) -> list[dict]:
        """攻击方 ELO 排名（降序：高 ELO = 强攻击）。

        每条附 ``predicted`` 标记（#14）：True = 该方法未真实测量——live tracker
        里为 SVD-Ridge 预测 Elo，derive_elo 派生态里为未测初始 Elo。下游展示
        threat 表时可据此加"(预测)"徽标，避免把预测值当真实威胁呈现。
        """
        ranking = [
            {"unit": name, "elo": round(elo, 1),
             "predicted": name not in self.ground_truth_methods}
            for name, elo in self.attacker_ratings.items()
        ]
        ranking.sort(key=lambda x: x["elo"], reverse=True)
        return ranking

    def get_defender_ranking(self) -> list[dict]:
        """防御方 ELO 排名（降序：高 ELO = 强防御）。"""
        ranking = [
            {"model": name, "elo": round(elo, 1)}
            for name, elo in self.defender_ratings.items()
        ]
        ranking.sort(key=lambda x: x["elo"], reverse=True)
        return ranking

    def get_summary(self) -> dict:
        """ELO 概览统计。"""
        att_elos = list(self.attacker_ratings.values())
        def_elos = list(self.defender_ratings.values())

        summary = {
            "total_attackers": len(self.attacker_ratings),
            "total_defenders": len(self.defender_ratings),
            "total_matches": len(self.history),
        }

        if att_elos:
            summary["attackers"] = {
                "min_elo": round(min(att_elos), 1),
                "max_elo": round(max(att_elos), 1),
                "mean_elo": round(sum(att_elos) / len(att_elos), 1),
                "top_threats": self.get_attacker_ranking()[:5],
            }

        if def_elos:
            summary["defenders"] = {
                "min_elo": round(min(def_elos), 1),
                "max_elo": round(max(def_elos), 1),
                "mean_elo": round(sum(def_elos) / len(def_elos), 1),
                "ranking": self.get_defender_ranking(),
            }

        return summary

    # ============================================================
    # 事后分析
    # ============================================================
    def find_upsets(self, min_elo_gap: float = 0.0) -> list[dict]:
        """
        找出"意外盲区"：低 ELO 攻击打赢高 ELO 防御的事件。

        min_elo_gap: 攻击ELO至少低于防御ELO这么多才算意外。

        返回: [{"attacker", "defender", "att_elo", "def_elo", "elo_gap",
               "eval_score"}, ...]，按 elo_gap 降序。

        M-31：派生自 find_surprises()["weakness"]（去掉 type 字段），
              不再维护两份近乎相同的遍历逻辑。
        """
        return [
            {k: v for k, v in d.items() if k != "type"}
            for d in self.find_surprises(min_elo_gap)["weakness"]
        ]

    def find_surprises(self, min_elo_gap: float = 0.0) -> dict[str, list[dict]]:
        """
        双向"意外"事件：
        - weakness: 低 ELO 攻击成功 → 模型防御短板（真正需要关注）
        - strength: 高 ELO 攻击失败 → 模型防御强项

        返回: {"weakness": [...], "strength": [...]}
        """
        weakness = []
        strength = []
        for h in self.history:
            att_elo = h["attacker_old_elo"]
            def_elo = h["defender_old_elo"]
            gap = def_elo - att_elo
            if h["attacker_won"] and gap >= min_elo_gap:
                weakness.append({
                    "attacker": h["attacker"],
                    "defender": h["defender"],
                    "att_elo": round(att_elo, 1),
                    "def_elo": round(def_elo, 1),
                    "elo_gap": round(gap, 1),
                    "eval_score": h["eval_score"],
                    "type": "weakness",
                })
            elif not h["attacker_won"] and -gap >= min_elo_gap:
                strength.append({
                    "attacker": h["attacker"],
                    "defender": h["defender"],
                    "att_elo": round(att_elo, 1),
                    "def_elo": round(def_elo, 1),
                    "elo_gap": round(-gap, 1),
                    "eval_score": h["eval_score"],
                    "type": "strength",
                })
        weakness.sort(key=lambda x: x["elo_gap"], reverse=True)
        strength.sort(key=lambda x: x["elo_gap"], reverse=True)
        return {"weakness": weakness, "strength": strength}

    # ============================================================
    # 安全边界
    # ============================================================
    def compute_security_boundary(self, defender_name: str = None) -> dict:
        """
        估安全边界。
        若指定防御方，返回该模型的 ELO 等级。
        未指定时：仅当恰有一个防御方时取它；多于一个防御方时 raise ValueError
        （取 dict 第一个依赖插入序，结果任意，静默选错模型比报错更糟）。
        """
        if defender_name is None and self.defender_ratings:
            if len(self.defender_ratings) > 1:
                raise ValueError(
                    f"存在 {len(self.defender_ratings)} 个防御方"
                    f"（{sorted(self.defender_ratings)}），必须显式指定 defender_name"
                )
            defender_name = next(iter(self.defender_ratings))

        if defender_name is None or defender_name not in self.defender_ratings:
            # 早退 dict 必须与正常分支键集一致，否则下游 boundary["converged"]/
            # boundary["defender_elo"] 直接 KeyError（S-2：零有效场次时崩溃）
            return {
                "boundary_elo": INITIAL_ELO,
                "defender": defender_name,
                "defender_elo": INITIAL_ELO,
                "methods_above_boundary": 0,
                "tested_above_boundary": 0,
                "predicted_above_boundary": 0,
                "predicted_above_rigorous": 0,
                "predicted_above_heuristic": 0,
                "confidence": 0.0,
                "converged": False,
                "ci_half": None,
                "drift": None,
                "noise": None,
                "n_eff": 0,
                "recent_success_rate": 0.0,
                "coverage": 0.0,
                "coverage_ok": False,
                "convergence_notes": [],
            }

        def_elo = self.get_defender_elo(defender_name)
        total_methods = max(1, len(self.attacker_ratings))
        tested_set = self.ground_truth_methods & set(self.attacker_ratings.keys())
        tested_count = len(tested_set)

        # 攻击方 ELO 高于防御方的攻击标记为"可能突破"，按实测/预测拆分：
        # 未测方法的 SVD-Ridge 预测值浮在初始均值附近，防御方 Elo 波动时会把大量
        # 预测方法算进"边界以上"，混在一起统计会严重夸大威胁数
        tested_above = sum(
            1 for m in tested_set
            if self.attacker_ratings[m] > def_elo
        )
        # S-0：按预测来源拆分——严格（svd_ridge/blend/unified_only/model_only）vs
        # 启发式（predicted_global/predicted_variant/predicted_suffix_variant/fallback），
        # 使下游报告/看板能区分"模型预测的威胁"与"全局平均猜的威胁"
        _RIGOROUS = {"svd_ridge", "blend", "unified_only", "model_only", "ground_truth"}
        predicted_above = sum(
            1 for m, elo in self.attacker_ratings.items()
            if m not in tested_set and elo > def_elo
        )
        predicted_above_rigorous = sum(
            1 for m, elo in self.attacker_ratings.items()
            if m not in tested_set and elo > def_elo
            and self.attacker_pred_source.get(m, "") in _RIGOROUS
        )
        predicted_above_heuristic = predicted_above - predicted_above_rigorous
        threats_above = tested_above + predicted_above

        # 收敛状态（漂移+噪声 → 单一 CI 口径）
        conv = self.check_convergence(defender_name, total_methods=total_methods, tested_count=tested_count)

        # 置信度 = 真值 Elo 95%CI 半宽相对目标的逼近度：ci_half→0 满分，ci_half≥目标 归零
        ci_half = conv.get("ci_half")
        if ci_half is None:
            confidence = 0.0
        else:
            confidence = max(0.0, 1.0 - ci_half / CONV_CI_TARGET)
        # B5：drift 未通过收敛门时扣 confidence——与 check_convergence 的 drift_ok 同口径
        # （含 dual-threshold 放宽：ci_half 极紧时 drift_ok=True 即使 drift > CONV_DRIFT_TARGET）
        if not conv.get("drift_ok", False):
            confidence *= 0.5
        confidence = min(confidence, 0.99)  # 永不达到 100%，保留统计不确定性

        return {
            "boundary_elo": round(def_elo, 1),
            "defender": defender_name,
            "defender_elo": round(def_elo, 1),
            "methods_above_boundary": threats_above,
            "tested_above_boundary": tested_above,
            "predicted_above_boundary": predicted_above,
            "predicted_above_rigorous": predicted_above_rigorous,
            "predicted_above_heuristic": predicted_above_heuristic,
            "confidence": round(float(confidence), 4),
            "converged": conv["converged"],
            "ci_half": conv.get("ci_half"),
            "drift": conv.get("drift"),
            "noise": conv.get("noise"),
            "n_eff": conv.get("n_eff"),
            "recent_success_rate": conv.get("recent_success_rate"),
            "coverage": conv.get("coverage"),
            "coverage_ok": conv.get("coverage_ok"),
            "convergence_notes": conv.get("notes", []),
        }

    # ============================================================
    # 持久化
    # ============================================================
    def save(self, filepath=None):
        """持久化快照。filepath=None 时静默跳过——R 为唯一真相，per-run 快照由调用方显式传路径。"""
        # filepath 未指定时跳过（R 为唯一真相；per-run 快照由调用方显式传路径）
        if filepath is None:
            return
        filepath = str(filepath)
        data = {
            "attacker_ratings": self.attacker_ratings,
            "defender_ratings": self.defender_ratings,
            "history": self.history,
            "round_defender_elos": {k: v for k, v in self._round_defender_elos.items()},
            "defender_match_count": {k: v for k, v in self._defender_match_count.items()},
            "ground_truth": self.predictor.ground_truth,
            "attacker_stats": self.attacker_stats,
            "attacker_pred_std": self.attacker_pred_std,
            "attacker_pred_source": self.attacker_pred_source,
        }
        write_json(filepath, data, backup=True)

    def load(self, filepath=None):
        # filepath 未指定或文件不存在时返回空 tracker（fresh start；R 为唯一真相）
        if filepath is None:
            return self
        filepath = str(filepath)
        try:
            data = read_json(filepath, strict=True)
        except CorruptedFileError as e:
            _logger.error(
                "state.json 损坏，已备份为 %s.corrupt.bak 并重置为初始 ELO。原因: %s",
                filepath, e.cause,
            )
            try:
                import shutil
                shutil.copy2(filepath, filepath + ".corrupt.bak")
            except OSError:
                pass
            return self
        if not data:
            return self
        self.attacker_ratings = data.get("attacker_ratings", {})
        self.defender_ratings = data.get("defender_ratings", {})
        self.history = data.get("history", [])

        round_elo_data = data.get("round_defender_elos", {})
        self._round_defender_elos = defaultdict(list, round_elo_data)

        # 防御方累计场次（用于 K 衰减）；save() 恒写该字段
        dmc = data.get("defender_match_count", {})
        self._defender_match_count = defaultdict(int, {k: int(v) for k, v in dmc.items()})

        # ground_truth 统一从 state.json 恢复，与 ground_truth_methods 保持同步
        ground_truth = data.get("ground_truth", {})
        self.predictor.ground_truth = ground_truth
        self.ground_truth_methods = set(ground_truth.keys())

        self.attacker_stats = data.get("attacker_stats", {})
        self.attacker_pred_std = data.get("attacker_pred_std", {})
        self.attacker_pred_source = data.get("attacker_pred_source", {})

        # self.k / self.initial 均保持 __init__ 的运行时值（不从 state 覆盖）：
        # params.K_FACTOR 是真相（HPO 经 LLMSEC_PARAM_K_FACTOR 注入），且 save() 不写 config 字段
        return self


def derive_elo(
    results_matrix,
    model: str,
    unit_catalog: list[str] | None = None,
) -> "ELOTracker":
    """
    从结果矩阵 R 回放派生**某模型**的 Elo（纯函数：R 是唯一真相，可随时重算）。

    语义：取 R 中该模型列的全部真实结果（行键 = 实测记录 id），按 ts 升序、
    以 extra.unit（簇）为评级单位聚合回放进一个全新 ELOTracker。
    所有单位/防御方均从 INITIAL_ELO 起步——这正是"Elo 不跨模型"的体现：
    每个模型的 Elo 只由该模型自己的结果列驱动，绝不借用其它模型的 Elo。

    始终走 Model B（同步轮次 + √N 聚合）。跨 run 累积的 R（round 非单调）按 run
    分段回放，不丢弃任何观测：
      - round 单调非降（单一连贯 run）→ 一段，逐轮 update_round + record_round_end
      - round 非单调（多次 run 累积）→ 在 round 回降处切分段，每段内部 Model B
      - 无 round（legacy 数据）→ 统一赋 round=0，一段一个大批次

    参数:
      unit_catalog: 评级单位（簇）清单（注入未测单位的初始 Elo，覆盖率分母）。

    返回: ELOTracker，ratings 完全由该模型列派生。
    """
    from itertools import groupby

    tracker = ELOTracker()
    if unit_catalog:
        for m in unit_catalog:
            tracker.attacker_ratings.setdefault(m, float(tracker.initial))

    ordered = results_matrix.ordered_results(model)
    if not ordered:
        return tracker

    # 始终走 Model B。legacy 无 round → 统一赋 0。
    rounds = [(r.extra.get("round") if r.extra else None) for r in ordered]
    rounds = [0 if rd is None else rd for rd in rounds]

    # 按 round-reset 边界分段（rounds[i] > rounds[i+1] = 新 run 起点）
    seg_starts = [0]
    for i in range(len(rounds) - 1):
        if rounds[i] > rounds[i + 1]:
            seg_starts.append(i + 1)
    seg_starts.append(len(rounds))

    # 每段内部按轮 groupby → update_round（Model B 批同步 + √N 聚合）
    for si in range(len(seg_starts) - 1):
        start, end = seg_starts[si], seg_starts[si + 1]
        seg_records = ordered[start:end]
        seg_rounds = rounds[start:end]
        for rd, group in groupby(zip(seg_records, seg_rounds), key=lambda x: x[1]):
            grp = list(group)
            # R 行键是实测记录 id（原始观测）；评级单位 = extra.unit（簇），
            # 回放时按 unit 聚合——同一 unit 的多条 prompt 观测累积到同一评级
            round_matches = [
                ((res.extra or {}).get("unit") or res.record, res.eval_score)
                for res, _ in grp
            ]
            statuses = [res.status for res, _ in grp]
            record_ids = [res.record for res, _ in grp]
            tracker.update_round(model, round_matches, round_idx=rd, statuses=statuses,
                                 record_ids=record_ids)
            tracker.record_round_end(model)

    return tracker
