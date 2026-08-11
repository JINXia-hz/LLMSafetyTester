"""
evaluation.elo_convergence — ELOTracker 的收敛判断逻辑（Mixin）。

从 elo.py 拆出（M-44）：收敛检测是独立关注点——漂移/噪声分解、Theil-Sen 斜率、
AR(1) 自相关校正、预测区间 95%CI。与 ELOTracker 的"评级追踪"核心分离后，
elo.py 聚焦于状态与更新，本模块聚焦于"何时停"。

作为 Mixin 被 ELOTracker 继承，访问 self._round_defender_elos / attacker_ratings /
ground_truth_methods / history。API 不变（tracker.check_convergence() 照常工作）。
"""

import numpy as np
from scipy.stats import t as _t_dist
from scipy.stats import theilslopes

from llmsec.params import (
    CONV_CI_TARGET,
    CONV_DRIFT_TARGET,
    CONV_WINDOW_MIN,
    MIN_COVERAGE_ABSOLUTE,
    MIN_COVERAGE_RATIO,
)


class ConvergenceMixin:
    """收敛判断 Mixin：由 ELOTracker 继承，方法访问 host 的状态属性。"""

    # ============================================================
    # 收敛判断
    # ============================================================
    def _recent_success_rate(self, window_methods: int = 15) -> float:
        """返回最近 window_methods 个被测方法（按方法去重，各取最近一次结果）的成功率。

        history 条目是场次而非方法——重复测试的方法若不去重会被重复计入，
        窗口实际覆盖的方法数随重测缩水，故按 attacker 去重后再数 window_methods 个。
        """
        recent_wins = 0
        seen: set[str] = set()
        for h in reversed(self.history):
            if len(seen) >= window_methods:
                break
            attacker = h["attacker"]
            if attacker in seen:
                continue
            seen.add(attacker)
            if h["attacker_won"]:
                recent_wins += 1
        if not seen:
            return 0.0
        return recent_wins / len(seen)

    @staticmethod
    def _trajectory_stats(elos: list[float]) -> dict | None:
        """
        对防御方 Elo 轨迹做漂移+噪声分解，合成真值 Elo 的 95%CI 半宽。

        - drift = OLS 斜率（Elo 分/轮）：朝真值移动的系统性趋势（好事）。
        - noise = 去趋势残差的标准差：随机抖动。
        - 自相关校正有效样本量 k_eff = m·(1−ρ)/(1+ρ)（Bartlett），ρ=残差 lag-1 自相关。
          k_eff 仅作为诊断量返回（量化轨迹的有效信息量）。
        - ci_half = t₀.₉₇₅(m−2) · noise · √(1 + 1/m + (m−t̄)²/S_tt)：
          防御方真值 Elo **当前水平**的 95%CI 半宽（预测区间口径）。√(...) 含不可约噪声(1)、
          截距不确定性(1/m)、端点杠杆((m−t̄)²/S_tt)；端点外推不确定性最大。

        口径说明（S-1 + H3/H4 修正）：边界点估计用的是**最后一次观测** current_elo，其
          围绕真值的误差由预测区间刻画。旧的 1.96·noise/√k_eff 是"轨迹均值"的标准误，对
          "当前水平"过窄（覆盖率 ~0.46）；S-1 改用 t·noise 但漏杠杆项仍偏窄；
          H3/H4 补预测区间杠杆 √(1+1/m+lev_end)，蒙特卡洛经验覆盖率回到 ~0.95。

        小样本收尾（S-1）：noise 用 ddof=2（OLS 拟合截距+斜率损失 2 个自由度，
        m≤2 时无法估计 → noise/ci_half 为 None）；分位数用 t(m−2) 而非 1.96——
        m=8 时 t₀.₉₇₅(6)≈2.447，AR(1) ρ=0 蒙特卡洛经验覆盖率由 ~0.907 回到 ~0.95。

        返回 None 表示样本不足以拟合（m<2）。
        """
        m = len(elos)
        if m < 2:
            return None
        t = np.arange(1, m + 1, dtype=float)
        y = np.asarray(elos, dtype=float)
        t_mean = t.mean()
        y_mean = y.mean()
        s_tt = float(np.sum((t - t_mean) ** 2))
        if s_tt == 0:
            slope, intercept = 0.0, float(y_mean)
        else:
            slope = float(np.sum((t - t_mean) * (y - y_mean)) / s_tt)
            intercept = float(y_mean - slope * t_mean)
        resid = y - (intercept + slope * t)
        # ddof=2：OLS 残差损失 2 个自由度（截距+斜率），m<=2 时无法估计 → None
        noise = float(np.std(resid, ddof=2)) if m >= 3 else None

        # lag-1 自相关（需足够残差点，否则保守取 ρ=0）
        rho = 0.0
        if m >= 4 and noise and noise > 0:
            a = resid[:-1]
            b = resid[1:]
            if np.std(a) > 0 and np.std(b) > 0:
                rho = float(np.corrcoef(a, b)[0, 1])
                if not np.isfinite(rho):
                    rho = 0.0
        denom = 1.0 + rho
        k_eff = m * (1.0 - rho) / denom if denom > 0 else float(m)
        k_eff = max(1.0, min(float(m), k_eff))

        if noise is None or not np.isfinite(noise):
            # 不可估计时返回 None（而非 inf）——inf 会经 runner_report.json 落成非法
            # JSON 字面量 `Infinity`，下游浏览器 JSON.parse 报错（M-1）
            ci_half = None
        else:
            # S-1 + H3/H4：当前水平 95%CI 半宽 = t₀.₉₇₅(m−2)·noise·√(1 + 1/m + lev_end)
            # 补预测区间杠杆项 lev_end = (m−t̄)²/S_tt：端点 t=m 处的 OLS 外推不确定性最大，
            # 原实现裸 t·noise 漏掉此项 → CI 系统性偏窄 → 收敛提前、置信度虚高。
            # `1` = 新观测的不可约噪声；`1/m` = 截距估计不确定性；`lev_end` = 斜率在端点的外推。
            lev_end = float((m - t_mean) ** 2 / s_tt) if s_tt > 0 else 0.0
            pi_factor = float(np.sqrt(1.0 + 1.0 / m + lev_end))
            ci_half = float(_t_dist.ppf(0.975, m - 2)) * noise * pi_factor

        return {
            "slope": slope,
            "intercept": intercept,
            "noise": noise,
            "rho": rho,
            "k_eff": k_eff,
            "ci_half": ci_half,
        }

    def check_convergence(
        self,
        defender_name: str,
        total_methods: int | None = None,
        tested_count: int | None = None,
    ) -> dict:
        """
        检查指定防御方是否收敛（漂移+噪声 → 单一 CI 口径）。

        核心思想：把"漂移"（Elo 朝真值移动，是好事）与"噪声"（随机抖动）分开。
          - drift = 最近 CONV_WINDOW_MIN 轮的斜率（当前是否还在移动）
          - noise / ci_half = 近期窗口（砍掉前 CONV_WINDOW_MIN 轮早期瞬态）去趋势残差
            → 真值 Elo 95%CI 半宽
        收敛当且仅当：
            ci_half < CONV_CI_TARGET   （水平估计足够精确）
          ∧ |drift| < CONV_DRIFT_TARGET（已不再系统性移动）
          ∧ 覆盖率达标 ∧ 轮次 ≥ CONV_WINDOW_MIN

        返回字段:
            converged, ci_half, drift, noise, n_eff, current_elo, n_rounds,
            recent_success_rate, coverage, coverage_ok, ci_ok, drift_ok, notes
        """
        round_elos = self._round_defender_elos.get(defender_name, [])
        current_elo = self.get_defender_elo(defender_name)

        if total_methods is None:
            total_methods = max(1, len(self.attacker_ratings))
        if tested_count is None:
            tested_count = len(self.ground_truth_methods)
        coverage = tested_count / total_methods
        coverage_ok = coverage >= MIN_COVERAGE_RATIO or tested_count >= MIN_COVERAGE_ABSOLUTE

        recent_success_rate = self._recent_success_rate()
        notes: list[str] = []
        n_rounds = len(round_elos)

        if n_rounds == 0:
            notes.append("尚无完整轮次")
            return {
                "converged": False,
                "ci_half": None,
                "drift": None,
                "noise": None,
                "n_eff": 0,
                "current_elo": round(current_elo, 1),
                "n_rounds": 0,
                "recent_success_rate": round(recent_success_rate, 4),
                "coverage": round(coverage, 4),
                "coverage_ok": coverage_ok,
                "ci_ok": False,
                "drift_ok": False,
                "notes": notes,
            }

        # B1 修复：noise/ci_half 改用近期窗口（砍掉早期瞬态），消除"早期从 INITIAL_ELO
        # 快速移动"的残差永久 inflate noise → ci_half 难降到目标 → 收敛过度延迟。
        # 原实现用全轨迹，注释称"更多数据→更稳"；但早期瞬态非噪声而是信号，混入会偏乐观方向
        # 的反面（偏保守）。砍掉前 CONV_WINDOW_MIN 轮后，noise 反映当前稳态抖动。
        if n_rounds > CONV_WINDOW_MIN * 2:
            noise_elos = round_elos[CONV_WINDOW_MIN:]
        else:
            noise_elos = round_elos
        stats = self._trajectory_stats(noise_elos)
        noise = stats["noise"] if stats else None
        ci_half = stats["ci_half"] if stats else None
        n_eff = stats["k_eff"] if stats else 0

        # drift 取"近期窗口"斜率（最近 CONV_WINDOW_MIN 轮），反映当前是否仍在移动；
        # 全窗口斜率会被早期快速上升永久拖高，导致已平稳的轨迹误判为仍在漂移。
        recent_n = min(n_rounds, CONV_WINDOW_MIN)
        recent = round_elos[-recent_n:] if recent_n >= 2 else round_elos
        if len(recent) >= 2:
            # #9：近期窗口小(≤CONV_WINDOW_MIN)，OLS 斜率对单点离群敏感（翻转 drift 符号
            # → 误判"仍在漂移/已稳"）。改 Theil-Sen（两两斜率中位）：4 点仅 6 个斜率取中位，
            # 成本可忽略；n=2 时退化为唯一两两斜率（与 OLS 等价，向后兼容）
            tr = np.arange(1, len(recent) + 1, dtype=float)
            yr = np.asarray(recent, dtype=float)
            drift = float(theilslopes(yr, tr)[0])
        else:
            drift = stats["slope"] if stats else None

        rounds_sufficient = n_rounds >= CONV_WINDOW_MIN
        if not rounds_sufficient:
            notes.append(f"轮次不足({n_rounds}/{CONV_WINDOW_MIN})，暂不判收敛")

        # 即便轮次不足也计算指标供展示，但 converged 必为 False
        ci_ok = ci_half is not None and ci_half < CONV_CI_TARGET
        # CI 极紧时放宽 drift 门槛：ci_half < target/2 → drift 容忍按 (target/2)/ci_half 放大
        # （漂移比测量精度小时无实质影响——"还在动"但"动多少我们算得清"）。上限 5× 防过度放宽。
        if ci_half is not None and 0 < ci_half < CONV_CI_TARGET / 2:
            _relax = min(5.0, (CONV_CI_TARGET / 2) / ci_half)
            _drift_target = CONV_DRIFT_TARGET * _relax
        else:
            _drift_target = CONV_DRIFT_TARGET
        drift_ok = drift is not None and abs(drift) < _drift_target

        if rounds_sufficient and not ci_ok:
            notes.append(f"真值 Elo 95%CI ±{ci_half:.1f} >= 目标 ±{CONV_CI_TARGET:.0f}")
        if rounds_sufficient and not drift_ok:
            _extra = f"（CI 极紧已放宽至 ±{_drift_target:.1f}/轮）" if _drift_target > CONV_DRIFT_TARGET else ""
            notes.append(f"仍在漂移 {drift:+.1f}/轮 >= ±{_drift_target:.1f}/轮{_extra}")
        if not coverage_ok:
            notes.append(f"覆盖率 {coverage:.1%} 不足")

        converged = rounds_sufficient and ci_ok and drift_ok and coverage_ok

        return {
            "converged": converged,
            "ci_half": None if ci_half is None else round(ci_half, 2),
            "drift": None if drift is None else round(drift, 4),
            "noise": None if noise is None else round(noise, 2),
            "n_eff": round(n_eff, 2),
            "current_elo": round(current_elo, 1),
            "n_rounds": n_rounds,
            "recent_success_rate": round(recent_success_rate, 4),
            "coverage": round(coverage, 4),
            "coverage_ok": coverage_ok,
            "ci_ok": ci_ok,
            "drift_ok": drift_ok,
            "notes": notes,
        }
