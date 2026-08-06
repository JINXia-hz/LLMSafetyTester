"""
evaluation.blend_predictor — 统一预测 + 模型预测，自适应加权（贝叶斯收缩）

设计（对应"预测矩阵"管理）：
  预测器有两层，最终输出按该模型的样本量自适应加权：
    • 统一预测 P_u(method)：跨所有模型池化训练，捕获"方法内在威胁"
      （强越狱对多数模型都强），是冷启动时新模型唯一的先验来源。
    • 模型预测 P_m(method, model)：仅用该模型列训练，捕获模型特异性弱点。
    • 混合 pred = w_u·P_u + w_m·P_m，其中 w_m = n_model/(n_model + K)。
      样本少 → w_m→0，全靠统一（向群体均值收缩）；样本多 → w_m→1，信任自身。

数学本质 = 经验贝叶斯收缩：统一预测是"群体先验"，模型预测是"个体似然"，
K 是先验强度。这天然实现了"0.5+0.5 / 0.7+0.3"且无需手调——权重随证据增长。

数据来源（R 唯一真相）：
  • 训练目标 y：由 derive_elo(R, model) 从结果矩阵派生的每方法 Elo。
  • 特征 X：方法级先验特征（与防御方无关），由上层 extract_all_features 提供。
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from llmsec.core.config import INITIAL_ELO, PREDICTORS_DIR
from llmsec.core.io import load_artifact, save_artifact
from llmsec.core.logging import get_logger
from llmsec.core.results import ResultsMatrix
from llmsec.evaluation.elo import derive_elo
from llmsec.evaluation.elo_cluster import EloPredictorModel
from llmsec.params import BLEND_PRIOR_K, RIDGE_PRED_STD_CAP_MIN, RIDGE_PRED_STD_CAP_MULT

logger = get_logger(__name__)


class BlendPredictor:
    """统一 + 模型双层预测器，按样本量自适应混合。"""

    def __init__(self, prior_k: float = BLEND_PRIOR_K):
        self.prior_k = float(prior_k)
        # 发现层 D+A：unified 从"全局1个"改为"每 target 1个(sim-加权)"。
        # unified_fallback = 均匀 universal（首模型/无 donor 时用）；unified = {target: sim-加权}
        self.unified_fallback: EloPredictorModel | None = None
        self.unified: dict[str, EloPredictorModel] = {}
        self.models: dict[str, EloPredictorModel] = {}
        self._features: dict = {}             # method -> 特征向量
        self._catalog: list[str] = []         # 规范方法清单
        self._tested: dict[str, set[str]] = {}      # model -> 已测方法集
        self._measured_elo: dict[str, dict[str, float]] = {}  # model -> {method: elo}
        self._model_n: dict[str, int] = {}    # 每模型样本量
        self._gt_std: dict[str, float] = {}   # 每模型 GT Elo 的 std（std 封顶用）
        self._cov: dict[str, float] = {}      # 统一层/模型层 OOS 残差协方差（#3：blend 方差含交叉项）

    # ---------- 训练 ----------
    def fit(
        self,
        results: ResultsMatrix,
        features: dict,
        method_catalog: list[str] | None = None,
    ) -> BlendPredictor:
        """
        从 R + 方法特征训练统一/模型双层预测器。

        参数:
          results: 结果矩阵 R（唯一真相）
          features: {method: 特征向量}，方法级、防御方无关
          method_catalog: 全部方法清单（含未测，决定预测覆盖）
        """
        self._features = dict(features)   # {method: {block: vector}}，保持块字典结构
        self._catalog = list(method_catalog) if method_catalog else list(features.keys())
        models = results.all_models()

        # 1) 每模型派生 Elo（R → Elo），收集团测值与样本量
        per_model_elo: dict[str, dict[str, float]] = {}
        for model in models:
            tracker = derive_elo(results, model, method_catalog=self._catalog)
            elo_map = {m: float(tracker.get_attacker_elo(m)) for m in tracker.ground_truth_methods}
            per_model_elo[model] = elo_map
            self._tested[model] = set(elo_map.keys())
            self._measured_elo[model] = elo_map
            self._model_n[model] = len(elo_map)
            if elo_map:
                self._gt_std[model] = float(np.std(list(elo_map.values())))

        # 2) 统一预测器：池化所有 (method, model) 为独立训练样本
        # #4：池化样本里同一方法跨模型重复出现，按方法分组做 GroupKFold，
        # 否则随机 K-Fold 会让同方法同时进训练/测试折 → CV 泄漏、σ² 系统性乐观
        pooled_gt: dict[str, dict] = {}
        pooled_groups: dict[str, str] = {}  # 合成 key → 方法名（GroupKFold 分组键）
        pooled_donor: dict[str, str] = {}   # 合成 key → donor 模型名（发现层 sim-加权用）
        pooled_feat: dict = dict(self._features)  # 真实 method key 保留（供预测用）
        for model, elo_map in per_model_elo.items():
            for m, elo in elo_map.items():
                if m not in self._features:
                    continue
                key = f"{m}#{model}"
                pooled_gt[key] = {"elo": elo}
                pooled_feat[key] = self._features[m]  # 合成 key → 同一方法特征
                pooled_groups[key] = m
                pooled_donor[key] = model

        # 2a) 均匀 universal（fallback：首模型/无 donor 时用，与历史行为一致）
        # 模型数≥2 才池化——单模型时 pooled = 该模型自身数据，训练 unified_fallback 等同
        # models[target]（冗余双训练，不同 fold 策略还会产出不同 λ*），故跳过
        self.unified_fallback = None
        if len(per_model_elo) >= 2 and len(pooled_gt) >= 2:
            try:
                self.unified_fallback = EloPredictorModel()
                self.unified_fallback.fit(pooled_feat, pooled_gt, groups=pooled_groups)
            except Exception as e:
                logger.warning("均匀 universal fit 失败（blend 退化为仅模型层）: %s", e)

        # 2b) 发现层 D+A：每个有指纹的 target 训练 sim-加权 unified
        # target 自身样本权重=1，有指纹 donor=sim(target,donor)，无指纹 donor=0(排除)；
        # 无相似 donor 的 target 不建 → predict 时回退 unified_fallback
        self.unified = {}
        if self.unified_fallback is not None:
            try:
                from llmsec.evaluation.model_fingerprint import donor_similarities, load_probes

                probes = load_probes()
            except Exception as e:
                # B-4：probes 加载失败 → 全部 target 静默退化为均匀 fallback，核心 sim-加权能力消失。
                # 原实现无 warning，排查困难。补 warning 让降级可见。
                logger.warning("probes 加载失败，Blend 退化为均匀 universal（无 sim-加权）: %s", e)
                probes = {}
            for target in models:
                sims = donor_similarities(target, probes) if probes else {}
                if not sims:
                    continue  # 首模型/无相似 donor → 用 fallback
                sw = {
                    key: (1.0 if pooled_donor[key] == target
                          else sims.get(pooled_donor[key], 0.0))
                    for key in pooled_gt
                }
                if sum(1 for v in sw.values() if v > 0) < 2:
                    continue  # 有效样本不足 → 用 fallback
                try:
                    pm = EloPredictorModel()
                    pm.fit(pooled_feat, pooled_gt, groups=pooled_groups, sample_weights=sw)
                    self.unified[target] = pm
                except Exception as e:
                    logger.warning("target %s sim-加权 unified fit 失败（用 fallback）: %s", target, e)

        # 3) 每模型预测器
        for model, elo_map in per_model_elo.items():
            if len(elo_map) < 2:
                continue
            gt = {m: {"elo": e} for m, e in elo_map.items() if m in self._features}
            if len(gt) < 2:
                continue
            try:
                pm = EloPredictorModel()
                pm.fit(self._features, gt)
                self.models[model] = pm
            except Exception as e:
                logger.warning("模型 %s 预测器 fit 失败（跳过）: %s", model, e)

        # 3) 层间协方差（#3）：统一层与模型层共用方法特征 → 预测残差正相关，
        # blend 方差须含交叉项 2·w_u·w_m·cov，否则 CI 系统性偏窄、过度自信。
        # 用各自 λ* 的 OOS 残差估计；统一层取 target 的 sim-加权 unified（无则 fallback）。
        self._cov = {}
        for model, pm in self.models.items():
            u_src = self.unified.get(model) or self.unified_fallback
            u_oos = getattr(u_src, "oos_by_key_", None) if u_src is not None else None
            m_oos = getattr(pm, "oos_by_key_", {})
            if not u_oos or not m_oos:
                continue
            elo_map = per_model_elo.get(model, {})
            # 取两层均有 OOS 预测的方法（统一层 key = f"{m}#{model}"）
            common = [
                m for m in elo_map
                if f"{m}#{model}" in u_oos and m in m_oos
            ]
            if len(common) < 3:
                continue
            y_true = np.array([elo_map[m] for m in common], dtype=np.float64)
            r_u = y_true - np.array([u_oos[f"{m}#{model}"] for m in common])
            r_m = y_true - np.array([m_oos[m] for m in common])
            cov = float(np.mean(r_u * r_m) - np.mean(r_u) * np.mean(r_m))
            self._cov[model] = cov
        return self

    # ---------- 权重 ----------
    def blend_weights(self, model: str) -> tuple[float, float]:
        """返回 (w_model, w_unified)，按该模型样本量自适应。"""
        n = self._model_n.get(model, 0)
        w_m = n / (n + self.prior_k) if (n + self.prior_k) > 0 else 0.0
        return w_m, 1.0 - w_m

    # ---------- 预测 ----------
    def _std_cap(self, model: str) -> float:
        y_std = self._gt_std.get(model, 0.0) or 0.0
        return max(RIDGE_PRED_STD_CAP_MULT * y_std, RIDGE_PRED_STD_CAP_MIN)

    def predict(self, method: str, model: str) -> dict:
        """
        预测某方法在某模型上的 Elo。

        返回: {elo, std, w_model, w_unified, source}
          source:
            ground_truth   — 已实测，直接返回派生 Elo（std=0）
            blend          — 统一+模型混合
            unified_only   — 该模型无独立预测器，仅用统一
            model_only     — 无统一预测器，仅用模型
            fallback       — 两层都无，返回初始 Elo + 大 std
        """
        # 实测优先（R 是真相）
        if method in self._tested.get(model, set()):
            elo = self._measured_elo[model][method]
            return {"elo": elo, "std": 0.0, "w_model": 1.0, "w_unified": 0.0, "source": "ground_truth"}

        feat = self._features.get(method)
        if feat is None:
            return {"elo": float(INITIAL_ELO), "std": self._std_cap(model),
                    "w_model": 0.0, "w_unified": 0.0, "source": "fallback"}

        w_m, w_u = self.blend_weights(model)
        # 发现层：优先用 target 的 sim-加权 unified（从相似 donor 借），无则均匀 fallback
        u_mean, u_var = self._predict_one(
            self.unified.get(model) or self.unified_fallback, method, feat
        )
        m_mean, m_var = (self._predict_one(self.models[model], method, feat)
                         if model in self.models else (None, None))

        cap = self._std_cap(model)

        if u_mean is not None and m_mean is not None:
            elo = w_u * u_mean + w_m * m_mean
            # #3：两层预测残差正相关，blend 方差须含交叉项 2·w_u·w_m·cov；
            # cov>0 时方差较原"层间独立"假设更大、CI 更诚实；max(var,0) 兜底防负
            cov = self._cov.get(model, 0.0)
            var = w_u * w_u * (u_var or 0.0) + w_m * w_m * (m_var or 0.0) + 2.0 * w_u * w_m * cov
            source = "blend"
        elif m_mean is not None:
            elo, var, source = m_mean, (m_var or 0.0), "model_only"
        elif u_mean is not None:
            elo, var, source = u_mean, (u_var or 0.0), "unified_only"
        else:
            return {"elo": float(INITIAL_ELO), "std": cap,
                    "w_model": w_m, "w_unified": w_u, "source": "fallback"}

        std = min(math.sqrt(max(var, 0.0)), cap)
        return {"elo": float(elo), "std": float(std), "w_model": w_m, "w_unified": w_u, "source": source}

    def _predict_one(self, model: EloPredictorModel | None, method: str, feat) -> tuple[float | None, float | None]:
        """单层预测，返回 (mean, variance)；模型缺失/失败返回 (None, None)。"""
        if model is None or model.w is None:
            return None, None
        try:
            means, variances = model.predict({method: feat}, [method])
            return float(means[0]), float(variances[0])
        except Exception as e:
            logger.warning("模型 %s predict 失败（返回 None）: %s", model, e)
            return None, None

    # ---------- 诊断 ----------
    def summary(self) -> dict:
        return {
            "unified_fallback_trained": self.unified_fallback is not None,
            "unified_sim_weighted_models": sorted(self.unified.keys()),
            "models_trained": sorted(self.models.keys()),
            "samples_per_model": dict(self._model_n),
            "weights_per_model": {m: {"w_model": round(self.blend_weights(m)[0], 3),
                                      "w_unified": round(self.blend_weights(m)[1], 3)}
                                  for m in self._model_n},
        }

    def diagnostics(self) -> dict:
        """看板"多模型层"诊断：summary() + 每 target 的 donor 相似度 + sim-加权/均匀的 λ。

        暴露发现层 D+A 的实际状态——哪些 target 启用了 sim-加权、从哪些 donor 借、相似度多少，
        以及 sim-加权 unified 与均匀 fallback 各自的正则化强度 λ。
        """
        base = self.summary()
        try:
            from llmsec.evaluation.model_fingerprint import donor_similarities, load_probes

            probes = load_probes()
        except Exception:
            probes = {}
        donor_sims = {t: donor_similarities(t, probes) for t in self.unified} if probes else {}
        per_target_lambda: dict[str, float | None] = {
            t: (float(pm.lambda_opt) if pm.lambda_opt is not None else None)
            for t, pm in self.unified.items()
        }
        if self.unified_fallback is not None and self.unified_fallback.lambda_opt is not None:
            per_target_lambda["_fallback_uniform"] = float(self.unified_fallback.lambda_opt)
        base.update({
            "donor_similarities": donor_sims,
            "per_target_lambda": per_target_lambda,
        })
        return base

    # ---------- 持久化（PREDICTORS_DIR 派生缓存）----------
    @staticmethod
    def _features_signature(features: dict | None) -> str:
        """features 的结构签名（每方法的块名 + 各块维度）。

        H-11 修复：用于检测 embedding 切换 / TF-IDF 降级 / 特征代码变更，
        避免仅凭 R+方法清单复用过期预测器（旧 features 被 pickle 进缓存，
        predict 用的是旧特征空间的模型）。
        """
        if not features:
            return "none"
        parts = []
        for m in sorted(features.keys()):
            blocks = features.get(m) or {}
            block_sig = ",".join(
                f"{k}:{len(v) if hasattr(v, '__len__') else 1}"
                for k, v in sorted(blocks.items())
            )
            parts.append(f"{m}={block_sig}")
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def cache_key(results: ResultsMatrix, method_catalog: list[str],
                  features: dict | None = None) -> str:
        """预测器依赖 (R 内容 + 方法清单 + features 结构 + 探针指纹) 的指纹——四者不变才可复用。

        发现层：sim-加权 unified 依赖各模型指纹（probes.json），指纹变 → 缓存失效。
        """
        # R 内容指纹：每模型列的 (method, score, ts)
        parts = []
        for model in sorted(results.all_models()):
            col = results.model_column(model)
            payload = ",".join(f"{m}:{r.eval_score}:{r.ts}" for m, r in sorted(col.items()))
            parts.append(f"{model}={payload}")
        r_fp = hashlib.md5(("|".join(parts)).encode("utf-8")).hexdigest()
        cat_fp = hashlib.md5(",".join(method_catalog).encode("utf-8")).hexdigest()
        feat_fp = BlendPredictor._features_signature(features)
        # 探针指纹签名（probes.json 的 {model: fingerprint} 内容）
        try:
            from llmsec.evaluation.model_fingerprint import load_probes

            probes = load_probes()
            probe_payload = "|".join(
                f"{mdl}:{','.join(f'{k}:{v}' for k, v in sorted((e or {}).get('fingerprint', {}).items()))}"
                for mdl, e in sorted(probes.items())
            )
            probe_fp = hashlib.md5(probe_payload.encode("utf-8")).hexdigest()[:8]
        except Exception:
            probe_fp = "noprobes"
        return f"blend_{r_fp[:12]}_{cat_fp[:8]}_{feat_fp[:8]}_{probe_fp}"

    def save(self, path) -> None:
        save_artifact(path, {
            "unified_fallback": self.unified_fallback,
            "unified": self.unified,
            "models": self.models,
            "_features": self._features,
            "_catalog": self._catalog,
            "_tested": self._tested,
            "_measured_elo": self._measured_elo,
            "_model_n": self._model_n,
            "_gt_std": self._gt_std,
            "_cov": self._cov,
            "prior_k": self.prior_k,
        })

    @classmethod
    def load(cls, path) -> BlendPredictor | None:
        data = load_artifact(path)
        if not data:
            return None
        bp = cls(prior_k=data.get("prior_k", BLEND_PRIOR_K))
        bp.unified_fallback = data.get("unified_fallback")
        bp.unified = data.get("unified", {})
        bp.models = data.get("models", {})
        bp._features = data.get("_features", {})
        bp._catalog = data.get("_catalog", [])
        bp._tested = data.get("_tested", {})
        bp._measured_elo = data.get("_measured_elo", {})
        bp._model_n = data.get("_model_n", {})
        bp._gt_std = data.get("_gt_std", {})
        bp._cov = data.get("_cov", {})
        return bp


def load_or_fit_blend_predictor(
    results: ResultsMatrix,
    features: dict,
    method_catalog: list[str] | None = None,
) -> BlendPredictor:
    """优先复用 PREDICTORS_DIR 中按 (R 指纹 + 方法清单) 缓存的预测器；
    未命中或加载失败则重新 fit 并缓存。R/方法清单不变时免去重复训练。"""
    catalog = list(method_catalog) if method_catalog else list(features.keys())
    key = BlendPredictor.cache_key(results, catalog, features)
    cache_path = PREDICTORS_DIR / f"{key}.pkl"

    cached = BlendPredictor.load(cache_path)
    if cached is not None:
        return cached

    bp = BlendPredictor().fit(results, features, method_catalog=catalog)
    try:
        bp.save(cache_path)
    except Exception:
        pass  # 缓存写失败不影响功能
    return bp
