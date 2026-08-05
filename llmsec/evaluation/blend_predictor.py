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
        self.unified: EloPredictorModel | None = None
        self.models: dict[str, EloPredictorModel] = {}
        self._features: dict = {}             # method -> 特征向量
        self._catalog: list[str] = []         # 规范方法清单
        self._tested: dict[str, set[str]] = {}      # model -> 已测方法集
        self._measured_elo: dict[str, dict[str, float]] = {}  # model -> {method: elo}
        self._model_n: dict[str, int] = {}    # 每模型样本量
        self._gt_std: dict[str, float] = {}   # 每模型 GT Elo 的 std（std 封顶用）

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
        pooled_gt: dict[str, dict] = {}
        pooled_feat: dict = dict(self._features)  # 真实 method key 保留（供预测用）
        for model, elo_map in per_model_elo.items():
            for m, elo in elo_map.items():
                if m not in self._features:
                    continue
                key = f"{m}#{model}"
                pooled_gt[key] = {"elo": elo}
                pooled_feat[key] = self._features[m]  # 合成 key → 同一方法特征
        if len(pooled_gt) >= 2:
            try:
                self.unified = EloPredictorModel()
                self.unified.fit(pooled_feat, pooled_gt)
            except Exception as e:
                logger.warning("统一预测器 fit 失败（退化为仅模型层）: %s", e)
                self.unified = None

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
        u_mean, u_var = self._predict_one(self.unified, method, feat)
        m_mean, m_var = (self._predict_one(self.models[model], method, feat)
                         if model in self.models else (None, None))

        cap = self._std_cap(model)

        if u_mean is not None and m_mean is not None:
            elo = w_u * u_mean + w_m * m_mean
            var = w_u * w_u * (u_var or 0.0) + w_m * w_m * (m_var or 0.0)
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
            "unified_trained": self.unified is not None,
            "models_trained": sorted(self.models.keys()),
            "samples_per_model": dict(self._model_n),
            "weights_per_model": {m: {"w_model": round(self.blend_weights(m)[0], 3),
                                      "w_unified": round(self.blend_weights(m)[1], 3)}
                                  for m in self._model_n},
        }

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
        """预测器依赖 (R 内容 + 方法清单 + features 结构) 的指纹——三者不变才可复用。"""
        # R 内容指纹：每模型列的 (method, score, ts)
        parts = []
        for model in sorted(results.all_models()):
            col = results.model_column(model)
            payload = ",".join(f"{m}:{r.eval_score}:{r.ts}" for m, r in sorted(col.items()))
            parts.append(f"{model}={payload}")
        r_fp = hashlib.md5(("|".join(parts)).encode("utf-8")).hexdigest()
        cat_fp = hashlib.md5(",".join(method_catalog).encode("utf-8")).hexdigest()
        feat_fp = BlendPredictor._features_signature(features)
        return f"blend_{r_fp[:12]}_{cat_fp[:8]}_{feat_fp[:8]}"

    def save(self, path) -> None:
        save_artifact(path, {
            "unified": self.unified,
            "models": self.models,
            "_features": self._features,
            "_catalog": self._catalog,
            "_tested": self._tested,
            "_measured_elo": self._measured_elo,
            "_model_n": self._model_n,
            "_gt_std": self._gt_std,
            "prior_k": self.prior_k,
        })

    @classmethod
    def load(cls, path) -> BlendPredictor | None:
        data = load_artifact(path)
        if not data:
            return None
        bp = cls(prior_k=data.get("prior_k", BLEND_PRIOR_K))
        bp.unified = data.get("unified")
        bp.models = data.get("models", {})
        bp._features = data.get("_features", {})
        bp._catalog = data.get("_catalog", [])
        bp._tested = data.get("_tested", {})
        bp._measured_elo = data.get("_measured_elo", {})
        bp._model_n = data.get("_model_n", {})
        bp._gt_std = data.get("_gt_std", {})
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
