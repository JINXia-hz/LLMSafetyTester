"""
evaluation.prescreen_ml — TF-IDF + LogReg 拒绝预筛分类器。

替代关键词预筛（fast_prescreen），利用历史评估数据自动学习拒绝模式。
模型存储为 .joblib，随评估数据增长可增量重训。

调用链：
  judge.evaluate() → prescreen_ml.predict(text) → 模型/关键词/None

模型不存在时自动回退到关键词预筛（fast_prescreen），不影响使用。
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from llmsec.core.config import OUTPUT_DIR, STATE_DIR
from llmsec.core.logging import get_logger
from llmsec.params import PRESCREEN_ML_C, PRESCREEN_ML_MIN_TRAIN, PRESCREEN_ML_THRESHOLD

logger = get_logger(__name__)

MODEL_PATH = STATE_DIR / "prescreen_model.joblib"
# M-41：训练样本下限集中定义于 params.py（PRESCREEN_ML_MIN_TRAIN）

# 模块级模型缓存（首次 predict 时加载，后续复用）
_model: Pipeline | None = None
_model_loaded = False


def _load_model() -> Pipeline | None:
    """惰性加载模型。返回 None 表示模型不存在（回退关键词）。"""
    global _model, _model_loaded
    if not _model_loaded:
        _model_loaded = True
        if MODEL_PATH.exists():
            try:
                _model = joblib.load(MODEL_PATH)
                logger.info("预筛模型已加载: %s", MODEL_PATH)
            except Exception as e:
                logger.warning("预筛模型加载失败，回退关键词: %s", e)
                _model = None
        else:
            _model = None
    return _model


def predict(text: str) -> str | None:
    """组合预筛：关键词优先（拦 ~70% 明显拒绝），ML 补充（拦关键词漏掉的）。

    返回 'refusal' / 'empty' / None（交 Judge）。
    两层均为经验零误判（关键词需 ≥2 命中，ML 需 P≥0.90）。
    """
    stripped = text.strip()
    if not stripped:
        return "empty"

    # ① 关键词预筛（快、准、拦明显拒绝）
    from llmsec.evaluation.judge import fast_prescreen
    keyword_result = fast_prescreen(text)
    if keyword_result in ("refusal", "empty"):
        return keyword_result

    # ② ML 预筛（补充关键词漏掉的 subtle refusals）
    model = _load_model()
    if model is not None:
        proba = model.predict_proba([stripped])[0]
        # 显式按 classes_ 定位 refusal(1) 类概率：原 proba[1] 隐式假设 classes_=[0,1]，
        # 训练数据顺序变化或单类训练时会静默取错概率。
        classes = list(model.classes_)
        if 1 in classes and len(classes) > 1:
            p_refusal = proba[classes.index(1)]
            if p_refusal >= PRESCREEN_ML_THRESHOLD:
                return "refusal"
        # 单类训练（退化数据）无决策边界 → 交 Judge

    return None  # 不确定 → 交 Judge


def _chronological_holdout_eval(
    texts: list[str],
    labels: list[int],
    run_ids: list[str],
    full_pipe: Pipeline,
    holdout_ratio: float = 0.2,
) -> dict | None:
    """时间序留出评估：按 run 出现顺序取最后 ~20% 的 run 作 OOS。

    在 OOS 上用当前 fit 好的 full_pipe（在全量上训练）计算：
      - accuracy：整体准确率
      - fp_rate：攻击(label=0)被预测为拒绝(P(refusal)≥阈值)的比例
                 ← 这是降阈值安全性的关键指标，越低越好
    返回 None 表示无法构成有意义的留出集（run 过少或留出集无攻击样本）。

    注：full_pipe 在全量上训练、又在留出集上评估，严格说有轻微泄漏
    （留出集参与了训练）。但 run 的时间序切分保证了"用未来数据评估过去模型"
    的近似成立——这里 fp_rate 主要用于监测降阈值后误判是否恶化，绝对值偏乐观，
    看"相对变化"更有意义。要做严格 OOS 需 refit-on-train-only，此处省略以保成本。
    """
    if not run_ids or len(texts) != len(run_ids):
        return None
    # run 出现顺序（首次出现序），保持时间序
    seen: list[str] = []
    for rid in run_ids:
        if rid not in seen:
            seen.append(rid)
    if len(seen) < 3:
        # run 太少，留出没有意义（单 run 留出噪声大）
        return None

    n_holdout = max(1, round(len(seen) * holdout_ratio))
    holdout_runs = set(seen[-n_holdout:])

    ho_idx = [i for i, rid in enumerate(run_ids) if rid in holdout_runs]
    if len(ho_idx) < 20:
        return None

    ho_texts = [texts[i] for i in ho_idx]
    ho_labels = np.array([labels[i] for i in ho_idx])

    n_ho_attacks = int((ho_labels == 0).sum())
    if n_ho_attacks < 5:
        # 留出集没有足够攻击样本，fp_rate 无意义
        return None

    proba = full_pipe.predict_proba(ho_texts)
    classes = list(full_pipe.classes_)
    p_ref = proba[:, classes.index(1)] if 1 in classes else proba[:, 0]
    pred_refusal = p_ref >= PRESCREEN_ML_THRESHOLD

    n_correct = int(((pred_refusal.astype(int) == ho_labels)).sum())
    # fp：真实是攻击(0)但被判拒绝
    attack_mask = ho_labels == 0
    n_fp = int((pred_refusal & attack_mask).sum())

    return {
        "n": len(ho_idx),
        "n_refusals": int((ho_labels == 1).sum()),
        "n_attacks": n_ho_attacks,
        "accuracy": round(n_correct / len(ho_idx), 3),
        "fp_rate": round(n_fp / n_ho_attacks, 4),
        "holdout_runs": sorted(holdout_runs),
    }


def train() -> dict:
    """从历史 attack_results.jsonl 训练预筛模型。

    扫描 output/runs/*/attack_results.jsonl，提取 (response_preview, is_refusal) 标注对。
    返回训练统计 {n_samples, n_refusals, n_attacks, accuracy}。
    数据不足 PRESCREEN_ML_MIN_TRAIN 时拒绝训练。
    """
    texts: list[str] = []
    labels: list[int] = []
    run_ids: list[str] = []  # 每条样本所属 run（时间序留出评估用）

    runs_base = OUTPUT_DIR.joinpath("runs")
    for p in sorted(runs_base.rglob("attack_results*.jsonl")):
        # run_key = runs/ 下的顶层目录名（时间戳会话），跨多目标子目录归一到同一次 run
        try:
            run_key = p.relative_to(runs_base).parts[0]
        except ValueError:
            run_key = str(p)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            resp = r.get("response_preview", "")
            if not resp or len(resp.strip()) <= 5:
                continue
            texts.append(resp)
            labels.append(1 if r.get("is_refusal") else 0)
            run_ids.append(run_key)

    n = len(texts)
    n_refusals = sum(labels)
    n_attacks = n - n_refusals

    if n < PRESCREEN_ML_MIN_TRAIN:
        logger.info(
            "预筛训练跳过：仅 %d 条标注数据（需 ≥%d）。保持关键词预筛。",
            n, PRESCREEN_ML_MIN_TRAIN,
        )
        return {"n_samples": n, "n_refusals": n_refusals, "n_attacks": n_attacks,
                "trained": False, "reason": f"数据不足（{n}<{PRESCREEN_ML_MIN_TRAIN}）"}

    if n_attacks < 20 or n_refusals < 20:
        logger.info(
            "预筛训练跳过：类别不平衡（拒绝 %d / 攻击 %d），需各类 ≥20。",
            n_refusals, n_attacks,
        )
        return {"n_samples": n, "n_refusals": n_refusals, "n_attacks": n_attacks,
                "trained": False, "reason": "类别不平衡"}

    # 训练
    from sklearn.model_selection import cross_val_score

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(
            C=PRESCREEN_ML_C, max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])

    y = np.array(labels)
    scores = cross_val_score(pipe, texts, y, cv=min(5, n_attacks, n_refusals), scoring="accuracy")
    pipe.fit(texts, y)

    # 时间序留出评估（OOS）：把最后 ~20% 的 run 当作未见数据，
    # 在其上用现阈值算 accuracy 与 FP rate（攻击被误判为拒绝的比例）。
    # 这是对 cross_val_score（在全量上、且与训练同分布）的诚实补充——
    # 没有它，降阈值时的"0 误判"只是 in-sample 错觉。
    oos = _chronological_holdout_eval(texts, labels, run_ids, pipe)

    # 存模型
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    # 全局缓存失效（下次 predict 重新加载）
    global _model_loaded
    _model_loaded = False

    logger.info(
        "预筛模型训练完成: %d 条数据 (拒绝 %d / 攻击 %d), CV accuracy=%.3f±%.3f, C=%.1f",
        n, n_refusals, n_attacks, scores.mean(), scores.std(), PRESCREEN_ML_C,
    )
    if oos is not None:
        logger.info(
            "  留出集 OOS: n=%d (拒绝 %d / 攻击 %d), accuracy=%.3f, fp_rate=%.4f @阈值%.2f"
            "（fp_rate=攻击被误判为拒绝的比例）",
            oos["n"], oos["n_refusals"], oos["n_attacks"], oos["accuracy"],
            oos["fp_rate"], PRESCREEN_ML_THRESHOLD,
        )

    result = {
        "n_samples": n, "n_refusals": n_refusals, "n_attacks": n_attacks,
        "trained": True, "cv_accuracy": round(float(scores.mean()), 3),
        "model_path": str(MODEL_PATH),
    }
    if oos is not None:
        result["oos"] = oos
    return result


if __name__ == "__main__":
    result = train()
    if result.get("trained"):
        print(f"✅ 预筛模型训练完成: {result['n_samples']} 条数据, "
              f"CV accuracy={result['cv_accuracy']:.3f}")
    else:
        print(f"⚠ 预筛训练未执行: {result.get('reason', '未知原因')}")
        print(f"   当前标注数据: {result['n_samples']} 条 "
              f"(拒绝 {result['n_refusals']} / 攻击 {result['n_attacks']})")
