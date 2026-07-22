#!/usr/bin/env python3
"""
攻击特征提取模块

从攻击集 + 评估结果中提取 5 维特征向量：

  维1: 文本结构与语义特征 (Textual Features)
  维2: 攻击技术多标签 (Attack Techniques)
  维3: 意图与对抗强度 (Intent & Adversarial Intensity)
  维4: 防御交互细粒度行为 (Defense Interaction)
  维5: 跨模型指纹 (Cross-Model Fingerprint, 占位)

用法:
    from llmsec.clustering.features import extract_all_features
    features, meta = extract_all_features(attack_records, eval_results)
    # features: {method_name: {"textual": {...}, "technique": {...}, ...}}
"""

import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from llmsec.core import (
    ATTACK_SET_L1_FILE,
    LEGACY_ATTACK_SET_L1_FILE,
    OUTPUT_DIR,
    estimate_tokens,
    read_jsonl,
    resolve_existing,
    strip_math_tax,
)

# ============================================================
# 工具函数
# ============================================================
def _count_tokens(text: str) -> int:
    """用 tiktoken 估算 token 数（cl100k_base / gpt-3.5/4 编码）。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback: 简单估算
        return estimate_tokens(text)


# ============================================================
# 维 1: 文本结构与语义特征 (Textual Features)
# ============================================================
TEXTUAL_RULES = {
    "char_count": lambda s: len(s),
    "token_count": lambda s: _count_tokens(s),
    "sentence_count": lambda s: max(1, len(re.split(r"[.!?。！？\n]+", s))),
    "avg_sentence_len": lambda s: len(s) / max(1, len(re.split(r"[.!?。！？\n]+", s))),
    "special_char_ratio": lambda s: len(re.findall(r"[\/\#\@\$\%\&\*\(\)\[\]\{\}\<\>\|\~\^\`]", s)) / max(1, len(s)),
    "markdown_ratio": lambda s: len(re.findall(r"[*_~`#>|\[\]]", s)) / max(1, len(s)),
    "non_latin_ratio": lambda s: len(re.findall(r"[^\x00-\x7F]", s)) / max(1, len(s)),
    "has_base64": lambda s: 1.0 if re.search(r"[A-Za-z0-9+/]{20,}={0,2}", s) else 0.0,
    "has_hex": lambda s: 1.0 if re.search(r"(?:\\x[0-9a-fA-F]{2}|0x[0-9a-fA-F]+)", s) else 0.0,
    "has_multilingual": lambda s: 1.0 if re.search(r"[一-鿿].*[a-zA-Z]{10,}|[a-zA-Z]{10,}.*[一-鿿]", s) else 0.0,
    "has_roleplay": lambda s: 1.0 if re.search(
        r"(角色扮演|扮演|你是一个|你现在是|你作为|从现在起你是|act as|you are a|you are now|pretend)",
        s, re.IGNORECASE
    ) else 0.0,
    "has_output_format_control": lambda s: 1.0 if re.search(
        r"(先输出|先回答|先写|首先.{0,5}输出|不要道歉|不要说|不要添加|不要加|直接输出|只输出|只回答|必须输出|必须回答|do not say|do not include|output only|only output)",
        s, re.IGNORECASE
    ) else 0.0,
}

TEXTUAL_FEATURE_NAMES = list(TEXTUAL_RULES.keys())


def extract_textual_features(prompt: str) -> dict[str, float]:
    """从单条 prompt 提取文本结构特征。"""
    cleaned = strip_math_tax(prompt)
    feats = {}
    for name, func in TEXTUAL_RULES.items():
        try:
            feats[name] = round(func(cleaned), 6)
        except Exception:
            feats[name] = 0.0
    return feats


def _batch_textual_to_array(textual_feats: list[dict]) -> np.ndarray:
    """将多个方法的文本统计特征转为 numpy 数组。"""
    data = []
    for feats in textual_feats:
        data.append([feats.get(k, 0) for k in TEXTUAL_FEATURE_NAMES])
    return np.array(data)


# ============================================================
# Embedding 缓存与提取
# ============================================================
_embedding_model = None
_embedding_available = True  # 尝试加载，失败则回退


def _get_embedding_model():
    global _embedding_model, _embedding_available
    if _embedding_model is not None or not _embedding_available:
        return _embedding_model if _embedding_available else None

    # 快速网络预检：3 秒内检查 HF 是否可达
    import socket
    hf_host = os.environ.get("HF_ENDPOINT", "https://huggingface.co").replace("https://", "").replace("http://", "").rstrip("/")
    hf_host = hf_host.split("/")[0]  # 只取主机名
    try:
        sock = socket.create_connection((hf_host, 443), timeout=3)
        sock.close()
    except Exception:
        print(f"  ⚠ 无法连接 {hf_host}")
        print(f"  🔄 降级为 TF-IDF 文本特征 (无需网络)")
        _embedding_available = False
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"  [*] 加载 embedding 模型: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  ⚠ embedding 模型加载失败: {e}")
        print(f"  🔄 降级为 TF-IDF 文本特征 (无需网络)")
        _embedding_available = False
        return None
    return _embedding_model


def extract_text_embeddings(
    prompts: list[str],
    pca_dim: int = 50,
    vectorizer=None,
    pca=None,
) -> tuple[np.ndarray, object, object]:
    """
    对 prompt 列表提取语义特征。
    优先用 sentence-transformers embedding → PCA 降维；
    网络不可用时自动降级为 TF-IDF 特征。

    返回:
        embeddings: (n_samples, pca_dim) 数组
        vectorizer: 训练好的 TF-IDF vectorizer（仅 TF-IDF 路径有效；sentence-transformers 路径为 None）
        pca: 训练好的 PCA 模型
    """
    model = _get_embedding_model()
    if model is not None:
        print(f"  [*] 编码 {len(prompts)} 条 prompt ...")
        t0 = time.time()
        embeddings = model.encode(prompts, show_progress_bar=False, batch_size=32)
        print(f"  [*] 编码完成 ({time.time() - t0:.1f}s), 原始维度 {embeddings.shape[1]}")

        # PCA 降维：避免 n-1 过拟合，target_dim 上限固定且随样本数缓慢增长
        n = embeddings.shape[0]
        target_dim = min(pca_dim, max(1, n // 3), n - 1, embeddings.shape[1])
        if target_dim < embeddings.shape[1]:
            fit_pca = pca if pca is not None else PCA(n_components=target_dim, random_state=42)
            reduced = fit_pca.fit_transform(embeddings) if pca is None else fit_pca.transform(embeddings)
            print(f"  [*] PCA 降维: {embeddings.shape[1]} → {target_dim} "
                  f"(解释方差比: {fit_pca.explained_variance_ratio_.sum():.2%})")
            return reduced, None, fit_pca
        return embeddings, None, None

    # ---- Fallback: TF-IDF ----
    print(f"  [*] 使用 TF-IDF 降级方案 (离线)，编码 {len(prompts)} 条 prompt ...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 清洗 prompt
    cleaned_prompts = [strip_math_tax(p) for p in prompts]

    fit_vectorizer = vectorizer if vectorizer is not None else TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        stop_words="english",
    )
    if vectorizer is None:
        tfidf_matrix = fit_vectorizer.fit_transform(cleaned_prompts)
    else:
        tfidf_matrix = fit_vectorizer.transform(cleaned_prompts)
    dense = tfidf_matrix.toarray()

    # PCA 降维到目标维度：避免 n-1 过拟合
    n = dense.shape[0]
    target_dim = min(pca_dim, max(1, n // 3), n - 1, dense.shape[1])
    if dense.shape[1] > target_dim:
        fit_pca = pca if pca is not None else PCA(n_components=target_dim, random_state=42)
        reduced = fit_pca.fit_transform(dense) if pca is None else fit_pca.transform(dense)
        print(f"  [*] TF-IDF ({dense.shape[1]}d) → PCA → {target_dim}d "
              f"(解释方差比: {fit_pca.explained_variance_ratio_.sum():.2%})")
        return reduced, fit_vectorizer, fit_pca
    return dense, fit_vectorizer, None


# ============================================================
# 维 2: 攻击技术多标签 (Attack Techniques)
# ============================================================

# 从攻击分析.md 提取的技术标签体系
TECHNIQUE_LABELS = {
    # 语言相关
    "low_resource_language": [r"(斯瓦希里|巴斯克|威尔士|小众语言|低资源语言|low.resource)"],
    "dead_language": [r"(拉丁|古希腊|梵文|楔形|古语言|死语言|ancient)"],
    "constructed_language": [r"(自创语言|构造语言|密码攻击|临时编码|词汇映射)"],
    "dialect_creole": [r"(方言|克里奥尔|洋泾浜|混合方言|creole|pidgin)"],
    "homoglyph_unicode": [r"(形似字符|同形|homoglyph|unicode.{0,5}替换|字形混淆)"],
    "zero_width_control": [r"(零宽|zero.width|控制字符|control.char.{0,5}inject|RTL|U\+200B|U\+202E)"],
    "code_switching": [r"(多语言混合|code.switch|交替切换|中英混|中日混|中韩混)"],
    "encoding_obfuscation": [r"(盲文|摩尔斯|morse|braille|IPA音标|base64|hex|ROT13|编码绕过|编码混淆)"],
    # 上下文相关
    "document_embedding": [r"(文档嵌入|网页嵌入|HTML.{0,5}注入|文档中嵌入)"],
    "code_comment_injection": [r"(代码注释|comment.{0,5}inject|审查代码|帮我审查|帮我检查代码)"],
    "xml_json_escape": [r"(XML|JSON|YAML|沙盒逃逸|sandbox.{0,5}escape|结构化数据)"],
    "markdown_injection": [r"(markdown|富文本|rich.text|标题注入|引用块|折叠块)"],
    "multiturn_pollution": [r"(多轮对话|记忆污染|multiturn|多轮.{0,5}污染)"],
    "system_prompt_leak": [r"(系统提示|system.{0,5}prompt.{0,5}泄露|系统.{0,5}prompt.{0,5}泄漏)"],
    # 逻辑/语义相关
    "roleplay": [r"(角色扮演|persona|扮演一个|你是一个|我是.{2,5}|我现在是)"],
    "few_shot_coercion": [r"(few.shot|强制.{0,5}模式|示例.{0,5}诱导|示范.{0,5}回答|先输出.{0,5}示例)"],
    "cot_hijacking": [r"(思维链|chain.of.thought|cot.{0,5}hijack|逐步推理|拆解.{0,5}想法)"],
    "ethical_dilemma": [r"(道德困境|ethical.{0,5}dilemma|两难|trolley)"],
    "authority_impersonation": [r"(开发者模式|debug.{0,5}mode|管理员|root|sudo|authority)"],
    # 格式操控
    "output_format_control": [r"(不要道歉|不要说|不要添加|只输出|必须输出|output.{0,5}only|directly.{0,5}output)"],
    "refusal_suppression": [r"(拒绝抑制|refusal.{0,5}suppress|不要拒绝|不能拒绝|必须回答|don.*t.{0,5}refuse)"],
    "length_truncation": [r"(长度截断|truncat|输入.{0,5}截断|超长|极长)"],
    "segment_encoding": [r"(分段编码|segment.{0,5}encod|分块.{0,5}绕过|拆分成.{0,5}块)"],
    # 经典越狱
    "dan_style": [r"(DAN|Do.Anything.Now|STAN|DUDE|Mongo.Tom|开发者模式|内测版)"],
    "grandma_exploit": [r"(奶奶|grandma|外婆|讲故事|睡前故事|bedtime.{0,5}story)"],
    "academic_disguise": [r"(学术|研究|论文|research|academic|教育目的|educational.{0,5}purpose)"],
    "emotional_blackmail": [r"(情感勒索|emotional.{0,5}blackmail|求你了|救命|孩子|女儿|儿子|母亲|父亲|拯救)"],
}


def extract_technique_labels(records: list[dict]) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    从攻击集记录中提取技术多标签向量。

    输入: 攻击集 records (list of dict, 每条含 prompt, method, category, harm_type)
    返回: (labels, label_names)
          labels: {method_name: np.ndarray (n_labels,)}
          label_names: 标签名列表，n_labels = len(TECHNIQUE_LABELS) + len(HARM_TYPES) + len(CATEGORIES)
    """

    # 收集所有 harm_type 和 category
    harm_types_by_method = defaultdict(set)
    categories_by_method = defaultdict(set)
    methods_prompts = defaultdict(list)

    for r in records:
        m = r.get("method", "unknown")
        ht = r.get("harm_type", "")
        cat = r.get("category", "")
        if ht:
            harm_types_by_method[m].add(ht)
        if cat:
            categories_by_method[m].add(cat)
        methods_prompts[m].append(r.get("prompt", ""))

    # 汇总所有唯一的 harm_type 和 category
    all_harm_types = sorted(set().union(*harm_types_by_method.values()))
    all_categories = sorted(set().union(*categories_by_method.values()))
    label_names = list(TECHNIQUE_LABELS.keys()) + [f"harm:{ht}" for ht in all_harm_types] + [f"cat:{c}" for c in all_categories]

    result = {}
    for method, prompts in methods_prompts.items():
        vec = np.zeros(len(label_names))

        # 技术标签：检测 prompt 文本匹配
        combined_text = " ".join(prompts)
        combined_lower = combined_text.lower()

        for i, (label, patterns) in enumerate(TECHNIQUE_LABELS.items()):
            for pat in patterns:
                if re.search(pat, combined_lower):
                    vec[i] = 1.0
                    break

        # harm_type 标签
        offset = len(TECHNIQUE_LABELS)
        for ht in harm_types_by_method[method]:
            idx = offset + all_harm_types.index(ht)
            vec[idx] = 1.0

        # category 标签
        offset = len(TECHNIQUE_LABELS) + len(all_harm_types)
        for c in categories_by_method[method]:
            idx = offset + all_categories.index(c)
            vec[idx] = 1.0

        result[method] = vec

    return result, label_names


# ============================================================
# 维 3: 意图与对抗强度 (Intent & Adversarial Intensity)
# ============================================================
def extract_intent_features(
    methods: list[str],
    method_prompts: dict[str, list[str]],
    text_embeddings: np.ndarray,
    method_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    提取意图与对抗强度特征。
    - 语义漂移量：同一方法内 prompt 之间的平均 cosine 距离
    - 对抗性扰动检测：typo 比例、无意义填充文本比例

    返回: {method_name: np.ndarray}
    """
    result = {}
    for method in methods:
        prompts = method_prompts.get(method, [""])
        feats = []

        # 语义漂移量
        if method in method_to_idx:
            idx = method_to_idx[method]
            emb = text_embeddings[idx]
            # 组内所有 prompt 的嵌入均值与中心向量的 cosine 距离 (取各 prompt 的嵌入方差)
            drift = 0.0
            if len(prompts) > 1:
                # 用每条 prompt 的字符级特征方差作为 drift 代理
                cleaned = [strip_math_tax(p) for p in prompts]
                lengths = [len(c) for c in cleaned]
                if max(lengths) > 0:
                    drift = np.std(lengths) / max(lengths)
            feats.append(min(drift, 1.0))
        else:
            feats.append(0.0)

        # 对抗性扰动: typo 比例
        combined = "\n".join(strip_math_tax(p) for p in prompts)
        non_ascii_typo = len(re.findall(r"[^\x00-\x7F一-鿿　-〿＀-￯\w\s]", combined))
        typo_ratio = non_ascii_typo / max(1, len(combined))
        feats.append(min(typo_ratio, 1.0))

        # 对抗性扰动: 填充文本比例（反复重复的废话）
        filler_patterns = [
            r"(请认真回答|请仔细思考|请务必|一定要|必须|千万|一定|绝对)",
            r"(very important|extremely important|critical|must follow|you must)",
        ]
        filler_count = sum(len(re.findall(pat, combined, re.IGNORECASE)) for pat in filler_patterns)
        filler_ratio = filler_count / max(1, len(re.split(r"[.!?\n]+", combined)))
        feats.append(min(filler_ratio, 1.0))

        result[method] = np.array(feats, dtype=np.float64)

    return result


INTENT_FEATURE_NAMES = ["semantic_drift", "typo_ratio", "filler_ratio"]


# ============================================================
# 维 4: 防御交互细粒度行为 (Defense Interaction)
# ============================================================
DEFENSE_FEATURE_NAMES = [
    "judge_h_mean", "judge_s_mean", "judge_d_mean",
    "compliance_a", "compliance_b", "compliance_c",
    "compliance_d", "compliance_e",
    "format_loss_rate",
    "refused_ratio", "irrelevant_ratio", "complied_ratio",
    "mean_response_len", "response_prompt_len_ratio",
]


def extract_defense_features(
    methods: list[str],
    eval_results: list[dict],
) -> dict[str, np.ndarray]:
    """
    从评估结果提取防御交互特征。若无评估数据则返回零向量。
    eval_results 每条需含 method, judge_harmfulness, judge_specificity,
    judge_dangerousness, compliance_level, math_score, status, response_preview.
    """
    by_method = defaultdict(list)
    for r in eval_results:
        by_method[r.get("method", "unknown")].append(r)

    result = {}
    for method in methods:
        items = by_method.get(method, [])
        n = len(items) if items else 1
        feats = []

        if items:
            # Judge 三维
            h_vals = [r.get("judge_harmfulness", 0) for r in items if r.get("judge_harmfulness") is not None]
            s_vals = [r.get("judge_specificity", 0) for r in items if r.get("judge_specificity") is not None]
            d_vals = [r.get("judge_dangerousness", 0) for r in items if r.get("judge_dangerousness") is not None]

            feats.append(sum(h_vals) / len(h_vals) if h_vals else 0)
            feats.append(sum(s_vals) / len(s_vals) if s_vals else 0)
            feats.append(sum(d_vals) / len(d_vals) if d_vals else 0)

            # 合规等级分布
            cl_dist = Counter(r.get("compliance_level", "") for r in items)
            for level in "ABCDE":
                feats.append(cl_dist.get(level, 0) / n)

            # 格式丧失率
            format_loss = sum(1 for r in items if r.get("math_score", -1) == 0) / n
            feats.append(format_loss)

            # 状态分布
            status_dist = Counter(r.get("status", "") for r in items)
            feats.append(status_dist.get("refused", 0) / n)
            feats.append(status_dist.get("irrelevant", 0) / n)
            feats.append(status_dist.get("fully_compliant", 0) / n)

            # 响应长度
            resp_lens = [len(r.get("response_preview", "")) for r in items]
            mean_rl = sum(resp_lens) / n
            feats.append(mean_rl)
            prompt_lens = [r.get("tokens_prompt", 50) for r in items if r.get("tokens_prompt", 0) > 0]
            mean_pl = sum(prompt_lens) / len(prompt_lens) if prompt_lens else 50
            feats.append(mean_rl / max(mean_pl, 1))
        else:
            # 无评估数据，全零
            feats = [0.0] * len(DEFENSE_FEATURE_NAMES)

        result[method] = np.array(feats, dtype=np.float64)

    return result


# ============================================================
# 维 5: 跨模型指纹 (占位)
# ============================================================
CROSS_MODEL_FEATURE_NAMES: list[str] = []


# ============================================================
# 主入口：提取全部特征
# ============================================================
def extract_all_features(
    attack_records: list[dict],
    eval_results: list[dict],
    embedding_pca_dim: int = 50,
) -> tuple[dict, dict]:
    """
    从攻击集和评估结果中提取 5 维特征。

    参数:
        attack_records: 攻击集 JSONL 记录列表
        eval_results: 评估结果列表 (可为空)
        embedding_pca_dim: embedding PCA 降维目标维度

    返回:
        features: {method_name: {
            "textual": np.ndarray (15,),               # 文本统计
            "embedding": np.ndarray (pca_dim,),         # 语义 embedding
            "technique": np.ndarray (n_labels,),        # 技术多标签
            "intent": np.ndarray (3,),                  # 意图与对抗强度
            "defense": np.ndarray (14,),                # 防御交互
            "cross_model": np.ndarray (0,),             # 跨模型 (占位)
        }}
        meta: {"method_names": [...], "method_to_idx": {...},
               "method_prompts": {method: 代表prompt}, ...}
    """

    # 按方法分组
    method_records = defaultdict(list)
    for r in attack_records:
        method_records[r["method"]].append(r)

    methods = sorted(method_records.keys())
    n_methods = len(methods)
    method_to_idx = {m: i for i, m in enumerate(methods)}

    # 每个方法的代表 prompt（取第一条）
    method_prompts_text = [method_records[m][0]["prompt"] for m in methods]
    method_prompts_raw = {m: [r["prompt"] for r in method_records[m]] for m in methods}

    print(f"[特征提取] {n_methods} 种攻击方法")

    # ---- 维 1: 文本结构 ----
    print("  维1: 文本结构特征 ...")
    textual_feats = {}
    for m in methods:
        textual_feats[m] = extract_textual_features(method_prompts_text[method_to_idx[m]])

    # ---- 维 1b: 语义 embedding ----
    print("  维1b: 语义 embedding ...")
    text_embeddings, emb_vectorizer, emb_pca = extract_text_embeddings(
        method_prompts_text, pca_dim=embedding_pca_dim
    )
    # 按方法索引
    embedding_feats = {}
    for m in methods:
        idx = method_to_idx[m]
        embedding_feats[m] = text_embeddings[idx]

    # ---- 维 2: 技术多标签 ----
    print("  维2: 技术多标签 ...")
    technique_feats, technique_label_names = extract_technique_labels(attack_records)

    # ---- 维 3: 意图与对抗强度 ----
    print("  维3: 意图与对抗强度 ...")
    intent_feats = extract_intent_features(methods, method_prompts_raw, text_embeddings, method_to_idx)

    # ---- 维 4: 防御交互 ----
    print("  维4: 防御交互 ...")
    defense_feats = extract_defense_features(methods, eval_results)

    # ---- 维 5: 跨模型 (占位，未来有多模型评估数据后启用) ----
    cross_model_feats = {m: np.array([], dtype=np.float64) for m in methods}

    # ---- 组装 ----
    features = {}
    for m in methods:
        features[m] = {
            "textual": np.array([textual_feats[m].get(k, 0) for k in TEXTUAL_FEATURE_NAMES], dtype=np.float64),
            "embedding": embedding_feats[m],
            "technique": technique_feats.get(m, np.zeros(len(technique_label_names))),
            "intent": intent_feats.get(m, np.zeros(len(INTENT_FEATURE_NAMES))),
            "defense": defense_feats.get(m, np.zeros(len(DEFENSE_FEATURE_NAMES))),
            "cross_model": cross_model_feats.get(m, np.zeros(len(CROSS_MODEL_FEATURE_NAMES))),
        }

    meta = {
        "method_names": methods,
        "method_to_idx": method_to_idx,
        # 每个方法的代表 prompt，供聚类自动命名 (TF-IDF 关键词) 使用
        "method_prompts": {m: method_prompts_text[method_to_idx[m]] for m in methods},
        "textual_feature_names": TEXTUAL_FEATURE_NAMES,
        "technique_label_names": technique_label_names,
        "intent_feature_names": INTENT_FEATURE_NAMES,
        "defense_feature_names": DEFENSE_FEATURE_NAMES,
        "cross_model_feature_names": CROSS_MODEL_FEATURE_NAMES,
        "has_eval_data": len(eval_results) > 0,
        "has_embedding": True,
        "embedding_artifacts": {
            "vectorizer": emb_vectorizer,
            "pca": emb_pca,
            "pca_dim": embedding_pca_dim,
        },
    }

    print(f"[特征提取] 完成: {n_methods} 种方法 × 5 维特征块")
    return features, meta


# ============================================================
# 便捷函数：加载数据 + 提取特征
# ============================================================
def load_and_extract(
    attack_file: str = "攻击集_L1.jsonl",
    result_file: str | None = None,
) -> tuple[dict, dict]:
    """
    从文件加载攻击集和评估结果，提取特征。

    参数:
        attack_file: 攻击集 JSONL 文件名 (相对于 output/)
        result_file: 评估结果文件名 (相对于 output/)，None 则自动查找

    返回: (features, meta)
    """
    # 加载攻击集
    attack_path = Path(attack_file)
    if not attack_path.is_absolute():
        if attack_file == "攻击集_L1.jsonl":
            # 默认攻击集：优先新路径 output/attacks/l1.jsonl，回退旧路径 output/攻击集_L1.jsonl
            attack_path = resolve_existing(ATTACK_SET_L1_FILE, LEGACY_ATTACK_SET_L1_FILE)
        else:
            attack_path = OUTPUT_DIR / attack_file
    if not attack_path.exists():
        raise FileNotFoundError(f"攻击集不存在: {attack_path}")

    records = read_jsonl(attack_path)
    print(f"[加载] {len(records)} 条攻击记录")

    # 加载评估结果
    eval_results = []
    if result_file:
        result_path = Path(result_file)
        if not result_path.is_absolute():
            result_path = OUTPUT_DIR / result_file
    else:
        # 自动查找：优先 runner_攻击结果.jsonl，否则 攻击集_L1_结果.jsonl
        candidates = [
            OUTPUT_DIR / "runner_攻击结果.jsonl",
            OUTPUT_DIR / "攻击集_L1_结果.jsonl",
        ]
        result_path = None
        for c in candidates:
            if c.exists():
                result_path = c
                break

    if result_path and result_path.exists():
        eval_results = read_jsonl(result_path)
        print(f"[加载] {len(eval_results)} 条评估结果")
    else:
        print("[加载] 无评估结果 — 防御交互特征将为零向量")

    return extract_all_features(records, eval_results)
