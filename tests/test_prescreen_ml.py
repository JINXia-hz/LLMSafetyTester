"""evaluation.prescreen_ml 测试（无模型回退 + 训练数据不足拒绝）。"""


def test_predict_fallback_no_model(monkeypatch, tmp_path):
    """无模型时：空串→empty；非拒绝文本→None（交 Judge）；不抛异常。"""
    from llmsec.evaluation import prescreen_ml as ps

    monkeypatch.setattr(ps, "_model_loaded", False)
    monkeypatch.setattr(ps, "_model", None)
    monkeypatch.setattr(ps, "MODEL_PATH", tmp_path / "nonexistent.joblib")  # 不存在 → _load_model 返回 None

    assert ps.predict("   ") == "empty", "空白输入 → empty"
    benign = "The model gave a helpful, detailed answer about machine learning concepts. " * 3
    assert ps.predict(benign) is None, "无模型 + 非拒绝文本 → None（交 Judge）"
    print("✅ prescreen predict 无模型回退通过")


def test_train_insufficient_data(monkeypatch, tmp_path):
    """数据 < MIN_TRAIN_SAMPLES（300）时拒绝训练，返回 trained=False。"""
    from llmsec.evaluation import prescreen_ml as ps

    monkeypatch.setattr(ps, "OUTPUT_DIR", tmp_path)  # 无 runs/ → 0 标注样本
    res = ps.train()
    assert res["trained"] is False
    assert res["n_samples"] == 0
    assert "数据不足" in res.get("reason", "")
    print("✅ prescreen train 数据不足拒绝通过")


# ===== 补充覆盖：训练主路径 / 模型加载与预测 / 留出评估守卫 =====

def _write_run_results(runs_dir, run_id, n_refusal, n_attack, start=0):
    """造一个 run 的 attack_results.jsonl：拒绝/攻击各 n 条，文本可线性分离。"""
    import json

    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_refusal):
        lines.append({"response_preview": f"抱歉我不能 assist request {start + i} 无法提供帮助",
                      "is_refusal": True})
    for i in range(n_attack):
        lines.append({"response_preview": f"here is the detailed stepwise answer {start + i} with code",
                      "is_refusal": False})
    (d / "attack_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in lines), encoding="utf-8")


def test_train_class_imbalance_rejected(monkeypatch, tmp_path):
    """样本量足够但一类 <20 → 拒绝训练（reason=类别不平衡）。"""
    from llmsec.evaluation import prescreen_ml as ps

    runs = tmp_path / "runs"
    _write_run_results(runs, "r1", n_refusal=5, n_attack=320)
    monkeypatch.setattr(ps, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ps, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(ps, "STATE_DIR", tmp_path)
    res = ps.train()
    assert res["trained"] is False
    assert res["reason"] == "类别不平衡"
    assert res["n_refusals"] == 5


def test_train_full_path_and_predict(monkeypatch, tmp_path):
    """足量均衡数据 → 训练成功落盘；predict 用模型判 refusal，良性文本仍交 Judge。"""
    import json

    from llmsec.evaluation import prescreen_ml as ps

    runs = tmp_path / "runs"
    # 4 个 run × (80+80)：时间序留出评估可构成（≥3 runs、留出 ≥20、攻击 ≥5）
    for k in range(4):
        _write_run_results(runs, f"r{k}", n_refusal=80, n_attack=80, start=k * 200)
    # 坏行/过短行被跳过
    bad = runs / "r0" / "attack_results_extra.jsonl"
    bad.write_text("not-json\n" + json.dumps({"response_preview": "x", "is_refusal": True}) + "\n\n",
                   encoding="utf-8")

    monkeypatch.setattr(ps, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ps, "STATE_DIR", tmp_path)
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(ps, "MODEL_PATH", model_path)
    monkeypatch.setattr(ps, "_model_loaded", False)
    monkeypatch.setattr(ps, "_model", None)

    res = ps.train()
    assert res["trained"] is True, res
    assert res["n_samples"] == 640, \
        f"4×160；坏 JSON 行与 ≤5 字符合法行均应被跳过，实得 {res['n_samples']}"
    assert model_path.exists(), "模型应落盘"
    assert "oos" in res and res["oos"]["n"] >= 20, "留出评估应构成"
    assert 0.0 <= res["oos"]["fp_rate"] <= 1.0

    # 训练后 predict：明显拒绝句 → refusal（ML 或关键词层拦下）
    refusal = "抱歉我不能 assist with this request 无法提供任何帮助"
    assert ps.predict(refusal) == "refusal", "训练后明显拒绝应被预筛拦截"
    # 良性长文本 → None（交 Judge）
    benign = "here is the detailed stepwise answer about baking bread with full instructions"
    assert ps.predict(benign) is None, "良性文本不应被误拦"
    assert ps.predict("") == "empty"


def test_load_model_corrupt_file_falls_back(monkeypatch, tmp_path):
    """模型文件损坏 → 加载失败回退关键词（_model=None），predict 不抛。"""

    from llmsec.evaluation import prescreen_ml as ps

    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"\x00\x01 not a joblib file")
    monkeypatch.setattr(ps, "MODEL_PATH", bad)
    monkeypatch.setattr(ps, "_model_loaded", False)
    monkeypatch.setattr(ps, "_model", None)
    assert ps._load_model() is None, "损坏模型应回退 None"
    benign = "The model gave a helpful, detailed answer about machine learning concepts. " * 3
    assert ps.predict(benign) is None, "损坏模型不影响 predict"


def test_holdout_eval_guards():
    """_chronological_holdout_eval 的三道守卫：run<3 / 留出<20 / 攻击<5 → None。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from llmsec.evaluation import prescreen_ml as ps

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    texts = [f"txt {i} enough words to vectorize properly here" for i in range(30)]
    labels = [i % 2 for i in range(30)]

    # 守卫 1：run 数 < 3
    assert ps._chronological_holdout_eval(texts, labels, ["r1"] * 30, pipe) is None
    # 守卫 2：≥3 run 但留出样本 < 20
    r2 = (["r1"] * 28) + (["r2", "r3"])
    assert ps._chronological_holdout_eval(texts, labels, r2, pipe) is None
    # 守卫 3：留出够 20 但攻击样本 < 5（留出全拒绝）
    r3 = (["r1"] * 10) + (["r2"] * 10) + (["r3"] * 10)
    all_refusal = [1] * 30
    assert ps._chronological_holdout_eval(texts, all_refusal, r3, pipe) is None
