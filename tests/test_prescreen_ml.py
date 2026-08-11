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
