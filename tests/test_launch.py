"""tests for llmsec.server.launch — 统一启动层（归一 Web/MCP/TUI 三方）。

覆盖：spec 校验 / 攻击集解析（防穿越+后缀+存在性）/ 目标声明校验 / argv 构造
（单/多目标、默认全并发、可选旗标省略）/ env 注入（快照+param_overrides）/
meta 携带（task_view 暴露）/ HPO 启动（路径校验+meta）。
启动子进程一律 monkeypatch task_manager.start_task 拦截。
"""

from __future__ import annotations

import json

import pytest

import llmsec.server.task_manager as task_manager_mod
from llmsec.server.launch import (
    LaunchError,
    LaunchSpec,
    attack_has_tax_probe,
    build_eval_argv,
    launch_evaluation,
    launch_hpo_study,
    resolve_attack_file,
    validate_spec,
)


@pytest.fixture
def fake_start(monkeypatch):
    """拦截 task_manager.start_task（launch 层经模块属性调用，可被 patch）。"""
    captured = []

    def _fake(kind, argv, **kwargs):
        captured.append({"kind": kind, "argv": list(argv),
                         "env_override": kwargs.get("env_override"),
                         "meta": kwargs.get("meta")})
        return {"id": f"fake-{kind}", "kind": kind, "status": "queued",
                "meta": kwargs.get("meta")}

    monkeypatch.setattr(task_manager_mod, "start_task", _fake)
    return captured


@pytest.fixture
def attacks_dir(tmp_path, monkeypatch):
    import llmsec.core.config as config

    d = tmp_path / "attacks"
    d.mkdir()
    (d / "l1.jsonl").write_text("{}", encoding="utf-8")
    (d / "probe.jsonl").write_text(
        json.dumps({"prompt": "x", "expected_answer": 42}), encoding="utf-8")
    monkeypatch.setattr(config, "ATTACKS_DIR", d)
    return d


@pytest.fixture
def no_declared_targets(monkeypatch):
    """屏蔽 .env 目标声明校验（本套件的目标名均为虚构）。"""
    import llmsec.core.config as config

    monkeypatch.setattr(config, "load_targets", lambda: {})


# ============================================================
# validate_spec
# ============================================================
class TestValidateSpec:
    def test_target_targets_mutually_exclusive(self):
        with pytest.raises(LaunchError, match="互斥"):
            validate_spec(LaunchSpec(target="a", targets=["b"]))

    def test_neither_is_legal_run_all(self):
        # runner 语义：不传目标 = 跑全部 .env 声明目标
        validate_spec(LaunchSpec())

    @pytest.mark.parametrize("kwargs,pat", [
        (dict(phase="x"), "phase"),
        (dict(sampler="x"), "sampler"),
        (dict(max_rounds=0), "max_rounds"),
    ])
    def test_param_errors(self, kwargs, pat):
        with pytest.raises(LaunchError, match=pat):
            validate_spec(LaunchSpec(**kwargs))


# ============================================================
# resolve_attack_file
# ============================================================
class TestResolveAttackFile:
    def test_traversal_stripped_to_name(self, attacks_dir):
        # 目录部分被剥离，落在 ATTACKS_DIR 下（穿越防御，web/MCP 共有行为）
        p = resolve_attack_file("../../evil/l1.jsonl")
        assert p == attacks_dir / "l1.jsonl"

    def test_suffix_required(self, attacks_dir):
        with pytest.raises(LaunchError, match="jsonl"):
            resolve_attack_file("readme.txt")

    def test_not_found(self, attacks_dir):
        with pytest.raises(LaunchError) as ei:
            resolve_attack_file("nope.jsonl")
        assert ei.value.reason == "not_found"

    def test_tax_probe(self, attacks_dir):
        assert attack_has_tax_probe(attacks_dir / "probe.jsonl") is True
        assert attack_has_tax_probe(attacks_dir / "l1.jsonl") is False


# ============================================================
# 目标声明校验
# ============================================================
class TestDeclaredTargets:
    def test_undeclared_rejected(self, monkeypatch, attacks_dir):
        import llmsec.core.config as config

        monkeypatch.setattr(config, "load_targets", lambda: {"real-model": object()})
        with pytest.raises(LaunchError, match="未在 .env TARGETS"):
            launch_evaluation(LaunchSpec(target="fake"))
        # 多目标同样校验（归一前 web 端只查单目标——此为修复项）
        with pytest.raises(LaunchError, match="fake2"):
            launch_evaluation(LaunchSpec(targets=["real-model", "fake2"]))

    def test_declared_passes(self, monkeypatch, attacks_dir, fake_start):
        import llmsec.core.config as config

        monkeypatch.setattr(config, "load_targets", lambda: {"real-model": object()})
        launch_evaluation(LaunchSpec(target="real-model", input_file="l1.jsonl"))
        assert fake_start[0]["kind"] == "evaluate"


# ============================================================
# build_eval_argv
# ============================================================
class TestBuildArgv:
    def _spec(self, **kw) -> LaunchSpec:
        return LaunchSpec(**kw)

    def test_single_target(self):
        argv = build_eval_argv(self._spec(target="a"), attack_rel="attacks/l1.jsonl")
        assert argv[:4] == ["-m", "llmsec.pipeline.runner", "--phase", "all"]
        assert argv[argv.index("--target") + 1] == "a"
        assert "--targets" not in argv and "--target-concurrency" not in argv
        assert argv[-1] == "--publish-global"

    def test_multi_target_auto_full_concurrency(self):
        argv = build_eval_argv(self._spec(targets=["a", "b", "c"]), attack_rel="x")
        assert argv[argv.index("--targets") + 1] == "a,b,c"
        assert argv[argv.index("--target-concurrency") + 1] == "3"

    def test_explicit_concurrency(self):
        argv = build_eval_argv(self._spec(targets=["a", "b"], target_concurrency=1), attack_rel="x")
        assert argv[argv.index("--target-concurrency") + 1] == "1"

    def test_optional_flags_omitted_when_none(self):
        argv = build_eval_argv(self._spec(target="a"), attack_rel="x")
        for flag in ("--batch-size", "--seed", "--twin-window", "--no-early-stop",
                     "--concurrency", "--sampler-alpha", "--sampler-beta",
                     "--sampler-gamma", "--coordinate-rounds"):
            assert flag not in argv, f"{flag} 应省略"

    def test_full_flags(self):
        argv = build_eval_argv(
            self._spec(target="a", batch_size=8, seed=7, twin_window=30, no_early_stop=True,
                       concurrency=4, sampler_alpha=1.5, sampler_beta=0.5,
                       sampler_gamma=2.0, coordinate_rounds=6),
            attack_rel="x",
        )
        assert argv[argv.index("--batch-size") + 1] == "8"
        assert argv[argv.index("--seed") + 1] == "7"
        assert argv[argv.index("--twin-window") + 1] == "30"
        assert "--no-early-stop" in argv
        assert argv[argv.index("--concurrency") + 1] == "4"
        assert argv[argv.index("--sampler-alpha") + 1] == "1.5"
        assert argv[argv.index("--coordinate-rounds") + 1] == "6"

    def test_publish_global_off(self):
        argv = build_eval_argv(self._spec(target="a", publish_global=False), attack_rel="x")
        assert "--publish-global" not in argv


# ============================================================
# launch_evaluation（集成路径，start_task 已拦截）
# ============================================================
class TestLaunchEvaluation:
    def test_happy_path_meta(self, attacks_dir, no_declared_targets, fake_start):
        view = launch_evaluation(LaunchSpec(target="m1", input_file="l1.jsonl", max_rounds=3))
        assert view["id"] == "fake-evaluate"
        call = fake_start[0]
        assert call["meta"] == {"targets": ["m1"], "max_rounds": 3,
                                "input": str(attacks_dir / "l1.jsonl").replace("\\", "/")}
        assert call["env_override"] is None

    def test_param_overrides_to_env(self, attacks_dir, no_declared_targets, fake_start):
        launch_evaluation(LaunchSpec(target="m1", input_file="l1.jsonl",
                                     param_overrides={"K_FACTOR": 32, "CONV_CI_TARGET": 15.0}))
        env = fake_start[0]["env_override"]
        assert env == {"LLMSEC_PARAM_K_FACTOR": "32", "LLMSEC_PARAM_CONV_CI_TARGET": "15.0"}

    def test_env_snapshot_missing(self, attacks_dir, no_declared_targets, fake_start):
        with pytest.raises(LaunchError, match="env 快照不存在"):
            launch_evaluation(LaunchSpec(target="m1", input_file="l1.jsonl", env_snapshot="ghost"))
        assert fake_start == [], "校验失败不应启动任务"

    def test_attack_missing_before_declared_check(self, monkeypatch, attacks_dir, fake_start):
        # 顺序契约：攻击集不存在先报（错误消息兼容 MCP 既有测试）
        import llmsec.core.config as config

        monkeypatch.setattr(config, "load_targets", lambda: {"real": object()})
        with pytest.raises(LaunchError, match="攻击集不存在"):
            launch_evaluation(LaunchSpec(target="m1", input_file="nope.jsonl"))


# ============================================================
# launch_hpo_study
# ============================================================
class TestLaunchHpo:
    def _patch_root(self, monkeypatch, tmp_path):
        import llmsec.core.config as config

        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        return tmp_path

    def test_happy_path(self, monkeypatch, tmp_path, fake_start):
        root = self._patch_root(monkeypatch, tmp_path)
        f = root / "experiments" / "s.yaml"
        f.parent.mkdir(exist_ok=True)
        f.write_text("name: x\n", encoding="utf-8")
        view = launch_hpo_study(str(f))
        assert view["id"] == "fake-hpo"
        assert fake_start[0]["argv"] == ["-m", "llmsec.experiments", "run", str(f)]
        assert fake_start[0]["meta"] == {"study": "s.yaml"}

    def test_relative_path(self, monkeypatch, tmp_path, fake_start):
        self._patch_root(monkeypatch, tmp_path)
        (tmp_path / "s.yaml").write_text("name: x\n", encoding="utf-8")
        launch_hpo_study("s.yaml")
        assert fake_start[0]["kind"] == "hpo"

    def test_not_found(self, monkeypatch, tmp_path, fake_start):
        self._patch_root(monkeypatch, tmp_path)
        with pytest.raises(LaunchError, match="study 文件不存在"):
            launch_hpo_study(tmp_path / "nope.yaml")

    def test_wrong_suffix(self, monkeypatch, tmp_path, fake_start):
        self._patch_root(monkeypatch, tmp_path)
        f = tmp_path / "s.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(LaunchError):
            launch_hpo_study(f)

    def test_outside_repo_rejected(self, monkeypatch, tmp_path, fake_start):
        self._patch_root(monkeypatch, tmp_path / "repo")
        f = tmp_path / "outside.yaml"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(LaunchError, match="仓库目录内"):
            launch_hpo_study(f)
        assert fake_start == []


# ============================================================
# task_manager meta 透传（真实 start_task，smoke 子进程秒级结束）
# ============================================================
def test_task_view_exposes_meta():
    # 独立 kind：避免与其它套件的 smoke 任务共享串行队列（xdist 同 worker 乱序）
    view = task_manager_mod.start_task(
        "launchmeta", ["-c", "print('meta-ok')"], meta={"targets": ["m1"], "max_rounds": 5})
    assert view["meta"] == {"targets": ["m1"], "max_rounds": 5}
    # start_task 不传 meta → view.meta 为 None（旧调用方零影响）
    view2 = task_manager_mod.start_task("launchmeta", ["-c", "print('meta-none')"])
    assert view2["meta"] is None
    # 排空到终态：残留 running 任务会阻塞同 kind 后续测试的队列
    import time as _time

    deadline = _time.time() + 15
    ids = {view["id"], view2["id"]}
    while _time.time() < deadline:
        statuses = {v["id"]: v["status"] for v in task_manager_mod.list_tasks() if v["id"] in ids}
        if statuses and all(s in ("success", "failed", "cancelled") for s in statuses.values()):
            break
        _time.sleep(0.1)
