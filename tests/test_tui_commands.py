"""commands.py 转译层单测——parse / complete / fuzzy 纯逻辑。

动作分发与后端映射在 test_tui_console.py（run_test 环境）覆盖；
本文件只测语法面：分词、旗标类型、资源校验、纠错阈值、补全位置感知。
"""

from __future__ import annotations

from llmsec.tui.commands import (
    REGISTRY,
    complete,
    looks_natural,
    parse,
    strong_match,
    usage,
    weak_matches,
)


# ============================================================
# parse：旗标与类型
# ============================================================
class TestParseFlags:
    def test_eval_full(self):
        p = parse("eval -t glm4,glm4-air -r 5 --sampler hybrid --seed 42 --param K=V,X=1")
        assert p.ok
        assert p.name == "eval"
        assert p.values["target"] == ["glm4", "glm4-air"]
        assert p.values["max-rounds"] == 5
        assert p.values["sampler"] == "hybrid"
        assert p.values["seed"] == 42
        assert p.values["param"] == {"K": "V", "X": "1"}

    def test_eval_repeated_flag_aggregates(self):
        p = parse("eval -t a -t b --t c")
        # --t 是 --target 的长名前缀吗？不是：--t 未知旗标
        assert "未知旗标 --t" in p.errors[0]

    def test_eval_repeatable_target(self):
        p = parse("eval -t a -t b")
        assert p.ok
        assert p.values["target"] == ["a", "b"]

    def test_eval_requires_target_or_all(self):
        p = parse("eval")
        assert not p.ok
        assert any("target" in e for e in p.errors)

    def test_eval_all_flag(self):
        p = parse("eval --all -r 3")
        assert p.ok
        assert p.values["all"] is True

    def test_flag_equals_form(self):
        p = parse("eval --target=x --max-rounds=3")
        assert p.ok
        assert p.values["target"] == ["x"]
        assert p.values["max-rounds"] == 3

    def test_bool_flag(self):
        p = parse("eval -t x --no-early-stop")
        assert p.ok and p.values["no-early-stop"] is True

    def test_int_type_error(self):
        p = parse("eval -t x -r abc")
        assert any("整数" in e for e in p.errors)

    def test_float_flag(self):
        p = parse("eval -t x --sampler-alpha 0.5")
        assert p.ok and p.values["sampler-alpha"] == 0.5
        p = parse("eval -t x --sampler-beta zz")
        assert any("数字" in e for e in p.errors)

    def test_kv_bad_segment(self):
        p = parse("eval -t x --param abc")
        assert any("KEY=V" in e for e in p.errors)

    def test_unknown_flag_suggests_nearest(self):
        p = parse("eval -t x --taret glm4")
        assert any("是 --target" in e for e in p.errors)

    def test_missing_flag_value(self):
        p = parse("eval -t x -r")
        assert any("缺值" in e for e in p.errors)


class TestParseShellVerbs:
    def test_ls_defaults(self):
        p = parse("ls")
        assert p.ok and p.positionals == []

    def test_ls_combined_short_bools(self):
        p = parse("ls -al tasks")
        assert p.ok
        assert p.values["all"] is True and p.values["long"] is True
        assert p.positionals == ["tasks"]

    def test_ls_resource_correction(self):
        p = parse("ls taks")
        assert p.ok
        assert p.corrections == ["taks → tasks"]
        assert p.positionals == ["tasks"]

    def test_ls_unknown_resource(self):
        p = parse("ls zzz")
        assert not p.ok
        assert any("未知资源" in e for e in p.errors)

    def test_ls_runs_path_kept(self):
        p = parse("ls runs/glm4")
        assert p.ok and p.positionals == ["runs/glm4"]

    def test_cat_prefix_correction(self):
        p = parse("cat task/ab12")
        assert p.ok
        assert p.positionals == ["tasks/ab12"]
        assert p.corrections == ["task → tasks"]

    def test_cat_bad_prefix(self):
        p = parse("cat foo/bar")
        assert not p.ok
        assert any("tasks/<id前缀>" in e for e in p.errors)

    def test_cat_missing_arg(self):
        p = parse("cat")
        assert any("缺参数" in e for e in p.errors)

    def test_rm_variadic(self):
        p = parse("rm run_a run_b run_c")
        assert p.ok
        assert p.positionals == ["run_a", "run_b", "run_c"]

    def test_extra_positional_error(self):
        p = parse("help a b")
        assert any("多余参数" in e for e in p.errors)


class TestParseCommands:
    def test_multiword_command(self):
        p = parse("snapshot set ws1 GENERATOR_MODEL=x")
        assert p.ok and p.name == "snapshot set"
        assert p.positionals == ["ws1", "GENERATOR_MODEL=x"]

    def test_multiword_needs_subcommand(self):
        p = parse("snapshot")
        assert not p.ok
        assert any("需要子命令" in e for e in p.errors)

    def test_aliases(self):
        assert parse("q").name == "quit"
        assert parse("exit").name == "quit"
        assert parse("evaluate -t x").name == "eval"

    def test_agent_quoted_text(self):
        p = parse('/agent "对比 run1 和 run2"')
        assert p.ok and p.name == "/agent"
        assert p.positionals == ["对比 run1 和 run2"]

    def test_agent_bare_words(self):
        p = parse("/agent 列出 最近的 run")
        assert p.positionals == ["列出", "最近的", "run"]

    def test_unmatched_quote(self):
        p = parse('/agent "未闭合')
        assert any("引号" in e for e in p.errors)

    def test_empty(self):
        assert parse("   ").errors == ["空输入"]


# ============================================================
# parse：拼写纠错
# ============================================================
class TestFuzzyParse:
    def test_verb_corrected_lss(self):
        p = parse("lss tasks")
        assert p.ok and p.name == "ls"
        assert p.corrections == ["lss → ls"]

    def test_verb_corrected_transposition(self):
        p = parse("evla -t x")
        assert p.ok and p.name == "eval"
        assert p.corrections == ["evla → eval"]

    def test_slash_verb_corrected(self):
        p = parse("/agnt 你好")
        assert p.ok and p.name == "/agent"
        assert p.corrections == ["/agnt → /agent"]

    def test_unknown_verb_lists_candidates(self):
        p = parse("zzzzzz")
        assert not p.ok
        assert any("未知命令" in e for e in p.errors)

    def test_verb_correction_only_once(self):
        # 纠错后第二词仍非法资源 → 走资源校验（不再整体失败于未知命令）
        p = parse("lss zzz")
        assert p.name == "ls"
        assert p.corrections == ["lss → ls"]
        assert any("未知资源" in e for e in p.errors)


class TestFuzzyMatchers:
    def test_strong_distance1(self):
        assert strong_match("lss", ["ls", "top", "rm"]) == "ls"

    def test_strong_transposition(self):
        assert strong_match("evla", ["eval"]) == "eval"

    def test_strong_exact(self):
        assert strong_match("ls", ["ls"]) == "ls"

    def test_strong_none_when_far(self):
        assert strong_match("zzzz", ["ls", "eval"]) is None

    def test_strong_none_when_ambiguous(self):
        # apple / aples 与 aple 距离都为 1 → 二义，不自动纠错
        assert strong_match("aple", ["apple", "aples"]) is None

    def test_weak(self):
        assert "eval" in weak_matches("evl", ["eval", "ls", "top"])


# ============================================================
# looks_natural：自然语言判定（agent 模式自动进入）
# ============================================================
class TestLooksNatural:
    def test_cjk_question(self):
        assert looks_natural("你好，能告诉我ASR是什么")
        assert looks_natural("列出最近的run")  # 无问号有汉字

    def test_ascii_question(self):
        assert looks_natural("what is the asr metric?")

    def test_multiword_english_sentence(self):
        # 多词且首词与命令动词毫无相似 → 句子而非拼错的命令
        assert looks_natural("show me the best model please")

    def test_single_unknown_token_not_natural(self):
        # 手滑的单个未知词 → 保留 did-you-mean 纠错路径
        assert not looks_natural("asdf")

    def test_known_command_not_natural(self):
        assert not looks_natural("ls tasks")

    def test_typo_of_command_not_natural(self):
        # 近似命令（evl→eval）像拼错，不进 agent
        assert not looks_natural("evl -t x")

    def test_empty_and_flags(self):
        assert not looks_natural("")
        assert not looks_natural("--all")


# ============================================================
# complete：位置感知
# ============================================================
class TestComplete:
    def _labels(self, r):
        return [c.label for c in r.items]

    def test_empty_line_lists_commands(self):
        r = complete("")
        labels = self._labels(r)
        # 空行列出命令（8 条上限，shell 动词在前）；字母前缀过滤可达其余
        assert "ls" in labels
        assert "eval" in self._labels(complete("e"))
        assert "/agent" in self._labels(complete("/"))

    def test_first_token_prefix(self):
        r = complete("l")
        assert self._labels(r) == ["ls"]

    def test_slash_namespace(self):
        r = complete("/a")
        assert self._labels(r) == ["/agent"]
        r = complete("/z")
        assert r.items == [] and r.hint_error

    def test_subcommand_completion(self):
        r = complete("snapshot ")
        assert set(self._labels(r)) == {"snapshot list", "snapshot new", "snapshot set", "snapshot rm"}
        r = complete("snapshot n")
        assert self._labels(r) == ["snapshot new"]

    def test_flag_name_completion(self):
        r = complete("ls -")
        labels = self._labels(r)
        assert "--all" in labels and "--long" in labels
        r = complete("eval --tar")
        assert self._labels(r) == ["--target"]

    def test_flag_value_completion_prev_token(self):
        r = complete("eval -t ", sources={"targets": lambda: ["glm4", "glm4-air"]})
        assert self._labels(r) == ["glm4", "glm4-air"]

    def test_flag_value_completion_inline_equals(self):
        r = complete("eval --target=gl", sources={"targets": lambda: ["glm4", "glm4-air"]})
        assert self._labels(r) == ["--target=glm4", "--target=glm4-air"]

    def test_positional_completion(self):
        r = complete("kill ", sources={"taskids": lambda: ["eval-1430-a1b2"]})
        assert self._labels(r) == ["eval-1430-a1b2"]

    def test_cat_path_completion(self):
        r = complete("cat ", sources={"cat_objects": lambda: ["tasks/eval-1-ab", "runs/run_x"]})
        assert set(self._labels(r)) == {"tasks/eval-1-ab", "runs/run_x"}

    def test_ls_resource_completion_with_paths(self):
        src = {"ls_resources": lambda: ["tasks", "runs"] + [f"runs/{t}" for t in ("glm4",)]}
        r = complete("ls runs/g", sources=src)
        assert self._labels(r) == ["runs/glm4"]

    def test_positional_prefix_filter(self):
        r = complete("kill ev", sources={"taskids": lambda: ["eval-1", "hpo-2"]})
        assert self._labels(r) == ["eval-1"]

    def test_variadic_uses_last_arg_completer(self):
        r = complete("rm run_", sources={"runs": lambda: ["run_a", "run_b"]})
        assert self._labels(r) == ["run_a", "run_b"]

    def test_unknown_command_hint_error(self):
        r = complete("zzzz ")
        assert r.hint_error

    def test_hint_usage(self):
        r = complete("eval ")
        assert "eval" in r.hint and "红队评估" in r.hint


# ============================================================
# usage
# ============================================================
class TestUsage:
    def test_usage_shape(self):
        u = usage(REGISTRY["compare"])
        assert u.startswith("compare <a> <b>")
