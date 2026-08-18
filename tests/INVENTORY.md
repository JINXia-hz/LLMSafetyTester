# 测试清单（自动生成，勿手改）

> 由 `scripts/gen_test_inventory.py` 生成于 2026-08-18T12:59:56；
> CI 会校验本文件与实际收集结果一致（`--check`），过期即失败。
> 本地刷新：`python scripts/gen_test_inventory.py`。

合计 **70** 个文件 / **1028** 个用例（含 parametrize 展开；
含默认排除的 real_api/e2e 用例——它们需手动 `pytest -m real_api` / `-m e2e` 触发）。

| 测试文件 | 用例数 |
|---|---|
| tests/test_allergy.py | 8 |
| tests/test_audit_r1_high.py | 17 |
| tests/test_audit_r2_control.py | 12 |
| tests/test_audit_r3_llmsec.py | 7 |
| tests/test_audit_r4_cleanup.py | 5 |
| tests/test_audit_r6_root.py | 7 |
| tests/test_audit_r7_cleanup.py | 8 |
| tests/test_audit_r7_eval.py | 6 |
| tests/test_audit_r7_high.py | 2 |
| tests/test_audit_r7_isolation.py | 5 |
| tests/test_audit_r7_server.py | 12 |
| tests/test_audit_r7_storage.py | 8 |
| tests/test_audit_r8_rootfix.py | 11 |
| tests/test_audit_r9_guard.py | 10 |
| tests/test_clustering.py | 17 |
| tests/test_control.py | 42 |
| tests/test_control_router.py | 10 |
| tests/test_core_infra.py | 26 |
| tests/test_core_regressions.py | 10 |
| tests/test_correctness.py | 17 |
| tests/test_dashboard.py | 34 |
| tests/test_data_integrity.py | 17 |
| tests/test_e2e_dashboard.py | 3 |
| tests/test_elo.py | 23 |
| tests/test_embedding_cache.py | 4 |
| tests/test_env_example.py | 2 |
| tests/test_evaluation_failure_modes.py | 13 |
| tests/test_evaluator.py | 11 |
| tests/test_experiments.py | 18 |
| tests/test_fix_judge_none.py | 33 |
| tests/test_gazette.py | 4 |
| tests/test_generators.py | 10 |
| tests/test_hpo_router.py | 6 |
| tests/test_isolation.py | 4 |
| tests/test_jailbreak_tax.py | 8 |
| tests/test_launch.py | 27 |
| tests/test_management.py | 33 |
| tests/test_mcp.py | 30 |
| tests/test_mcp_tools.py | 61 |
| tests/test_merge.py | 9 |
| tests/test_pipeline_review.py | 10 |
| tests/test_precluster_hdb.py | 10 |
| tests/test_predictors.py | 17 |
| tests/test_prescreen_ml.py | 6 |
| tests/test_print_pdf.py | 4 |
| tests/test_probe.py | 7 |
| tests/test_progress.py | 3 |
| tests/test_queue_menxia.py | 8 |
| tests/test_real_api.py | 6 |
| tests/test_report.py | 9 |
| tests/test_results_matrix.py | 4 |
| tests/test_retry.py | 11 |
| tests/test_review_regressions.py | 17 |
| tests/test_rstore.py | 8 |
| tests/test_run_issues.py | 13 |
| tests/test_runner.py | 4 |
| tests/test_samplers.py | 13 |
| tests/test_scoring.py | 11 |
| tests/test_shangshu_menxia.py | 31 |
| tests/test_storage_catalog.py | 16 |
| tests/test_targets.py | 12 |
| tests/test_targets_backends.py | 5 |
| tests/test_taxonomy.py | 37 |
| tests/test_tui_app.py | 1 |
| tests/test_tui_commands.py | 55 |
| tests/test_tui_console.py | 29 |
| tests/test_tui_render.py | 31 |
| tests/test_tui_task_store.py | 25 |
| tests/test_tui_widgets.py | 28 |
| tests/test_units.py | 7 |
