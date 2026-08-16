# 测试说明（tests/）

本目录是 llmsec 的回归 + 冒烟测试集，**按被测子系统组织**（文件名 = 被测模块名）。分两层：
- **离线测试**（默认）：mock/stub 目标与 Judge，秒级、零费用、`-n 4` 并行。
- **真实 API / 端到端测试**（默认排除）：打真实外部 API，产生费用，需手动触发。

## 怎么跑

```bash
pytest tests/                               # 离线全量（默认 -n 4 并行，真实 API/e2e 自动排除）
pytest tests/test_elo.py                    # 单文件
pytest tests/test_elo.py::test_convergence_true_positive   # 单个用例
pytest -n auto                              # 强制更多 worker（注意冷启动开销）

# —— 覆盖率 ——
pytest tests/ --cov=llmsec --cov=control --cov-report=term   # 需 pip install pytest-cov
# 注意：--cov 只能用包名（llmsec / control），不要用点分子模块（如 --cov=llmsec.mcp）。
# 根因（numpy 2.4 + coverage 7.15 实测可独立复现）：coverage 对点分子模块源的解析
# 会在启动期额外 import 一遍包链（llmsec/__init__ → numpy），与后续正常导入叠加，
# 触发 numpy C 扩展（_multiarray_umath）的"cannot load module more than once per
# process"二次初始化守卫，conftest 加载即崩。包名形式不触发；要看子模块覆盖率用
# 包形式跑完整表再 grep，或 coverage run --source=llmsec/mcp（路径形式）。

# —— 真实 API / 端到端测试（默认不跑，需手动触发）——
pytest -m real_api tests/test_real_api.py -v -n 0    # 真实目标/Judge/Generator 连通性
pytest -m e2e tests/test_e2e_dashboard.py -v -n 0    # 看板评估全流程（子进程 + 真实 API）
```

- `conftest.py` 统一做路径注入 + Windows 控制台 UTF-8 + 网络隔离。
- **离线测试**：autouse fixture `_isolate_network_env` 每个用例清空网络 env，保证不打外部 API。
- **真实 API 测试**：标 `@pytest.mark.real_api`，fixture 不清 env（保留 .env 凭证）；`require_real_api` fixture 在无凭证时优雅 skip。

## 真实 API / 端到端测试

默认被 `addopts = "-m 'not real_api and not e2e'"` 排除，`pytest tests/` 行为与离线套件完全一致。

| marker | 文件 | 测什么 | 触发方式 |
|---|---|
| `real_api` | `test_real_api` | 目标模型/Judge/Generator 真实连通性 + OpenAI 契约 + 安全基线（拒绝有害）| `pytest -m real_api` |
| `e2e` | `test_e2e_dashboard` | 看板 `POST /api/run/evaluate` → 子进程 → 进度轮询 → 取消，全链路真实 API | `pytest -m e2e` |
| `e2e` | `test_tui_app` | Textual TUI 无头冒烟（轮询→表格更新/面板切换/发起表单弹窗） | `pytest -m e2e` |

**前提**：项目 `.env` 已配真实凭证（`TARGET_API_KEY`/`TARGET_BASE_URL`/`GENERATOR_API_KEY` 非占位符 `sk-yyy`/`sk-xxx`）。未配置时用例自动 skip 并给出提示。

**费用**：`real_api` 约 3~5 次 API 调用（极低）；`e2e` 看板约 2~4 次目标 + 2~4 次 Judge；`e2e` TUI 仅轮询本地文件。

**隔离**：`e2e` 看板测试通过 `isolated_tasks` fixture patch `tasks._start_task`，把 `--publish-global` 改写为 `--work-dir <tmp>`，全局 `output/state/results.json` 零污染。

**建议串行**（`-n 0`）：真实 API 并行易触发限速；子进程轮询不适合 xdist worker。
> 注意：禁用并行用 `-n 0`（覆盖 addopts 的 `-n 4`），**不要**用 `-p no:xdist`（会让 addopts 的 `-n 4` 因插件缺失而报错）。

## 文件清单（按子系统分组，完整）

用例数不在本表硬编码（曾随测试演进失真）——**以 [`tests/INVENTORY.md`](INVENTORY.md) 为准**：由 `scripts/gen_test_inventory.py` 从 `pytest --collect-only` 生成，CI 会校验不过期。

### 评估核心（evaluation/）

| 文件 | 测什么 |
|---|---|
| test_elo | ELO 收敛判据（CI 口径 + drift/CI 双阈值）、边界健壮性（零场次/字符串分数/σ² 下限）、R-cutover 派生访问层（指纹缓存失效/迁移幂等/tracker 进程内缓存/active_model）、derive_elo 按 round 重建、成功率窗口按 distinct 方法去重、多防御方未指定报错
| test_predictors | SVD-Ridge 批量预测精度（MAP 不确定性/退化列封顶/模型缓存）、BlendPredictor 双层收缩、发现层指纹 + 相似度加权池化、加权 K-Fold 泄漏 + σ² 有效样本量自由度
| test_results_matrix | 结果矩阵 R（唯一真相）upsert/时序/round-trip + 多目标扫描 + derive_elo 幂等
| `test_units` | 簇粒度单元化（unit 指纹/构建/特征质心/assemble）+ R v2 记录级 schema（extra.unit 聚合）|
| test_samplers | 四种采样器（gap/infogain/coordinate/hybrid）参数生效性、坐标下降簇轮询/边界聚焦/耗尽补足、已测方法历史成功率口径、空候选短路
| test_evaluator | 断点筛选（ID 数值序/段级匹配）、有害记录 eval_score>0、judge_mode 三模式标签、token_ratio=None 口径、H/S/D 均值只计真 Judge 记录、_eval_no_judge ≥2 命中判拒
| test_scoring | compliance 去宽泛兜底、B 级折扣、JUDGE_MODEL 回退、有害度钳位、math 取末标签、短响应不判 empty、解析回退大小写、judge_calls 线程隔离
| test_prescreen_ml | TF-IDF+LogReg 拒绝预筛：无模型回退、训练主路径（足量数据→落盘→predict）、类别不平衡拒绝、损坏模型回退、时间序留出评估三守卫
| `test_jailbreak_tax` | 越狱税全链路（注入/剥离一致性、math_score 三档、哨兵不测税、聚合）|
| test_allergy | 过敏检测窗口选取（一侧不足向另一侧借、上方按 Elo 距离取最近）、OR 口径（关键词命中即过敏）、safe_twin 缺键防护
| test_retry | 统一重试 retry_call（成功/前 N-1 失败/全失败/retry_on/on_retry）+ 各模块重试参数冒烟

### 聚类（clustering/）

| 文件 | 测什么 |
|---|---|
| test_clustering | 阻尼白化空间方差 + HDBSCAN 端到端 + auto-k（已知簇数 ±2）+ ANOVA 簇效 + 弱监督加权 + D-optimal 种子覆盖 + embedding 降级链 + 契约健壮性
| `test_precluster_hdb` | 预聚类复用 HDBSCAN 核心（不触发命名/画像）、簇恢复 ARI、确定性、ImportError/核心异常→KMeans 回退→双失败 None（无 hdbscan 环境也跑回退用例）|
| test_embedding_cache | embedding 磁盘缓存：二次全命中零 encode、部分命中只编码新增、缓存键随 source 变化

### 管线与目标（pipeline/ + targets/ + attacks/）

| 文件 | 测什么 |
|---|---|
| `test_runner` | runner 集成（增量落盘/publish_tracker/Phase2 降级/多目标 canonical report）|
| test_run_issues | 运行期问题：全量/部分 resume 从 R 回放、blend 缓存键稳定、fallback 报告、sweep 外扩、并发写安全
| test_pipeline_review | 终审交接：no_early_stop/--work-dir 透传、phase2 无 state 报错、过敏早退零调用、r_snapshot 不读活 R
| test_targets | legacy fallback 注册 + 路由可达性 + 空配置兜底、并发单例客户端、pcap 重试/4xx 短路、env 惰性读
| test_targets_backends | openai_backend / local_sim 后端行为
| test_generators | generate 注入越狱税 + 5 类危害、MD 解析/类别轮转/两轮生成重试与退回、main 全流程 + 断点续传、harmbench functional_category、safe_twin --no-generate
| test_probe | 探测脚本路由分发（openai/pcap）、字段扫描递归、SSL/连接/超时错误分支
| `test_isolation` | rebind_to_workdir 把 9 组产物路径全部重绑（单元化隔离核心契约）|
| `test_taxonomy` | harm_type 归一化（标准词直通/HarmBench 别名/中文别名/批量去重）|
| test_progress | core.progress 并发写、OSError 静默、attack_phase 落盘字段口径

### 服务与看板（server/）

| 文件 | 测什么 |
|---|---|
| test_dashboard | Web 面板全 API 冒烟 + 路径校验 + 任务生命周期 + HPO/趋势/SSE/报告下载 + 任务状态刷新/LRU/上限校验 + task_manager core（队列串行/queued 与已结束取消/超限淘汰/僵尸告警/Popen 失败/env_override 注入/失败告警/日志与进度容错）+ 攻击集上传（后缀/空文件/首行 JSON/防穿越）+ /api/env 掩码与 .env 原子更新 + /api/targets/add 四件套追加
| `test_hpo_router` | HPO 看板端点（study 创建/状态/trial 明细）|
| `test_launch` | 统一启动层（LaunchSpec 校验/攻击集防穿越+后缀/目标声明校验/argv 构造含默认全并发/env 快照+param_overrides 注入/meta 携带/HPO 路径校验）|

### 实验框架（experiments/）

| 文件 | 测什么 |
|---|---|
| `test_experiments` | HPO 框架（schema 解析、grid/random/bayesian、指标聚合、work-dir 隔离、续跑）+ metrics（产物定位双布局、state 回放收敛轮、未收敛惩罚、aggregate 口径）|

### 管理面（management/）

| 文件 | 测什么 |
|---|---|
| test_management | management 包：remove_model/record 行列清理、runs 发现/过滤/垃圾检测/软删除、caches 清单/clean dry-run→软删/legacy 判定、snapshot 导出（global/run: 源、tar.gz 打包、越界拒绝）、merge、CLI
| test_merge | management.merge plan/execute/dry-run 不写盘/多源/models 过滤/ws↔ws/同记录覆盖

### 基础设施（core/）

| 文件 | 测什么 |
|---|---|
| `test_core_infra` | io（read_json 容错/write_jsonl 原子重试/append/write_json NaN 拒绝与备份/save_artifact/write_csv）+ monitoring（事件落盘/去抖/级别过滤/webhook 通道/异常兜底/标准告警包装）|
| `test_print_pdf` | 前端打印管线静态契约（PRINT_PARTS/@page/卸载钩子接线）|

### MCP 工具层（mcp/）

| 文件 | 测什么 |
|---|---|
| test_mcp | 纯函数（混淆/math/特征/指标聚合）、两步确认 token 生命周期（一次性/peek/TTL）、run_evaluation 参数校验、create_server 注册
| test_mcp_tools | query 工具（runs 发现与过滤/对比/报告读取防穿越/elo 四件/allergy/targets 掩码探活/plan/gazette/workspace/cluster/params）、actions 工具（merge spec 校验全分支/delete/clean/merge 全流程、env_snapshot CRUD、fork/gc/export）、tasks 工具（status/log/progress/cancel/run_evaluation argv 与 env 注入/orchestrate）、server main 双模式 + 54 工具注册核对

### 终端界面（tui/）

| 文件 | 测什么 |
|---|---|
| `test_tui_render` | 字符渲染（盲文条边界/OLS 平滑单调性/sparkline 归一化/行格式/CJK 对齐/进度回放 done-active 判定/占位声明）|
| `test_tui_widgets` | 自定义 widget（盲文 sparkline/进度条等纯逻辑：归一化/边界/空数据）|
| test_tui_panels | 面板层（tasks/runs/hpo 行构造与刷新逻辑、公共辅助），离线组合测试
| `test_tui_task_store` | TUI 任务状态层（磁盘扫描 detached 视图/增量 tail 回放含半行/损坏行/meta 占位/study 启动校验）|

### 控制层（control/，三省架构）

| 文件 | 测什么 |
|---|---|
| test_shangshu_menxia | 三省架构主干：bus 发布订阅/过滤/异常隔离、plan 拓扑分层/持久化、capabilities 清单契约、env_snapshot CRUD/编辑/merge/备份、menxia assess_step 判据、executor 封驳+依赖传播
| test_control | control 模块层主干：invoker argv/PYTHONUNBUFFERED、compare/run_metrics/ws: 前缀观测、workspace fork/mark_merged、tools schema 分发、fallback 意图解析、session CRUD/滑窗、menxia review 判据/阈值/全流程
| test_control_router | control HTTP 端点冒烟（llm-status/tools/index）、chat 三模式兜底、fork→list→delete/重复 400/merge dry-run 沙箱闭环
| `test_queue_menxia` | PlanQueue submit/duplicate/status/cancel + 门下省三阶段订阅（plan_drafted/approved → review 报告）|
| `test_gazette` | 文牍存储基础行为（append/read、read_plan_context 重建、list 过滤、空查询）|

### 报告（reporting/）

| 文件 | 测什么 |
|---|---|
| test_report | 多源结果加载、_load_elo_tracker 优先级、叙事 timeout 透传/空响应与异常退回 fallback、load_elo 缺省活跃模型、load_allergy 按模型分文件与 run 回退、fallback 报告三段渲染

### 回归套件（历史审计批次，按修复来源保留）

| 文件 | 来源 |
|---|---|
| `test_core_regressions` | P0 致命修复 F1-F7（math 哨兵/弱监督位置/技术标签/括号反转/template hash）|
| `test_correctness` | P2 正确性（tree inf/NaN 守卫/conv_rounds 恢复/report inconclusive/K-Fold/predict schema）|
| `test_data_integrity` | P0 数据完整性（NaN 校验/原子写/多目标映射/local_sim 数学优先级）|
| `test_review_regressions` | 全面审查 FR1-FR13（落盘顺序/logger/stale GT/boundary 键集/no-judge 兜底/dedup）|
| test_fix_judge_none | judge/generate/safe_twin 的 reasoning_content 回退、重试确定性、探活 chat smoke、报告/配置口径等 14 类契约
| `test_audit_r1_high` | 第 1 轮审查 H1-H14（种子路由/扁平 fpr/TASKS 注册/发现缓存/start-from 数值序/打包/PlanQueue/封驳回执/SSE/.env 原子写/gazette 并发）|
| `test_audit_r2_control` | 第 2 轮审查 control M1-M13（plan 校验/fallback 异常转文案/session 配对/store 自治/阈值 TTL/invoker 超时/env 备份唯一）|
| `test_audit_r3_llmsec` | 第 3 轮审查 L1-L8（_validate_run 防穿越/hpo study 名/tell 失败归还/write_jsonl 并发/log_tail/500 脱敏/前端卸载钩子）|
| `test_audit_r4_cleanup` | 第 4 轮清理守卫（死端点 404/探活统一/write_csv 逗号/KMeans 兜底统一/list_all_runs 口径）|
| `test_audit_r6_root` | R6 收口（session 会话锁/gazette unblocked 标记/partition_publish_names/fsig 签名/ModelsProbeResult 契约/update 写回）|

### 真实 API / 端到端（默认排除，手动触发）

| 文件 | marker | 测什么 |
|---|---|
| `test_real_api` | `real_api` | 目标模型真实连通性 + OpenAI 契约 + 安全基线（拒绝有害）；Judge 评分契约；Generator 客户端可达 |
| `test_e2e_dashboard` | `e2e` | 看板 `POST /api/run/evaluate` 全链路（子进程 + 真实 API）：触发→轮询→success；progress.jsonl 进度记录；`POST /cancel` 任务取消 |
| `test_tui_app` | `e2e` | Textual 无头冒烟（轮询→表格更新/面板切换/发起表单弹窗）|

## 速查：出问题该看哪个测试

| 症状 | 看这里 |
|---|---|
| 收敛提前/置信度虚高 | `test_elo` |
| 换模型后 Elo/FPR 串味 | `test_elo`、`test_results_matrix` |
| 采样旋钮调了没反应 | `test_samplers` |
| 聚类质量/簇效 | `test_clustering`、`test_precluster_hdb` |
| 预测值离谱/CI 爆炸 | `test_predictors` |
| JSON 损坏/丢数据 | `test_data_integrity`、`test_core_infra` |
| 看板接口/任务 | `test_dashboard` |
| 目标路由/legacy .env | `test_targets` |
| 越狱税/数学探针 | `test_jailbreak_tax` |
| MCP 工具行为 | `test_mcp`、`test_mcp_tools` |
| TUI 显示/任务视图 | `test_tui_render`、`test_tui_widgets`、`test_tui_panels` |
| 三省架构/工作流 | `test_shangshu_menxia`、`test_control`、`test_queue_menxia` |
| 目标/Judge 真实不通 | `test_real_api`（`-m real_api`）|
| 看板评估端到端跑不通 | `test_e2e_dashboard`（`-m e2e`）|

## 维护约定

- **新测试优先放进既有子系统文件**（文件名 = 被测模块名）；只有新子系统才开新文件。
- **修复批次类用例**（audit/review/issue）留在对应批次文件里，但**墓碑用例**（断言"已删符号仍不存在"）在清理期结束后应删除——它们永不发现新 bug。
- 小而散的评审文件（如历史上的 `test_eval_review_*`）已并入对应子系统文件；新增请直接归位，不再单独开文件。
- **用例数表 CI 生成**：新增/删除测试后运行 `python scripts/gen_test_inventory.py` 刷新 `tests/INVENTORY.md` 并一并提交——CI 的 `--check` 会拦截过期清单。
