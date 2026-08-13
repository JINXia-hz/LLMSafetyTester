# 测试说明（tests/）

本目录是 llmsec 的回归 + 冒烟测试集，按被测子系统组织。分两层：
- **离线测试**（默认）：mock/stub 目标与 Judge，秒级、零费用、`-n 4` 并行。
- **真实 API / 端到端测试**（默认排除）：打真实外部 API，产生费用，需手动触发。

## 怎么跑

```bash
pytest tests/                               # 离线全量（默认 -n 4 并行，真实 API/e2e 自动排除）
pytest tests/test_elo.py                    # 单文件
pytest tests/test_elo.py::test_convergence_true_positive   # 单个用例
pytest -n auto                              # 强制更多 worker（注意冷启动开销）

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
|---|---|---|---|
| `real_api` | `test_real_api` | 目标模型/Judge/Generator 真实连通性 + OpenAI 契约 + 安全基线（拒绝有害）| `pytest -m real_api` |
| `e2e` | `test_e2e_dashboard` | 看板 `POST /api/run/evaluate` → 子进程 → 进度轮询 → 取消，全链路真实 API | `pytest -m e2e` |

**前提**：项目 `.env` 已配真实凭证（`TARGET_API_KEY`/`TARGET_BASE_URL`/`GENERATOR_API_KEY` 非占位符 `sk-yyy`/`sk-xxx`）。未配置时用例自动 skip 并给出提示。

**费用**：`real_api` 约 3~5 次 API 调用（极低）；`e2e` 约 2~4 次目标 + 2~4 次 Judge（`batch=2 + rounds=1`，最小攻击集 `_unit_smoke.jsonl`）。

**隔离**：`e2e` 测试通过 `isolated_tasks` fixture patch `tasks._start_task`，把 `--publish-global` 改写为 `--work-dir <tmp>`，全局 `output/state/results.json` 零污染。

**建议串行**（`-n 0`）：真实 API 并行易触发限速；子进程轮询不适合 xdist worker。
> 注意：禁用并行用 `-n 0`（覆盖 addopts 的 `-n 4`），**不要**用 `-p no:xdist`（会让 addopts 的 `-n 4` 因插件缺失而报错）。

## 文件清单（20 个，按子系统分组）

### 评估核心

| 文件 | tests | 测什么 |
|---|---|---|
| `test_elo` | 13 | ELO 收敛判据（CI 口径 + drift/CI dual-threshold）、边界健壮性（零场次/字符串分数/σ² 下限）、R-cutover 派生访问层（指纹缓存失效/迁移幂等/Blend 缓存）、derive_elo 按 round 重建 |
| `test_predictors` | 14 | SVD-Ridge 批量预测精度（MAP 不确定性/退化列封顶/模型缓存）、BlendPredictor 双层收缩、发现层指纹 + 相似度加权池化 |
| `test_results_matrix` | 5 | 结果矩阵 R（唯一真相）upsert/时序/round-trip + 多目标扫描 + derive_elo 幂等 |

### 采样与编排

| 文件 | tests | 测什么 |
|---|---|---|
| `test_samplers` | 5 | 采样器参数生效性（hybrid 透传 α/β/γ、InfoGain pred_std、coordinate-rounds 默认） |
| `test_runner` | 9 | runner 集成（增量落盘/publish_tracker/Phase2 降级/多目标 canonical report）、韧性（过敏按模型隔离/单方法聚类 error）、.venv re-exec/resume/input 默认 |
| `test_evaluator` | 3 | 断点筛选（ID 数值序/段级匹配）、有害记录 eval_score>0、attack_accuracy 排除拒绝 |

### 评分与目标

| 文件 | tests | 测什么 |
|---|---|---|
| `test_scoring` | 6 | compliance 去宽泛兜底、B 级折扣、JUDGE_MODEL 回退、有害度钳位、math 取末标签、短响应不判 empty |
| `test_jailbreak_tax` | 8 | 越狱税全链路（注入/剥离一致性、math_score 三档、哨兵不测税、聚合） |
| `test_targets` | 12 | legacy fallback 注册 + 路由可达性 + 空配置兜底、并发单例客户端、pcap 重试/4xx 短路、env 惰性读 |
| `test_allergy` | 4 | 过敏检测窗口选取（一侧不足向另一侧借、上方按 Elo 距离取最近） |

### 聚类

| 文件 | tests | 测什么 |
|---|---|---|
| `test_clustering` | 19 | 阻尼白化空间方差 + HDBSCAN 端到端 + auto-k（已知簇数 ±2）+ ANOVA 簇效 + 弱监督加权 + D-optimal 种子覆盖 + embedding 降级链 + 契约健壮性（空 tracker/路径解析/缺字段） |

### 外围

| 文件 | tests | 测什么 |
|---|---|---|
| `test_dashboard` | 13 | Web 面板全 API 冒烟 + 路径校验 + 任务生命周期 + HPO/趋势/SSE/报告下载 + 任务状态刷新/LRU/上限校验 |
| `test_experiments` | 7 | HPO 框架（schema 解析、grid/random/bayesian、指标聚合、work-dir 隔离、续跑） |
| `test_generators` | 3 | generate 注入越狱税 + 5 类危害、harmbench functional_category、safe_twin --no-generate |
| `test_report` | 3 | report 多源结果加载、_load_elo_tracker 优先级、叙事 timeout 透传 |
| `test_retry` | 11 | 统一重试 retry_call（成功/前 N-1 失败/全失败/retry_on/on_retry）+ 各模块重试参数冒烟 |

### 回归套件（历史审计批次，按修复来源保留）

| 文件 | tests | 来源 |
|---|---|---|
| `test_core_regressions` | 10 | P0 致命修复 F1-F7（math 哨兵/弱监督位置/技术标签/括号反转/template hash） |
| `test_correctness` | 17 | P2 正确性（tree inf/NaN 守卫/conv_rounds 恢复/report inconclusive/K-Fold/predict schema） |
| `test_data_integrity` | 19 | P0 数据完整性（NaN 校验/原子写/多目标映射/local_sim 数学优先级） |
| `test_review_regressions` | 17 | 全面审查 FR1-FR13（落盘顺序/logger/stale GT/boundary 键集/no-judge 兜底/dedup） |

### 真实 API / 端到端（默认排除，手动触发）

| 文件 | marker | 测什么 |
|---|---|---|
| `test_real_api` | `real_api` | 目标模型真实连通性 + OpenAI 契约 + 安全基线（拒绝有害）；Judge 评分契约（正常→A、有害→C/D/E）；Generator 客户端可达 |
| `test_e2e_dashboard` | `e2e` | 看板 `POST /api/run/evaluate` 全链路（子进程 + 真实 API）：触发→轮询→success；progress.jsonl 进度记录；`POST /cancel` 任务取消 |

## 速查：出问题该看哪个测试

| 症状 | 看这里 |
|---|---|
| 收敛提前/置信度虚高 | `test_elo` |
| 换模型后 Elo/FPR 串味 | `test_elo`、`test_results_matrix` |
| 采样旋钮调了没反应 | `test_samplers` |
| 聚类质量/簇效 | `test_clustering` |
| 预测值离谱/CI 爆炸 | `test_predictors` |
| JSON 损坏/丢数据 | `test_data_integrity` |
| 看板接口/任务 | `test_dashboard` |
| 目标路由/legacy .env | `test_targets` |
| 越狱税/数学探针 | `test_jailbreak_tax` |
| 目标/Judge 真实不通 | `test_real_api`（`-m real_api`）|
| 看板评估端到端跑不通 | `test_e2e_dashboard`（`-m e2e`）|
