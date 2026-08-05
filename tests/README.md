# 测试说明（tests/）

本目录是 llmsec 的回归 + 冒烟测试集，**pytest 用例**（共 174 个），覆盖某个模块或某轮审计修复。

## 怎么跑

```bash
pytest tests/                                            # 全量
pytest tests/test_elo_convergence.py                     # 单文件
pytest tests/test_scoring.py::test_b_level_discount      # 单个用例
pytest -n auto                                           # 并行（已装 pytest-xdist）
```

- `conftest.py` 统一做路径注入（让 tests/ 能 import llmsec）+ Windows 控制台 UTF-8（经 `setup_console()`），各 test 模块不再各自处理。
- **绝大多数离线可跑**（mock/stub 目标与 Judge）；只有少数端到端冒烟（如 `test_p1_runner` 的 .venv re-exec）依赖本机环境。

## 命名约定（看不懂的根源）

文件名混用了两套风格，对应不同来源：

| 前缀 | 含义 |
|---|---|
| `test_p0_*` | **P0 致命修复**的回归（对应审计 F1–F7 等会导致崩溃/数据损坏的修复） |
| `test_p1_*` | **P1 高危修复**的回归（对应审计 H/M 系列单项修复） |
| `test_p2_*` | **P2 正确性/数据完整性**回归 |
| `test_<模块>` | **按模块/特性**组织的回归（不绑定某轮审计，覆盖该模块的核心不变量） |

> `p0/p1/p2` 是**修复批次代号，不是测试优先级**。比如 `test_p1_clustering` 不是"聚类的高优先级测试"，而是"在 P1 修复批次中，针对聚类模块那批修复写的回归"。新增模块级测试请用 `test_<模块>.py` 风格。

## 测试矩阵（按被测领域分组）

### 评估核心（Elo / 收敛 / 预测）

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_elo_convergence` | Elo 收敛判据（抗假阳性 / 真收敛）+ 变体后缀兜底预测 | S-1（CI 口径）、变体 fallback |
| `test_elo_edge_cases` | 零场次 boundary 键完整、update 字符串分数、σ² 下限、artifacts 优先 cluster_result | S-2、M-3、M-7、M-4 |
| `test_svd_ridge` | SVD-Ridge 批量预测精度（MAP 不确定性）、PCAP 防御方名、首轮 `ci_half=None`、K-Fold λ、模型缓存、退化列封顶 | S-1、M-1、M-7 |
| `test_blend_predictor` | Blend 双层预测（统一+模型）贝叶斯收缩权重 | P2 双层预测 |
| `test_elo_access` | R→派生 Elo 访问层（指纹缓存失效、迁移幂等）+ Blend 缓存 | R-cutover、M-17（动态路径） |
| `test_results_matrix` | 结果矩阵 R（唯一真相）upsert/时序/round-trip + 多目标扫描 + `derive_elo` | R 架构、M-4 |

### 采样与编排

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_samplers` | 采样器参数**生效性**：hybrid 透传 α/β/γ、InfoGain 用 pred_std 区分候选、coordinate-rounds 默认接 params | M-8、M-9、M-10 |
| `test_p1_runner` | runner：.venv re-exec、round_idx 守卫、resume `tested` 初始化、`--input` 默认 | H1–H4 |
| `test_evaluator_resume` | evaluator 断点筛选（ID 数值序/段级匹配）、有害记录保持 eval_score>0、attack_accuracy 排除拒绝 | M-14、M-19、M-21 |
| `test_runner_resilience` | 过敏检测按模型隔离（换模型不跳过/不串味）、单方法聚类返回 error 不写文件 | S-3、M-30 |
| `test_runner_integration` | runner 主循环 + 多目标集成（stub 不触网）：每轮增量落盘 attack_file、每轮 publish_tracker、Phase2 judge 故障降级、twin entry 缺字段兜底、多目标写 canonical runner_report | M-11、M-12、M-15、M-35、M-36 |
| `test_allergy_window` | 过敏检测窗口选取（一侧不足向另一侧借、上方按 Elo 距离取最近） | twin window 逻辑 |

### 评分与越狱税

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_jailbreak_tax` | 越狱税全链路：注入/剥离一致性、math_score 三档、哨兵（`expected_answer=0` 不测税不扣分）、聚合 | 越狱税设计 |
| `test_scoring` | 判官/评分链：compliance 去宽泛兜底、B 级折扣、JUDGE_MODEL 回退、有害度钳位、math 取末标签、短响应不判 empty | M-20、M-22、M-23、M-25、M-26、M-28 |
| `test_p1_targets_judge` | targets（pcap/多线程）+ judge/evaluator：线程安全 `call_target`、分级匹配、pcap 重试/4xx 短路、env 惰性读、`update_elo` defender_name | M10–M12、M16–M19 |
| `test_p1_retry` | 统一重试 `retry_call`（成功/前 N-1 失败/全失败/`retry_on`/`on_retry`）+ 各模块重试参数冒烟 | M5、M-24 |

### 聚类

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_whitened_tree` | 阻尼白化空间 + HDBSCAN + auto-k（已知簇数 ±2）+ ANOVA 簇效 + 弱监督加权 + D-optimal 种子覆盖 + **embedding 降级链本地优先** | M-27、白化/树/auto-k |
| `test_p1_clustering` | reaction_validation 有限值/JSON-safe、n=1 PCA 路径、damp 默认 + 旧空间 transform、TREE_K_MIN 小样本、**embedding 降级链顺序** | H7、H8、M3/M15、M-27、M-29 |
| `test_clustering_kdistance` | 离线构造 3 已知簇（base64/rot13/code）验 HDBSCAN ≥3 簇 + 小簇命名（`write=False` 不污染） | 聚类端到端 |
| `test_cluster_contract` | cluster_analysis 空 tracker 健壮、load_and_extract result_file 路径解析一致、缺字段记录不崩 | M-31、M-32、M-33 |

### 数据完整性与基础设施

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_p2_data_integrity` | `elo.update` NaN/inf 校验、`io` 原子写+strict+损坏备份、`config` 多目标 name↔idx 映射、local_sim 数学/有害优先级 | F1、F3、F4、F5 |
| `test_p2_correctness` | tree 的 inf/NaN 守卫、`_compute_conv_rounds` 轨迹恢复、report inconclusive 分支、K-Fold 打乱、predict schema 一致性 | F2、H3、H4、H9、H10 |
| `test_p0_regressions` | P0 致命修复：math 哨兵、弱监督加权位置、技术标签大小写、`math_score_distribution`、local_sim 识别数学题、括号反转、template hash | F1–F7 |

### 外围（生成 / 报告 / 面板）

| 文件 | 测什么 | 覆盖的审计点 |
|---|---|---|
| `test_p1_generators` | generate 注入越狱税+5 类危害、harmbench `functional_category`、safe_twin `--no-generate` | M6、M7、M20 |
| `test_p1_report` | report 多源结果加载、`_load_elo_tracker` 优先级、叙事 timeout 透传 | H9、H10、M9 |
| `test_dashboard_api` | Web 面板全 API 冒烟 + 路径校验 + 任务生命周期 + **新增端点**（趋势/批次富化/取消/SSE 流/完整日志/报告下载/树叶标签/空状态原因） | dashboard 契约 |
| `test_p1_dashboard` | dashboard 内部修复：任务状态刷新关句柄、投影缓存 LRU 淘汰、`batch_size` 上限校验 | H6、M8、M18 |
| `test_experiments` | 实验框架：schema 解析、grid/random/bayesian 搜索、指标聚合（mean/mean_plus_std）、work-dir 隔离、编排续跑 | HPO 框架、M-34、S-6 |

## 速查：出问题时该看哪个测试

- **收敛提前/置信度虚高** → `test_elo_convergence`、`test_svd_ridge`
- **换模型后 Elo/FPR 串味** → `test_elo_access`、`test_results_matrix`
- **采样旋钮调了没反应** → `test_samplers`
- **聚类质量/簇效** → `test_whitened_tree`、`test_clustering_kdistance`
- **JSON 损坏/丢数据** → `test_p2_data_integrity`
- **看板接口/任务** → `test_dashboard_api`、`test_p1_dashboard`
