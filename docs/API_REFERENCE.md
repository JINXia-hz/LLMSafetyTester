# LLMSEC 接口参考手册

> **对应代码版本**：2026-08 收尾阶段（`main` 分支）
> **维护方式**：本文档手工编写，接口增删改后需同步更新。FastAPI 另在 `/docs`（Swagger UI）与 `/redoc` 提供自动生成的交互式文档，两者互补。

llmsec 通过 **三类接口** 对外暴露能力：

| 接口类型 | 消费者 | 启动方式 | 文档章节 |
|----------|--------|----------|----------|
| MCP 工具库（54 工具） | 外部 Agent（ZCode / Cursor / Claude） | `llmsec-mcp` | [一](#一mcp-工具库) |
| HTTP REST API（54 端点） | 人类（Web 看板 UI） / 脚本 | `uvicorn llmsec.server.dashboard_api:app` | [二](#二http-rest-api) |
| CLI（3 console script + 2 模块入口） | 脚本自动化 | `llmsec` / `llmsec-manage` / `python -m control` 等 | [三](#三cli-入口) |

三套接口共享同一套底层能力（ELO 评估、R 矩阵、workspace fork/merge、子进程任务管理），但面向不同消费者。配置项统一从仓库根 `.env` 读取，参见 [四、配置参考](#四配置参考)。

---

## 目录

- [一、MCP 工具库](#一mcp-工具库)
  - [1.1 传输与启动](#11-传输与启动)
  - [1.2 两步确认机制](#12-两步确认机制)
  - [1.3 工具总览](#13-工具总览)
  - [1.4 Tier 1 — 纯函数（7）](#14-tier-1--纯函数7)
  - [1.5 Tier 2 — 只读查询（23）](#15-tier-2--只读查询23)
  - [1.6 Tier 3 — 写操作（17）](#16-tier-3--写操作17)
  - [1.7 Tier 4 — 长任务（7）](#17-tier-4--长任务7)
- [二、HTTP REST API](#二http-rest-api)
  - [2.1 端点总览](#21-端点总览)
  - [2.2 数据查询](#22-数据查询data_query)
  - [2.3 任务管理](#23-任务管理tasks)
  - [2.4 聚类可视化](#24-聚类可视化cluster_viz)
  - [2.5 HPO 配置台](#25-hpo-配置台hpo)
  - [2.6 控制层（三省制）](#26-控制层三省制control)
- [三、CLI 入口](#三cli-入口)
  - [3.1 `llmsec` — 评估流水线](#31-llmsec--评估流水线)
  - [3.2 `llmsec-manage` — 信息管理](#32-llmsec-manage--信息管理)
  - [3.3 `llmsec-mcp` — MCP 服务器](#33-llmsec-mcp--mcp-服务器)
  - [3.4 `python -m control` — 控制层](#34-python--m-control--控制层)
  - [3.5 `python -m llmsec.experiments` — 实验框架](#35-python--m-llmsecexperiments--实验框架)
- [四、配置参考](#四配置参考)
  - [4.1 `.env` 字段表](#41-env-字段表)
  - [4.2 `params.py` 行为参数](#42-paramspy-行为参数)

---

## 一、MCP 工具库

把 llmsec 的全部能力以 **工具** 形式暴露给外部 Agent。Agent 通过 MCP 协议调用工具，获得 JSON 结构化返回。所有工具的 schema 从 Python type hint + docstring 自动推断（FastMCP 机制）。

### 1.1 传输与启动

```bash
llmsec-mcp                       # 默认 stdio（适配 IDE / ZCode 集成）
llmsec-mcp --transport http      # HTTP 远程访问（默认 127.0.0.1:8765）
llmsec-mcp --transport http --host 0.0.0.0 --port 9000
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--transport` | choices | `stdio` | `stdio`（默认）或 `http` |
| `--host` | str | `127.0.0.1` | HTTP 模式监听地址 |
| `--port` | int | `8765` | HTTP 模式监听端口 |

- **stdio 模式**：适配 IDE/ZCode 集成，进程退出时注册了 `atexit` + 信号处理器做防僵尸兜底。
- **http 模式**：供远程访问，默认只绑 `127.0.0.1`。

> ⚠ MCP server 有自己独立的子进程任务队列（`llmsec.server.task_manager`），与 HTTP 看板的任务队列互不干扰。

### 1.2 两步确认机制

涉及**不可逆写操作**（删除 run、清理缓存、合并 R 矩阵、快照写回 .env）的工具采用 preview / confirm 成对设计：

```
Agent → xxx_preview(args)  →  返回 {summary, confirm_token, ttl_seconds: 300}
Agent 审阅 summary
Agent → xxx_confirm(token) →  返回 {status: "executed", result}
```

| 属性 | 值 |
|------|-----|
| token 生成 | `secrets.token_urlsafe(8)` |
| 存储 | 内存 dict（仅 MCP server 单进程生命周期内有效，重启失效） |
| TTL | **300 秒（5 分钟）**，过期自动失效 |
| 一次性 | 确认后即删，**不可重放** |
| 线程安全 | `threading.Lock` 保护所有读写 |

> token 不存在 / 已过期 / 已确认，confirm 均返回 `{"status": "expired_or_already_confirmed"}`。Agent 需重新调 preview 获取新 token。

### 1.3 工具总览

54 个工具按风险分四层。Tier 越高，副作用越大。

| Tier | 文件 | 数量 | 特性 |
|------|------|------|------|
| **Tier 1** | `compute.py` | 7 | 纯函数（零副作用、零 IO） |
| **Tier 2** | `query.py` | 23 | 只读查询（幂等、无副作用） |
| **Tier 3** | `actions.py` | 17 | 写操作（部分带两步确认） |
| **Tier 4** | `tasks.py` | 7 | 长任务（异步提交 + 轮询） |

<details>
<summary>完整工具一览表（点击展开）</summary>

| 工具名 | T | 简述 | 异步 | 确认 |
|--------|:-:|------|:----:|:----:|
| `obfuscate_prompt` | 1 | 攻击 prompt 混淆/编码（b64/rot13/code/story/raw） | | |
| `compute_eval_score` | 1 | 根据 Judge + 越狱税计算综合 eval_score | | |
| `compute_math_score` | 1 | 越狱税探针回答判定 | | |
| `extract_math_answer` | 1 | 提取最后一个 `[MATH:x]` 答案 | | |
| `extract_textual_features` | 1 | 提取 12 维文本结构特征 | | |
| `extract_report_metrics` | 1 | 从 runner_report 抽取核心度量 | | |
| `aggregate_metrics` | 1 | 跨 repeats 聚合度量（mean / mean_plus_std） | | |
| `list_runs` | 2 | 列出所有评估 run（多维过滤） | | |
| `compare_runs` | 2 | 对比多个 run 的评估指标 | | |
| `read_run_report` | 2 | 读单个 run 完整报告 + 安全树 | | |
| `assess_run_findings` | 2 | 阈值规则审查 run 异常发现 | | |
| `review_run` | 2 | 完整审查 run（规则 + 中文叙事摘要） | | |
| `get_thresholds` | 2 | 读取安全审查阈值常量 | | |
| `get_results_summary` | 2 | R 矩阵概要 | | |
| `elo_ranking` | 2 | 指定模型攻击方 Elo 排名 | | |
| `elo_security_boundary` | 2 | 指定模型安全边界（含收敛/置信度） | | |
| `elo_find_surprises` | 2 | 双向意外（防御短板 / 强项） | | |
| `elo_suggest_next_pairing` | 2 | 下一批测试配对建议 | | |
| `get_allergy_report` | 2 | 读取过敏检测（FPR）报告 | | |
| `list_targets` | 2 | 列出 .env 声明的目标模型（脱敏） | | |
| `probe_targets` | 2 | 探测目标模型/服务 API 连通性 | | |
| `list_workspaces` | 2 | 列出所有 fork 工作区 | | |
| `list_workspace_runs` | 2 | 列出工作区内的 run | | |
| `get_cluster_report` | 2 | 读取聚类分析报告 | | |
| `get_params` | 2 | 读取 params.py 全部行为参数 | | |
| `list_plans` | 2 | 列出最近的 Plan（编排计划） | | |
| `get_plan` | 2 | 读单个 Plan 完整详情 | | |
| `list_gazettes` | 2 | 列出最近的文牍（事件流索引） | | |
| `get_plan_context` | 2 | 从文牍重建 Plan 上下文快照 | | |
| `read_plan_events` | 2 | 读 Plan 完整事件流 | | |
| `delete_runs_preview` | 3 | 预览删除 run 影响 | | ✅ |
| `delete_runs_confirm` | 3 | 用 token 执行 run 删除 | | ✅ |
| `clean_caches_preview` | 3 | 预览清理缓存影响 | | ✅ |
| `clean_caches_confirm` | 3 | 用 token 执行缓存清理 | | ✅ |
| `fork_workspace` | 3 | fork 隔离工作区 | | |
| `export_snapshot` | 3 | 导出 R 矩阵快照 | | |
| `create_env_snapshot` | 3 | 创建 .env 配置快照 | | |
| `edit_env_snapshot` | 3 | 修改快照里的配置项 | | |
| `list_env_snapshots` | 3 | 列出所有 .env 配置快照 | | |
| `get_env_config` | 3 | 读全局 .env 配置（脱敏） | | |
| `delete_env_snapshot` | 3 | 删除一个 .env 配置快照 | | |
| `merge_workspaces_preview` | 3 | 预览 R 矩阵合并影响 ⚠ critical | | ✅ |
| `merge_workspaces_confirm` | 3 | 用 token 执行 R 矩阵合并 ⚠ critical | | ✅ |
| `merge_env_snapshot_to_global_preview` | 3 | 预览快照写回全局 .env ⚠ critical | | ✅ |
| `merge_env_snapshot_to_global_confirm` | 3 | 用 token 执行快照写回 ⚠ critical | | ✅ |
| `delete_workspace` | 3 | 删除一个 fork 工作区 | | |
| `gc_merged_workspaces` | 3 | 清理已 merge 且超期的工作区 | | |
| `run_evaluation` | 4 | 提交一次评估任务 | ✅ | |
| `get_task_status` | 4 | 查任务状态 + 日志尾部 | | |
| `get_task_progress` | 4 | 查进度快照（逐目标/逐轮） | | |
| `get_task_log` | 4 | 读完整任务日志 | | |
| `cancel_task` | 4 | 取消排队中/运行中任务 | | |
| `list_tasks` | 4 | 列出所有任务 | | |
| `orchestrate_runs` | 4 | 批量并行评估（A/B 对比/参数扫描） | ✅ | |

</details>

---

### 1.4 Tier 1 — 纯函数（7）

零副作用、零 IO。给定输入必有确定输出，适合 Agent 无风险调用。

#### `obfuscate_prompt(prompt, method)`
对攻击 prompt 应用指定混淆/编码变换，保持原意只改变表达形式。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `prompt` | str | ✅ | 原始攻击 prompt |
| `method` | str | ✅ | `b64` / `rot13` / `code`（伪装 Python docstring）/ `story`（虚构场景）/ `raw`（不变换） |

**返回**：变换后的 prompt 文本（str）。

#### `compute_eval_score(judge_result, math_score=None)`
根据 Judge 判定 + 越狱税得分计算综合 eval_score。标度：-2.0 ~ -1.0 = 拒绝/引导/无关（安全）；0.0 ~ 5.0 = 越狱成功。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `judge_result` | dict | ✅ | Judge 判定，需含 `compliance_level`(A/B/C/D/E)、`is_refusal`、`is_harmful`、`combined_score` |
| `math_score` | int \| None | | 越狱税得分（2=正确, 1=格式对答案错, 0=格式缺失）；None=无越狱税探针 |

**返回**：`{eval_score, jailbreak_tax, status, compliance_level}`。

#### `compute_math_score(raw_text, expected)`

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `raw_text` | str | ✅ | 模型响应文本 |
| `expected` | int | ✅ | 期望答案 |

**返回**：`{math_score, actual_answer}`。`math_score`: 2=格式对且答案对, 1=格式对答案错, 0=格式缺。

#### `extract_math_answer(text)`
从响应文本提取最后一个 `[MATH:数字]` 答案（模板约定"最后一行=最终答案"，CoT 中复述标签取首个会高估越狱税）。

**返回**：提取到的整数，或 `None`（格式不存在）。

#### `extract_textual_features(prompt)`
提取 12 维文本结构特征（长度/标点密度/疑问句率等）。纯规则计算，不依赖外部模型。

**返回**：`{特征名: 浮点值, ...}`。

#### `extract_report_metrics(report)`

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `report` | dict | ✅ | 完整的 `runner_report.json` 内容 |

**返回**：`{asr, rounds, total_tested, boundary_elo, boundary_confidence, ci_half, drift, converged, coverage, conv_rounds, fpr}`。report 为空或缺段时对应字段为 `None`。

#### `aggregate_metrics(values, mode="mean")`

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `values` | list[float \| None] | ✅ | 度量值列表，可含 None |
| `mode` | str | | `mean`（均值）/ `mean_plus_std`（均值+标准差，风险厌恶） |

**返回**：聚合浮点值。自动过滤 None / inf / nan；空列表返回 `inf`。

---

### 1.5 Tier 2 — 只读查询（23）

幂等、无副作用。所有查询从磁盘读取现有产物（R 矩阵 / run 报告 / 索引），不触发评估。

#### Run 查询

##### `list_runs(target=None, since=None, junk_only=False, level=None, has_report=None, min_size=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | str | 只列指定目标模型的 run |
| `since` | str | 起始时间（ISO 或 `yyyy-mm-dd`） |
| `junk_only` | bool | 只列无报告的垃圾/失败 run |
| `level` | str | 按安全等级过滤（`safe`/`allergic`/`vulnerable`/`broken`/`inconclusive`） |
| `has_report` | bool | 只列有/无 `runner_report.json` 的 run |
| `min_size` | int | 最小字节数过滤 |

**返回**：run 元数据 dict 列表（时间倒序），每条含 `name, target, security_level, asr, boundary_elo, has_report, mtime, size`。

##### `compare_runs(run_names)`

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `run_names` | list[str] | ✅ | 要对比的 run 名称（≥2 个） |

**返回**：结构化对比报告 dict（安全等级/ASR/Elo 边界/收敛/覆盖率对比），不存在的 run 会在报告中标注。

##### `read_run_report(run_name)`

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `run_name` | str | ✅ | 格式 `batch/target`（用 `list_runs` 查） |

**返回**：`{report, tree, run_dir, run_name}`；run 不存在返回 `None`。

##### `assess_run_findings(run_name)`
用内置阈值规则审查 run，产出异常发现列表（findings）。纯规则判定，不调 LLM。

**返回**：`{run_name, findings: [...], thresholds: {...}}`。每个 finding 含 `severity / metric / value / threshold / interpretation`。

##### `review_run(run_name, use_llm=True)`
完整审查：读报告 → 规则判定 → 生成中文叙事摘要。比 `assess_run_findings` 更完整。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `run_name` | str | — | run 名 |
| `use_llm` | bool | `True` | `True`=用 LLM 润色叙事（需 GENERATOR_*）；`False`=规则模板兜底（离线可用） |

**返回**：`{run_name, summary, findings, digest, metrics}`。

##### `get_thresholds()`
读取安全审查阈值常量（来自 `params.py`）：`PORTRAIT_MIN_TESTED`、`PORTRAIT_ASR_SAFE`、`ALLERGY_FPR_SAFE`、`CONV_CI_TARGET` 等。

#### R 矩阵与 Elo 派生

> R 观测（统一库 `output/state/catalog.db` 的 observations 表）是 llmsec 唯一不可重算的状态。以下 Elo 工具均从 R 纯函数回放派生，进程内按列指纹缓存（elo_cache 表）。

##### `get_results_summary()`
**返回**：`{models, records, total_observations}`。R 不存在或空时返回提示。

##### `elo_ranking(model)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | str | 目标模型名（R 矩阵一列） |

**返回**：攻击方排名 dict 列表（降序：高 Elo = 强攻击），每条含 `attacker / elo / played / predicted`。

##### `elo_security_boundary(model)`
**返回**：安全边界 dict，含 `boundary_elo / converged / confidence / ci_half / methods_above_boundary` 等。

##### `elo_find_surprises(model, min_elo_gap=0.0)`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | str | — | 目标模型名 |
| `min_elo_gap` | float | `0.0` | 最小 Elo 差距阈值，过滤噪声 |

**返回**：`{"weakness": [...], "strength": [...]}`。weakness = 低 Elo 攻击成功（防御短板），strength = 高 Elo 攻击失败（防御强项）。

##### `elo_suggest_next_pairing(model, n=5)`
选 |攻击Elo - 防御Elo| 最小的 n 对（不确定性最大，测试获益最高）。

**返回**：配对建议列表，每条含 `attacker / defender / elo_gap`。

#### 过敏检测

##### `get_allergy_report()`
**返回**：过敏检测报告 dict，含 `summary.false_positive_rate` 等。过敏侧（FPR）评估"会不会误伤"（对无害请求过度拒绝），与攻击侧（Elo/ASR）互补。

#### 目标与连通性

##### `list_targets()`
**返回**：`.env` 中 `TARGETS` 声明的目标列表，每条含 `name / base_url / model / api_key`（脱敏）。

##### `probe_targets(name=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 只探测指定目标（不探 services）。None 探全部 + generator + judge |

两阶段探测：① `models.list`（GET）校验端点，不耗 token；② `chat smoke`（max_tokens=64）校验鉴权，401/403 判不可达。

**返回**：`{targets: [{name, model, reachable, latency_ms, error, warning}], services: [...]}`。

> 💡 强烈建议在 `run_evaluation` 前先探测——模型不可达时跑完整评估只会得到全 ASR=0 的假阴性。

#### Workspace 查询

##### `list_workspaces()`
**返回**：所有 fork 工作区元数据列表，每条含 `name / source / note / created`。

##### `list_workspace_runs()`
**返回**：所有工作区内的 run（含报告），与 `list_runs`（只扫 `output/runs/`）互补。每条含 `name / workspace / target / security_level / asr / boundary_elo`。

#### 聚类

##### `get_cluster_report()`
**返回**：聚类分析报告（`cluster_report.json`）；未跑过聚类返回 `None`。需安装 `[cluster]` extras。

#### 调参与配置

##### `get_params(category=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `category` | str | 只返回指定分组（`pipeline`/`elo`/`ridge`/`sampler`/`judge`/`cluster`/`blend`/`twin`/`report`/`sim`）；None=全部分组 |

**返回**：`{分组名: {参数名: {value, type, description}}}`。配合 `run_evaluation(param_overrides={...})` 临时覆写。

#### Plan / 文牍（三省制）

##### `list_plans(recent=20)`
**返回**：最近 Plan 摘要列表（倒序），每条含 `id / intent / status / created`。

##### `get_plan(plan_id)`
**返回**：Plan 详情（含 `steps / topological_layers / status`）；不存在返回 `None`。

##### `list_gazettes(recent=20)`
**返回**：文牍（执行历史）索引列表，每条含 `plan_id / intent / created / events_count`。

##### `get_plan_context(plan_id)`
从文牍事件流重建 Plan 上下文快照（聚合视图）：意图、各步状态、封驳记录、审查记录。

**返回**：上下文 dict；Plan 不存在返回 `None`。

##### `read_plan_events(plan_id)`
**返回**：某 Plan 的完整原始事件流（按时间排序），每条含 `ts / kind / dept / detail`。比 `get_plan_context` 更细。

---

### 1.6 Tier 3 — 写操作（17）

有副作用。其中 4 对 preview/confirm 需两步确认。

#### 删除 run（两步确认）

##### `delete_runs_preview(names, delete_r=False) → {confirm_token}`

| 参数 | 类型 | 说明 |
|------|------|------|
| `names` | list[str] | 要删除的 run 名（用 `list_runs` 查） |
| `delete_r` | bool | 同时从 R 矩阵移除对应模型的观测列 |

**返回**：`{action, summary, total_size_human, confirm_token, ttl_seconds, impact_note, next_step}`。删除是**软删除**（移到 `.trash/`，可恢复）。

##### `delete_runs_confirm(token) → {status, result}`
`status`: `"executed"` 或 `"expired_or_already_confirmed"`。

#### 清理缓存（两步确认）

##### `clean_caches_preview(categories) → {confirm_token}`

| 参数 | 类型 | 说明 |
|------|------|------|
| `categories` | list[str] | 可选值见下 |

| 类别 | 说明 | 可恢复 |
|------|------|:------:|
| `predictors` | 混合预测器 pkl | ✅（重训） |
| `feature_cluster` | 特征缓存 + 聚类产物 | 特征可重建 / cluster 需重跑 |
| `model_state` | 预筛 ML 模型 joblib | ✅（重训） |

（Elo 派生缓存已表化进 catalog.db 的 elo_cache 表，指纹自动失效，无需清理类别；任务日志清理走 `storage gc-tasks`。）

##### `clean_caches_confirm(token) → {status, result}`

#### 合并 R 矩阵（两步确认 ⚠ critical）

##### `merge_workspaces_preview(sources, target="global", models=None) → {confirm_token}`
合并语义：对每个源的每个模型（或 `models` 子集），把该列全部观测 upsert 到目标 R（同 record+model 覆盖，不同 record 累加）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `sources` | list[str] | — | 源描述符列表 |
| `target` | str | `"global"` | 目标描述符 |
| `models` | list[str] \| None | `None` | 只合并指定模型 |

**描述符格式**：`"global"` → 统一库 `output/state/catalog.db`；`"ws:<name>"` → `output/workspaces/<name>/catalog.db`（卫星库）；其他 → 视为目录路径。

典型场景：fork 工作区跑完实验后，`sources=["ws:exp1"], target="global"` 合并回全局。

##### `merge_workspaces_confirm(token) → {status, result}`

#### 快照写回 .env（两步确认 ⚠ critical）

##### `merge_env_snapshot_to_global_preview(name) → {confirm_token}`
快照里有的 key 覆盖全局同名 key，快照里没有的不动。全局 `.env` 先备份到 `.env.bak.<timestamp>`。

##### `merge_env_snapshot_to_global_confirm(token) → {status, result}`

#### 直接执行工具（无需确认）

##### `fork_workspace(name, source="global", note="")`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | str | — | 工作区名（唯一） |
| `source` | str | `"global"` | `"global"` 或 `"run:<run_name>"` |
| `note` | str | `""` | 备注（记入索引） |

**返回**：工作区信息 dict（`name / path / source / models / records`）。

##### `export_snapshot(source="global", out=None)`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `source` | str | `"global"` | `"global"` 或 `"run:<name>"` |
| `out` | str \| None | `None` | 输出路径（目录或 `.tar.gz`）；None 默认到 `output/snapshots/<时间戳>/` |

快照是统一库副本（sqlite backup API，WAL 安全），elo_cache/probes 等派生表随库自带。

##### `create_env_snapshot(name, source="global", note="")`

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | str | — | 快照名 |
| `source` | str | `"global"` | `"global"` / `"blank"`（空快照）/ 另一个快照名 |
| `note` | str | `""` | 备注 |

典型流程：`create(source="blank")` → `edit_env_snapshot` 写入 API key → `run_evaluation(env_snapshot=...)`。

##### `edit_env_snapshot(name, key, value)`
修改快照里某个配置项。受管理的 key 前缀：`TARGETS / TARGET_*`、`GENERATOR_*`、`JUDGE_* / JUDGE_API_KEY`、`CONTROL_*`、`LLMSEC_PARAM_*`。

##### `list_env_snapshots()`
**返回**：快照列表（倒序），每条含 `name / source / keys / note / created`。

##### `get_env_config()`
**返回**：`{configured: {...}, missing_essential: [...], total_keys}`。检查 `GENERATOR_API_KEY`、`GENERATOR_BASE_URL` 等关键配置。

##### `delete_env_snapshot(name)`
**返回**：`{deleted: name, info: {...}}`。

##### `delete_workspace(name)`
**返回**：`{deleted: name}` 或错误信息。仅删隔离副本，不影响全局 R。

##### `gc_merged_workspaces(older_than_days=7)`
清理已 merge 且超期的工作区目录（延迟 GC）。merge 后不立即删（orchestrator 对比/历史可能引用），按 `merged_at` 延迟清理。

**返回**：`{cleaned: [{name, size}], skipped_fresh: N, gc_log_size: N}`。

---

### 1.7 Tier 4 — 长任务（7）

评估跑几分钟到几十分钟（调 LLM API），不能同步阻塞。采用**提交 → 返回 task_id → 轮询状态**模式。

#### `run_evaluation(...)`
提交一次评估任务，立即返回 task_id。校验/argv 构造统一走 `llmsec.server.launch`
（与 Web 看板 `POST /api/run/evaluate`、TUI 同一链路，参数面一致）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `target` | str \| None | `None` | 单目标（与 `targets` 二选一）。须在 `.env TARGETS` 中声明 |
| `targets` | list[str] \| None | `None` | 多目标列表，默认全并发 |
| `input_file` | str | `"attacks/l1.jsonl"` | 攻击集路径（相对仓库根；取末段文件名防穿越，须 `.jsonl`） |
| `max_rounds` | int | `5` | 自适应最大轮数（1-50） |
| `phase` | str | `"all"` | `"all"` / `"1"`(仅攻击) / `"2"`(仅过敏) |
| `batch_size` | int \| None | `None` | 每轮批量大小 |
| `sampler` | str | `"hybrid"` | `"hybrid"` / `"gap"` / `"infogain"` / `"coordinate"` |
| `sampler_alpha` | float \| None | `None` | InfoGain 不确定性权重（不传用 params 默认值） |
| `sampler_beta` | float \| None | `None` | InfoGain 簇覆盖权重 |
| `sampler_gamma` | float \| None | `None` | InfoGain 成功潜力权重 |
| `coordinate_rounds` | int \| None | `None` | Hybrid 模式下前多少轮使用 InfoGain 探索 |
| `seed` | int \| None | `None` | 随机种子（可复现） |
| `env_snapshot` | str \| None | `None` | .env 快照名，指定时覆盖全局 .env（隔离评估） |
| `twin_window` | int \| None | `None` | 过敏检测方法数上限 |
| `no_early_stop` | bool | `False` | 跑满 max_rounds 不提前停止 |
| `concurrency` | int \| None | `None` | 批内并行度；None=默认全并发 |
| `param_overrides` | dict \| None | `None` | 覆写 params.py 参数 `{"PARAM_NAME": value}`，类型推断 bool/int/float/str |

**返回**：task_view dict（含 `id / status / started_at / meta`；`meta.targets/max_rounds`
为结构化任务摘要，`GET /api/tasks/{id}/progress` 的目标占位行同源）。用 `id` 轮询 `get_task_status`。

> 💡 `param_overrides` 用法：先用 `get_params` 查参数名和当前值，再如 `{"K_FACTOR": 32, "CONV_CI_TARGET": 15.0}` 传入。只在本次评估生效，不改全局。

#### `orchestrate_runs(specs, max_workers=2, compare_after=True, env_snapshot=None)`
批量并行评估（A/B 对比 / 参数扫描）。每个 spec fork 一个隔离工作区跑 runner，全部并行。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `specs` | list[dict] | — | 工作单元规格列表 |
| `max_workers` | int | `2` | 并行度 |
| `compare_after` | bool | `True` | 全部完成后是否自动跑 compare |
| `env_snapshot` | str \| None | `None` | 批次共享快照名（与每个 spec 的 `param_overrides` 叠加） |

**`specs` 每条字段**：

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `name` | ✅ | workspace 名 |
| `target` | | 目标模型名 |
| `source` | | fork 来源（默认 `"global"`） |
| `input_file` | | 攻击集（默认 `"attacks/l1.jsonl"`） |
| `max_rounds` | | 最大轮数（默认 5） |
| `seed` | | 随机种子 |
| `note` | | 备注 |
| `param_overrides` | | 覆写 params.py 参数 |

**返回**：task_view dict（含 `id / status`）。完整结果在任务完成后通过 `get_task_log` / `read_run_report` 查看。

#### 任务轮询工具

##### `get_task_status(task_id)`
**返回**：task_view dict（`id / status / returncode / log_tail(末 4KB) / ...`）；不存在返回 `None`。status: `queued / running / success / failed / cancelled`。

##### `get_task_progress(task_id)`
比 `get_task_status` 更详细——含每目标当前轮次、ASR、Elo 变化。数据来自子进程的 `progress.jsonl`。

**返回**：`{kind, status, targets, max_rounds, progress: {target: {...}}, log_tail}`。

##### `get_task_log(task_id)`
**返回**：`{id, log}`，log 为完整日志文本。

##### `cancel_task(task_id)`
取消排队中/运行中任务。queued 直接标记；running 发 SIGTERM（5s 宽限后 SIGKILL）。已观测的评估结果保留在 R 矩阵中。

**返回**：取消后的 task_view；已结束返回错误。

##### `list_tasks()`
**返回**：所有任务 task_view 列表（倒序）。

---

## 二、HTTP REST API

Web 看板后端，FastAPI + Pydantic。服务于原生 HTML/JS 前端。

```bash
# 启动
uvicorn llmsec.server.dashboard_api:app --host 127.0.0.1 --port 8080
# 交互式文档（自动生成）
# http://localhost:8080/docs   (Swagger UI)
# http://localhost:8080/redoc   (ReDoc)
```

### 2.1 端点总览

共 54 个端点，按 5 个 router 模块组织。所有 `APIRouter()` 无 prefix、无 tags，完整路径直接写在装饰器中。

<details>
<summary>完整端点一览表（点击展开）</summary>

| 方法 | 路径 | 模块 | 简述 |
|------|------|------|------|
| GET | `/` | dashboard_api | 首页 HTML |
| GET | `/api/runs` | data_query | 列出运行批次（可分页） |
| GET | `/api/trend` | data_query | 跨批次安全趋势 |
| GET | `/api/overview` | data_query | 单批次总览（雷达/越狱税） |
| GET | `/api/threats` | data_query | 威胁/强防/意外事件 |
| GET | `/api/elo` | data_query | 攻防 ELO 排名 |
| GET | `/api/report-md` | data_query | 报告 markdown 正文 |
| GET | `/api/report/download` | data_query | 下载报告（.md 附件） |
| GET | `/api/clusters` | data_query | 聚类摘要 |
| GET | `/api/model` | data_query | 预测模型诊断 |
| GET | `/api/attack-sets` | data_query | 攻击集列表 |
| GET | `/api/targets` | data_query | 目标下拉 |
| POST | `/api/targets/add` | data_query | 新增目标到 .env |
| GET | `/api/env` | data_query | 连接配置（api_key 掩码） |
| PUT | `/api/env` | data_query | 更新连接配置 |
| GET | `/api/targets/probe` | data_query | 模型可达性探活 |
| POST | `/api/run/evaluate` | tasks | 启动评估任务 |
| GET | `/api/tasks` | tasks | 列出所有任务 |
| GET | `/api/tasks/{task_id}` | tasks | 查单任务 |
| GET | `/api/tasks/{task_id}/log` | tasks | 完整任务日志 |
| GET | `/api/tasks/{task_id}/progress` | tasks | 任务进度快照 |
| POST | `/api/tasks/{task_id}/cancel` | tasks | 取消任务 |
| GET | `/api/tasks/{task_id}/stream` | tasks | SSE 实时进度流 |
| POST | `/api/attack-sets/upload` | tasks | 上传攻击集 .jsonl |
| GET | `/api/cluster-projection` | cluster_viz | 聚类 2D 投影 |
| GET | `/api/cluster-tree` | cluster_viz | 层次树树图坐标 |
| GET | `/api/cluster-cut` | cluster_viz | 切 k 个簇 |
| GET | `/api/hpo/params` | hpo | HPO 可调参数清单 |
| POST | `/api/hpo/preview` | hpo | 预览搜索空间/成本 |
| POST | `/api/run/hpo` | hpo | 启动 HPO study 任务 |
| GET | `/api/control/workspaces` | control | 列出工作区 |
| POST | `/api/control/fork` | control | fork 新工作区 |
| POST | `/api/control/fork-and-run` | control | fork 并起 runner |
| DELETE | `/api/control/workspaces/{name}` | control | 删除工作区 |
| POST | `/api/control/compare` | control | 对比 run |
| POST | `/api/control/merge` | control | 合并 R 矩阵 |
| GET | `/api/control/llm-status` | control | LLM 是否配置 |
| GET | `/api/control/tools` | control | 中书省工具 schema |
| GET | `/api/control/capabilities` | control | 尚书省能力清单 |
| POST | `/api/control/chat` | control | 中书省对话 |
| POST | `/api/control/chat/reset` | control | 清空 session |
| POST | `/api/control/review` | control | 门下省审查 run |
| POST | `/api/control/plan/approve` | control | 准奏 Plan 并入队 |
| POST | `/api/control/plan/reject` | control | 驳回 Plan |
| POST | `/api/control/plan/block/approve` | control | 放行某步封驳 |
| GET | `/api/control/plan/queue` | control | Plan 队列状态 |
| GET | `/api/control/plan/{plan_id}/status` | control | 查 Plan 状态 |
| GET | `/api/control/plans` | control | 列出最近 Plan |
| GET | `/api/control/bus/feed` | control | 总线消息流（轮询） |
| GET | `/api/control/blocks` | control | 待确认封驳列表 |
| GET | `/api/control/env-snapshots` | control | 列出 .env 快照 |
| POST | `/api/control/env-snapshots` | control | 创建 .env 快照 |
| DELETE | `/api/control/env-snapshots/{name}` | control | 删除 .env 快照 |

</details>

---

### 2.2 数据查询（data_query）

只读数据 API，供看板各面板消费。所有 GET 端点支持 `run` 查询参数指定批次（格式 `YYYY-MM-DD_HHMMSS` 或 `YYYY-MM-DD_HHMMSS/target`，缺省=最新）。

| 端点 | 查询参数 | 说明 |
|------|----------|------|
| `GET /api/runs` | `limit`, `offset` | 分页列出运行批次，富化 target/level/asr，标记 active |
| `GET /api/trend` | `target` | 跨批次趋势时序（asr/fpr/elo/level） |
| `GET /api/overview` | `run` | 单批次总览（雷达图/harm_type asr/边界 ELO/越狱税） |
| `GET /api/threats` | `run` | 威胁榜单（top_threats/strong_defenses/upsets） |
| `GET /api/elo` | `run` | 攻击者 ELO 排名 + 防御者评分 + 每轮轨迹 |
| `GET /api/report-md` | `run` | 返回 security_report.md 正文 |
| `GET /api/report/download` | `run`, `format` | `.md` 附件下载（`format != md` → 400） |
| `GET /api/clusters` | `run` | 聚类摘要（簇/高危/盲点/稳定/validation/hdbscan） |
| `GET /api/model` | `run` | 模型诊断（svd_ridge/blend_predictor） |
| `GET /api/attack-sets` | — | 列出 attacks/ 下攻击集（名称/大小/记录数） |
| `GET /api/targets` | — | 列出 .env TARGETS（仅 name/model，不出 api_key） |
| `GET /api/targets/probe` | `name` | 探测可达性（两段式：models.list + chat smoke） |
| `GET /api/env` | — | 当前连接配置（api_key 掩码） |

**写端点**：

##### `POST /api/targets/add` — 新增目标到 .env
请求体 `AddTargetRequest`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `name` | str | ✅ | 目标名 |
| `model` | str | ✅ | 模型名 |
| `base_url` | str | ✅ | 端点 URL |
| `api_key` | str | ✅ | API key（写入 .env，不回显） |

原子写入 `.env`（写前备份 `.env.bak`），追加 `TARGET_<N>_*` 四件套 + `TARGETS` 列表。

##### `PUT /api/env` — 更新连接配置
请求体 `EnvUpdate`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `target_base_url` | str \| None | |
| `target_model` | str \| None | |
| `target_api_key` | str \| None | 留空=不改 |
| `generator_base_url` | str \| None | |
| `generator_model` | str \| None | |
| `generator_api_key` | str \| None | 留空=不改 |
| `judge_model` | str \| None | |

只写入提供的字段。`updates` 为空 → 400。

---

### 2.3 任务管理（tasks）

子进程任务队列（`subprocess.Popen`），同 kind FIFO 串行，超 64 个淘汰最旧终态。

##### `POST /api/run/evaluate` — 启动评估任务
请求体 `EvaluateRequest`。校验/argv 构造统一在 `llmsec.server.launch`
（与 MCP `run_evaluation`、TUI 同一链路）——目标未在 `.env TARGETS` 声明会 400。

| 字段 | 类型 | 默认 | 校验 | 说明 |
|------|------|------|------|------|
| `phase` | str | `"all"` | `^(all\|1\|2)$` | 运行阶段 |
| `input` | str | `"l1.jsonl"` | `.jsonl` + 防穿越 | 攻击集文件名（attacks/ 下） |
| `batch_size` | int | `min(DEFAULT, ADAPTIVE_MAX)` | `ge=1, le=ADAPTIVE_BATCH_MAX` | 每轮批量 |
| `max_rounds` | int | `DEFAULT_MAX_ROUNDS` | `ge=1, le=MAX_ROUNDS_LIMIT` | 最大轮次 |
| `sampler` | str | `"hybrid"` | `^(gap\|infogain\|coordinate\|hybrid)$` | 采样策略 |
| `sampler_alpha` | float \| None | `None` | — | InfoGain 不确定性权重 |
| `sampler_beta` | float \| None | `None` | — | InfoGain 簇覆盖权重 |
| `sampler_gamma` | float \| None | `None` | — | InfoGain 成功潜力权重 |
| `coordinate_rounds` | int \| None | `None` | — | Hybrid 探索轮数 |
| `target` | str \| None | `None` | `^[\w.\-:]+$` | 单目标 |
| `targets` | str \| None | `None` | — | 多目标子集，逗号分隔 |
| `target_concurrency` | int \| None | `None` | `ge=1, le=32` | 多目标并发数 |
| `no_early_stop` | bool | `False` | — | 跑满轮数不早停 |
| `env_snapshot` | str \| None | `None` | — | .env 快照名（隔离评估，能力与 MCP 对齐） |
| `param_overrides` | dict \| None | `None` | — | 覆写 params.py 参数（同 MCP） |

自动注入 `--publish-global`，预检越狱税探针。返回 task_view（额外带 `has_tax_probe`
与 `meta` 结构化摘要）。

##### 任务操作端点

| 端点 | 参数 | 说明 |
|------|------|------|
| `GET /api/tasks` | — | 列出所有任务（倒序，含 log_tail 末 4KB） |
| `GET /api/tasks/{task_id}` | path: `task_id` | 查单任务状态（不存在→404） |
| `GET /api/tasks/{task_id}/log` | path: `task_id`, query: `download` | 完整日志（`download=1` 返回 .txt 附件） |
| `GET /api/tasks/{task_id}/progress` | path: `task_id` | 进度快照（evaluate=每目标最后一条；hpo=汇总+最近 30 trial） |
| `POST /api/tasks/{task_id}/cancel` | path: `task_id` | 取消任务（已结束→409） |

##### `GET /api/tasks/{task_id}/stream` — SSE 实时进度流
`StreamingResponse(media_type="text/event-stream")`，跟随 `progress.jsonl` 增量行，每行发 `event:progress`；子进程结束发 `event:done` 再关闭。

> 响应头 `Cache-Control: no-cache, X-Accel-Buffering: no`（禁用代理缓冲）。SSE 不可用时降级为轮询 `/progress`。

##### `POST /api/attack-sets/upload` — 上传攻击集
`multipart/form-data`，字段 `file: UploadFile`。校验：后缀 `.jsonl`、非空、首行可 JSON parse（任一不满足→400）。防路径穿越，只取文件名。

---

### 2.4 聚类可视化（cluster_viz）

| 端点 | 参数 | 说明 |
|------|------|------|
| `GET /api/cluster-projection` | query: `method`（`pca`/`tsne`，默认 `pca`） | 高维特征 2D 投影，按 (method, mtime) 缓存 |
| `GET /api/cluster-tree` | — | 层次树树图坐标（dendrogram 的 icoord/dcoord）+ auto-k；降级返回 `{available: false}` |
| `GET /api/cluster-cut` | query: **`k`**（必填，int） | 在层次树切 k 个簇（`[2, n]`，越界→400），按 (k, mtime) 缓存 |

---

### 2.5 HPO 配置台（hpo）

| 端点 | 说明 |
|------|------|
| `GET /api/hpo/params` | HPO 可调 key params 清单（名/类型/当前值/建议范围/分组） |
| `POST /api/hpo/preview` | 预览搜索空间（configs 数 = grid 笛卡尔积 / random / bayesian=max_trials，总 trial = ×repeats ×targets）+ warnings |
| `POST /api/run/hpo` | 写临时 study.yaml 启动 hpo 任务（名为空→400；无 targets 且 fixed 无 target→400） |

##### `HpoRequest` 请求体（preview 与 run/hpo 共用）

| 字段 | 类型 | 默认 | 校验 | 说明 |
|------|------|------|------|------|
| `name` | str | — | 必填 | study 名 |
| `objective` | `_Objective` | `_Objective()` | — | 见下 |
| `strategy` | str | `"bayesian"` | — | 搜索策略（`grid`/`random`/`bayesian`） |
| `max_trials` | int | `20` | `ge=1, le=500` | 最大 config 数 |
| `max_wall_minutes` | int | `0` | `ge=0` | 墙钟上限（0=不限） |
| `trial_timeout_minutes` | int | `30` | `ge=1` | 单 trial 超时 |
| `repeats` | int | `1` | `ge=1, le=5` | 每 config 重复次数 |
| `seed_base` | int | `0` | — | 随机种子基 |
| `space` | dict[str, `_FactorSpec`] | `{}` | — | 因子搜索空间 |
| `fixed` | dict | `{}` | — | 固定参数 |
| `targets` | list[str] | `[]` | — | 目标模型列表 |
| `max_concurrent` | int | `1` | `ge=1, le=8` | 并发数 |
| `est_methods_per_trial` | int | `50` | `ge=1` | 成本预估用 |

**`_Objective`**：`metric`(str, 默认 `"conv_rounds"`)、`direction`(`"minimize"`/`"maximize"`)、`aggregate`(`"mean"` 等)。

**`_FactorSpec`**：`type`(默认 `"float"`)、`low`、`high`、`step`、`log`(bool)、`choices`(list)。

---

### 2.6 控制层（三省制）（control）

三省制编排 UI 的后端。前缀 `/api/control/`。

#### Workspace 管理

| 端点 | 请求体 | 说明 |
|------|--------|------|
| `GET /api/control/workspaces` | — | 列出所有 fork 工作区 |
| `POST /api/control/fork` | `ForkRequest` | fork 新工作区 |
| `POST /api/control/fork-and-run` | `ForkRunRequest` | fork 后异步起 runner（后台子进程任务） |
| `DELETE /api/control/workspaces/{name}` | — | 删除工作区（不存在→404） |

**`ForkRequest`**：`name`(必填)、`source`(默认 `"global"`)、`note`(默认 `""`)。

**`ForkRunRequest`**：`name`(必填)、`source`、`note`、`target`、`input_file`(默认 `"attacks/l1.jsonl"`)、`max_rounds`(默认 5)、`seed`。

#### 对比 / 合并

| 端点 | 请求体 | 说明 |
|------|--------|------|
| `POST /api/control/compare` | `CompareRequest` | 对比多个 run（`runs < 2`→400） |
| `POST /api/control/merge` | `MergeRequest` | 合并 R 矩阵（`confirm=True` 时执行+回写） |

**`CompareRequest`**：`runs`(list[str], 必填)。
**`MergeRequest`**：`sources`(list[str], 必填)、`target`(str, 必填)、`models`(list[str] \| None)、`confirm`(bool, 默认 False)。

#### LLM / 对话

| 端点 | 请求体 | 说明 |
|------|--------|------|
| `GET /api/control/llm-status` | — | LLM 是否配置（`{configured: bool}`） |
| `GET /api/control/tools` | — | 中书省保留工具 schema 列表 |
| `GET /api/control/capabilities` | — | 尚书省完整能力清单 |
| `POST /api/control/chat` | `ChatRequest` | 中书省对话（简单自处理，复杂转尚书省拟案） |
| `POST /api/control/chat/reset` | `ResetRequest` | 清空 session 历史 |
| `POST /api/control/review` | `ReviewRequest` | 门下省审查 run |

**`ChatRequest`**：`text`(str, 必填)、`session_id`(str \| None)。
**`ResetRequest`**：`session_id`。
**`ReviewRequest`**：`run`(str, 必填)、`use_llm`(bool, 默认 True)。

#### Plan 管理（尚书省执行）

| 端点 | 请求体 | 说明 |
|------|--------|------|
| `POST /api/control/plan/approve` | `PlanApproveRequest` | 准奏 Plan → 提交执行队列（异步） |
| `POST /api/control/plan/reject` | `PlanRejectRequest` | 驳回 Plan（同时清封驳） |
| `POST /api/control/plan/block/approve` | `BlockApproveRequest` | 放行某步封驳 |
| `GET /api/control/plan/queue` | — | 查执行队列状态 |
| `GET /api/control/plan/{plan_id}/status` | path: `plan_id` | 查 Plan 状态（不存在→404） |
| `GET /api/control/plans` | — | 列出最近 Plan |

**`PlanApproveRequest`**：`plan_id`(必填)、`session_id`。
**`PlanRejectRequest`**：`plan_id`(必填)。
**`BlockApproveRequest`**：`plan_id`(必填)、`step_id`(必填)。若 Plan 已 done 且有 blocked 步骤被放行，自动重新提交执行队列。

#### 总线 / 封驳 / 快照

| 端点 | 参数 | 说明 |
|------|------|------|
| `GET /api/control/bus/feed` | query: `since`(float), `dept` | 总线消息流（前端轮询补全） |
| `GET /api/control/blocks` | — | 当前待确认封驳列表 |
| `GET /api/control/env-snapshots` | — | 列出 .env 快照 |
| `POST /api/control/env-snapshots` | `EnvSnapshotCreateRequest` | 创建 .env 快照 |
| `DELETE /api/control/env-snapshots/{name}` | path: `name` | 删除 .env 快照（不存在→404） |

**`EnvSnapshotCreateRequest`**：`name`(必填)、`source`(默认 `"global"`)、`note`(默认 `""`)。

---

## 三、CLI 入口

| 命令 | 注册方式 | 入口 |
|------|----------|------|
| `llmsec` | console script | `llmsec.pipeline.runner:main` |
| `llmsec-manage` | console script | `llmsec.management.__main__:main` |
| `llmsec-mcp` | console script | `llmsec.mcp.server:main` |
| `python -m control` | 模块 | `control.cli:main` |
| `python -m llmsec.experiments` | 模块 | `llmsec.experiments.__main__:main` |

### 3.1 `llmsec` — 评估流水线

```bash
llmsec --phase all --input attacks/l1.jsonl --max-rounds 5 --sampler hybrid
llmsec --target minimax --publish-global
llmsec --targets model-a,model-b --target-concurrency 2
llmsec --work-dir output/workspaces/exp1 --no-early-stop   # 实验隔离
```

**22 个参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--phase` | choices | `all` | `all` / `1`(攻击) / `2`(过敏) |
| `--input` | str | `attacks/l1.jsonl` | 攻击集（相对路径锚 PROJECT_ROOT） |
| `--batch-size` | int(≥1) | `DEFAULT_BATCH_SIZE`(10) | 每轮测试攻击数 |
| `--max-rounds` | int(≥1) | `DEFAULT_MAX_ROUNDS`(5) | 最大自适应轮次 |
| `--twin-window` | int(≥1) | None | 过敏检测方法数上限；None=自适应 |
| `--ridge-refit-threshold` | int | `RIDGE_REFIT_THRESHOLD`(10) | 触发 SVD-Ridge 重跑 K-Fold 的阈值 |
| `--refresh-features` | flag | False | 强制重建特征缓存 |
| `--sampler` | choices | `hybrid` | `gap` / `infogain` / `coordinate` / `hybrid` |
| `--sampler-alpha` | float | `1.0` | InfoGain 不确定性权重 |
| `--sampler-beta` | float | `0.3` | InfoGain 簇覆盖权重 |
| `--sampler-gamma` | float | `1.0` | InfoGain 成功潜力权重 |
| `--coordinate-rounds` | int(≥1) | `2` | Hybrid 前多少轮用 InfoGain 探索 |
| `--coord-min-per-cluster` | int(≥1) | `3` | 坐标下降采样器每簇最少实测数 |
| `--targets` | str | None | 多目标，逗号分隔（与 `--target` 互斥） |
| `--target` | str | None | 单目标（与 `--targets` 互斥） |
| `--seed` | int | `get_global_seed()` | 全局随机种子 |
| `--work-dir` | str | None | 实验隔离模式，所有产物写该目录 |
| `--publish-global` | flag | False | 全局模式结束时 publish 进全局 R |
| `--no-early-stop` | flag | False | 跑满 max_rounds 不提前收敛 |
| `--concurrency` | int | None | 批内并行度；不传=全并发；0=串行 |
| `--no-parallel` | flag | False | 禁用批内并行（等价 `--concurrency 0`） |
| `--target-concurrency` | int(≥1) | `1` | 多目标并发数 |

> ⚠ `--work-dir` 设置后**强制 `no_early_stop=True`**，全部 9 个写入点重绑到 work-dir（经 `core.isolation.rebind_to_workdir`），全局 output/ 零写入。`--target` 与 `--targets` 互斥，违反则 `sys.exit(1)`。

### 3.2 `llmsec-manage` — 信息管理

```bash
llmsec-manage runs list --json
llmsec-manage runs delete batch_20260814 --delete-r --yes
llmsec-manage cache clean elo_cache predictors --yes
llmsec-manage snapshot export --source global --out backup.tar.gz
llmsec-manage merge --sources ws:exp1 --target global --yes
llmsec-manage thresholds --json
```

结构：`llmsec-manage <group> <cmd> [options]`，5 个分组。所有写操作默认 **dry-run**，需 `--yes` 才执行。删除走软删除（`.trash/`）。

| 分组 | 子命令 | 关键参数 |
|------|--------|----------|
| `runs` | `list` | `--json`, `--target`, `--since`, `--until`, `--level`, `--no-report`, `--min-size`, `--junk-only` |
| | `delete` | 位置参数 `names`(+)，`--delete-r`，`--yes`，`--json` |
| `cache` | `list` | `--json` |
| | `clean` | 位置参数 `categories`(+)：`predictors`/`feature_cluster`/`model_state`，`--yes`，`--json` |
| `snapshot` | `export` | `--source`(global 或 run:\<name\>)，`--out`，`--json` |
| `merge` | — | `--sources`(+)，`--target`，`--models`(*)，`--yes`，`--json` |
| `thresholds` | — | `--json`（导出 params.py 审查阈值常量） |

### 3.3 `llmsec-mcp` — MCP 服务器

见 [1.1 传输与启动](#11-传输与启动)。

### 3.4 `python -m control` — 控制层

```bash
python -m control workspace fork exp1 --source global --run --target minimax
python -m control workspace list --json
python -m control compare batch_a batch_b --json
python -m control orchestrate specs.json --workers 2
python -m control chat               # 交互式对话
python -m control tool review_run '{"run_name": "batch/model"}'
```

结构：`python -m control <cmd> [options]`，5 个子命令。

| 子命令 | 说明 |
|--------|------|
| `workspace fork <name>` | fork 工作区。`--source`、`--note`、`--run`(fork 后起 run)、`--target`、`--max-rounds`、`--seed`、`--json` |
| `workspace list` | 列出工作区。`--json` |
| `workspace delete <name>` | 删除工作区。`--json` |
| `workspace gc` | 清理已 merge 超期工作区。`--older-than-days`(默认 7)、`--json` |
| `compare <runs...>` | 对比 run。`--json` |
| `orchestrate <specs.json>` | 批量编排。`--workers`(默认 2)、`--json`。specs.json 为 RunSpec 列表 |
| `chat` | 交互式对话中间者 |
| `tool <name> [args_json]` | 直接调工具。位置参数 `name`、`args`(默认 `"{}"`)、`--json` |

### 3.5 `python -m llmsec.experiments` — 实验框架

```bash
python -m llmsec.experiments run study.yaml
python -m llmsec.experiments report my-study
python -m llmsec.experiments trials my-study
```

> ⚠ 该 CLI **不用 argparse**，手动解析 `sys.argv`。参数无类型校验/默认值/help 元数据。

| 子命令 | 位置参数 | 行为 |
|--------|----------|------|
| `run` | `<study.yaml>` | 运行/续跑 study（支持断点续跑） |
| `report` | `<name>` | 打印最佳 config + 对比表（读 `study_dir(name)/study.yaml`） |
| `trials` | `<name>` | 列出全部 trial（读统一库 trials 表） |

---

## 四、配置参考

### 4.1 `.env` 字段表

复制 `.env.example` 为 `.env` 后填写。带 ✱ 的为必填。

#### 连接配置

| 字段 | 必填 | 默认 | 说明 |
|------|:----:|------|------|
| `TARGET_TYPE` | ✱ | `openai` | 目标类型：`openai` / `local_sim` / `pcap_judge` |
| `TARGET_API_KEY` | ✱ | — | 目标 API key |
| `TARGET_BASE_URL` | ✱ | — | 目标端点 URL |
| `TARGET_MODEL` | ✱ | — | 目标模型名 |
| `TARGETS` | | — | 多目标声明（逗号分隔名列表，每名配 `TARGET_<N>_*` 四件套） |
| `GENERATOR_API_KEY` | ✱ | — | 生成模型 key（攻击生成/安全孪生/报告叙事） |
| `GENERATOR_BASE_URL` | ✱ | — | 生成模型端点 |
| `GENERATOR_MODEL` | ✱ | — | 生成模型名 |
| `JUDGE_MODEL` | | 回退 GENERATOR_MODEL | Judge 模型 |

#### 超时 / token 预算

| 字段 | 默认 | 说明 |
|------|------|------|
| `GENERATOR_TIMEOUT` | `60` | 生成模型单次请求超时（秒） |
| `JUDGE_TIMEOUT` | `90` | Judge 单次请求超时（秒） |
| `REPORT_TIMEOUT` | `180` | 叙事报告超时（长任务） |
| `JUDGE_MAX_TOKENS` | `1024` | Judge 最大输出 token（推理模型建议 ≥1024） |
| `GENERATOR_MAX_TOKENS` | `4096` | 生成模型最大输出 token |

#### Embedding

| 字段 | 说明 |
|------|------|
| `HF_ENDPOINT` | HuggingFace 镜像（默认 `https://hf-mirror.com`） |
| `SENTENCE_TRANSFORMERS_HOME` | 本地缓存目录（默认 `llmsec/.models`） |
| `EMBEDDING_MODEL` | embedding 模型（默认 `all-MiniLM-L6-v2`） |
| `EMBEDDING_API_BASE` | API embedding 端点（三项齐全时优先于本地缓存） |
| `EMBEDDING_API_KEY` | API embedding key |
| `EMBEDDING_API_MODEL` | API embedding 模型（如 `bge-m3`） |

#### PCAP Judge

| 字段 | 说明 |
|------|------|
| `PCAP_JUDGE_URL` | `TARGET_TYPE=pcap_judge` 时必填 |

#### 监控告警（新增）

| 字段 | 默认 | 说明 |
|------|------|------|
| `LLMSEC_LOG_LEVEL` | `INFO` | 日志级别（`DEBUG`/`INFO`/`WARNING`/`ERROR`） |
| `LLMSEC_LOG_FILE` | `output/logs/llmsec.log` | 日志落盘路径；置空则不落盘（仅 stdout） |
| `LLMSEC_ALERT_WEBHOOK` | _(空)_ | 告警 webhook URL（飞书/钉钉/企业微信/Slack 入站 webhook）；空=不启用 |
| `LLMSEC_ALERT_LEVEL` | `warning` | 最低告警级别（`info`/`warning`/`error`） |
| `LLMSEC_ZOMBIE_MINUTES` | `60` | 任务 running 超该分钟数无产出则告警（僵尸检测） |

> 告警走双通道：logger.warning（落 llmsec.log，人工可 grep）+ webhook。webhook POST 非阻塞（线程池提交），失败不影响主流程。

### 4.2 `params.py` 行为参数

`llmsec/params.py` 定义了大量控制评估行为的"旋钮"（ELO 因子、收敛阈值、采样权重等），按 10 个分组组织。**完整清单请用工具查看**：

```bash
# MCP 工具
get_params()              # 全部
get_params(category="elo")  # 指定分组

# CLI
llmsec-manage thresholds --json   # 仅审查阈值常量
```

可用 `run_evaluation(param_overrides={...})` 或 `LLMSEC_PARAM_<NAME>=value` 环境变量临时覆写。分组：`pipeline` / `elo` / `ridge` / `sampler` / `judge` / `cluster` / `blend` / `twin` / `report` / `sim`。
