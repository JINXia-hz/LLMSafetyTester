# 更新日志 / Changelog

本项目的所有显著变更记录于此。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增

- 攻击记录契约（`llmsec/attacks/schema.py`）：`AttackRecord` 单一 schema 权威——必填三件套
  硬校验、harm_type/source 宽松枚举、harmbench 溯源字段透传、进化血统字段预留
  （evolved/operator/parent_id/generation）
- 攻击集体检校验器（`python -m llmsec.attacks.validate`）：schema 违规 / harm_type 分布
  与 other 占比 / UTF-8→GBK mojibake 特征命中 / 文件内外重复 / method-category 基数 /
  跨文件重复组，明细落盘 `output/attack_set_health.json`。定位是体检不是门禁
- 首次全量体检结论（22,726 条 / 10 文件）：契约违规 0；other 危害占比 61.9%；
  jailbreakv28k 有 2,090 条 mojibake（占其 46%）；五份外部数据集 method 字段
  逐条唯一（无法用于方法级聚合）；all_merged.jsonl 为重排 id + 重新注题的
  再生数据，与成员文件 id 不可连接

## [1.1.0] - 2026-08-18

### 新增

- TUI v4：shell 式终端控制台（ls/cat/eval/top 等动词直译后端动作），面板/表单交互整体退役
- MCP 服务器：50+ 工具按四风险层级暴露给外部 agent，写操作走 preview→confirm 两步确认；probe_targets 连通性探测、get_params + param_overrides 调参闭环
- 三省 Agent 控制层：中书省规划对话 / 尚书省结构化 Plan 异步执行队列 / 门下省审查封驳，文牍机制 + 消息总线驱动
- ML 预筛：TF-IDF + LogReg 两层拦截，run 结束后静默自动重训（≥300 条数据自动启用）
- 监控告警轻量闭环（webhook + alerts.jsonl）与僵尸任务检测
- 周末 HPO 实验框架（Optuna study 编排 + 启动器集成）

### 变更

- 存储治理 P1–P9：统一 SQLModel/SQLite 目录库 `catalog.db` 独占全部状态——meta.json/trials.jsonl/json 链退役、elo_cache 表化、predictor_cache 真 LRU、reconcile 退出热路径
- control 层全量库化：文件模拟机器清零、守卫单源化、冻结导入白名单清零
- 并发机制重构：评测与实验批次并行加速

### 修复

- 全库多轮审计（r1–r9）：Elo/R 观测正确性、Judge 故障保守失败（ASR 不再虚报）、推理模型 content=None 全链路、报告 asr=0 显示与越狱税空响应假退化
- 测试环境隔离：输出目录全量隔离，堵住 cluster/allergy/.env.bak 三族真实数据覆盖泄漏
- CI 平台竞态：TUI top 直播表断言收敛化、队列 worker 与引擎 dispose 的 teardown 竞态
- CodeQL 安全告警清零 + 自定义 path-injection sanitizer

### 工程（v1.1.0 发布冲刺）

- 测试清单 INVENTORY 校验从死声明变为 CI 实门禁（生成器入库 + `--check` 步骤）
- 覆盖率棘轮：CI 加 `--cov-fail-under=79`（基线 81%，只升不降）
- `.env.example` 补全 35+ 缺失键（多目标方案 / PCAP / 三省 Agent / LLMSEC_PARAM_*），新增双向防漂移测试
- PCAP 后端 TLS 校验可配置（`PCAP_VERIFY_TLS`），三处 `verify=False` 收敛到单开关
- 依赖审计分车道：PR advisory + 每周 blocking audit（带白名单复审纪律）
- flake-check 每日竞态重放（热点文件 ×5）；`_wait_until` 助手四处副本收敛到 `tests/utils.py`
- env 快照 merge 备份保留策略（最近 5 份，密钥明文副本不再无限堆积）
- 发布脚手架：release workflow（tag → GitHub Release + Docker 版本标签）、本 CHANGELOG、PyPI 就绪的包元数据

## [1.0.0] - 2026-08-06

首个正式版本：黑盒 LLM 安全评估框架全链路。

- 评估内核：自适应采样（gap/infogain/coordinate/hybrid 四采样器）、双边 Elo 评级与 95% CI 收敛判据、冷启动预测（SVD-Ridge + Blend）、攻击聚类单元化
- 三阶段管线：攻击评估 → 安全孪生误杀检测（FPR）→ 量化安全报告（含越狱税）
- Web 看板（FastAPI + SSE 任务流）、CLI 四入口、Docker 部署（full/slim 双镜像）
- 质量基建：约 970 个离线测试、ruff + CodeQL + pre-commit 门禁、GitHub Actions CI（ubuntu/windows 双平台）
