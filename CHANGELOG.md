# 更新日志 / Changelog

本项目的所有显著变更记录于此。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增

- 攻击记录契约（`llmsec/attacks/schema.py`）：`AttackRecord` 单一 schema 权威——必填三件套
  硬校验（含空白 prompt 拒收）、harm_type/source 宽松枚举、harmbench 溯源字段透传、
  进化血统字段预留（evolved/operator/parent_id/generation）
- 攻击集体检校验器（`python -m llmsec.attacks.validate`）：schema 违规 / harm_type 分布
  与 other 占比 / UTF-8→GBK mojibake 特征命中 / 文件内外重复 / method-category 基数 /
  跨文件重复组，明细落盘 `output/attack_set_health.json`。定位是体检不是门禁
- 攻击集清洗器（`python -m llmsec.attacks.clean`）：mojibake 分段确定性修复（GBK 逆解码，
  真 UTF-8 中文段保留）+ 孤立标记启发式补全（em-dash/右引号，逐条记入 repaired 字段）；
  method 去序号恢复模板族聚合（原值存 method_raw）；harm_original 保全非六类标签；
  all_merged.jsonl 从清洗后成员重建（保留成员原 id，恢复可连接性）。产物落
  `attacks/cleaned/`，原件零改动。实测：mojibake 2090→0，jailbreakv28k method
  4530→326 个模板族
- harm_type 抽样重标校准器（`python -m llmsec.attacks.relabel`）：按 source 分层抽样
  （默认 500）LLM 语义归类，独立报告不写回数据文件；`--dry-run` 零 API 看样本构成
- 生成器薄接口（`llmsec/attacks/base.py`：`AttackGenerator` 协议 + `ensure_contract`
  自检）；generate.py / harmbench.py 输出端接入契约自检（违规即停写，行为不变）
- 外部产物导入通道（`llmsec-manage attacks import`）：契约校验 → source 登记 →
  三空间 id 冲突检测 → `attacks/imported/<source>.jsonl`；dry-run 默认。
  配套 `docs/攻击集导入.md`（契约字段表 / 避坑清单）——外部产物走通道即合规，
  不要求交出生成代码
- 攻击有效性评估（V1/V2/V3）：静态质量评估器（`python -m llmsec.attacks.quality`，
  方法贯彻度/危害实质性/构造质量三维锚定量表 + 问题标签，缓存于
  `attack_quality.json` 支持续跑）+ 融合层（`python -m llmsec.attacks.assess <run_dir>`，
  低 ASR × 低质量 = 假防御嫌疑——修正安全边界的解释层，不重算 Elo）+
  `generate_reports` 自动挂接（产出 `attack_validity.json` / `attack_rectification.md`
  整改需求报告并并入 runner_report）

### 修复

- 上线前终审第五轮（P0/P1 十项 + 缓后两项收口，全部活体复现后修复）：
  - **P0 假防御甄别整链失效**：质量缓存键加 prompt 指纹（C-6）时评估明细行
    不带 prompt，两侧键恒不等——攻击有效性评估全部单位误判"质量分缺失"。
    现明细行落 `prompt_sha16`（`_build_attack_row`），`quality_key` 双侧同源，
    assess 增加零命中 ERROR 哨兵防键口径再次静默漂移
  - **门下省 fail-closed 故障票序列化崩溃**（E-3 路径）：`issue_block` 返回的
    BlockTicket 对象未 `to_dict()`，门下省回调一旦异常文牍/Plan 落库即抛
    TypeError、Plan 永久卡 executing。现与正常路径同口径返回 dict
  - **执行期放行被静默吞掉**：worker 摘牌窗口内 executor 收尾重入队恒被
    submit 判重拒绝。现 running 中再提交 = "当前轮结束后重跑"语义（ctl_queue
    行 per-entry 生命周期：mark 只迁 queued、finish 只关 running、恢复去重）
  - **用户驳回被收尾改判**：最后一层执行期间的驳回会被 done/approved 覆盖。
    现收尾前复查圣裁终局（与层间 E-5 检查共用 `_abort_finish`）
  - **judge_parse_fallback 污染 live Elo**：attack_phase 两处过滤漏滤第二种
    降级模式，live 与 R 派生 Elo 分叉。抽 `scoring.elo_eligible` 单源三处共用
  - **生产链路 FPR 失真**：allergy_phase（runner 主链路）Judge 降级条目计入
    FPR 分子分母且单关键词命中即判过敏——与 safe_twin（CLI 路径）口径分叉。
    现对齐：降级剔除（计 `judge_failed_count` 单列）+ 阈值同源
    `≥PRESCREEN_REFUSAL_HITS`（拒绝关键词计数四处收口为 `judge.refusal_hits`）
  - **外部任务 SSE 必崩**：`_current_status` 裸访问 Task.returncode（模型无该列），
    跨进程任务连流零事件即断。改 getattr 兜底
  - **clone_from_run 静默空 R**：state.json 损坏时非严格解析回退空 history 绕过
    守卫、空矩阵覆写 dest。现 strict 解析 + 显式 ValueError
  - **跨攻击集撞名 id**：R 行观测单位与本集单位不一致（重新生成攻击集的典型
    场景）时记录池被误标已测、漏测 prompt。resume 标记加 unit 一致性过滤
  - storage 契约 R 域收口（缓后 A-5）：service 层 8 处直连 rstore 改道
    `storage.contract`（R 域 API 补齐导出）+ AST 守卫禁止包外 import 子模块
  - real_api 判别力测试断言方向写反（A/B=配合有害请求=攻击成功），修正为与
    评级方案一致；此前"通过"只是 judge 端点返空、兜底猜 D 碰巧满足反断言
- 契约缺口：全空格 prompt 此前骗过 min_length=1 校验，现按空白拒收
- 门下省封驳待裁计数跨页/重放失配：放行（`plan/block/approve`）与 Plan 驳回
  （`plan/reject`）清封驳令时新增总线广播 `step_unblocked`（信封带 plan_id，
  payload 带 step_id/reason=approve|reject），门下省面板消费该消息幂等递减
  待裁计数并把封驳卡按钮翻成已放行印——此前只在"点按钮的那一页"本地递减，
  他页放行或刷新重放会让徽标恒卡「封驳 N 起 · 待圣裁」、按钮残留可点（404）。
  同修 menxia.js 两处既有 bug：封驳卡 `data-plan` 误读 `m.payload.plan_id`
  （plan_id 在消息信封顶层）导致纯 GUI 点「准奏放行」必 404；放行请求 404
  （令已被他处清除）时按已处理收场而非恢复按钮。新增
  `list_tickets_for_plan`（storage 契约）供驳回前取令清单逐令广播

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
