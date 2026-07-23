# LLM API 安全性评估系统

一个系统化的黑盒 LLM 安全评估框架。从攻击生成、目标测试、结果评分、ELO 排名、过敏检测到聚类分析，形成完整的评估链路。

## 项目动机

传统 LLM 安全测试存在三个核心问题：

1. **判断粗糙**：依靠关键词匹配（"抱歉"、"I cannot"）判断攻击是否成功，漏判率和误判率双高
2. **评估低效**：大量攻击 prompt 无差别全量测试，无法根据目标模型的防御水平自适应调整
3. **维度单一**：只看"攻没攻破"，不考量误杀率、越狱税、跨类别稳定性等多维指标

## 核心概念

- **越狱税（Jailbreak Tax）**：攻击 prompt 中嵌入数学题，目标模型需以 `[MATH:答案]` 格式作答。越狱成功后若数学推理能力退化（答错或丢失格式），说明模型为"配合"付出了能力代价。
- **反向 ELO**：攻击方法为进攻方、目标模型为防守方。攻击成功 → 攻击方法 ELO 上升；被防住 → 下降。ELO 越高威胁越大，防守方 ELO 即"安全边界"。
- **ASR + FPR 二维画像**：ASR（攻击成功率）衡量防线强度；FPR（误杀率/过敏率）通过"安全孪生"（语义安全但结构相似的 prompt）衡量模型是否误伤正常请求。
- **LLM-as-Judge**：三层递进评分——快速预筛（0 API 调用）→ 合规等级判断（A-E）→ 有害度三维评分（harmfulness / specificity / dangerousness），节省 70% 以上 Judge API 调用。
- **聚类冷启动 ELO**：对未评估过的攻击方法，按文本/技术/意图/防御交互特征近似性分到最近簇，取簇内真实 ELO 的**距离倒数加权平均**作为初始值；同一攻击基底的变体（如 `*_rot13`、`*_b64`、`*_code`、`*_story`）会优先相互借用 ELO，降低跨变体污染。首次运行无真实数据时自动采样少量种子方法做真实评估，再预测其余方法，实现海量攻击数据的免预热接入。
- **动态聚类**：聚类模型只基于真实评估数据构建，预测值绝不参与聚类，避免"死 ELO"污染簇结构。启动时系统会复用已保存的 `cluster_artifacts`；仅在以下情况触发重训练：无 artifacts、攻击集方法列表变化、新增 ground truth 跨过阈值（默认 10）、或显式指定 `--cluster-retrain-force`。攻击过程中簇固定不变，最终聚类在全部真实评估完成后运行一次。HDBSCAN 参数通过 **k-distance 图自动选择 `min_samples` 与 `cluster_selection_epsilon`**；若 ground truth 样本仍被标为噪声，会自动挂回到最近的非噪声簇，确保每个已测方法都是可用锚点。
- **抗假阳性收敛判定**：防御方 ELO 收敛不仅看最近轮次 ELO 标准差，还同时约束相对标准差、最近被测方法成功率必须接近 50%、以及已测方法覆盖率足够。避免"一轮全失败后因更新幅度小而误判收敛"的情况。

## 目录结构

`llmsec/` 是仓库唯一顶层目录，业务代码全部位于 `llmsec/` 包内；所有入口统一通过 `python -m llmsec.<子包>.<模块>` 执行。

```
llmsec/
├── core/         # 配置(.env加载/路径常量)、日志、JSONL IO、LLM客户端重试
├── targets/      # 目标模型后端路由: openai / local_sim / pcap_judge (TARGET_TYPE)
├── evaluation/   # evaluator(评估主流程) / judge(LLM评分) / elo(反向ELO) / safe_twin(过敏检测)
├── attacks/      # generate(L1攻击生成) / harmbench(HarmBench攻击集生成)
├── pipeline/     # runner(自适应编排) / launcher(交互启动器) / probe(受害者探测)
├── reporting/    # report(五维树形画像 + LLM叙事报告 + 方法注册表)
├── clustering/   # 特征提取 + hdbscan/kmeans/hierarchical 聚类 + CLI
└── server/       # local_model_server(本地模拟模型, OpenAI兼容)
```

## 安装

```bash
pip install -r llmsec/requirements.txt
```

Python 3.11。其中 `hdbscan`、`sentence-transformers`、`tiktoken` 为聚类模块的可选/惰性依赖。

## 环境配置

在 `llmsec/.env` 中配置（由 `llmsec/core/config.py` 统一加载）：

```env
# 目标模型（被测试对象）
TARGET_TYPE=openai                     # openai | local_sim | pcap_judge
TARGET_API_KEY=sk-yyy
TARGET_BASE_URL=https://api.deepseek.com/v1
TARGET_MODEL=deepseek-v4-flash

# 攻击生成 & 安全孪生 & 报告叙事（共用）
GENERATOR_API_KEY=sk-xxx
GENERATOR_BASE_URL=https://api.deepseek.com/v1
GENERATOR_MODEL=deepseek-v4-flash

# Judge（缺省回退 GENERATOR_API_KEY / GENERATOR_BASE_URL）
JUDGE_MODEL=deepseek-v4-flash

# 可选
SAFE_TWIN_MODEL=deepseek-v4-flash      # 安全孪生生成模型
EMBEDDING_MODEL=all-MiniLM-L6-v2       # 聚类语义嵌入模型
HF_ENDPOINT=https://huggingface.co     # 嵌入模型下载源
LLMSEC_LOG_LEVEL=INFO

# TARGET_TYPE=pcap_judge 时
PCAP_JUDGE_URL=...
PCAP_MODEL_VERSION=Qwen3.6-35B-A3B
PCAP_PROMPT_KEY=custom:dev
```

**本地无外部 API 测试**：`TARGET_TYPE=local_sim`，配合 `python -m llmsec.server.local_model_server` 即可离线跑通 evaluate 链路。

## 用法

```bash
python -m llmsec.attacks.generate [--dry-run] [--only 1.1.1] [--output PATH]
    # 解析 攻击分析.md 中的 L1 方法，生成攻击集（默认 output/attacks/l1.jsonl）

python -m llmsec.attacks.harmbench [--max N] [--seed N]
    # 生成 HarmBench 攻击集（默认 output/attacks/harmbench_jailbreak.jsonl）

python -m llmsec.evaluation.evaluator [--input attacks/l1.jsonl] [--max-samples N] [--repeat N]
                                      [--only ID] [--start-from ID] [--no-judge] [--judge-model M]
    # 全量评估：逐条发送目标 → Judge 评分 → 更新 ELO。--input 相对 output/ 目录。
    # --no-judge 跳过 Judge（不打外部 API），回退关键词检测。

python -m llmsec.pipeline.runner [--phase {all,1,2}] [--input FILE] [--batch-size N]
                                 [--max-rounds N] [--twin-window N]
                                 [--cluster-retrain-threshold N]
                                 [--cluster-retrain-force]
                                 [--sampler {gap,infogain,coordinate,hybrid}]
                                 [--sampler-alpha A] [--sampler-beta B] [--sampler-gamma G]
                                 [--coordinate-rounds R]
    # 自适应编排：ELO 驱动逐轮攻击（phase 1）→ 边界附近按需过敏检测（phase 2）。
    # --twin-window 未指定时按 ELO 边界置信度自适应窗口（置信度越低窗口越大）。
    # --cluster-retrain-threshold 控制新增多少真实样本后触发聚类重训练（默认 10）。
    #   启动时会优先复用已保存的 cluster_artifacts；仅在必要时（无 artifacts、攻击集变化、
    #   ground truth 增量达到 threshold、或 --cluster-retrain-force）才重训练。
    # --sampler 选择 Phase 1 采样策略：gap（分差最小）/ infogain（全局信息增益）/
    #   coordinate（簇坐标下降）/ hybrid（前 R 轮 InfoGain + 后接 Coordinate，默认）。
    # --sampler-alpha/beta/gamma 调节 InfoGain 的不确定性、簇覆盖、成功潜力权重。

python -m llmsec.evaluation.cluster_analysis [--defender NAME] [--output PATH]
    # 基于当前 ELO 与聚类结果输出簇级安全分析（高风险/盲区/稳定簇）。

python -m tests.clustering_kdistance
    # 离线验证聚类效果：构造 base64/rot13/code 三类攻击，断言簇数 ≥3 且噪声比 <30%。

python -m tests.test_elo_convergence
    # 回归测试：验证 predict_fixed 变体兜底与 check_convergence 抗假阳性。

python -m llmsec.evaluation.elo_cluster --status
    # 查看聚类-ELO 预测器状态：ground truth 数、预测数、簇数、下次触发训练阈值等。

python -m llmsec.evaluation.safe_twin [--generate|--evaluate|--all]
    # 安全孪生生成与过敏（FPR）检测。

python -m llmsec.clustering.cli [--method {hdbscan,kmeans,hierarchical}] [--k N]
                                [--min-cluster-size N] [--weights emb,tech,intent,defense]
                                [--dump-features]
    # 攻击方法聚类分析（默认 hdbscan 自动选簇）。

python -m llmsec.reporting.report [--output-dir DIR]
    # 扫描 DIR 下 *_结果.jsonl，生成五维树形画像 + 方法注册表 + LLM 叙事报告
    #（LLM 不可用时自动回退纯文本报告）。

python -m llmsec.pipeline.launcher
    # 交互式启动器：选择攻击集与模式后引导执行。

python -m llmsec.pipeline.probe [--text "测试文本"]
    # 目标 API 连通性探测。

python -m llmsec.server.local_model_server [--port 8000] [--refusal-rate 0.7] [--math-accuracy 0.55]
    # 本地模拟小模型服务器（OpenAI 兼容），用于无真实 LLM 的离线测试。
```

## 输出文件布局

```
llmsec/output/
├── attacks/                # 攻击集（l1.jsonl、harmbench_jailbreak.jsonl）
├── state/                  # 持久化状态
│   ├── state.json          #   统一 ELO + ground truth 状态（攻击方/防守方 + 历史 + 聚类锚点）
│   └── safe_twins.jsonl    #   安全孪生集
├── runs/<时间戳>/          # runner 单次运行产物
│   ├── attack_results.jsonl
│   ├── runner_report.json  #   综合报告
│   ├── allergy.json        #   过敏报告 + 2D 画像
│   ├── sampler_log.jsonl   #   每轮采样器决策日志
│   ├── cluster_security_analysis.json  # 簇级安全分析
│   ├── security_tree.json  #   五维树形画像
│   └── security_report.md  #   LLM 叙事报告
├── {输入名}_结果.jsonl      # evaluate 逐条结果
├── {输入名}_汇总.json       # evaluate 统计摘要
├── method_registry.json    # 方法注册表（ELO + 聚类标签 + prompt 清单）
├── cluster_report.json     # 聚类报告
├── cluster_matrix.csv      # 方法×特征矩阵
└── cluster_features.json   # --dump-features 导出的特征
```

## 许可

GPL
