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
    # 自适应编排：ELO 驱动逐轮攻击（phase 1）→ 边界附近按需过敏检测（phase 2）。
    # --twin-window 未指定时按 ELO 边界置信度自适应窗口（置信度越低窗口越大）。

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
│   ├── elo.json            #   ELO 评分（攻击方/防守方 + 历史）
│   └── safe_twins.jsonl    #   安全孪生集
├── runs/<时间戳>/          # runner 单次运行产物
│   ├── attack_results.jsonl
│   ├── runner_report.json  #   综合报告
│   ├── allergy.json        #   过敏报告 + 2D 画像
│   ├── security_tree.json  #   五维树形画像
│   └── security_report.md  #   LLM 叙事报告
├── {输入名}_结果.jsonl      # evaluate 逐条结果
├── {输入名}_汇总.json       # evaluate 统计摘要
├── method_registry.json    # 方法注册表（ELO + 聚类标签 + prompt 清单）
├── cluster_report.json     # 聚类报告
├── cluster_matrix.csv      # 方法×特征矩阵
└── cluster_features.json   # --dump-features 导出的特征
```

旧路径（`output/elo.json`、`output/攻击集_L1.jsonl` 等）保留兼容回退读取。

## 许可

GPL
