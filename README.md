# LLM API 安全性评估系统

一个系统化的黑盒LLM安全评估框架。从攻击生成、目标测试、结果评分、ELO排名、过敏检测到聚类分析，形成完整的评估链路。

## 项目动机

传统LLM安全测试存在三个核心问题：

1. **判断粗糙**：依靠关键词匹配（"抱歉"、"I cannot"）判断攻击是否成功，漏判率和误判率双高
2. **评估低效**：大量攻击prompt无差别全量测试，无法根据目标模型的防御水平自适应调整
3. **维度单一**：只看"攻没攻破"，不考量误杀率、越狱税、跨类别稳定性等多维指标

本项目通过多个独立但协同的模块解决上述问题。

---

项目仍处于开发阶段，以下信息属于数个版本之前AI自动生成。仅供参考。

## 整体架构

```
攻击分析.md  ──→  generate_attacks.py  ──→  攻击集_L1.jsonl
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                         evaluate.py                     runner.py
                     (全量手动评估)                   (自适应自动编排)
                              │                               │
                     ┌───────┼───────┐              ┌────────┼────────┐
                     ▼       ▼       ▼              ▼        ▼        ▼
               judge.py  elo.py  safe_twin.py   judge.py  elo.py  safe_twin.py
                     │       │       │              │        │        │
              各数据集结果  elo.json  过敏报告    实时ELO  按需孪生  综合报告
                              │                               │
                              └───────────┬───────────────────┘
                                          ▼                              ▼
                                   cluster.py                     report.py
                                   (含 TF-IDF 自动命名)                 │
                                          │                      security_tree.json
                                  cluster_report.json            security_report.md
                                                                method_registry.json
```

**两条使用路径**：
- `evaluate.py`：手动评估 — 全量遍历攻击集，适合单次完整测试
- `runner.py`：自适应评估 — ELO驱动逐轮搜索 + 按需过敏检测，适合高效定位安全边界

### 数据流说明

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 生成 | `攻击分析.md` (L1方法定义) | API 逐方法生成×5条 | `攻击集_L1.jsonl` (含越狱税数学题) |
| 评估 | 任意 `.jsonl` prompt 文件 | 目标LLM响应 → Judge评分 → ELO更新 | `{输入名}_结果.jsonl` + `{输入名}_汇总.json` + `elo.json` |
| 编排 | 攻击集 + ELO | 三阶段自适应: 攻击→过敏→评判 + 统一注册表 | `runner_report.json` + `security_tree.json` + `security_report.md` + `method_registry.json` |
| 聚类 | 评估结果 + `elo.json` + prompt文本 | 数值特征 + TF-IDF语义特征 → K-Means → 自动命名 | `cluster_report.json` + `cluster_matrix.csv` |

---

## 模块详解

### 1. `generate_attacks.py` — 攻击生成器

解析 `攻击分析.md` 中所有 L1 构建难度的攻击方法，调用 API 生成攻击 prompt 并嵌入越狱税数学题。

**越狱税机制**：数学题答案（`[MATH:数字]`）用于检测目标模型越狱后推理能力是否退化。

### 2. `evaluate.py` — 评估器（主流程编排）

读入攻击集，逐条发送给目标LLM，通过 Judge 评分并自动更新 ELO。支持 `--input` 切换不同攻击集，结果文件按输入文件自动命名。

### 3. `judge.py` — LLM-as-Judge 评分模块

三层递进设计：快速预筛（0 API调用）→ 合规等级判断（A-E）→ 有害度三维评分（harmfulness/specificity/dangerousness）。目标节省70%以上 Judge API 调用。

### 4. `elo.py` — ELO评分与自适应测试

反向ELO语义：攻击成功→ELO下降（高威胁），失败→ELO上升（被防住）。支持自适应测试策略和安全边界估算。

### 5. `safe_twin.py` — 过敏判断模块

为攻击prompt生成语义安全但结构相似的"安全孪生"，测试目标模型是否会误杀安全请求，计算FPR并与ASR组成2D安全画像。

### 6. `cluster.py` — 聚类分析模块

结合数值特征（ASR/ELO/越狱税等）和 TF-IDF 语义特征对攻击方法进行 K-Means 聚类。肘部法则自动选K，为每个簇提取关键词作为自动命名，输出簇画像和风险等级。

### 7. `report.py` — 层级报告生成器

构建五维树形安全画像（by_harm_type / by_attack_category / by_elo_tier / by_functional / by_source），调用 LLM 生成人类可读的 Markdown 安全评估报告。同时生成 `method_registry.json` 统一索引，确保每个攻击方法的ELO评分、聚类标签和 prompt 清单可通过 method 名一键查询。

### 8. `runner.py` — 统一编排器

串联三阶段自适应评估：
- Phase 1: ELO自适应攻击（实时更新 → 二分搜索 → 收敛判断）
- Phase 2: ELO边界附近按需生成安全孪生 → 过敏检测
- Phase 3: 调用 `report.py` 生成树形报告 + 叙事报告 + 方法注册表

---

## 环境配置

`.env` 文件中配置两个API：

```env
# 攻击生成 & Judge & 安全孪生（三者共用）
GENERATOR_API_KEY=sk-xxx
GENERATOR_BASE_URL=https://api.deepseek.com/v1
GENERATOR_MODEL=deepseek-v4-flash

# 目标模型（被测试对象）
TARGET_API_KEY=sk-yyy
TARGET_BASE_URL=https://api.deepseek.com/v1
TARGET_MODEL=deepseek-v4-flash
```

---

## 典型工作流

### 方式A：全流程自动编排（推荐）

```bash
python generate_attacks.py
python runner.py
python runner.py --batch-size 20 --max-rounds 3

# 输出
# output/runner_report.json      — 综合报告
# output/security_tree.json      — 树形数据
# output/security_report.md      — LLM叙事报告
# output/method_registry.json    — 方法统一注册表
# output/elo.json                — ELO排名
```

### 方式B：手动分步评估

```bash
python generate_attacks.py
python evaluate.py --max-samples 20
python safe_twin.py && python safe_twin.py --evaluate
python cluster.py
python report.py
```

---

## 输出文件清单

| 文件 | 格式 | 说明 |
|------|------|------|
| `output/攻击集_L1.jsonl` | JSONL | 生成的攻击prompt（含数学题） |
| `output/{输入名}_结果.jsonl` | JSONL | 逐条评估结果 |
| `output/{输入名}_汇总.json` | JSON | 统计摘要（含ELO + Judge统计） |
| `output/elo.json` | JSON | ELO评分持久化状态 |
| `output/runner_report.json` | JSON | 统一评估报告 |
| `output/safe_twins.jsonl` | JSONL | 安全孪生集 |
| `output/allergy_report.json` | JSON | 过敏报告 + 2D安全画像 |
| `output/security_tree.json` | JSON | 树形安全画像（5维分解） |
| `output/security_report.md` | Markdown | LLM润色的安全评估报告 |
| `output/method_registry.json` | JSON | 统一方法注册表（ELO + 聚类标签 + prompt清单） |
| `output/cluster_report.json` | JSON | K-Means聚类完整报告 |
| `output/cluster_matrix.csv` | CSV | 方法×特征矩阵 |

---

## 许可

GPL
