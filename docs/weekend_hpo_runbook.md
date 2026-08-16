# 周末 HPO 长期实验 Runbook

## 实验概览
| 阶段 | 策略 | 配置数 | 墙钟上限 | 说明 |
|---|---|---|---|---|
| smoke | grid | 2 | 25 min | 链路验证（l1.jsonl, 2 rounds） |
| stage1 粗扫 | bayesian TPE | 40 | 1200 min (20h) | 7 维收敛+聚类参数 |
| stage2 细化 | bayesian TPE | 25 | 720 min (12h) | top5 收窄空间, repeats=2, mean_plus_std |
| stage3 新族 | bayesian TPE | 25 | 780 min (13h) | 采样器/批量/判分/Ridge/Blend/HDBSCAN 8 维，锁定 stage2 最优 |

- **目标函数**：`ci_half` minimize（max_rounds=4 固定预算跑满，跨目标均值）
- **攻击集**：`attacks/all_merged.jsonl`（10498 行）
- **目标模型**：`minimax`、`gemma-4-12B-it`（CyberPal 服务未启动、DeepSeek key 失效，探活失败已排除——修复后加回 `weekend_stage1.yaml` 的 targets 重跑即自动补缺）
- 并发：`max_concurrent=4 × config_concurrency=3`（本地端点免费，尽量吃满）

## 命令（均用 .venv）
```bash
.venv/Scripts/python.exe scripts/weekend_hpo.py probe    # 探活
.venv/Scripts/python.exe scripts/weekend_hpo.py smoke    # 冒烟
.venv/Scripts/python.exe scripts/weekend_hpo.py run      # 全流程：stage1 → 自动生成并跑 stage2 → 报告
.venv/Scripts/python.exe scripts/weekend_hpo.py report   # 随时查看汇总 + 建议 .env 追加行
python -m llmsec.experiments trials weekend_stage1       # 逐 trial 明细
```

## 进度观察
- 看板：`python -m llmsec.server.dashboard_api` → http://localhost:8080 （HPO 页实时 trial 进度 + 当前最优）
- 产物：`output/experiments/weekend_stage{1,2}/`（trials.jsonl / summary.json / trial_* 工作目录）

## 中断恢复
所有 study 断点续跑：进程被杀后**重跑同一命令**即从 `trials.jsonl` 恢复，只补缺失的 (config, target, seed) 单元。

## 周一使用最优参数
`report` 会打印 `LLMSEC_PARAM_*` 行，追加到 `.env` 即让正式评估直接采用调优结果。

## 注意
- Windows 睡眠会中断实验：`powercfg /change standby-timeout-ac 0`（管理员）或保持接电+关盖不睡眠
- 框架自带保护：连续 3 个 trial 失败自动中止 + webhook 告警；单 trial 超 40 min 强杀
