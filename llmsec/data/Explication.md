# 内置攻击数据

本目录包含从 HarmBench 提取的静态数据，供 `llmsec.attacks.harmbench` 生成**测试与示范用**攻击集（非项目核心；用户可从任意来源自带攻击集，见项目 README「攻击集从哪来」）。

| 文件 | 内容 | 用途 |
|---|---|---|
| `harmbench_behaviors.csv` | 1528 条有害行为（Behavior / FunctionalCategory / SemanticCategory / Tags / ContextString / BehaviorID） | 攻击目标行为库 |
| `human_jailbreaks.json` | 114 个人工越狱模板（含 `{0}` / `{behavior}` 占位符） | 行为包装模板 |

## 出处与许可证

数据提取自 [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)（2026-07-27 提取），原始许可证为 **MIT License**（Copyright © 2024 centerforaisafety）：

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to inclusion of the license notice.

数据不会同步更新；如需新版请自行从上游重新提取（`data/behavior_datasets/harmbench_behaviors_text_all.csv` 与 `baselines/human_jailbreaks/jailbreaks.py`）。
