# Built-in Attack Data

> English | [中文](Explication.md)

This directory contains static data extracted from HarmBench, used by `llmsec.attacks.harmbench` to generate **test and demo** attack sets (not project-core; users can bring their own attack set from any source — see the project README's "Attack sets" section).

| File | Contents | Purpose |
|---|---|---|
| `harmbench_behaviors.csv` | 1528 harmful behaviors (Behavior / FunctionalCategory / SemanticCategory / Tags / ContextString / BehaviorID) | Attack target behavior library |
| `human_jailbreaks.json` | 114 human jailbreak templates (with `{0}` / `{behavior}` placeholders) | Behavior-wrapping templates |

## Attribution and license

Data was extracted from [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) (extracted 2026-07-27); the original license is **MIT License** (Copyright © 2024 centerforaisafety):

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to inclusion of the license notice.

The data is not synced; for a newer version, re-extract it from upstream yourself (`data/behavior_datasets/harmbench_behaviors_text_all.csv` and `baselines/human_jailbreaks/jailbreaks.py`).
