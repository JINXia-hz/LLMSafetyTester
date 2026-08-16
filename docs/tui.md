# llmsec-tui — 终端指挥台

CLI 的第一等终端界面（Textual）。**独立进程直连**：直接调用 task_manager 与 MCP
工具层，不需要启动 Web 看板（`dashboard_api`）；不开服务也能发起评估、看实时进度、
翻历史 run。

## 安装与启动

```bash
pip install -e ".[tui]"     # textual 是可选依赖（extra），核心安装不含
llmsec-tui                  # 或 python -m llmsec.tui
```

建议在 **Windows Terminal** 下运行：盲文进度条（⣿⣦⣀）、❯✓✗▋、CJK 对齐需要
UTF-8 终端（应用启动时会做 `setup_console()` 兜底，legacy conhost 下字符可能变形）。

## 三面板与快捷键

全局键（延续 Web 看板 `1-7/r` 的习惯）：

| 键 | 功能 |
|---|---|
| `1` / `2` / `3` / `4` | 切面板：任务中心 / HPO 直播 / Runs 浏览 / 宣政殿对话 |
| `r` | 立即刷新（唤醒 2s 轮询线程 + 重载 runs） |
| `?` | 键位速查帮助 |
| `q` | 退出 |

**任务中心（1）**：任务表（状态/进度/来源）+ 选中任务的终端进度窗（盲文条、OLS
平滑进度、`❯ 目标 R3/5 ELO↑ CI± [⣿⣿⣦⣀] 67% ▋`）。

| 键 | 功能 |
|---|---|
| `n` | 发起评估（表单：目标多选/攻击集/轮数/采样策略/种子/批量/采样器超参 αβγ·探索轮/env 快照隔离/参数覆写 KEY=V/no_early_stop） |
| `c` | 取消选中的运行中/排队任务（本机 SIGTERM→SIGKILL；外部任务经 PID 跨进程强杀） |
| `l` | 查看完整日志 |

**HPO 直播（2）**：hpo 任务的放大视图（config/trial 进度 + 目标值 sparkline +
最近 12 条 trial 流水）。

| 键 | 功能 |
|---|---|
| `s` | 启动 study（选 `experiments/` 或 `output/experiments/` 下的 yaml，或手输仓库内路径） |
| `c` | 取消选中的运行中/排队任务（与任务中心同行为） |
| `l` | 查看选中任务完整日志（与任务中心同行为） |

**Runs 浏览（3）**：历史 run 表（等级/ASR/边界 Elo）。

| 键 | 功能 |
|---|---|
| `enter` | 读报告（核心指标 + 门下省 findings + 完整 JSON） |
| `m` | 标记 run（最多 2 个，再按取消） |
| `v` | 对比已标记的 2 个 run（指标透视表） |
| `e` | 查看目标模型的攻击方 Elo 榜 |
| `b` | 安全边界（boundary_elo / 收敛 / 置信度 / CI / 边界上下方法数） |
| `p` | 意外发现（短板=低 Elo 攻击得手 / 强项=高 Elo 攻击失手） |
| `n` | 下一批测试建议（Elo 差距最小的配对，主动采样决策） |

**宣政殿（4）**：中书省对话面板——自然语言或 JSON 指令直接操作控制层
（列出 run / 对比 / 编排 / 快照……规则版意图引擎，无需开看板）。
输入指令回车发送（`help` 查看全部指令）；**Esc 离开输入框**后 `1/2/3/q`
等全局键才可用（输入框聚焦时会吃键）。

## 架构（与看板/MCP 的关系）

```
                    ┌─ dashboard router（FastAPI）─┐
task_manager ───────┼─ MCP server（fastmcp）      ├── 各自进程内 TASKS 注册表（互相隔离）
（子进程任务核心）    └─ TUI（llmsec-tui）          ┘
```

- **TUI 自己提交的任务**：本进程 `TASKS` 直管，可取消；
- **外部任务**（看板/MCP 启动的、或 TUI 重启前的）：task_manager 落盘
  `<task_id>.meta.json`（kind/cmd/pid/状态），TUI 扫描后显示**真实状态**（运行中
  以 PID 存活为准，持有进程崩溃且无人回写终态 → 「已结束」），进度照常直播
  （增量 tail 回放），带存活 PID 的可**跨进程取消**（taskkill /T 连子进程树）；
- 发起评估经统一启动层 `llmsec/server/launch.py`（与 Web/MCP 同链路全能力面：
  env 快照隔离 / 参数覆写 / 采样器超参）；发起 HPO 与看板 `POST /api/run/hpo`
  同一命令（`-m llmsec.experiments run <yaml>`）；
- 渲染层（`llmsec/tui/render.py`）移植自 Web 端 `run-control.js` 的终端拟真组件，
  配色延续「漆夜玄朱」暗色主题。

## 已知边界

- 宣政殿为规则版意图引擎（LLM 版对话在看板 `POST /api/control/chat`，需开服务）；
- HPO 因子选择表单是 Web 端强项，TUI 从简——直接跑已有 study yaml；
- 无 meta.json 的历史外部任务仍显示「外部」（状态未知）；
- orchestrate 批量编排、图表（plotext）未做。
