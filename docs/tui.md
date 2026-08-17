# llmsec-tui — shell 式终端指挥台

CLI 的第一等终端界面（Textual）。**v4 范式：TUI = 原生 CLI 外面加一层 shell 转译壳**——
常态是一台控制台（全屏日志流 + 底部命令行，无常驻可视化区域），所有功能手敲命令；
`/` 前缀保留给 TUI 特制功能（宣政殿对话）。独立进程直连，不需要启动 Web 看板。

```
用户输入（ls -al tasks / eval -t glm4 -r 5 / kill ab12）
   ↓ 转译层 tui/commands.py（动词×资源词典 · 旗标 · 路径对象 · 拼写纠错 · 补全）
   ↓ 动作分发 tui/console.py（线程 worker）
   ↓ 既有后端：launch 层 / task_manager / MCP query·actions —— 后端零改动
```

## 安装与启动

```bash
pip install -e ".[tui]"     # textual 是可选依赖（extra）
llmsec-tui                  # 或 python -m llmsec.tui
```

建议在 **Windows Terminal** 下运行：盲文进度条（⣿⣿⣦⣀）、❯✓✗▋、CJK 对齐需要
UTF-8 终端（应用启动时会做 `setup_console()` 兜底，legacy conhost 下字符可能变形）。

## 键位

输入行是唯一焦点（启动即聚焦）：

| 键 | 功能 |
|---|---|
| `Tab` | 应用补全（浮层高亮项；可链式：命令 → 旗标 → 值） |
| `↑` / `↓` | 浮层可见时导航浮层；否则翻命令历史 |
| `Enter` | 浮层可见且高亮项与已输入不同 → 先应用补全（浮层随之刷新/收起）；否则执行当前行 |
| `Esc` | 关浮层；再按清空输入行 |

历史持久化在 `output/state/tui_history.txt`（去重、上限 200 条）。

## 命令词汇表

**Shell 动词（转译特制）**

| 命令 | 说明 |
|---|---|
| `ls [-a] [-l] [资源]` | 列表进控制台流。资源：`tasks`(默认) / `runs` / `targets` / `attacks` / `studies` / `snapshots` / `workspaces` / `cache` / `params`；`ls runs/<目标>` 按目标过滤；`-a` 含已结束/外部，`-l` 长格式（完整 id/cmd/meta） |
| `cat tasks/<id前缀>` | 全屏查看完整日志（q/Esc 关） |
| `cat runs/<run名>` | 全屏查看报告（核心指标 + 门下省 findings + 完整 JSON） |
| `mkdir <名> [--source global\|run:<x>]` | 开辟隔离工作区 |
| `rmdir <名>` | 删除工作区 |
| `rm <run...> [--delete-r]` | 删除 run——预览 + `confirm <token>` 两步执行 |
| `clean <类别...>` | 清理缓存（elo_cache/predictors/feature_cluster/task_logs）——同样两步 |
| `kill <id前缀\|latest>` | 取消任务；外部任务跨进程强杀前 inline `y/N` 确认 |
| `top [id前缀\|hpo]` | 唤起任务直播全屏视图（表格 + 盲文进度/HPO sparkline，2s 刷新），q/Esc 返回 |

**Domain 动词**

| 命令 | 说明 |
|---|---|
| `eval -t <目标> [-r 轮] [-i 攻击集] [--sampler …] [--seed] [--batch-size] [--alpha/--beta/--gamma/--coord] [--phase all\|1\|2] [--twin-window] [--no-early-stop] [--env-snap 快照] [--param K=V,…]` | 发起红队评估（LaunchSpec 全能力面；`--all` 跑全部声明目标） |
| `hpo <study.yaml>` | 启动 HPO study |
| `probe [目标]` | 目标 API 连通性探测 |
| `elo [模型]` / `boundary <模型>` / `surprise [模型]` / `pairing [模型] [--n]` | Elo 榜 / 安全边界 / 双向意外 / 下一批测试建议 |
| `compare <a> <b>` | 对比两个 run（指标透视表） |
| `snapshot list / new <名> / set <名> K=V / rm <名>` | env 快照管理 |
| `confirm <token>` | 执行 rm/clean 预览过的写操作 |
| `help [命令]` / `clear` / `refresh` / `quit` | 速查 / 清屏 / 强制刷新 / 退出 |

**`/` 特制（仅 1 条）**：`/agent <文本>` —— 宣政殿对话（自然语言或 JSON 指令直调
控制层；无参打印引擎 help）。

## 补全与纠错

- **Tab 补全**位置感知：命令名 → 子命令（`snapshot n…`）→ 旗标名 → 旗标值
  （目标名来自 .env、攻击集/快照/study yaml 来自磁盘、run 名 60s 缓存、任务 id
  前缀、待确认 token）；`--target=gl⇥` 内联等号形式同样可补。
- **实时提示行**：输入 `e` 即提示 `eval · 发起红队评估`；未知命令红字 + 最近似建议。
- **拼错自动纠错**：动词/资源级强匹配（Damerau 编辑距离 ≤1 或相似度 ≥0.8 且唯一
  命中）直接执行并回显 `✎ 已纠错：lss → ls`；多候选/弱匹配只列候选不执行；
  未知旗标只建议不纠错。

## 示例会话

```
❯ eval -t glm4 -r 5
✓ 任务 evaluate-1430-a1b2 已入队 · glm4 · 5 轮
  top 看直播 · log evaluate-1430-a1b2 看日志
❯ top                      ← 全屏直播，q 返回
❯ ls -al tasks             ← 全量任务长格式
❯ lss runs
✎ 已纠错：lss → ls
（runs 表）
❯ elo glm4                 ← 攻击方 Elo 榜
❯ rm run_20260817_… 
✓ 已预览：删除 1 个 run（token ab12，5 分钟内 confirm ab12 执行）
❯ confirm ab12
✓ 已执行：删除 1 个 run
❯ /agent 对比 glm4 最近的两个 run
中书 ❯ ……
```

## 架构（与看板/MCP 的关系）

```
                    ┌─ dashboard router（FastAPI）─┐
task_manager ───────┼─ MCP server（fastmcp）      ├── 各自进程内 TASKS 注册表（互相隔离）
（子进程任务核心）    └─ TUI（llmsec-tui）          ┘
```

- **外部任务**（看板/MCP 启动、或 TUI 重启前的）：task_manager 每次状态迁移把
  任务行（kind/cmd/pid/状态/meta）upsert 进目录库（`output/state/catalog.db`，
  P4 起库行即跨进程唯一真相），TUI 轮询查询显示**真实状态**（运行中以 PID 存活
  为准；进程已死无人回写 → 「已结束」），进度照常直播（progress.jsonl 增量
  tail），带存活 PID 的可跨进程取消（`kill`，taskkill /T 连子进程树；取消状态
  回写库行）；
- 发起评估经统一启动层 `llmsec/server/launch.py`（与 Web/MCP 同链路全能力面）；
  HPO 与看板 `POST /api/run/hpo` 同一命令；
- 任务终态时控制台顶部 notify 一次（`top 查看`），不做进度自动刷屏；
- 渲染层 `tui/render.py` 移植自 Web 端 `run-control.js`，配色延续「漆夜玄朱」。

## 已知边界

- `/agent` 为规则版意图引擎（LLM 版对话在看板 `POST /api/control/chat`，需开服务）；
- merge 工作区（`merge_workspaces_*`）两步流未接入，走看板或 MCP；
- `cd` 切换工作区上下文未做（mkdir/rmdir/ls workspaces 可用）；
- 旧世代裸残留文件（无库行、无 meta.json）的历史外部任务显示「外部」（状态未知）；
