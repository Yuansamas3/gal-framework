# `gal_framework` (v2.1) — Universal Visual Novel / Galgame Localization Middleware

`gal_framework` 是一套**纯 Python 标准库实现（零外部依赖）、采用 `gt_input` / `transl_cache` / `gt_output` 物理三分离原子账本**、专为本地 AI Agent 高并发省流流水线打造的通用视觉小说 / Galgame 本地化中台框架。

---

## 核心特性 (v2.1)

- **物理三分离原子账本、跨进程锁与 `O(1)` 增量落盘 (`core/cache_ledger.py` + `core/rebuilder.py`)**
  - `gt_input/*.json`（只读原文）、`transl_cache/*.json`（裸初译 `pre_dst` / 二轮校对 `proofread_dst` / 预览 `post_dst_preview` / 上下文校验键）与 `gt_output/*.json`（成品交付）严格物理隔离。
  - Windows/POSIX 双端跨线程/跨进程可重入文件锁（`file_lock`）+ 唯一临时文件原子落盘（`PID.TID.UUID.tmp` -> `os.replace()` 退避重试），支持多 Worker 安全并发推拉；附带 `cache_history.jsonl` 全量审计与按 `task_id` 逆序一键回滚。
  - `rebuild` 零大模型成本全书重建：修改译后条件字典或人名映射后秒级重刷全书缓存与输出，并完整保留 `post_src_changed` 重翻标记。
  - `push_batch` 与 `edit_cache` 按目标文件 `O(1)` 选择性加载与差集重检，百文件级商业大作毫秒级响应。
- **高密度省流工作包协议与断路器一步连招 (`core/batcher.py`)**
  - **`--session <id>` 记忆去重锁**：首锅注入完整 `<project_bible>` 底色卡，同会话后续批次自动缩减为 2 行 `BibleHash` 指纹提醒（若中途修改底色卡则哈希变化自动重发），消除多轮重复上下文膨胀。
  - **按文件分组极简协议**：支持 `## @file: <name>` 分组输入与 `@@ file: <name>` + `<id>|<译文>` 单行补丁提交（同时 100% 兼容传统 `@@ file|id` + `+译文`、JSON 数组与 JSONL），直接削减 60%~75% I/O Token。
  - **带防绕过熔断的 `batch-push --pull-next`**：`0` 硬错误且单锅软告警未超标时一步返回下一锅未译文本；一旦检出硬错误或软告警飙升（`>8%`），立即熔断刹车（含 GBK 非法字符防崩延迟导出与跨轮未修债追踪），仅吐回问题条目要求原地修绿。
- **确定性剥壳/扣壳与 PUA 私用区控制符保护 (`core/normalizer.py` + `core/text_preserve.py`)**
  - 幂等剥离外层空白/换行及单行、嵌套、跨 2~3 条目连续对话引号（`「…」`/`『…』`）。
  - 采用 Unicode 私用区（PUA）无碰撞哨兵编码，`protect_text(include_ruby=True)` 全程免疫任意译后字典误改并罩住标点对齐（支持句尾控制符前标点对齐）。
  - 内置 `base` / `kag` / `scn_m2` / `renpy` / `rpgmaker` / `wolf` 六套安全控制符预设，支持 `translate` / `verbatim` / `flexible` 三种 Ruby 注音模式。
- **三级条件字典与单字人名构词防误伤 (`core/dictionary.py` + `core/glossary_tools.py`)**
  - 完整支持 `pre_dict`、`gpt_dict`、`post_dict` 前缀修饰（`^^` 句首、`1^` 仅首现）与条件语法（`pre_jp`/`post_jp`/`pre_dst` + `[and]`/`[or]`/`!` + `(同上)`）。
  - 内置单字汉字角色名（如 `潤`、`奏`）日文复合词与活用词干边界识别（自动豁免 `潤沢`、`潤んだ`、`奏でる` 等非人名构词，支持 `[exclude:...]` 标签）。
  - **`sample-debuts` 实证人设采样器**：秒级扫描全剧所有说话人的首次登场上下文窗口、第一人称代词频率（`僕/俺/私/わたし/あーし` 等）与句尾语气助词分布，从源头杜绝盲猜角色辈分与口癖。
- **双档质检分级与双轮自启发种子门禁 (`core/problems.py` + `core/seed_ledger.py`)**
  - 覆盖 16 大维度（22 项细粒度检查），区分必拦硬错误（`ERROR`：缺/多/错序控制符、假名/附着记号残留、擅加括号、引入英文、无依据器官强补、语域违和词等）与软提示（`WARN`：单句过长、比日文长、字典未使用等）。
  - 实施**「单锅 8% 比率熔断 + 全书 `min(2%, 50条)` 绝对硬顶」**双层防债机制，配合跨 3 区真采样、`>=2` 轮隔离消账的 `seed` 门禁。

---

## 目录架构

```text
gal_framework/
├── README.md
├── LICENSE
├── ARCHITECTURE_BLUEPRINT.md
├── gal_cli.py                         # 统一 CLI 总控入口（Markdown / JSON 双模输出）
├── core/
│   ├── models.py                      # CSentense / SentenceItem 模型与 ProjectConfig
│   ├── cache_ledger.py                # gt_input/transl_cache/gt_output 三分离原子账本与 history 日志
│   ├── dictionary.py                  # 三级条件字典、单字构词边界保护与局部词表注入器
│   ├── normalizer.py                  # 幂等外层剥壳扣壳、跨行对话识别与统一保护标点对齐
│   ├── text_preserve.py               # 控制符与 Ruby 保护器（防零长/防吞正文/嵌套追踪/错序校验）
│   ├── problems.py                    # 16 大维度双档（ERROR / WARN）全自动质检器
│   ├── batcher.py                     # Session 去重工作包、断路器一步连招、edit-cache 与 search-replace
│   ├── rebuilder.py                   # 零成本重建器 (rebuild) 与按 task_id 回滚器 (revert)
│   ├── glossary_tools.py              # 说话人统计、sample-debuts 初登场实证采样与异写巡检
│   └── seed_ledger.py                 # 自启发审校种子账本（真采样校验 + 软告警硬顶 + 轮次隔离门禁）
├── guidelines/                        # 通用宪章与三套题材翻译规范
├── profiles/                          # 结构化质检与语域探针配置 JSON
├── dicts/                             # 通用内置基础字典库
├── text_preserve_presets/             # 引擎控制符预设 (base / kag / scn_m2 / renpy / rpgmaker / wolf)
├── adapters/                          # 双向无损格式适配器 (gt_json / compact_pipe)
├── sampleProject/                     # 标准可运行全链路验收样板工程
├── skills/gal-framework/SKILL.md      # AI Agent 六阶省流本地化操作手册
└── tests/test_framework.py            # 24 项全模块与端到端单元测试套件
```

---

## CLI 命令速查

| 命令 | 用途 | 常用示例 |
| --- | --- | --- |
| `init` | 初始化三分离工程与默认词典 | `python gal_cli.py init --project ./my_game --profile modern_acg --preserve base,kag` |
| `import` | 导入外部脚本至 `gt_input/` | `python gal_cli.py import --project ./my_game --format compact_pipe --input ./raw_scripts` |
| `prep-scan` | 扫描说话人、重复句、专名与控制符 | `python gal_cli.py prep-scan --project ./my_game` |
| `sample-debuts` | 提取全角色首次登场上下文与自称词统计 | `python gal_cli.py sample-debuts --project ./my_game --top 25 --window 6` |
| `dict` | 查看/增删项目 pre/gpt/post 字典 | `python gal_cli.py dict --project ./my_game --kind gpt --action add --src "星見坂" --dst "星见坂"` |
| `guideline` | 查看/追加 `project_guideline.md` | `python gal_cli.py guideline --project ./my_game --action show` |
| `make-trial` | 生成 100 句试译定调工作包 | `python gal_cli.py make-trial --project ./my_game --limit 100` |
| `batch-pull` | 按会话去重拉取未命中工作包 | `python gal_cli.py batch-pull --project ./my_game --session w1 --batch-size 120` |
| `batch-push` | 提交补丁、增量质检并一步拉取下一锅 | `python gal_cli.py batch-push --project ./my_game --patch-file ./p.txt --session w1 --pull-next` |
| `edit-cache` | 按 index 定向修补缓存并记审计日志 | `python gal_cli.py edit-cache --project ./my_game --file ch01.json --index 12 --proofread-dst "精修译文"` |
| `search-replace` | 全局搜索替换（支持 `--dry-run`） | `python gal_cli.py search-replace --project ./my_game --find "旧译" --replace "新译" --dry-run` |
| `revert` | 查看历史或按 `task_id` 一键回滚 | `python gal_cli.py revert --project ./my_game --task-id <task_id>` |
| `rebuild` | 零大模型成本重刷全书缓存与输出 | `python gal_cli.py rebuild --project ./my_game` |
| `problems` | 按 `all\|error\|warn` 查看质检排行与明细 | `python gal_cli.py problems --project ./my_game --severity error` |
| `seed` | 驱动自启发审校循环与硬门禁 | `python gal_cli.py seed --project ./my_game --action status\|sample\|add\|consume\|complete-round\|check-gate` |
| `export` | 检查完成度并无损导出最终文本 | `python gal_cli.py export --project ./my_game --format compact_pipe --output ./out_cn` |
| `status` | 查看进度、缓存命中、软硬告警与门禁总览 | `python gal_cli.py status --project ./my_game` |

---

## 运行测试

需要 Python 3.10+（纯标准库，零外部依赖）：

```bash
python -m unittest discover -s tests -v
```

---

## 致谢与开源协议 (Acknowledgments & License)

本项目在设计理念与工程范式上深受以下优秀前辈项目的启发，特此致以诚挚感谢：

- **[GalTransl](https://github.com/GalTransl/GalTransl)**（GPL-3.0）：物理三分离缓存账本（`gt_input` / `transl_cache` / `gt_output`）、条件字典语法、外层对话壳剥离及高频质检维度等经典本地化工程思想的先驱。
- **[LinguaGacha](https://github.com/neavo/LinguaGacha)**：Agent 原生工作流编排、控制符保护审计与自启发多轮种子审校机制的设计参考。

本项目全部核心代码均为基于 Python 标准库的独立实现，遵循 **[GNU General Public License v3.0 (GPL-3.0)](./LICENSE)** 开源协议发布。仅供个人学习、技术研究与合法本地化辅助交流使用。

### 免责声明 (Disclaimer)

1. **纯通用文本处理工具**：本仓库仅提供面向通用结构化文本（JSON / TSV / 纯文本）的缓存账本、格式校验与本地化辅助框架，**不包含任何商业作品的受版权保护内容、原始剧本、加密密钥、可执行文件修改或DRM/签名绕过代码**。
2. **非商业与合规使用**：本项目为个人技术学习与自用工作流归档，严禁将本项目用于任何非法用途、盗版分发或商业倒卖牟利。
3. **责任界定**：用户在使用本工具处理任何第三方文本前，应自行确保已合法持有相关作品并符合所在地法律法规；因用户不当使用所引发的一切版权或法律责任均由用户本人独立承担，与本项目作者无关。
4. **维护说明**：本仓库主要作为个人自用备份公开存档（As-Is），不提供任何明示或暗示的担保，亦不承诺 Issue / PR 响应时效。
