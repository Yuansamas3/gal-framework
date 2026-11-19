# 通用本地化中台架构蓝图 (`gal_framework`)

> **设计哲学（「抄它的账本，别抄它的门面」）**：
> 摒弃重型桌面 GUI 外壳与不稳定反代接口，100% 吸收 `GalTransl` 的**核心账本机制（`gt_input` / `transl_cache` / `gt_output` 三分离原子缓存账本、确定性剥壳扣壳、三级条件字典、按批动态词表注入器、`rebuild` 零成本重建、经典 13 项机械问题检测）**与 `LinguaGacha` 的**（控制符正则保护、语域对等判据、自启发种子审校账本）**，打造一套纯标准库 Python CLI + `SKILL.md` 驱动、专为本地多代理并发而生的工业级通用视觉小说 / Galgame 本地化中台。

---

## 一、 运行环境与开源合规规范

1. **Python 运行时**：Python 3.10+（纯标准库实现，零外部第三方依赖，跨平台开箱即跑）。
2. **零单作耦合、零本地隐私残留**：框架源码、内置词典、`sampleProject/`、测试用例与文档中严禁写入任何商业游戏原文或本机私人绝对路径，确保仓库纯净、可随时开源发布。

---

## 二、 框架目录树与标准工程账本结构

### A. 中台框架目录树 (`gal_framework/`)
```text
gal_framework/
├── README.md                          # 框架架构、账本机制与 CLI 六阶流水线说明
├── ARCHITECTURE_BLUEPRINT.md          # 架构设计蓝图与验收契约
├── gal_cli.py                         # 统一 CLI 总控入口（Markdown / JSON 双模输出）
├── core/
│   ├── __init__.py
│   ├── models.py                      # CSentense / SentenceItem 内存模型与 ProjectConfig
│   ├── cache_ledger.py                # [核心账本] transl_cache 读写引擎：兼容新旧键名、原子替换 (.json.tmp -> os.replace)、7 种缓存未命中原因码、cache_history.jsonl 变更审计
│   ├── dictionary.py                  # [条件字典 + 动态注入器] pre/gpt/post 三级字典：完整支持 pre_jp/post_jp/pre_dst 条件 ([and]/[or]/!/(同上)/^^/1^) + 批次局部词表动态注入器
│   ├── normalizer.py                  # [确定性剥壳扣壳] 幂等对话引号剥离(含跨2~3句续接)、首尾空白/换行暂存、统一控制符+Ruby 保护罩内的标点对齐与原壳恢复
│   ├── text_preserve.py               # [控制符保护器] 正则安全核验(防零长/防吞正文)、嵌套 token 追踪、缺/多/错序全量校验与可配置 Ruby 校验模式 (translate | verbatim)
│   ├── problems.py                    # [全维度质检引擎] 16 大核心维度 / 22 项细粒度统一注册检查类型 + 细粒度正则过滤与区间白名单
│   ├── batcher.py                     # [Agent 工作包调度器] batch-pull (--only-miss 增量拉取 + 滑动上下文 + 动态局部词表) / batch-push (实时质检回执) / edit-cache / search-replace
│   ├── rebuilder.py                   # [零成本重建与回滚器] 等同 GalTransl rebuilda + 按 cache_history task_id 一键 revert
│   ├── glossary_tools.py              # [译前扫描器] 说话人频次与复合说话人覆盖统计、重复句提取、术语候选与二元组倒排索引异写巡检
│   └── seed_ledger.py                 # [自启发审校账本] seeds.json 状态机：跨 >=3 区真采样核验、禁止跨轮消账与 >=2 轮完结门禁
├── guidelines/                        # 可插拔通用文风规范库
│   ├── 00_agent_charter.md            # 虚构叙事忠实宪章 + 判据层级 + 残句不补全 + 功能性粗粝不抹平 + Avoid Overthinking
│   ├── zh_light_novel_v2.md           # 现代轻小说/校园日常喜剧版
│   ├── zh_dark_mystery.md             # 悬疑/怪谈/冷冽正剧版
│   └── zh_classical_fantasy.md        # 古风/和风剑戟/西式奇幻宫廷版
├── profiles/                          # 结构化质检阈值与语域探针配置 JSON
│   ├── modern_acg.json
│   ├── dark_mystery.json
│   └── classical_fantasy.json
├── dicts/                             # 通用基础字典库
│   ├── pre_h_phonetic.tsv             # H 场景口齿不清变形音与伏字预矫正字典 (200+ 条，含防误伤条件)
│   ├── pre_common.tsv                 # 通用译前归一化字典
│   ├── gpt_common_acg.tsv             # ACG 通用术语与日式惯用陷阱词参考字典
│   ├── gpt_common_menu.tsv            # 常见游戏系统菜单词典
│   └── post_symbol_common.tsv         # 译后标点与条件符号修复字典（含 pre_jp/post_jp 条件规则、…。->… 等）
├── text_preserve_presets/             # 引擎控制符正则预设 (base/kag/renpy/rpgmaker/wolf.json)
├── adapters/                          # 外部脚本与标准工程 gt_input / gt_output 双向无损桥接器
│   ├── __init__.py
│   ├── gt_json.py                     # 标准 GalTransl JSON (`[{"name", "message"}]`) 读写
│   └── compact_pipe.py                # 紧凑管道文本 (`ID|spk|text` / `ID|[spk]text`) 无损互转
├── sampleProject/                     # [标准可运行验收样板工程] 26 句综合原创合成测试剧本
│   ├── config.json
│   ├── project_guideline.md
│   ├── 项目GPT字典.tsv
│   ├── 项目字典_译前.tsv
│   ├── 项目字典_译后.tsv
│   ├── gt_input/chapter_01.json
│   ├── transl_cache/
│   └── gt_output/
├── skills/gal-framework/SKILL.md      # 供主控与并发翻译/校对子代理加载的标准操作手册
└── tests/test_framework.py            # 全模块单元测试 + sampleProject 全生命周期端到端验收测试
```

### B. 标准工程三分离账本结构 (`<project_dir>/`)
严格遵循物理三分离原则，**绝不允许把译后替换结果直接覆写回原始裸译文**：
1. **`gt_input/*.json`**：原始只读输入文件（含稳定 `index`、可选 `name`/`names`、`message` 及透传扩展元数据 `_meta`）。
2. **`transl_cache/*.json`**：核心缓存账本，每条记录区分保存：
   - `index`、`name` / `name_preview`
   - `pre_src`（兼容旧键 `pre_jp`）：不可变原文
   - `post_src`（兼容旧键 `post_jp`）：经剥壳与 `pre_dict` 后的送译文本（缓存命中校验键；若改动 `pre_dict` 导致某句 `post_src` 变化，仅该句标为 `post_src_changed` 增量重翻，校对锁定句不受影响）
   - `pre_dst`（兼容旧键 `pre_zh`）：AI 初译落盘裸译文（未经 `post_dict` 污染）
   - `proofread_dst`（兼容旧键 `proofread_zh`）：二轮校对/人工精修裸译文（存在时优先级高于 `pre_dst`）
   - `post_dst_preview`（兼容旧键 `post_zh_preview`）：经统一控制符保护下的标点对齐、`post_dict` 条件替换与原壳扣回后的成品预览
   - `trans_by` / `proofread_by` / `problem` / `problems` / `proofread_comment`（兼容 `doub_content`）/ `skip_check`
   - **原子落盘铁律**：所有写入先写 `<file>.json.tmp` 再调用 `os.replace()` 原子替换，任务启动自动清理残留 `.json.tmp`。
3. **`cache_history.jsonl`**：追加式审计账本，记录 `batch-push`、`edit-cache`、`search-replace`、`revert` 的 `task_id`、时间戳、`reason` 与前后字段 diff，支持一键回滚。
4. **`gt_output/*.json`**：最终交付目录，由 `batch-push`/`rebuild`/`export` 生成，改字典或人名映射后跑 `rebuild` 零大模型成本秒级刷新。

---

## 三、 核心齿轮关键算法契约

1. **条件字典与动态注入 (`dictionary.py`)**：
   - 支持 `pre_jp`/`post_jp`/`pre_dst`/`post_dst`/`speaker` 条件域，`[and]`、`[or]`、`!` 取反、`(同上)` 继承、`^^` 句首、`1^` 仅首次，并按查找词长度稳定降序排列。
   - `inject_glossary_for_batch` 仅提取当前批次 `pre_src`/`post_src`/`speaker` 字面命中的 `gpt_dict` 词条（项目词典优先覆盖通用词典）注入工作包。
2. **幂等剥壳与统一保护后处理 (`normalizer.py` + `text_preserve.py`)**：
   - `strip_outer_shell` 带 `shell_stripped` 幂等标记（二次调用不会静默抹除 `pre_dict` 结果），支持单行、嵌套及跨 2~3 条目对话引号剥离，同时防误剥行内旁白混合句。
   - 译后执行顺序固定为：`protect_text(include_ruby=True)` -> `_remove_first_symbol` -> `post_dict.apply` -> 标点/引号配对修复 -> `_fix_last_symbol`（先削多余句号/逗号，再补回 `♪`/`！？`）-> `restore_text` -> 换行策略 -> 扣回原壳。
   - `TextPreserver` 保留嵌套内层控制符不丢弃，同时校验 `缺控制符`、`多控制符` 与 `控制符错序`；`ruby_preserve_mode` 支持 `translate`（默认：允许注音本地化翻译，同时校验注音标签种类/个数与内部嵌套控制符）与 `verbatim`（要求注音逐字一致）两种显式模式。
3. **16 大维度（22 项细粒度）质检引擎 (`problems.py`)**：
   - 全部 22 项检查类型统一收录于 `ALL_PROBLEM_TYPES`，均受 `config.enabled_problems`、`problem_filter_regex` 与 `problem_whitelist` 统一管辖：`缺控制符`、`多控制符`、`控制符错序`、`丢失换行`、`多加换行`（与 `normalizer` 统一以 `post_src` 对 `inner_post_dst` 为唯一基准）、`残留日文`（区分 `原文输出`）、`附着记号残留`（`゛゜っッー・`）、`本无韩文`、`本无括号`/`本有括号`、`本无引号`/`本有引号`、`引入英文`、`字典未使用`、`比日文长`、`单句过长`、`词频过高`、`独白男他`（排除「吉他/维他命/其他」等非代词复合词）、`非GBK字符`、`无依据器官词强补`、`语域违和词`、`翻译失败`。
4. **CLI 命令集 (`gal_cli.py`)**：
   - `init` / `import` / `prep-scan` / `dict` / `guideline` / `make-trial` / `batch-pull` (`--only-miss`) / `batch-push` / `edit-cache` / `search-replace` (`--dry-run`) / `revert` / `rebuild` / `problems` / `seed` / `export` / `status`。

---

## 四、 验收标准

1. `sampleProject/` 提供 26 句综合原创测试剧本与完整项目词典/配置，可随时运行端到端生命周期验收。
2. `python -m unittest discover -s tests -v` 覆盖全部核心齿轮、11 项代码评审修复点与 `sampleProject/` 全生命周期验收，**100% PASS**。
3. 全仓库扫描确认 **0 条本地私人路径或商业私货残留**。
