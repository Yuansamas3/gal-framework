---
name: gal-framework
description: 使用 gal_framework v2.0 高效省配额视觉小说/Galgame 本地化中台（gt_input/transl_cache/gt_output 三分离原子账本 + 会话圣经去重 + 紧凑补丁协议 + 双层分级质检）执行导入、实证登场深挖、100句试译定调、轻量短生命周期并发生产、零成本重建、自启发种子审校与无损导出。
---

# `gal_framework` v2.0 原生代理高效率本地化操作手册（Quota-Lean & Two-Tier QA Edition）

本技能指导主控代理（Coordinator）与并发翻译/校对子代理（Worker Subagents）通过统一中台 `gal_cli.py` 协同作业。严禁手写临时脚本直接篡改 `gt_input/` 或绕过 `transl_cache/` 原子账本与质检门禁。

---

## 核心铁律（主控与子代理必读）

1. **三分离账本不可破坏**：
   - `gt_input/*.json`：只读原文输入，绝不覆写。
   - `transl_cache/*.json`：核心缓存账本，严格分离 `pre_src`/`post_src`、`pre_dst`、`proofread_dst` 与 `post_dst_preview`。
   - `gt_output/*.json`：最终交付产物，初译期间可用 `--no-refresh-output` 延后集中导出，修改词表后用 `rebuild` 一键零成本重刷。
   - 所有变更自动追加至 `cache_history.jsonl`，误操作可凭 `task_id` 执行 `revert` 一键回滚。
2. **剥壳扣壳与控制符保护交给机器**：
   - 框架已幂等剥离外层空白、换行及单行/嵌套/跨 2~3 行对话引号。子代理仅需翻译 `<input>` 中的纯净内核 `POST_SRC`，严禁重复套加外层 `「」`。
   - `post_dict` 与标点修复均运行在统一控制符+Ruby 保护罩内，不会改坏控制符与注音标签。
3. **双层分级质检（`ERROR` 熔断必改 vs `WARN` 债务预算）**：
   - **硬错误（`ERROR`）**：`缺控制符`、`多控制符`、`控制符错序`、`丢失换行`、`多加换行`、`残留日文`、`原文输出`、`附着记号残留`、`本无韩文`、`本无括号`、`引入英文`、`非GBK字符`、`无依据器官词强补`、`语域违和词`、`翻译失败` 及自定义 `arinashi` 规则——**当场阻断推进，必须立即修复！**
   - **软提示（`WARN`）**：`单句过长`、`比日文长`、`字典未使用`（含单字人名自动防复合词误报与 `[exclude:詞1,詞2]`）、`本有括号`、`本无引号`、`本有引号`、`独白男他`、`词频过高`——**初译流水线直接忽略，不返工重刷**；但全书软警告受 `max_warn_ratio`（默认 2%，最低 5 条）预算约束，超预算将在 `seed` 终审门禁中升级阻断。
4. **Avoid Overthinking**：先在脑中立起画面与角色声音，直觉落笔；只修真硬伤，绝不反复重刷已通过条目。

---

## 六阶 Quota-Lean v2.0 标准流水线

### 阶段 0：工程初始化与导入
```bash
python gal_cli.py init \
  --project <project_dir> \
  --profile modern_acg \
  --guideline zh_light_novel_v2.md \
  --encoding utf8 \
  --preserve base,kag \
  --ruby-mode translate

python gal_cli.py import \
  --project <project_dir> \
  --format gt_json \
  --input <source_dir>
```

### 阶段 1：译前筑基扫描 + 角色首秀实证深挖（严禁盲猜人设！）
```bash
# 1) 扫描说话人、高频重复句、专名候选与控制符样本
python gal_cli.py prep-scan --project <project_dir>

# 2) 自动挖掘 Top 角色首次登场 ±6 行真凭实据上下文、第一人称自称与句尾口癖统计
python gal_cli.py sample-debuts --project <project_dir> --top 25 --window 6

# 3) 录入确认后的项目术语（单字角色名支持内置复合词保护及可选 [exclude:潤沢,湿潤] 标签）
python gal_cli.py dict --project <project_dir> --kind gpt --action add --src "潤" --dst "润" --info "男主角 [exclude:潤沢,湿潤]"
```
- **铁律（凭 `prep-scan` + `sample-debuts` 实证建立三层 `<project_bible>`，严禁白纸盲翻！）**：
  1. **第一层·世界观与戏剧核心**：核实作品发售背景、题材基调与路线核心矛盾。
  2. **第二层·实证人物关系与声线矩阵**：直接依据 `sample-debuts` 输出的首次登场前后行（`index | speaker | pre_src`）与第一人称/句尾统计（`僕/俺/私/わたし/あーし`、`っす/ですわ/かしら/わね/だぜ/なの/じゃ`），锁定真实身份（如前辈后辈/师徒/青梅竹马/上下级）、辈分、互称与反差声线。
  3. **第三层·剧本家笔触与文体音叉**：明确句法呼吸、吐槽/抒情机制与 H 戏温度美学。
  4. 将浓缩后的三层圣经写入 `project_guideline.md`。

### 阶段 2：100 句试译定调（风格校准门禁）
```bash
python gal_cli.py make-trial --project <project_dir> --limit 100
```
- 主控完成试译并 `batch-push` 确认 0 硬错误后，校准文风再启动全量并发。

### 阶段 3：Quota-Lean 并发批次生产（轻量自定义 Worker + 短会话上限 + 一步推拉循环）

1. **严禁直接派发携带冗余工具链的重型默认子代理！**
   - 直接继承主控的全套外部插件、长系统提示词与无关工具会在每回合重复计费。主控应**先注册一个超轻量专属翻译子代理（`gal_translator`）**，仅保留最小文件/命令工具权限、关闭多余 MCP 与嵌套派生能力，并将精简翻译宪章、`<project_bible>` 与紧凑补丁规范一次性固化进其系统提示词：
   ```text
   Worker Role: gal_translator
   Tools: minimal file/CLI write tools only (disable extra MCP & nested subagents)
   System Prompt: <精简翻译宪章 + project_bible + 紧凑补丁规范>
   ```
   - 随后按波次并发派发多个轻量 `gal_translator` 实例（配合中低推理预算档位即可）。
2. **反 `O(T^2)` 短生命周期分片上限（Anti-`O(T^2)` Short-Lived Worker Cap）**：
   - **严禁给单个子代理对话分配 2,000+ 行（50+ 回合）长任务！**
   - 每个 `gal_translator` 子代理分片严格封顶在 **400–600 行（按 120–150 行/批 = 仅 3–4 批，总回合数 `T <= 8`）**，按波次（Wave）派生全新 `gal_translator` 实例。将单会话控制在 `T <= 8` 可彻底消除多轮历史平方级滚雪球（上下文从 `286k` 压至 `< 18k`，削减 `>90%` `cache_read` 开销）！
3. **会话圣经去重 + 分组紧凑补丁 + `--pull-next` 一步闭环（含硬错误熔断器）**：
   - 首轮拉取（自动写入 `.sessions/<worker_id>.json` 并在同会话后续批次自动省略未变圣经，且自动启用 `## @file:` 紧凑分组渲染）：
     ```bash
     python gal_cli.py batch-pull \
       --project <project_dir> \
       --files "sc01_*.json" \
       --session worker_w1_a \
       --batch-size 120 \
       --context 6
     ```
   - 紧凑分组补丁格式（无需重复写文件名或 `+` 前缀，`<id>|<dst>` 仅按首个 `|` 切分）：
     ```text
     @@ file: sc01_01.json
     1|莲，早上好
     2|喂喂，快迟到了啊
     ```
   - **一步提交并自动拉取下一批**（减少 50% 工具调用回合）：
     ```bash
     python gal_cli.py batch-push \
       --project <project_dir> \
       --patch-file <patch.txt> \
       --session worker_w1_a \
       --files "sc01_*.json" \
       --batch-size 120 \
       --context 6 \
       --no-refresh-output \
       --pull-next
     ```
   - **质检响应与熔断规则**：
     - 若本批存在 **`Hard Errors > 0`** 或解析/定位错误，`--pull-next` **自动触发熔断（`blocked_by_errors`）不翻页**，并当场吐出**仅含失败硬伤条目的 Delta Repair 修复包**；子代理必须立即提交修复补丁复核通过后，系统才会放行进入下一批未译文本！
     - 若仅有 `Soft Warnings`（`Hard Errors == 0`），直接忽略并继续翻译 `--pull-next` 返回的下一批正文。

### 阶段 4：全书硬错误清零、软警告控债、零成本重建与安全回滚
```bash
# 优先排查并清零全书硬错误（ERROR）
python gal_cli.py problems --project <project_dir> --severity error --group-by detail --limit 50

# 查看软警告（WARN）分布，确保控制在 2% 预算（max_warn_ratio）以内
python gal_cli.py problems --project <project_dir> --severity warn --group-by detail --limit 50

# 定向增量修补缓存（仅重算并落盘被修改的条目与文件）
python gal_cli.py edit-cache --project <project_dir> --file chapter_01.json --index 26 --proofread-dst "精修译文" --reason "二轮润色"

# 全局搜索替换（先 --dry-run 预览，再实跑；支持 revert 按 task_id 一键回滚）
python gal_cli.py search-replace --project <project_dir> --find "旧译" --replace "新译" --dry-run

# 全部初译落盘或修改词典后，零大模型成本一次性重建 transl_cache 与 gt_output
python gal_cli.py rebuild --project <project_dir>
```

### 阶段 5：自启发种子审校循环（真采样 + 硬伤清零 + 2% 警告预算硬门禁）
```bash
python gal_cli.py seed --project <project_dir> --action status
python gal_cli.py seed --project <project_dir> --action sample --regions 3 --per-region 10
python gal_cli.py seed --project <project_dir> --action add --pattern "<调查模式>" --query "<检索词>" --category diff
python gal_cli.py seed --project <project_dir> --action consume --seed-id S1-001 --resolution "<结论>" --complete-round

# 验收硬门禁（要求 completedRounds >= 2、真采样 >= 2 批且末批 >= 3 区、0 未消种子、0 硬错误、且软警告 <= 2% 预算上限）
python gal_cli.py seed --project <project_dir> --action check-gate
```

### 阶段 6：状态终检与无损导出交付
```bash
python gal_cli.py status --project <project_dir>
python gal_cli.py export --project <project_dir> --format gt_json --output <output_dir>
```
