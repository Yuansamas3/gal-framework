#!/usr/bin/env python3
"""Unified CLI controller for gal_framework (three-separation ledger & Agent-native dual-mode output)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_FRAMEWORK_ROOT = Path(__file__).resolve().parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from adapters import import_path
from core.batcher import (
    aggregate_problems,
    edit_cache_entries,
    make_trial,
    pull_batch,
    push_batch,
    search_replace_project,
)
from core.cache_ledger import (
    load_project_files,
    read_history_events,
    save_input_file,
)
from core.dictionary import load_project_dicts
from core.glossary_tools import (
    audit_variant_inconsistency,
    dump_names,
    extract_repeats,
    extract_term_candidates,
    sample_character_debuts,
)
from core.models import ItemStatus, ProjectConfig
from core.normalizer import preprocess_items
from core.problems import split_problems_by_severity
from core.rebuilder import export_project_outputs, rebuild_project, revert_task
from core.seed_ledger import SeedLedger
from core.text_preserve import TextPreserver


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8")
            except Exception:
                pass


def _split_csv_arg(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        for part in v.split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def _emit(payload: dict[str, Any], fmt: str, md_text: str, output_path: str | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) if fmt == "json" else md_text
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rendered + "\n", encoding="utf-8")
        print(f"Written output to: {p}")
    else:
        print(rendered)


def _project_dict_path(config: ProjectConfig, project_dir: Path, kind: str) -> Path:
    refs = {"pre": config.pre_dicts, "gpt": config.gpt_dicts, "post": config.post_dicts}[kind]
    for ref in reversed(refs):
        if ref.startswith("project:"):
            return config.resolve_resource_path(ref, project_dir)
    defaults = {
        "pre": "项目字典_译前.tsv",
        "gpt": "项目GPT字典.tsv",
        "post": "项目字典_译后.tsv",
    }
    return project_dir / defaults[kind]


def cmd_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("gt_input", "transl_cache", "gt_output", "work"):
        (project_dir / sub).mkdir(exist_ok=True)

    templates = {
        "项目字典_译前.tsv": "# 项目译前字典：查找词<TAB>替换词[<TAB>//注释]\n",
        "项目GPT字典.tsv": "# 项目GPT术语表：src<TAB>dst[<TAB>info]\n",
        "项目字典_译后.tsv": "# 项目译后字典：支持普通行与 pre_jp/post_jp/pre_dst 条件语法\n",
        "project_guideline.md": (
            f"# {args.name or project_dir.name} 项目专属翻译规范\n\n"
            "- 人物称谓、口癖与作品特有设定请补充于本文件。\n"
        ),
    }
    for fname, content in templates.items():
        fpath = project_dir / fname
        if not fpath.exists():
            fpath.write_text(content, encoding="utf-8")

    presets = _split_csv_arg(args.preserve) or ["base"]
    if "base" not in presets:
        presets.insert(0, "base")

    cfg = ProjectConfig(
        project_name=args.name or project_dir.name,
        profile=args.profile,
        guideline=args.guideline or "",
        target_encoding=args.encoding,
        preserve_presets=presets,
        ruby_preserve_mode=args.ruby_mode,
        strip_outer_quotes=not args.no_strip_quotes,
        align_trailing_period=not args.no_align_period,
    )

    profile_data = cfg.load_profile_data(project_dir)
    if not cfg.guideline:
        cfg.guideline = str(profile_data.get("guideline", "zh_light_novel_v2.md"))
    if "avg_sentence_length_threshold" in profile_data:
        cfg.avg_sentence_length_threshold = int(profile_data["avg_sentence_length_threshold"])
    if "enable_organ_injection_check" in profile_data:
        cfg.enable_organ_injection_check = bool(profile_data["enable_organ_injection_check"])
    if "enable_register_check" in profile_data:
        cfg.enable_register_check = bool(profile_data["enable_register_check"])

    cfg_path = cfg.save(project_dir)
    payload = {
        "status": "initialized",
        "project_dir": str(project_dir),
        "config_path": str(cfg_path),
        "profile": cfg.profile,
        "guideline": cfg.guideline,
        "target_encoding": cfg.target_encoding,
        "preserve_presets": cfg.preserve_presets,
        "ruby_preserve_mode": cfg.ruby_preserve_mode,
        "ledger_dirs": {
            "gt_input": str(cfg.input_path(project_dir)),
            "transl_cache": str(cfg.cache_path(project_dir)),
            "gt_output": str(cfg.output_path(project_dir)),
        },
    }
    md = (
        f"# Project Initialized: `{cfg.project_name}`\n\n"
        f"- Config: `{cfg_path}`\n"
        f"- Profile: `{cfg.profile}` | Guideline: `{cfg.guideline}` | RubyMode: `{cfg.ruby_preserve_mode}`\n"
        f"- Encoding: `{cfg.target_encoding}` | Presets: `{', '.join(cfg.preserve_presets)}`\n"
        f"- Ledger: `gt_input/` + `transl_cache/` + `gt_output/` + `cache_history.jsonl`"
    )
    _emit(payload, args.out_format, md)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)

    imported = import_path(args.format, Path(args.input).resolve())
    total_items = 0
    dialogue_items = 0
    stripped_quotes = 0
    file_stats: list[dict[str, Any]] = []

    for orig_rel, format_meta, items in imported:
        rel_name = orig_rel if orig_rel.endswith(".json") else f"{orig_rel}.json"
        if rel_name != orig_rel:
            format_meta = dict(format_meta)
            format_meta["orig_rel_name"] = orig_rel

        # Save immutable raw source into `gt_input/` before preprocessing
        save_input_file(project_dir, config, rel_name, items, format_meta=format_meta)

        # Preview preprocessing metrics
        preprocess_items(items, config, pre_dict=pre_dict, force=True)
        d_cnt = sum(1 for it in items if it.is_dialogue)
        q_cnt = sum(1 for it in items if it.left_quote or it.right_quote)
        total_items += len(items)
        dialogue_items += d_cnt
        stripped_quotes += q_cnt
        file_stats.append(
            {
                "file": rel_name,
                "items": len(items),
                "dialogues": d_cnt,
                "outer_quotes_stripped": q_cnt,
            }
        )

    config.source_format = args.format
    config.save(project_dir)

    payload = {
        "status": "imported",
        "format": args.format,
        "files_imported": len(file_stats),
        "total_items": total_items,
        "dialogue_items": dialogue_items,
        "outer_quotes_stripped": stripped_quotes,
        "files": file_stats,
    }
    lines = [
        f"# Imported {len(file_stats)} file(s) into `gt_input/` (`{args.format}`)",
        "",
        f"- Total items: **{total_items}** | Dialogues: **{dialogue_items}** | Outer-quote shells stripped: **{stripped_quotes}**",
        "",
        "| File | Items | Dialogues | Outer Quotes Stripped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for st in file_stats:
        lines.append(
            f"| `{st['file']}` | {st['items']} | {st['dialogues']} | {st['outer_quotes_stripped']} |"
        )
    _emit(payload, args.out_format, "\n".join(lines))
    return 0


def cmd_prep_scan(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, gpt_dict, _ = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=_split_csv_arg(args.files),
    )

    all_items = [it for _f, _meta, items, _h, _m in loaded for it in items]
    names_report = dump_names(all_items, gpt_dict=gpt_dict, speaker_map=config.speaker_map)
    repeats = extract_repeats(all_items, min_count=args.min_repeat, limit=args.limit)
    candidates = extract_term_candidates(
        all_items,
        gpt_dict=gpt_dict,
        min_count=args.min_term,
        limit=args.limit,
    )
    preserve_scan = preserver.scan_samples(all_items)
    variant_audit = audit_variant_inconsistency(all_items, gpt_dict=gpt_dict)

    payload = {
        "files_scanned": len(loaded),
        "total_items": len(all_items),
        "speakers": names_report,
        "repeats": repeats,
        "term_candidates": candidates,
        "control_scan": preserve_scan,
        "variant_audit": variant_audit,
    }

    md = [
        f"# Pre-Translation Foundation Scan ({len(loaded)} files, {len(all_items)} items)",
        "",
        "## 1. Speakers",
        f"- Unique: **{names_report['total_unique_speakers']}** | Covered: **{names_report['covered_speakers']}** "
        f"({names_report['coverage_ratio'] * 100:.1f}%) | Uncovered: **{names_report['uncovered_speakers']}**",
        "",
        "| Speaker | Count | Covered | Mapped | Sample |",
        "| --- | ---: | :---: | --- | --- |",
    ]
    for sp in names_report["speakers"][: args.limit]:
        sample = sp["examples"][0].replace("|", "\\|") if sp["examples"] else ""
        md.append(
            f"| `{sp['speaker']}` | {sp['count']} | {'Y' if sp['covered'] else 'N'} | {sp['mapped_dst'] or '-'} | {sample} |"
        )

    md.extend(["", "## 2. High-Frequency Repeated Lines"])
    if repeats:
        md.extend(["| Count | Text | Sample Refs |", "| ---: | --- | --- |"])
        for rep in repeats[:25]:
            txt = rep["text"].replace("\n", "<br>").replace("|", "\\|")
            md.append(f"| {rep['count']} | {txt} | {', '.join(rep['sample_refs'][:3])} |")
    else:
        md.append("(none)")

    md.extend(["", "## 3. Uncovered Term Candidates"])
    if candidates:
        md.extend(["| Term | Count | Kind | Example |", "| --- | ---: | --- | --- |"])
        for c in candidates[:30]:
            ex = c["examples"][0].replace("|", "\\|") if c["examples"] else ""
            md.append(f"| `{c['term']}` | {c['count']} | {c['source']} | {ex} |")
    else:
        md.append("(none)")

    md.extend(["", "## 4. Control Token Samples"])
    if preserve_scan["top_tokens"]:
        toks = ", ".join(f"`{t['token']}`(x{t['count']})" for t in preserve_scan["top_tokens"][:15])
        md.append(f"- Top tokens: {toks}")
    else:
        md.append("- No control tokens detected.")

    _emit(payload, args.out_format, "\n".join(md), output_path=args.output)
    return 0


def cmd_sample_debuts(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=_split_csv_arg(args.files),
    )
    res = sample_character_debuts(
        loaded,
        top_n=args.top,
        context_radius=args.window,
    )
    _emit(res, args.out_format, res["markdown"], output_path=args.output)
    return 0


def cmd_dict(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    dpath = _project_dict_path(config, project_dir, args.kind)
    dpath.parent.mkdir(parents=True, exist_ok=True)

    if args.action == "list":
        content = dpath.read_text(encoding="utf-8-sig") if dpath.exists() else ""
        lines = [ln for ln in content.splitlines() if ln.strip() and not ln.strip().startswith(("#", "//"))]
        payload = {"kind": args.kind, "path": str(dpath), "count": len(lines), "entries": lines}
        md = f"# Project `{args.kind}` Dictionary (`{dpath.name}`, {len(lines)} entries)\n\n" + (
            "```tsv\n" + "\n".join(lines[:100]) + "\n```" if lines else "(empty)"
        )
        _emit(payload, args.out_format, md)
        return 0

    if args.action == "add":
        if not args.src:
            print("Error: --src is required for dict --action add", file=sys.stderr)
            return 2
        if args.cond_type and args.cond_expr:
            new_line = f"{args.cond_type}\t{args.cond_expr}\t{args.src}\t{args.dst}"
            if args.info:
                new_line += f"\t{args.info}"
        else:
            new_line = f"{args.src}\t{args.dst}"
            if args.info:
                new_line += f"\t{args.info}"

        existing = dpath.read_text(encoding="utf-8-sig").splitlines() if dpath.exists() else []
        existing.append(new_line)
        dpath.write_text("\n".join(existing) + "\n", encoding="utf-8")
        payload = {"status": "added", "kind": args.kind, "path": str(dpath), "line": new_line}
        _emit(payload, args.out_format, f"# Added to `{dpath.name}`\n\n`{new_line}`")
        return 0

    if args.action == "remove":
        if not args.src or not dpath.exists():
            print("Error: --src is required and dictionary must exist", file=sys.stderr)
            return 2
        old_lines = dpath.read_text(encoding="utf-8-sig").splitlines()
        kept: list[str] = []
        removed = 0
        for ln in old_lines:
            parts = ln.split("\t")
            hit = (len(parts) >= 2 and parts[0].strip() == args.src) or (
                len(parts) >= 4 and parts[2].strip() == args.src
            )
            if hit:
                removed += 1
            else:
                kept.append(ln)
        dpath.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        payload = {"status": "removed", "kind": args.kind, "src": args.src, "removed_count": removed}
        _emit(payload, args.out_format, f"# Removed {removed} line(s) matching `{args.src}` from `{dpath.name}`")
        return 0

    return 2


def cmd_guideline(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    gpath = project_dir / config.project_guideline

    if args.action == "show":
        text = gpath.read_text(encoding="utf-8-sig") if gpath.exists() else ""
        payload = {"path": str(gpath), "content": text}
        _emit(payload, args.out_format, text or "(empty project guideline)")
        return 0

    if args.action == "append":
        if not args.text:
            print("Error: --text is required for guideline --action append", file=sys.stderr)
            return 2
        existing = gpath.read_text(encoding="utf-8-sig").rstrip("\n") if gpath.exists() else ""
        updated = (existing + "\n" if existing else "") + args.text.strip() + "\n"
        gpath.write_text(updated, encoding="utf-8")
        payload = {"status": "appended", "path": str(gpath), "appended": args.text.strip()}
        _emit(payload, args.out_format, f"# Appended to `{gpath.name}`\n\n{args.text.strip()}")
        return 0

    return 2


def cmd_make_trial(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    result = make_trial(project_dir, file_patterns=_split_csv_arg(args.files), limit=args.limit)
    _emit(result, args.out_format, result["markdown"], output_path=args.output)
    return 0


def cmd_batch_pull(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    only_miss = not args.all and not args.retransl_problems
    result = pull_batch(
        project_dir,
        file_patterns=_split_csv_arg(args.files),
        batch_size=args.batch_size,
        context_num=args.context,
        only_miss=only_miss,
        retransl_problems=args.retransl_problems,
        problem_type=args.problem_type,
        proofread_mode=args.proofread,
        offset=args.offset,
        session_id=args.session,
        compact=args.compact,
        no_compact=args.no_compact,
    )
    _emit(result, args.out_format, result["markdown"], output_path=args.output)
    return 0


def cmd_batch_push(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    if args.patch_file:
        patch_text = Path(args.patch_file).read_text(encoding="utf-8-sig")
    else:
        patch_text = sys.stdin.read()

    result = push_batch(
        project_dir,
        patch_text,
        default_file=args.file or "",
        proofread=args.proofread,
        trans_by=args.trans_by,
        reason=args.reason,
        refresh_output=not args.no_refresh_output,
        pull_next=args.pull_next,
        session_id=args.session,
        compact=args.compact,
        no_compact=args.no_compact,
        file_patterns=_split_csv_arg(args.files) or None,
        batch_size=args.batch_size,
        context_num=args.context,
    )

    lines = [
        "# Batch Push Instant Lint Report",
        "",
        f"- Task ID: `{result['task_id']}`",
        f"- Applied: **{result['applied_count']}** | Clean: **{result['clean_count']}** "
        f"| Flagged: **{result['problem_item_count']}** "
        f"(Hard Errors: **{result['hard_error_count']}**, Warnings: **{result['soft_warn_count']}**) "
        f"| Parse/Target Errors: **{result['error_count']}**",
    ]
    if "pull_next_status" in result:
        lines.append(f"- Pull-Next Status: `{result['pull_next_status']}`")
    if result["errors"]:
        lines.extend(["", "## Patch Errors"])
        for err in result["errors"]:
            lines.append(f"- {err}")
    if result["hard_error_items"]:
        lines.extend(
            [
                "",
                "## Hard Errors (MUST Fix & Re-Submit Immediately)",
                "| PATCH_KEY | ANCHOR | POST_SRC | POST_DST | HARD_ERRORS |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in result["hard_error_items"]:
            src = item["post_src"].replace("\n", "<br>").replace("|", "\\|")
            dst = item["post_dst"].replace("\n", "<br>").replace("|", "\\|")
            probs = "; ".join(item["problems"]).replace("|", "\\|")
            lines.append(f"| `{item['patch_key']}` | `{item['anchor']}` | {src} | {dst} | {probs} |")
    if result["soft_warn_items"]:
        lines.extend(
            [
                "",
                "## Soft Warnings (Advisory — Ignore During Drafting)",
                "| PATCH_KEY | ANCHOR | POST_SRC | POST_DST | WARNINGS |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in result["soft_warn_items"]:
            src = item["post_src"].replace("\n", "<br>").replace("|", "\\|")
            dst = item["post_dst"].replace("\n", "<br>").replace("|", "\\|")
            probs = "; ".join(item["problems"]).replace("|", "\\|")
            lines.append(f"| `{item['patch_key']}` | `{item['anchor']}` | {src} | {dst} | {probs} |")
    if not result["problem_items"] and not result["errors"]:
        lines.append("\nAll applied items passed QA with 0 issues.")

    if "next_batch" in result:
        lines.extend(["", "---", "", result["next_batch"]["markdown"]])

    _emit(result, args.out_format, "\n".join(lines))
    return 0 if result["error_count"] == 0 else 2


def cmd_edit_cache(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    if args.edits_file:
        edits = json.loads(Path(args.edits_file).read_text(encoding="utf-8-sig"))
    else:
        if not args.file or args.index is None:
            print("Error: --file and --index (or --edits-file) required", file=sys.stderr)
            return 2
        edit_obj: dict[str, Any] = {"file": args.file, "index": args.index}
        if args.pre_dst is not None:
            edit_obj["pre_dst"] = args.pre_dst
        if args.proofread_dst is not None:
            edit_obj["proofread_dst"] = args.proofread_dst
        if args.speaker_dst is not None:
            edit_obj["speaker_dst"] = args.speaker_dst
        if args.skip_check is not None:
            edit_obj["skip_check"] = args.skip_check.lower() in ("true", "1", "yes")
        if args.comment is not None:
            edit_obj["proofread_comment"] = args.comment
        edits = [edit_obj]

    res = edit_cache_entries(
        project_dir,
        edits,
        reason=args.reason,
        refresh_output=not args.no_refresh_output,
    )
    md = (
        f"# Cache Edited (`{res['task_id']}`)\n\n"
        f"- Edited: **{res['edited_count']}** | Errors: **{res['error_count']}** | Flagged: **{len(res['problem_items'])}**"
    )
    _emit(res, args.out_format, md)
    return 0 if res["error_count"] == 0 else 2


def cmd_problems(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    res = aggregate_problems(
        project_dir,
        problem_type=args.type,
        severity=args.severity,
        group_by=args.group_by,
        limit=args.limit,
        file_patterns=_split_csv_arg(args.files),
    )

    lines = [
        "# Project QA Problems Summary",
        "",
        f"- Items: **{res['translated_items']}/{res['total_items']}** translated | "
        f"Matched (`{res['severity']}`): **{res['problem_item_count']}** "
        f"(Hard errors: **{res['hard_error_items']}**, Warnings: **{res['soft_warn_items']}**)",
        "",
        "## Ranking by Problem Type",
    ]
    if res["by_type"]:
        lines.extend(["| Problem Type | Severity | Count |", "| --- | :---: | ---: |"])
        for row in res["by_type"]:
            lines.append(f"| `{row['type']}` | `{row.get('severity', 'error').upper()}` | {row['count']} |")
    else:
        lines.append("(0 problems)")

    if args.group_by == "file":
        lines.extend(["", "## By File"])
        for fname, rows in res.get("files", {}).items():
            lines.append(f"\n### `{fname}` ({len(rows)} shown)")
            for r in rows:
                lines.append(
                    f"- `{r['file']}|{r['id']}` ({r['anchor']}): {'; '.join(r['problems'])} "
                    f"| SRC=`{r['post_src']}` | DST=`{r['post_dst']}`"
                )
    else:
        items = res.get("items", [])
        if items:
            lines.extend(
                [
                    "",
                    f"## Problem Items (showing {len(items)})",
                    "| FILE\\|ID | ANCHOR | POST_SRC | POST_DST | PROBLEMS |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for r in items:
                src = r["post_src"].replace("\n", "<br>").replace("|", "\\|")
                dst = r["post_dst"].replace("\n", "<br>").replace("|", "\\|")
                probs = "; ".join(r["problems"]).replace("|", "\\|")
                lines.append(f"| `{r['file']}|{r['id']}` | `{r['anchor']}` | {src} | {dst} | {probs} |")

    _emit(res, args.out_format, "\n".join(lines))
    return 0


def cmd_search_replace(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    res = search_replace_project(
        project_dir,
        args.find,
        args.replace,
        field=args.field,
        use_regex=args.regex,
        file_patterns=_split_csv_arg(args.files),
        only_problem=args.only_problem,
        dry_run=args.dry_run,
        reason=args.reason,
        refresh_output=not args.no_refresh_output,
    )
    lines = [
        f"# Search & Replace ({'DRY RUN' if res['dry_run'] else 'APPLIED'})",
        "",
        f"- Task ID: `{res['task_id'] or '(dry-run)'}` | Field: `{res['field']}` | Matched: **{res['matched_count']}** | Touched files: **{len(res['touched_files'])}**",
    ]
    for ch in res["changes"][:30]:
        lines.append(f"- `{ch['file']}|{ch['id']}`: `{ch['before']}` -> `{ch['after']}`")
    _emit(res, args.out_format, "\n".join(lines))
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    if args.list or not args.task_id:
        events = read_history_events(project_dir, config)
        recent = list(reversed(events))[: args.limit]
        payload = {"count": len(events), "recent": recent}
        lines = [f"# Cache History ({len(events)} event(s))", ""]
        if recent:
            lines.extend(["| TASK_ID | ACTION | CHANGES | TIMESTAMP | REASON |", "| --- | --- | ---: | --- | --- |"])
            for ev in recent:
                lines.append(
                    f"| `{ev.get('task_id')}` | `{ev.get('action')}` | {ev.get('change_count', 0)} | "
                    f"{ev.get('timestamp', '')} | {ev.get('reason', '')} |"
                )
        else:
            lines.append("(no history events)")
        _emit(payload, args.out_format, "\n".join(lines))
        return 0

    res = revert_task(project_dir, args.task_id, refresh_output=not args.no_refresh_output)
    if res["status"] != "reverted":
        _emit(res, args.out_format, f"# Revert Failed\n\n- {res.get('error')}")
        return 1
    md = (
        f"# Reverted Task `{res['reverted_task_id']}`\n\n"
        f"- Reverted entries: **{res['reverted_count']}** | Revert Audit ID: `{res['revert_history_id']}`\n"
        f"- Touched files: `{', '.join(res['touched_files'])}`"
    )
    _emit(res, args.out_format, md)
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    res = rebuild_project(
        project_dir,
        file_patterns=_split_csv_arg(args.files),
        ignore_post_src=args.ignore_post_src,
        refresh_output=not args.no_refresh_output,
    )
    lines = [
        "# Zero-Cost Project Rebuild Complete",
        "",
        f"- Files rebuilt: **{res['files_rebuilt']}** | Cache hits: **{res['cache_hits']}** | Misses: **{res['cache_misses']}**",
        f"- Translated items refreshed: **{res['rebuilt_translated_items']}/{res['total_items']}**",
        f"- Problems before: **{res['problems_before']}** -> after: **{res['problems_after']}** (delta: **{res['delta']:+d}**)",
    ]
    if res["miss_breakdown"]:
        lines.append("- Miss reasons: " + ", ".join(f"`{k}`={v}" for k, v in res["miss_breakdown"].items()))
    if res["by_type"]:
        lines.extend(["", "| Problem Type | Count |", "| --- | ---: |"])
        for row in res["by_type"]:
            lines.append(f"| `{row['type']}` | {row['count']} |")
    _emit(res, args.out_format, "\n".join(lines))
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)
    ledger = SeedLedger(project_dir / "work" / "seeds.json")
    loaded = load_project_files(project_dir, config, pre_dict=pre_dict, ignore_post_src=True)
    all_items = [it for _f, _meta, items, _h, _m in loaded for it in items]
    open_problems = sum(1 for it in all_items if it.problems)
    open_errors = sum(1 for it in all_items if split_problems_by_severity(it.problems)[0])
    open_warns = sum(1 for it in all_items if split_problems_by_severity(it.problems)[1])
    warn_budget = config.compute_warn_budget(len(all_items))

    action = args.action
    if action == "status":
        res = ledger.status(
            open_problem_count=open_problems,
            open_error_count=open_errors,
            open_warn_count=open_warns,
            max_warn_budget=warn_budget,
        )
        md = (
            f"# Seed Review Ledger Status\n\n"
            f"- Completed Rounds: **{res['completedRounds']}/{res['minRequiredRounds']}**\n"
            f"- Dispersed Sample Batches: **{res['sampleBatches']}**\n"
            f"- Seeds: **{res['consumedSeeds']}/{res['totalSeeds']}** consumed (**{res['unconsumedCount']}** open)\n"
            f"- Open QA: **{res['openProblemCount']}** item(s) "
            f"(**{res['openErrorCount']}** hard errors, **{res['openWarnCount']}/{warn_budget}** warnings)\n"
            f"- Gate Passed: **{'YES' if res['gatePassed'] else 'NO'}**\n"
            f"- Next Action: {res['nextAction']}"
        )
        _emit(res, args.out_format, md)
        return 0

    if action == "sample":
        res = ledger.sample_dispersed(all_items, regions=args.regions, per_region=args.per_region)
        lines = [
            f"# Dispersed Review Sample (Round {res['round']}, {res['regions_sampled']} regions, {res['sampled_count']} items)",
            "",
            "| FILE\\|ID | SPEAKER | POST_SRC | POST_DST | PROBLEMS |",
            "| --- | --- | --- | --- | --- |",
        ]
        for it in res["items"]:
            src = it["post_src"].replace("\n", "<br>").replace("|", "\\|")
            dst = it["post_dst"].replace("\n", "<br>").replace("|", "\\|")
            probs = "; ".join(it["problems"]) or "-"
            lines.append(f"| `{it['file']}|{it['id']}` | {it['speaker'] or '-'} | {src} | {dst} | {probs} |")
        _emit(res, args.out_format, "\n".join(lines))
        return 0

    if action == "add":
        if not args.pattern:
            print("Error: --pattern is required for seed --action add", file=sys.stderr)
            return 2
        res = ledger.add_seed(
            args.pattern,
            query=args.query or "",
            category=args.category,
            round_num=args.round,
            items=all_items if args.query else None,
        )
        md = (
            f"# Seed Added: `{res['id']}` (Round {res['round']})\n\n"
            f"- Category: `{res['category']}`\n"
            f"- Pattern: {res['pattern']}\n"
            + (f"- Preview hits: **{res.get('previewHitCount', 0)}**" if args.query else "")
        )
        _emit(res, args.out_format, md)
        return 0

    if action == "consume":
        if not args.seed_id:
            print("Error: --seed-id is required for seed --action consume", file=sys.stderr)
            return 2
        res = ledger.consume_seed(
            args.seed_id,
            resolution=args.resolution or "",
            complete_round=args.complete_round,
        )
        md = (
            f"# Seed(s) Consumed\n\n"
            f"- Consumed IDs: `{', '.join(res['consumed_ids']) or '(none)'}`\n"
            f"- Skipped Future IDs: `{', '.join(res['skipped_future_ids']) or '(none)'}`\n"
            f"- Completed Rounds: **{res['completedRounds']}** (advanced={res['round_advanced']})\n"
            f"- Remaining Unconsumed: **{res['remaining_unconsumed']}**"
            + (f"\n- Warning: {res['round_block_reason']}" if res["round_block_reason"] else "")
        )
        _emit(res, args.out_format, md)
        return 0 if not res["round_block_reason"] else 1

    if action == "complete-round":
        ok, rounds, err = ledger.complete_round()
        res = ledger.status(
            open_problem_count=open_problems,
            open_error_count=open_errors,
            open_warn_count=open_warns,
            max_warn_budget=warn_budget,
        )
        res["round_advanced"] = ok
        res["error"] = err
        md = (
            f"# Round Completion: **{'OK (Round ' + str(rounds) + ')' if ok else 'BLOCKED'}**\n\n"
            + (f"- Error: {err}\n" if err else "")
            + f"- Gate Passed: **{'YES' if res['gatePassed'] else 'NO'}**\n"
            f"- Next Action: {res['nextAction']}"
        )
        _emit(res, args.out_format, md)
        return 0 if ok else 1

    if action == "check-gate":
        passed, reasons = ledger.check_gate(
            open_problem_count=open_problems,
            open_error_count=open_errors,
            open_warn_count=open_warns,
            max_warn_budget=warn_budget,
        )
        res = {
            "gatePassed": passed,
            "blockingReasons": reasons,
            **ledger.status(
                open_problem_count=open_problems,
                open_error_count=open_errors,
                open_warn_count=open_warns,
                max_warn_budget=warn_budget,
            ),
        }
        md = (
            f"# Review Gate: **{'PASSED' if passed else 'BLOCKED'}**\n"
            + ("\n".join(f"- Blocking: {r}" for r in reasons) if reasons else "- All review gate checks satisfied.")
        )
        _emit(res, args.out_format, md)
        return 0 if passed else 1

    return 2


def cmd_export(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=_split_csv_arg(args.files),
        ignore_post_src=True,
    )
    out_dir = Path(args.output).resolve() if args.output else config.output_path(project_dir)
    out_fmt = args.format or config.source_format
    encoding = args.encoding or config.target_encoding

    total_items = 0
    translated_items = 0
    problem_items = 0
    export_payload = []

    for rel_name, fmt_meta, items, _hits, _misses in loaded:
        for it in items:
            total_items += 1
            if it.status in (ItemStatus.PROCESSED.value, ItemStatus.PROOFREAD.value, ItemStatus.RULE_SKIPPED.value):
                translated_items += 1
            if it.problems:
                problem_items += 1
        export_payload.append((rel_name, fmt_meta, items))

    untranslated = total_items - translated_items
    if untranslated > 0 and not args.allow_incomplete:
        payload = {
            "status": "blocked",
            "reason": f"{untranslated} item(s) still untranslated; pass --allow-incomplete to export anyway",
            "total_items": total_items,
            "translated_items": translated_items,
            "problem_items": problem_items,
        }
        _emit(payload, args.out_format, f"# Export Blocked\n\n- {payload['reason']}")
        return 1

    exported_files = export_project_outputs(
        project_dir,
        config,
        export_payload,
        output_dir=out_dir,
        output_format=out_fmt,
        encoding=encoding,
        with_src=args.with_src,
    )

    payload = {
        "status": "exported",
        "format": out_fmt,
        "encoding": encoding,
        "output_dir": str(out_dir),
        "files_exported": len(exported_files),
        "total_items": total_items,
        "translated_items": translated_items,
        "problem_items": problem_items,
        "files": exported_files,
    }
    md = (
        f"# Export Complete (`{out_fmt}`, `{encoding}`)\n\n"
        f"- Output Dir: `{out_dir}`\n"
        f"- Files: **{len(exported_files)}** | Completion: **{translated_items}/{total_items}** | Open QA Problems: **{problem_items}**"
    )
    _emit(payload, args.out_format, md)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project_dir = Path(args.project).resolve()
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)
    loaded = load_project_files(project_dir, config, pre_dict=pre_dict)
    all_items = [it for _f, _meta, items, _h, _m in loaded for it in items]
    hits = sum(len(h) for _f, _meta, _it, h, _m in loaded)
    misses = sum(len(m) for _f, _meta, _it, _h, m in loaded)
    miss_reasons = Counter(it.cache_miss_reason for it in all_items if it.cache_miss_reason)
    translated = sum(
        1 for it in all_items if it.status in (ItemStatus.PROCESSED.value, ItemStatus.PROOFREAD.value)
    )
    problems = sum(1 for it in all_items if it.problems)
    hard_errors = sum(1 for it in all_items if split_problems_by_severity(it.problems)[0])
    soft_warns = sum(1 for it in all_items if split_problems_by_severity(it.problems)[1])
    warn_budget = config.compute_warn_budget(len(all_items))
    ledger = SeedLedger(project_dir / "work" / "seeds.json")
    seed_st = ledger.status(
        open_problem_count=problems,
        open_error_count=hard_errors,
        open_warn_count=soft_warns,
        max_warn_budget=warn_budget,
    )
    events = read_history_events(project_dir, config)

    payload = {
        "project_name": config.project_name,
        "profile": config.profile,
        "guideline": config.guideline,
        "encoding": config.target_encoding,
        "files": len(loaded),
        "total_items": len(all_items),
        "translated_items": translated,
        "cache_hits": hits,
        "cache_misses": misses,
        "miss_breakdown": dict(miss_reasons),
        "completion_ratio": round(translated / len(all_items), 4) if all_items else 0.0,
        "problem_items": problems,
        "hard_error_items": hard_errors,
        "soft_warn_items": soft_warns,
        "max_warn_budget": warn_budget,
        "warn_budget_exceeded": soft_warns > warn_budget,
        "history_events": len(events),
        "seed_gate": seed_st,
    }
    md = (
        f"# Project Status: `{config.project_name}`\n\n"
        f"- Profile: `{config.profile}` | Guideline: `{config.guideline}` | Encoding: `{config.target_encoding}`\n"
        f"- Progress: **{translated}/{len(all_items)}** ({payload['completion_ratio'] * 100:.1f}%) across **{len(loaded)}** file(s)\n"
        f"- Cache: **{hits}** hits / **{misses}** misses ({dict(miss_reasons)})\n"
        f"- Open QA: **{problems}** item(s) (**{hard_errors}** hard errors, **{soft_warns}** warnings; budget={warn_budget}) | History Events: **{len(events)}**\n"
        f"- Seed Review Gate: **{'PASSED' if seed_st['gatePassed'] else 'OPEN'}** "
        f"(rounds {seed_st['completedRounds']}/{seed_st['minRequiredRounds']}, samples {seed_st['sampleBatches']}, unconsumed {seed_st['unconsumedCount']})"
    )
    _emit(payload, args.out_format, md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gal_cli.py",
        description="Universal Visual Novel / Galgame Localization Middleware CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialize three-separation localization project")
    p_init.add_argument("--project", required=True)
    p_init.add_argument("--name", default="")
    p_init.add_argument("--profile", default="modern_acg")
    p_init.add_argument("--guideline", default="")
    p_init.add_argument("--encoding", default="utf8", choices=["utf8", "gbk"])
    p_init.add_argument("--preserve", action="append", default=[])
    p_init.add_argument("--ruby-mode", default="translate", choices=["translate", "verbatim", "flexible"])
    p_init.add_argument("--no-strip-quotes", action="store_true")
    p_init.add_argument("--no-align-period", action="store_true")
    p_init.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_init.set_defaults(func=cmd_init)

    # import
    p_imp = sub.add_parser("import", help="Import external script files into gt_input/")
    p_imp.add_argument("--project", required=True)
    p_imp.add_argument("--format", required=True, choices=["gt_json", "compact_pipe"])
    p_imp.add_argument("--input", required=True)
    p_imp.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_imp.set_defaults(func=cmd_import)

    # prep-scan
    p_scan = sub.add_parser("prep-scan", help="Scan speakers, repeats, term candidates, and control tokens")
    p_scan.add_argument("--project", required=True)
    p_scan.add_argument("--files", action="append", default=[])
    p_scan.add_argument("--min-repeat", type=int, default=3)
    p_scan.add_argument("--min-term", type=int, default=2)
    p_scan.add_argument("--limit", type=int, default=50)
    p_scan.add_argument("--output", default=None)
    p_scan.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_scan.set_defaults(func=cmd_prep_scan)

    # sample-debuts
    p_deb = sub.add_parser("sample-debuts", help="Mine character first appearances (±window), pronouns, and endings")
    p_deb.add_argument("--project", required=True)
    p_deb.add_argument("--files", action="append", default=[])
    p_deb.add_argument("--top", type=int, default=25)
    p_deb.add_argument("--window", type=int, default=6)
    p_deb.add_argument("--output", default=None)
    p_deb.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_deb.set_defaults(func=cmd_sample_debuts)

    # dict
    p_dict = sub.add_parser("dict", help="Inspect or edit project pre/gpt/post dictionary")
    p_dict.add_argument("--project", required=True)
    p_dict.add_argument("--kind", required=True, choices=["pre", "gpt", "post"])
    p_dict.add_argument("--action", required=True, choices=["list", "add", "remove"])
    p_dict.add_argument("--src", default="")
    p_dict.add_argument("--dst", default="")
    p_dict.add_argument("--info", default="")
    p_dict.add_argument("--cond-type", default="")
    p_dict.add_argument("--cond-expr", default="")
    p_dict.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_dict.set_defaults(func=cmd_dict)

    # guideline
    p_guide = sub.add_parser("guideline", help="Show or append to project_guideline.md")
    p_guide.add_argument("--project", required=True)
    p_guide.add_argument("--action", required=True, choices=["show", "append"])
    p_guide.add_argument("--text", default="")
    p_guide.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_guide.set_defaults(func=cmd_guideline)

    # make-trial
    p_trial = sub.add_parser("make-trial", help="Generate a 100-line trial style-calibration work package")
    p_trial.add_argument("--project", required=True)
    p_trial.add_argument("--files", action="append", default=[])
    p_trial.add_argument("--limit", type=int, default=100)
    p_trial.add_argument("--output", default=None)
    p_trial.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_trial.set_defaults(func=cmd_make_trial)

    # batch-pull
    p_pull = sub.add_parser("batch-pull", help="Pull work unit from cache misses with local glossary & context")
    p_pull.add_argument("--project", required=True)
    p_pull.add_argument("--files", action="append", default=[])
    p_pull.add_argument("--batch-size", type=int, default=None)
    p_pull.add_argument("--context", type=int, default=None)
    p_pull.add_argument("--session", default="")
    p_pull.add_argument("--compact", action="store_true", default=False)
    p_pull.add_argument("--no-compact", action="store_true", default=False)
    p_pull.add_argument("--only-miss", "--only-untranslated", dest="only_miss", action="store_true", default=True)
    p_pull.add_argument("--retransl-problems", action="store_true", default=False)
    p_pull.add_argument("--proofread", action="store_true", default=False)
    p_pull.add_argument("--all", action="store_true", default=False)
    p_pull.add_argument("--problem-type", default=None)
    p_pull.add_argument("--offset", type=int, default=0)
    p_pull.add_argument("--output", default=None)
    p_pull.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_pull.set_defaults(func=cmd_batch_pull)

    # batch-push
    p_push = sub.add_parser("batch-push", help="Apply patch to transl_cache with instant QA lint & history audit")
    p_push.add_argument("--project", required=True)
    p_push.add_argument("--patch-file", default=None)
    p_push.add_argument("--file", default="")
    p_push.add_argument("--files", action="append", default=[])
    p_push.add_argument("--batch-size", type=int, default=None)
    p_push.add_argument("--context", type=int, default=None)
    p_push.add_argument("--session", default="")
    p_push.add_argument("--compact", action="store_true", default=False)
    p_push.add_argument("--no-compact", action="store_true", default=False)
    p_push.add_argument("--pull-next", action="store_true", default=False)
    p_push.add_argument("--proofread", action="store_true")
    p_push.add_argument("--trans-by", default="agent")
    p_push.add_argument("--reason", default="batch-push")
    p_push.add_argument("--no-refresh-output", action="store_true")
    p_push.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_push.set_defaults(func=cmd_batch_push)

    # edit-cache
    p_ec = sub.add_parser("edit-cache", help="Patch specific index entries in transl_cache with audit log")
    p_ec.add_argument("--project", required=True)
    p_ec.add_argument("--edits-file", default=None)
    p_ec.add_argument("--file", default="")
    p_ec.add_argument("--index", type=int, default=None)
    p_ec.add_argument("--pre-dst", default=None)
    p_ec.add_argument("--proofread-dst", default=None)
    p_ec.add_argument("--speaker-dst", default=None)
    p_ec.add_argument("--skip-check", default=None)
    p_ec.add_argument("--comment", default=None)
    p_ec.add_argument("--reason", default="edit-cache")
    p_ec.add_argument("--no-refresh-output", action="store_true")
    p_ec.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_ec.set_defaults(func=cmd_edit_cache)

    # problems
    p_prob = sub.add_parser("problems", help="Aggregate and drill down into project QA problems")
    p_prob.add_argument("--project", required=True)
    p_prob.add_argument("--type", default=None)
    p_prob.add_argument("--severity", default="all", choices=["all", "error", "warn"])
    p_prob.add_argument("--group-by", default="detail", choices=["detail", "file"])
    p_prob.add_argument("--limit", type=int, default=50)
    p_prob.add_argument("--files", action="append", default=[])
    p_prob.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_prob.set_defaults(func=cmd_problems)

    # search-replace
    p_sr = sub.add_parser("search-replace", help="Search & replace cache translations with history & re-lint")
    p_sr.add_argument("--project", required=True)
    p_sr.add_argument("--find", required=True)
    p_sr.add_argument("--replace", required=True)
    p_sr.add_argument("--field", default="pre_dst", choices=["pre_dst", "proofread_dst", "speaker_dst"])
    p_sr.add_argument("--regex", action="store_true")
    p_sr.add_argument("--files", action="append", default=[])
    p_sr.add_argument("--only-problem", default=None)
    p_sr.add_argument("--dry-run", action="store_true")
    p_sr.add_argument("--reason", default="search-replace")
    p_sr.add_argument("--no-refresh-output", action="store_true")
    p_sr.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_sr.set_defaults(func=cmd_search_replace)

    # revert
    p_rev = sub.add_parser("revert", help="List cache_history.jsonl events or revert a specific task_id")
    p_rev.add_argument("--project", required=True)
    p_rev.add_argument("--task-id", default="")
    p_rev.add_argument("--list", action="store_true")
    p_rev.add_argument("--limit", type=int, default=20)
    p_rev.add_argument("--no-refresh-output", action="store_true")
    p_rev.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_rev.set_defaults(func=cmd_revert)

    # rebuild
    p_reb = sub.add_parser("rebuild", help="Zero-cost rebuild of transl_cache & gt_output (GalTransl rebuilda)")
    p_reb.add_argument("--project", required=True)
    p_reb.add_argument("--files", action="append", default=[])
    p_reb.add_argument("--ignore-post-src", action="store_true")
    p_reb.add_argument("--no-refresh-output", action="store_true")
    p_reb.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_reb.set_defaults(func=cmd_rebuild)

    # seed
    p_seed = sub.add_parser("seed", help="Drive self-heuristic review seed ledger (`work/seeds.json`)")
    p_seed.add_argument("--project", required=True)
    p_seed.add_argument(
        "--action",
        required=True,
        choices=["status", "sample", "add", "consume", "complete-round", "check-gate"],
    )
    p_seed.add_argument("--regions", type=int, default=3)
    p_seed.add_argument("--per-region", type=int, default=10)
    p_seed.add_argument("--pattern", default="")
    p_seed.add_argument("--query", default="")
    p_seed.add_argument("--category", default="expand", choices=["expand", "diff", "link", "sample"])
    p_seed.add_argument("--round", type=int, default=None)
    p_seed.add_argument("--seed-id", default="")
    p_seed.add_argument("--resolution", default="")
    p_seed.add_argument("--complete-round", action="store_true")
    p_seed.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_seed.set_defaults(func=cmd_seed)

    # export
    p_exp = sub.add_parser("export", help="Export final localized files from transl_cache")
    p_exp.add_argument("--project", required=True)
    p_exp.add_argument("--format", default="", choices=["", "gt_json", "compact_pipe"])
    p_exp.add_argument("--output", default="", help="Output directory (defaults to gt_output/)")
    p_exp.add_argument("--encoding", default="", choices=["", "utf8", "gbk"])
    p_exp.add_argument("--files", action="append", default=[])
    p_exp.add_argument("--with-src", action="store_true")
    p_exp.add_argument("--allow-incomplete", action="store_true")
    p_exp.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_exp.set_defaults(func=cmd_export)

    # status
    p_st = sub.add_parser("status", help="Show project progress, cache hit/miss stats, QA count, and seed gate")
    p_st.add_argument("--project", required=True)
    p_st.add_argument("--out-format", default="markdown", choices=["markdown", "json"])
    p_st.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
