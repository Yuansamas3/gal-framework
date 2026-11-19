"""Agent work-unit compiler, real-time lint patch applier, edit-cache, and search-replace on three-separation ledger."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .cache_ledger import (
    append_history_event,
    atomic_write_json,
    load_project_files,
    save_cache_file,
    snapshot_item_fields,
)
from .dictionary import GptDict, inject_glossary_for_batch, load_project_dicts
from .models import ItemStatus, ProjectConfig, SentenceItem
from .normalizer import normalize_and_restore_dst
from .problems import check_item_problems, problem_severity, split_problems_by_severity
from .rebuilder import export_project_outputs
from .text_preserve import TextPreserver

_ANCHOR_RE = re.compile(r"^[0-9a-f]{3}$")
_FILE_BLOCK_RE = re.compile(r"^(?:@@\s*@?file\s*:\s*|##\s*@file\s*:\s*|@file\s*:\s*)(.+)$", re.IGNORECASE)


def _is_int_token(val: str) -> bool:
    return val.isdigit() or (val.startswith("-") and val[1:].isdigit())


def _sanitize_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id.strip()).strip("._")
    return cleaned or "session"


def _extract_guideline_digest(config: ProjectConfig, project_dir: Path, profile: dict[str, Any]) -> str:
    """Build a concise, token-efficient charter + genre guideline + project guideline digest."""
    lines: list[str] = []
    charter_path = config.resolve_guideline_path(config.charter, project_dir)
    guide_path = config.resolve_guideline_path(config.guideline, project_dir)
    proj_guide_path = project_dir / config.project_guideline

    lines.append(
        f"Profile: `{config.profile}` | Guideline: `{config.guideline}` | "
        f"Encoding: `{config.target_encoding}` | OuterQuotesStripped: `{config.strip_outer_quotes}` | "
        f"RubyMode: `{config.ruby_preserve_mode}`"
    )
    if charter_path.exists():
        lines.append(
            "- 宪章红线：虚构叙事如实忠实呈现，禁止回避/删减/粉饰/弱化；残句不补全；"
            "功能性粗粝不抹平；零假名/促音/浊点/长音残留；严禁擅自加括号或英文译注；Avoid Overthinking。"
        )
    if guide_path.exists():
        raw = guide_path.read_text(encoding="utf-8-sig")
        bullets: list[str] = []
        for ln in raw.splitlines():
            s = ln.strip()
            if s.startswith(("核心要求", "**最终准绳")) or (
                s.startswith(("1. **", "2. **", "3. **", "4. **", "5. **")) and "「" not in s
            ):
                bullets.append(s)
            if len(bullets) >= 4:
                break
        if bullets:
            lines.extend(bullets)
    if proj_guide_path.exists():
        raw_proj = proj_guide_path.read_text(encoding="utf-8-sig").strip()
        if raw_proj:
            lines.append("\n<project_bible>\n" + raw_proj + "\n</project_bible>")
    forbidden = profile.get("forbidden_words", [])
    if forbidden:
        lines.append(f"- 语域违禁词示例：{'、'.join(forbidden[:12])}")
    return "\n".join(lines)


def _render_src_for_prompt(item: SentenceItem) -> str:
    text = item.post_src
    if "\r\n" in text:
        text = text.replace("\r\n", "<br>")
    elif "\n" in text:
        text = text.replace("\n", "<br>")
    return text.replace("|", "\\|")


def _render_dst_for_context(item: SentenceItem) -> str:
    text = item.post_dst or item.post_src or item.pre_src
    text = text.replace("\r\n", "<br>").replace("\n", "<br>")
    return text.replace("|", "\\|")


def pull_batch(
    project_dir: Path,
    *,
    file_patterns: Sequence[str] | None = None,
    batch_size: int | None = None,
    context_num: int | None = None,
    only_miss: bool = True,
    retransl_problems: bool = False,
    problem_type: str | None = None,
    proofread_mode: bool = False,
    offset: int = 0,
    session_id: str = "",
    compact: bool = False,
    no_compact: bool = False,
) -> dict[str, Any]:
    """Compile a token-efficient Agent work unit from cache misses (or problem/all items)."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, _ = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    sid = session_id.strip()
    use_compact = False if no_compact else (compact or bool(sid))

    b_size = batch_size if batch_size is not None else config.batch_size
    c_num = context_num if context_num is not None else config.context_num
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=file_patterns,
        proofread_mode=proofread_mode,
    )

    candidates: list[tuple[str, int, SentenceItem, list[SentenceItem]]] = []
    miss_summary: Counter[str] = Counter()

    for file_name, _meta, items, _hits, _misses in loaded:
        for pos, it in enumerate(items):
            if it.cache_miss_reason:
                miss_summary[it.cache_miss_reason] += 1
            if it.status in (ItemStatus.EXCLUDED.value, ItemStatus.RULE_SKIPPED.value):
                continue
            if retransl_problems:
                if not it.problems:
                    continue
                if problem_type and not any(problem_type in p for p in it.problems):
                    continue
            elif only_miss:
                if not it.cache_miss_reason and it.status != ItemStatus.NONE.value:
                    continue
                if not it.post_src.strip():
                    continue
            candidates.append((file_name, pos, it, items))

    total_matching = len(candidates)
    selected_tuples = candidates[offset : offset + b_size]
    selected_items = [t[2] for t in selected_tuples]

    # Sliding preceding context window
    context_rows: list[dict[str, Any]] = []
    seen_ctx: set[tuple[str, int]] = set()
    selected_keys = {(it.file_name, it.index) for it in selected_items}

    if c_num > 0 and selected_tuples:
        prev_file = None
        prev_pos = -999
        for file_name, pos, _it, all_items in selected_tuples:
            if file_name != prev_file or pos != prev_pos + 1:
                start = max(0, pos - c_num)
                for c_idx in range(start, pos):
                    cit = all_items[c_idx]
                    key = (cit.file_name, cit.index)
                    if key in selected_keys or key in seen_ctx:
                        continue
                    seen_ctx.add(key)
                    context_rows.append(
                        {
                            "anchor": cit.anchor,
                            "file": cit.file_name,
                            "id": cit.index,
                            "speaker": cit.speaker_dst or cit.speaker,
                            "text": _render_dst_for_context(cit),
                        }
                    )
            prev_file = file_name
            prev_pos = pos

    active_glossary, glossary_block = inject_glossary_for_batch(
        selected_items,
        gpt_dict,
        include_speaker=config.use_gpt_dict_in_name,
    )

    batch_tokens: Counter[str] = Counter()
    for it in selected_items:
        for tok in preserver.extract_control_tokens(it.post_src):
            batch_tokens[tok] += 1

    guideline_digest = _extract_guideline_digest(config, project_dir, profile)
    bible_hash = hashlib.sha1(guideline_digest.encode("utf-8")).hexdigest()
    bible_deduplicated = False
    rendered_guideline_anchor = guideline_digest

    if sid:
        session_file = project_dir / "work" / ".sessions" / f"{_sanitize_session_id(sid)}.json"
        prev_hash = ""
        if session_file.exists():
            try:
                sess_data = json.loads(session_file.read_text(encoding="utf-8-sig"))
                if isinstance(sess_data, dict):
                    prev_hash = str(sess_data.get("bible_hash", ""))
            except (OSError, json.JSONDecodeError):
                prev_hash = ""
        if prev_hash == bible_hash:
            bible_deduplicated = True
            header_line = guideline_digest.splitlines()[0] if guideline_digest else ""
            rendered_guideline_anchor = (
                f"Session: {sid} | BibleHash: {bible_hash[:8]} "
                "(unchanged — keep character voices, zero added parens, 100% control tokens)"
                + (f"\n{header_line}" if header_line else "")
            )
        else:
            atomic_write_json(
                session_file,
                {"session_id": sid, "bible_hash": bible_hash},
            )

    md_lines: list[str] = [
        f"# Work-Unit ({len(selected_items)} items | remaining={max(0, total_matching - offset - len(selected_items))})",
        "",
        "<guideline_anchor>",
        rendered_guideline_anchor,
        "</guideline_anchor>",
    ]
    if batch_tokens:
        samples = ", ".join(f"`{tok}`(x{cnt})" for tok, cnt in batch_tokens.most_common(12))
        md_lines.extend(["", "<control_samples>", f"必须 100% 原样保留：{samples}", "</control_samples>"])

    md_lines.extend(["", "<glossary>", glossary_block, "</glossary>"])

    if context_rows:
        if use_compact:
            md_lines.extend(["", "<context>"])
            cur_ctx_file = None
            for row in context_rows:
                if row["file"] != cur_ctx_file:
                    cur_ctx_file = row["file"]
                    md_lines.append(f"## @file: {cur_ctx_file}")
                spk_ctx = str(row["speaker"] or "-").replace("|", "\\|")
                md_lines.append(f"{row['anchor']}|{row['id']}|{spk_ctx}|{row['text']}")
            md_lines.append("</context>")
        else:
            md_lines.extend(["", "<context>", "| FILE | ID | SPEAKER | TEXT |", "| --- | --- | --- | --- |"])
            for row in context_rows:
                md_lines.append(
                    f"| {row['file']} | {row['id']} | {row['speaker'] or '-'} | {row['text']} |"
                )
            md_lines.append("</context>")

    if use_compact:
        md_lines.extend(["", "<input>"])
    else:
        md_lines.extend(
            [
                "",
                "<input>",
                "| ANCHOR\\|ID | FILE | SPEAKER | POST_SRC |",
                "| --- | --- | --- | --- |",
            ]
        )

    input_records: list[dict[str, Any]] = []
    cur_in_file = None
    for it in selected_items:
        rendered_src = _render_src_for_prompt(it)
        spk_col = it.speaker.replace("|", "\\|") if it.speaker else "-"
        suffix = ""
        if it.problems and retransl_problems:
            suffix = " [PROBLEMS: " + "; ".join(it.problems) + "]"
        elif it.cache_miss_reason and it.cache_miss_reason != "key_not_found":
            suffix = f" [MISS: {it.cache_miss_reason}]"
        if use_compact:
            if it.file_name != cur_in_file:
                cur_in_file = it.file_name
                md_lines.append(f"## @file: {cur_in_file}")
            md_lines.append(f"{it.anchor}|{it.index}|{spk_col}|{rendered_src}{suffix}")
        else:
            md_lines.append(
                f"| `{it.anchor}|{it.index}` | `{it.file_name}` | {spk_col} | {rendered_src}{suffix} |"
            )
        input_records.append(
            {
                "anchor": it.anchor,
                "anchor_id": it.anchor_id,
                "file": it.file_name,
                "id": it.index,
                "patch_key": it.patch_key,
                "speaker": it.speaker,
                "post_src": it.post_src,
                "pre_dst": it.pre_dst,
                "cache_miss_reason": it.cache_miss_reason,
                "problems": list(it.problems),
            }
        )
    md_lines.append("</input>")

    if use_compact:
        md_lines.extend(
            [
                "",
                "<patch_format>",
                "提交译文请使用分组紧凑补丁格式（不含外层「」引号，换行用 `<br>`，系统自动扣壳过译后字典）：",
                "```",
                "@@ file: <file_name>",
                "<id>|<纯净译文内核>",
                "```",
                "</patch_format>",
            ]
        )
    else:
        md_lines.extend(
            [
                "",
                "<patch_format>",
                "提交译文请使用 `@@ file|id`（或 `@@ anchor|id`）+ `+译文` 补丁格式（不含外层「」引号，系统会自动削标点、过译后字典并扣回原壳）：",
                "```",
                "@@ <file>|<id>",
                "+<纯净译文内核>",
                "```",
                "</patch_format>",
            ]
        )

    return {
        "mode": "retransl_problems" if retransl_problems else ("only_miss" if only_miss else "all"),
        "session_id": sid,
        "bible_hash": bible_hash,
        "bible_deduplicated": bible_deduplicated,
        "compact": use_compact,
        "selected_count": len(selected_items),
        "total_matching": total_matching,
        "remaining": max(0, total_matching - offset - len(selected_items)),
        "miss_breakdown": dict(miss_summary),
        "active_glossary_count": len(active_glossary),
        "active_glossary": [e.format_prompt_line() for e in active_glossary],
        "control_samples": [{"token": k, "count": v} for k, v in batch_tokens.most_common(12)],
        "context": context_rows,
        "items": input_records,
        "markdown": "\n".join(md_lines),
    }


def make_trial(
    project_dir: Path,
    *,
    file_patterns: Sequence[str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Generate a representative trial work unit (default 100 lines) for user style calibration."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, _ = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)
    loaded = load_project_files(project_dir, config, pre_dict=pre_dict, file_patterns=file_patterns)

    if not file_patterns and len(loaded) > 3:
        loaded = [loaded[0], loaded[len(loaded) // 2], loaded[-1]]

    chosen: list[SentenceItem] = []
    if loaded:
        quota = max(1, (limit + len(loaded) - 1) // len(loaded))
        for _fname, _meta, items, _hits, _misses in loaded:
            eligible = [
                it
                for it in items
                if it.status not in (ItemStatus.EXCLUDED.value, ItemStatus.RULE_SKIPPED.value)
                and it.post_src.strip()
            ]
            if not eligible:
                continue
            take = min(quota, len(eligible), limit - len(chosen))
            chosen.extend(eligible[:take])
            if len(chosen) >= limit:
                break

    active_glossary, glossary_block = inject_glossary_for_batch(
        chosen,
        gpt_dict,
        include_speaker=config.use_gpt_dict_in_name,
    )
    guideline_digest = _extract_guideline_digest(config, project_dir, profile)

    batch_tokens: Counter[str] = Counter()
    for it in chosen:
        for tok in preserver.extract_control_tokens(it.post_src):
            batch_tokens[tok] += 1

    md_lines = [
        f"# Trial Style-Calibration Package ({len(chosen)} lines)",
        "",
        "<guideline_anchor>",
        guideline_digest,
        "</guideline_anchor>",
        "",
        "<glossary>",
        glossary_block,
        "</glossary>",
        "",
        "<input>",
        "| ANCHOR\\|ID | FILE | SPEAKER | POST_SRC |",
        "| --- | --- | --- | --- |",
    ]
    records: list[dict[str, Any]] = []
    for it in chosen:
        md_lines.append(
            f"| `{it.anchor}|{it.index}` | `{it.file_name}` | {it.speaker or '-'} | {_render_src_for_prompt(it)} |"
        )
        records.append(
            {
                "anchor": it.anchor,
                "file": it.file_name,
                "id": it.index,
                "speaker": it.speaker,
                "post_src": it.post_src,
            }
        )
    md_lines.append("</input>")

    return {
        "trial_count": len(chosen),
        "files": list(dict.fromkeys(it.file_name for it in chosen)),
        "active_glossary_count": len(active_glossary),
        "active_glossary": [e.format_prompt_line() for e in active_glossary],
        "control_samples": [{"token": k, "count": v} for k, v in batch_tokens.most_common(12)],
        "items": records,
        "markdown": "\n".join(md_lines),
    }


@dataclass
class PatchEntry:
    """Single parsed translation submission entry."""

    file_name: str = ""
    index: int | None = None
    anchor: str = ""
    dst: str = ""
    speaker_dst: str = ""
    comment: str = ""
    raw_header: str = ""


def _parse_header_spec(header: str, current_file: str = "") -> tuple[str, int | None, str]:
    spec = header.strip().strip("`").strip()
    if _is_int_token(spec):
        return current_file, int(spec), ""
    parts = [p.strip() for p in spec.split("|") if p.strip()]
    if len(parts) == 2:
        p0, p1 = parts
        try:
            idx = int(p1)
        except ValueError:
            return p0, None, ""
        if _ANCHOR_RE.match(p0) and "." not in p0 and "/" not in p0:
            return current_file, idx, p0
        return p0, idx, ""
    if len(parts) == 3:
        p0, p1, p2 = parts
        if _is_int_token(p1):
            return p0, int(p1), p2
        if _is_int_token(p2):
            return p1, int(p2), p0
    return spec, None, ""


def _try_parse_compact_entry_line(raw_ln: str, current_file: str) -> PatchEntry | None:
    """Parse `<id>|<dst>` or `<anchor>|<id>|<dst>`, preserving any `|` inside `<dst>`."""
    line = raw_ln.rstrip("\r\n").lstrip()
    if "|" not in line:
        return None
    first, rest = line.split("|", 1)
    f_clean = first.strip().strip("`").strip()

    if "|" in rest:
        second, third = rest.split("|", 1)
        s_clean = second.strip().strip("`").strip()
        if _ANCHOR_RE.match(f_clean) and _is_int_token(s_clean):
            return PatchEntry(
                file_name=current_file,
                index=int(s_clean),
                anchor=f_clean,
                dst=third,
                raw_header=line[:80],
            )

    if _is_int_token(f_clean) and not rest.lstrip().startswith("{"):
        return PatchEntry(
            file_name=current_file,
            index=int(f_clean),
            anchor="",
            dst=rest,
            raw_header=line[:80],
        )
    return None


def parse_patch_text(raw_text: str) -> tuple[list[PatchEntry], list[str]]:
    """Parse Grouped Compact Patch (`@@ file: name` + `id|dst`), Anchor Patch (`@@ file|id` + `+dst`), JSON array, or JSONLines."""
    text = raw_text.strip()
    if not text:
        return [], ["empty patch input"]

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return [], ["empty patch input after stripping fences"]

    if cleaned.startswith("["):
        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                entries: list[PatchEntry] = []
                for row in arr:
                    if not isinstance(row, dict):
                        continue
                    entries.append(
                        PatchEntry(
                            file_name=str(row.get("file", row.get("file_name", ""))),
                            index=int(row["id"]) if "id" in row else (int(row["index"]) if "index" in row else None),
                            anchor=str(row.get("anchor", "")),
                            dst=str(row.get("dst", row.get("message", row.get("translation", "")))),
                            speaker_dst=str(row.get("name_dst", row.get("speaker_dst", row.get("name", "")))),
                            comment=str(row.get("comment", row.get("proofread_comment", ""))),
                            raw_header=json.dumps(row, ensure_ascii=False)[:80],
                        )
                    )
                return entries, []
        except json.JSONDecodeError:
            pass

    has_patch_blocks = any(
        _FILE_BLOCK_RE.match(ln.strip()) or ln.lstrip().startswith("@@") for ln in lines
    )
    if has_patch_blocks:
        entries = []
        errors: list[str] = []
        current_file = ""
        cur_header = ""
        cur_dst_lines: list[str] = []
        cur_spk = ""
        cur_comment = ""

        def _flush() -> None:
            nonlocal cur_header, cur_dst_lines, cur_spk, cur_comment
            if not cur_header:
                return
            fname, idx, anchor = _parse_header_spec(cur_header, current_file=current_file)
            if idx is None:
                errors.append(f"invalid patch header: @@ {cur_header}")
            else:
                entries.append(
                    PatchEntry(
                        file_name=fname,
                        index=idx,
                        anchor=anchor,
                        dst="\n".join(cur_dst_lines),
                        speaker_dst=cur_spk,
                        comment=cur_comment,
                        raw_header=cur_header,
                    )
                )
            cur_header = ""
            cur_dst_lines = []
            cur_spk = ""
            cur_comment = ""

        for raw_ln in lines:
            stripped = raw_ln.strip()
            if not stripped:
                continue
            if stripped in ("*** Begin Patch", "*** End Patch") or stripped.startswith("```"):
                continue
            m_file = _FILE_BLOCK_RE.match(stripped)
            if m_file:
                _flush()
                current_file = m_file.group(1).strip().strip("`").strip()
                continue
            if stripped.startswith(("#", "//")):
                continue
            if raw_ln.lstrip().startswith("@@"):
                _flush()
                cur_header = raw_ln.lstrip()[2:].strip()
                continue
            if cur_header:
                if raw_ln.startswith("+"):
                    cur_dst_lines.append(raw_ln[1:])
                    continue
                if stripped.startswith("@name:"):
                    cur_spk = stripped[len("@name:") :].strip()
                    continue
                if stripped.startswith("@comment:"):
                    cur_comment = stripped[len("@comment:") :].strip()
                    continue
                if raw_ln.startswith("="):
                    cur_spk = raw_ln[1:].strip()
                    continue
                compact_in_block = _try_parse_compact_entry_line(raw_ln, current_file)
                if compact_in_block is not None:
                    _flush()
                    entries.append(compact_in_block)
                continue

            if stripped.startswith("@name:") and entries:
                entries[-1].speaker_dst = stripped[len("@name:") :].strip()
                continue
            if stripped.startswith("@comment:") and entries:
                entries[-1].comment = stripped[len("@comment:") :].strip()
                continue
            compact_entry = _try_parse_compact_entry_line(raw_ln, current_file)
            if compact_entry is not None:
                entries.append(compact_entry)
            else:
                errors.append(f"invalid compact patch line: {stripped[:80]}")
        _flush()
        return entries, errors

    entries = []
    errors = []
    for line_no, raw_ln in enumerate(lines, start=1):
        s = raw_ln.strip()
        if not s or s.startswith("```") or s.startswith(("#", "//")):
            continue
        if "{" not in s:
            compact_bare = _try_parse_compact_entry_line(raw_ln, "")
            if compact_bare is not None:
                entries.append(compact_bare)
                continue
        anchor_prefix = ""
        json_part = s
        if "|" in s and not s.startswith("{"):
            maybe_anchor, rest = s.split("|", 1)
            if _ANCHOR_RE.match(maybe_anchor.strip()) and rest.strip().startswith("{"):
                anchor_prefix = maybe_anchor.strip()
                json_part = rest.strip()
        try:
            obj = json.loads(json_part)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSONL ({exc})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {line_no}: JSON object expected")
            continue
        idx_val = obj.get("id", obj.get("index"))
        entries.append(
            PatchEntry(
                file_name=str(obj.get("file", obj.get("file_name", ""))),
                index=int(idx_val) if idx_val is not None else None,
                anchor=str(obj.get("anchor", anchor_prefix)),
                dst=str(obj.get("dst", obj.get("message", obj.get("translation", "")))),
                speaker_dst=str(obj.get("name_dst", obj.get("speaker_dst", obj.get("name", "")))),
                comment=str(obj.get("comment", obj.get("proofread_comment", ""))),
                raw_header=s[:80],
            )
        )
    return entries, errors


def _build_delta_repair_unit(
    hard_error_items: list[dict[str, Any]],
    parse_or_target_errors: list[str],
    *,
    use_compact: bool = True,
) -> dict[str, Any]:
    """Build a Circuit-Breaker Delta Repair Work-Unit containing ONLY failed items from the current push."""
    md_lines: list[str] = [
        f"# CIRCUIT-BREAKER: Delta Repair Work-Unit ({len(hard_error_items)} hard-error item(s) | Advance BLOCKED)",
        "",
        "存在硬性 QA 错误或补丁定位错误，已阻断翻页！请仅修复并重新提交以下条目：",
    ]
    if parse_or_target_errors:
        md_lines.extend(["", "<patch_errors>"])
        for err in parse_or_target_errors:
            md_lines.append(f"- {err}")
        md_lines.append("</patch_errors>")

    if hard_error_items:
        md_lines.extend(["", "<input>"])
        if use_compact:
            cur_file = None
            for row in hard_error_items:
                if row["file"] != cur_file:
                    cur_file = row["file"]
                    md_lines.append(f"## @file: {cur_file}")
                spk = str(row.get("speaker") or "-").replace("|", "\\|")
                src = str(row.get("post_src", "")).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")
                cur_dst = str(row.get("post_dst", "")).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")
                probs = "; ".join(row.get("problems", []))
                md_lines.append(
                    f"{row['anchor']}|{row['id']}|{spk}|{src} [CUR_DST: {cur_dst}] [HARD_ERRORS: {probs}]"
                )
        else:
            md_lines.extend(["| ANCHOR\\|ID | FILE | SPEAKER | POST_SRC | HARD_ERRORS |", "| --- | --- | --- | --- | --- |"])
            for row in hard_error_items:
                spk = str(row.get("speaker") or "-").replace("|", "\\|")
                src = str(row.get("post_src", "")).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")
                probs = "; ".join(row.get("problems", [])).replace("|", "\\|")
                md_lines.append(f"| `{row['anchor']}|{row['id']}` | `{row['file']}` | {spk} | {src} | {probs} |")
        md_lines.append("</input>")

    md_lines.extend(
        [
            "",
            "<patch_format>",
            "```",
            "@@ file: <file_name>",
            "<id>|<修复后的纯净译文内核>",
            "```",
            "</patch_format>",
        ]
    )
    return {
        "mode": "delta_repair",
        "blocked": True,
        "selected_count": len(hard_error_items),
        "total_matching": len(hard_error_items),
        "remaining": len(hard_error_items),
        "errors": list(parse_or_target_errors),
        "items": list(hard_error_items),
        "markdown": "\n".join(md_lines),
    }


def push_batch(
    project_dir: Path,
    patch_text: str,
    *,
    default_file: str = "",
    proofread: bool = False,
    trans_by: str = "agent",
    reason: str = "batch-push",
    refresh_output: bool = True,
    pull_next: bool = False,
    session_id: str = "",
    compact: bool = False,
    no_compact: bool = False,
    file_patterns: Sequence[str] | None = None,
    batch_size: int | None = None,
    context_num: int | None = None,
) -> dict[str, Any]:
    """Apply patch to `transl_cache` with O(1) selective loading, delta QA lint, history audit, and optional `--pull-next`."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, post_dict = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    parsed_entries, parse_errors = parse_patch_text(patch_text)
    if not parsed_entries:
        loaded = []
    elif all(bool(e.file_name or default_file) for e in parsed_entries):
        selective_files = sorted({e.file_name or default_file for e in parsed_entries})
        loaded = load_project_files(
            project_dir,
            config,
            pre_dict=pre_dict,
            file_patterns=selective_files,
            ignore_post_src=True,
        )
    else:
        loaded = load_project_files(project_dir, config, pre_dict=pre_dict, ignore_post_src=True)

    file_map: dict[str, tuple[dict[str, Any], list[SentenceItem]]] = {}
    by_file_id: dict[tuple[str, int], SentenceItem] = {}
    by_basename_id: dict[tuple[str, int], list[SentenceItem]] = defaultdict(list)
    by_anchor_id: dict[tuple[str, int], list[SentenceItem]] = defaultdict(list)

    for rel_name, fmt_meta, items, _hits, _misses in loaded:
        file_map[rel_name] = (fmt_meta, items)
        bname = Path(rel_name).name
        for it in items:
            by_file_id[(rel_name, it.index)] = it
            by_basename_id[(bname, it.index)].append(it)
            by_anchor_id[(it.anchor, it.index)].append(it)

    errors: list[str] = list(parse_errors)
    touched_files: set[str] = set()
    applied_items: list[SentenceItem] = []
    history_changes: list[dict[str, Any]] = []
    config_dirty = False

    for entry in parsed_entries:
        if entry.index is None:
            errors.append(f"missing index in entry: {entry.raw_header}")
            continue

        target: SentenceItem | None = None
        fname = entry.file_name or default_file

        if fname:
            target = by_file_id.get((fname, entry.index))
            if target is None:
                cands = by_basename_id.get((Path(fname).name, entry.index), [])
                if len(cands) == 1:
                    target = cands[0]
        elif entry.anchor:
            cands = by_anchor_id.get((entry.anchor, entry.index), [])
            if len(cands) == 1:
                target = cands[0]
            elif len(cands) > 1:
                errors.append(f"ambiguous anchor|id `{entry.anchor}|{entry.index}`; specify file name")
                continue
        elif len(file_map) == 1:
            only_file = next(iter(file_map.keys()))
            target = by_file_id.get((only_file, entry.index))

        if target is None:
            errors.append(
                f"target item not found: file={fname!r}, id={entry.index}, anchor={entry.anchor!r}"
            )
            continue

        if entry.anchor and entry.anchor != target.anchor:
            errors.append(
                f"anchor mismatch for {target.patch_key}: expected `{target.anchor}`, got `{entry.anchor}`"
            )
            continue

        before_snap = snapshot_item_fields(target)

        dst_val = entry.dst
        if "<br>" in dst_val and "<br>" not in target.pre_src:
            if "\r\n" in target.post_src:
                dst_val = dst_val.replace("<br>", "\r\n")
            elif "\n" in target.post_src:
                dst_val = dst_val.replace("<br>", "\n")

        if proofread:
            target.proofread_dst = dst_val
            target.proofread_by = trans_by
            target.status = ItemStatus.PROOFREAD.value
        else:
            target.pre_dst = dst_val
            target.proofread_dst = ""
            target.trans_by = trans_by
            target.status = ItemStatus.PROCESSED.value

        if entry.comment:
            target.proofread_comment = entry.comment

        if entry.speaker_dst:
            target.speaker_dst = entry.speaker_dst
            if target.speaker and config.speaker_map.get(target.speaker) != entry.speaker_dst:
                config.speaker_map[target.speaker] = entry.speaker_dst
                config_dirty = True
        elif target.speaker and not target.speaker_dst and target.speaker in config.speaker_map:
            target.speaker_dst = config.speaker_map[target.speaker]

        normalize_and_restore_dst(target, config, post_dict=post_dict, preserver=preserver)
        check_item_problems(target, config, profile=profile, preserver=preserver, gpt_dict=gpt_dict)

        history_changes.append(
            {
                "file": target.file_name,
                "index": target.index,
                "before": before_snap,
                "after": snapshot_item_fields(target),
            }
        )
        touched_files.add(target.file_name)
        applied_items.append(target)

    export_payload: list[tuple[str, dict[str, Any], list[SentenceItem]]] = []
    for rel_name in sorted(touched_files):
        fmt_meta, items = file_map[rel_name]
        save_cache_file(project_dir, config, rel_name, items)
        export_payload.append((rel_name, fmt_meta, items))

    if config_dirty:
        config.save(project_dir)

    task_id = append_history_event(
        project_dir,
        config,
        action="batch-push",
        reason=reason,
        changes=history_changes,
    )

    if refresh_output and export_payload:
        export_project_outputs(project_dir, config, export_payload)

    problem_items: list[dict[str, Any]] = []
    hard_error_items: list[dict[str, Any]] = []
    soft_warn_items: list[dict[str, Any]] = []

    for it in applied_items:
        if not it.problems:
            continue
        h_errs, s_warns = split_problems_by_severity(it.problems)
        base_rec = {
            "file": it.file_name,
            "id": it.index,
            "anchor": it.anchor,
            "patch_key": it.patch_key,
            "speaker": it.speaker,
            "post_src": it.post_src,
            "pre_dst": it.effective_core_dst,
            "post_dst": it.post_dst,
            "severity": "error" if h_errs else "warn",
            "hard_errors": h_errs,
            "soft_warnings": s_warns,
            "problems": list(it.problems),
        }
        problem_items.append(base_rec)
        if h_errs:
            hard_error_items.append({**base_rec, "problems": h_errs})
        if s_warns:
            soft_warn_items.append({**base_rec, "problems": s_warns})

    result: dict[str, Any] = {
        "task_id": task_id,
        "applied_count": len(applied_items),
        "clean_count": len(applied_items) - len(problem_items),
        "problem_item_count": len(problem_items),
        "hard_error_count": len(hard_error_items),
        "soft_warn_count": len(soft_warn_items),
        "error_count": len(errors),
        "touched_files": sorted(touched_files),
        "errors": errors,
        "problem_items": problem_items,
        "hard_error_items": hard_error_items,
        "soft_warn_items": soft_warn_items,
    }

    batch_warn_cap = config.compute_batch_warn_budget(len(applied_items))
    result["batch_warn_budget"] = batch_warn_cap

    if pull_next:
        use_compact = False if no_compact else (compact or bool(session_id.strip()))
        if hard_error_items or errors:
            result["pull_next_status"] = "blocked_by_errors"
            result["next_batch"] = _build_delta_repair_unit(
                hard_error_items,
                errors,
                use_compact=use_compact,
            )
        elif applied_items and len(soft_warn_items) > batch_warn_cap:
            result["pull_next_status"] = "blocked_by_warn_spike"
            result["next_batch"] = _build_delta_repair_unit(
                soft_warn_items,
                [
                    f"per-batch soft warnings ({len(soft_warn_items)}) exceed batch warn cap ({batch_warn_cap}); trim overly long/drifted lines before advancing"
                ],
                use_compact=use_compact,
            )
        else:
            result["pull_next_status"] = "advanced"
            result["next_batch"] = pull_batch(
                project_dir,
                file_patterns=file_patterns or ([default_file] if default_file else None),
                batch_size=batch_size,
                context_num=context_num,
                only_miss=True,
                proofread_mode=proofread,
                session_id=session_id,
                compact=compact,
                no_compact=no_compact,
            )

    return result


def edit_cache_entries(
    project_dir: Path,
    edits: Sequence[dict[str, Any]],
    *,
    reason: str = "edit-cache",
    refresh_output: bool = True,
) -> dict[str, Any]:
    """Batch-patch specific `(file, index)` entries in `transl_cache` with O(1) delta file loading & history audit."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, post_dict = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    if not edits:
        loaded = []
    elif all(bool(str(ed.get("file", "")).strip()) for ed in edits):
        edit_files = sorted({str(ed.get("file", "")).strip() for ed in edits})
        loaded = load_project_files(
            project_dir,
            config,
            pre_dict=pre_dict,
            file_patterns=edit_files,
            ignore_post_src=True,
        )
    else:
        loaded = load_project_files(project_dir, config, pre_dict=pre_dict, ignore_post_src=True)

    file_map = {rel: (meta, items) for rel, meta, items, _h, _m in loaded}
    by_file_id = {(rel, it.index): it for rel, _meta, items, _h, _m in loaded for it in items}

    touched: set[str] = set()
    changes: list[dict[str, Any]] = []
    errors: list[str] = []
    updated_items: list[SentenceItem] = []

    for ed in edits:
        fname = str(ed.get("file", ""))
        idx = int(ed.get("index", 0))
        target = by_file_id.get((fname, idx))
        if target is None:
            errors.append(f"item not found: {fname}|{idx}")
            continue
        before_snap = snapshot_item_fields(target)
        if ed.get("pre_dst") is not None:
            target.pre_dst = str(ed["pre_dst"])
            target.status = ItemStatus.PROCESSED.value
        if ed.get("proofread_dst") is not None:
            target.proofread_dst = str(ed["proofread_dst"])
            if target.proofread_dst:
                target.status = ItemStatus.PROOFREAD.value
                target.proofread_by = str(ed.get("by", "editor"))
        if ed.get("speaker_dst") is not None:
            target.speaker_dst = str(ed["speaker_dst"])
        if ed.get("skip_check") is not None:
            target.skip_check = bool(ed["skip_check"])
        if ed.get("proofread_comment") is not None:
            target.proofread_comment = str(ed["proofread_comment"])

        if target.effective_core_dst != "":
            normalize_and_restore_dst(target, config, post_dict=post_dict, preserver=preserver)
            check_item_problems(target, config, profile=profile, preserver=preserver, gpt_dict=gpt_dict)
        else:
            target.post_dst = ""
            target.problems = []

        changes.append(
            {
                "file": target.file_name,
                "index": target.index,
                "before": before_snap,
                "after": snapshot_item_fields(target),
            }
        )
        touched.add(target.file_name)
        updated_items.append(target)

    export_payload = []
    for rel_name in sorted(touched):
        fmt_meta, items = file_map[rel_name]
        save_cache_file(project_dir, config, rel_name, items)
        export_payload.append((rel_name, fmt_meta, items))

    task_id = append_history_event(
        project_dir,
        config,
        action="edit-cache",
        reason=reason,
        changes=changes,
    )
    if refresh_output and export_payload:
        export_project_outputs(project_dir, config, export_payload)

    return {
        "task_id": task_id,
        "edited_count": len(updated_items),
        "error_count": len(errors),
        "errors": errors,
        "problem_items": [
            {"file": it.file_name, "id": it.index, "post_dst": it.post_dst, "problems": list(it.problems)}
            for it in updated_items
            if it.problems
        ],
    }


def aggregate_problems(
    project_dir: Path,
    *,
    problem_type: str | None = None,
    severity: str = "all",
    group_by: str = "detail",
    limit: int = 50,
    file_patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate QA problems across the project with severity filtering, ranking, and drill-down."""
    config = ProjectConfig.load(project_dir)
    pre_dict, _, _ = load_project_dicts(config, project_dir)
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=file_patterns,
        ignore_post_src=True,
    )

    sev_norm = (severity or "all").lower()
    total_items = 0
    translated_items = 0
    problem_items_total = 0
    hard_error_item_count = 0
    soft_warn_item_count = 0
    type_counter: Counter[str] = Counter()
    file_counter: Counter[str] = Counter()
    detail_rows: list[dict[str, Any]] = []

    for file_name, _meta, items, _hits, _misses in loaded:
        for it in items:
            total_items += 1
            if it.status in (ItemStatus.PROCESSED.value, ItemStatus.PROOFREAD.value):
                translated_items += 1
            if not it.problems:
                continue
            type_filtered = [p for p in it.problems if not problem_type or problem_type in p]
            if not type_filtered:
                continue
            h_errs, s_warns = split_problems_by_severity(type_filtered)
            if h_errs:
                hard_error_item_count += 1
            if s_warns:
                soft_warn_item_count += 1

            if sev_norm == "error":
                matched_probs = h_errs
            elif sev_norm == "warn":
                matched_probs = s_warns
            else:
                matched_probs = type_filtered
            if not matched_probs:
                continue

            problem_items_total += 1
            file_counter[file_name] += 1
            for p in matched_probs:
                cat = p.split("：", 1)[0]
                type_counter[cat] += 1
            detail_rows.append(
                {
                    "file": file_name,
                    "id": it.index,
                    "anchor": it.anchor,
                    "speaker": it.speaker,
                    "post_src": it.post_src,
                    "post_dst": it.post_dst,
                    "severity": "error" if split_problems_by_severity(matched_probs)[0] else "warn",
                    "problems": matched_probs,
                }
            )

    summary: dict[str, Any] = {
        "total_items": total_items,
        "translated_items": translated_items,
        "severity": sev_norm,
        "problem_item_count": problem_items_total,
        "hard_error_items": hard_error_item_count,
        "soft_warn_items": soft_warn_item_count,
        "by_type": [
            {"type": k, "severity": problem_severity(k), "count": v}
            for k, v in type_counter.most_common()
        ],
        "by_file": [{"file": k, "count": v} for k, v in file_counter.most_common()],
        "group_by": group_by,
    }
    if group_by == "file":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in detail_rows:
            if len(grouped[row["file"]]) < limit:
                grouped[row["file"]].append(row)
        summary["files"] = dict(grouped)
    else:
        summary["items"] = detail_rows[:limit]
    return summary


def search_replace_project(
    project_dir: Path,
    find: str,
    replace: str,
    *,
    field: str = "pre_dst",
    use_regex: bool = False,
    file_patterns: Sequence[str] | None = None,
    only_problem: str | None = None,
    dry_run: bool = False,
    reason: str = "search-replace",
    refresh_output: bool = True,
) -> dict[str, Any]:
    """Search and replace across `transl_cache`, recording history for one-click `revert`."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, post_dict = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    rx = re.compile(find) if use_regex else None
    loaded = load_project_files(
        project_dir,
        config,
        pre_dict=pre_dict,
        file_patterns=file_patterns,
        ignore_post_src=True,
    )

    changes_preview: list[dict[str, Any]] = []
    history_changes: list[dict[str, Any]] = []
    touched_files: list[str] = []
    export_payload: list[tuple[str, dict[str, Any], list[SentenceItem]]] = []

    for rel_name, fmt_meta, items, _hits, _misses in loaded:
        file_dirty = False
        for it in items:
            if only_problem and not any(only_problem in p for p in it.problems):
                continue
            if field == "speaker_dst":
                old_val = it.speaker_dst
            elif field == "proofread_dst":
                old_val = it.proofread_dst
            else:
                old_val = it.effective_core_dst
            if not old_val:
                continue

            new_val = rx.sub(replace, old_val) if (use_regex and rx is not None) else old_val.replace(find, replace)
            if new_val == old_val:
                continue

            changes_preview.append(
                {
                    "file": rel_name,
                    "id": it.index,
                    "field": field,
                    "before": old_val,
                    "after": new_val,
                }
            )
            if not dry_run:
                before_snap = snapshot_item_fields(it)
                if field == "speaker_dst":
                    it.speaker_dst = new_val
                elif field == "proofread_dst" or it.proofread_dst:
                    it.proofread_dst = new_val
                else:
                    it.pre_dst = new_val
                normalize_and_restore_dst(it, config, post_dict=post_dict, preserver=preserver)
                check_item_problems(it, config, profile=profile, preserver=preserver, gpt_dict=gpt_dict)
                history_changes.append(
                    {
                        "file": rel_name,
                        "index": it.index,
                        "before": before_snap,
                        "after": snapshot_item_fields(it),
                    }
                )
                file_dirty = True

        if file_dirty and not dry_run:
            save_cache_file(project_dir, config, rel_name, items)
            touched_files.append(rel_name)
            export_payload.append((rel_name, fmt_meta, items))

    task_id = ""
    if not dry_run and history_changes:
        task_id = append_history_event(
            project_dir,
            config,
            action="search-replace",
            reason=reason,
            changes=history_changes,
        )
        if refresh_output and export_payload:
            export_project_outputs(project_dir, config, export_payload)

    return {
        "dry_run": dry_run,
        "task_id": task_id,
        "field": field,
        "matched_count": len(changes_preview),
        "touched_files": touched_files,
        "changes": changes_preview[:100],
    }
