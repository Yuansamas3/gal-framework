"""Zero-cost full project rebuilder (GalTransl `rebuilda` equivalent) & history revert engine."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from adapters import export_file
from .cache_ledger import (
    MISS_POST_SRC_CHANGED,
    append_history_event,
    cleanup_stale_cache_temp_files,
    load_project_files,
    project_ledger_lock,
    read_history_events,
    save_cache_file,
    snapshot_item_fields,
)
from .dictionary import load_project_dicts
from .models import ItemStatus, ProjectConfig
from .normalizer import normalize_and_restore_dst
from .problems import check_item_problems
from .text_preserve import TextPreserver


def export_project_outputs(
    project_dir: Path,
    config: ProjectConfig,
    loaded_files: Sequence[tuple[str, dict[str, Any], list[Any]]],
    *,
    output_dir: Path | None = None,
    output_format: str | None = None,
    encoding: str | None = None,
    with_src: bool = False,
) -> list[str]:
    """Export `post_dst_preview` from cached items into `gt_output/` (or custom output_dir)."""
    target_dir = output_dir or config.output_path(project_dir)
    fmt = output_format or "gt_json"
    enc = encoding or config.target_encoding
    written: list[str] = []

    for rel_name, fmt_meta, items in loaded_files:
        out_rel = rel_name
        if fmt == "compact_pipe" and out_rel.endswith(".json") and fmt_meta.get("orig_rel_name"):
            out_rel = str(fmt_meta["orig_rel_name"])
        out_path = target_dir / out_rel
        export_file(
            fmt,
            out_path,
            items,
            fmt_meta,
            speaker_map=config.speaker_map,
            encoding=enc,
            with_src=with_src,
        )
        written.append(str(out_path))
    return written


def rebuild_project(
    project_dir: Path,
    *,
    file_patterns: Sequence[str] | None = None,
    ignore_post_src: bool = False,
    refresh_output: bool = True,
) -> dict[str, Any]:
    """Zero-LLM rebuild: re-apply latest pre/post/gpt dicts, shell restore, QA, and gt_output."""
    config = ProjectConfig.load(project_dir)
    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, post_dict = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    with project_ledger_lock(project_dir):
        cleanup_stale_cache_temp_files(config.cache_path(project_dir), min_age_seconds=300.0)
        loaded = load_project_files(
            project_dir,
            config,
            pre_dict=pre_dict,
            file_patterns=file_patterns,
            ignore_post_src=ignore_post_src,
            is_rebuild=True,
        )

        before_problem_items = 0
        after_problem_items = 0
        total_items = 0
        total_hits = 0
        total_misses = 0
        rebuilt_translated = 0
        miss_reasons: Counter[str] = Counter()
        miss_details: list[dict[str, Any]] = []
        type_counter: Counter[str] = Counter()
        export_payload: list[tuple[str, dict[str, Any], list[Any]]] = []

        for rel_name, fmt_meta, items, hits, misses in loaded:
            total_items += len(items)
            total_hits += len(hits)
            total_misses += len(misses)
            before_problem_items += sum(1 for it in items if it.problems)

            for m in misses:
                miss_reasons[m.cache_miss_reason or "unknown"] += 1
                if len(miss_details) < 50:
                    miss_details.append(
                        {
                            "file": rel_name,
                            "id": m.index,
                            "reason": m.cache_miss_reason,
                            "post_src": m.post_src,
                        }
                    )

            for it in items:
                if it.speaker:
                    if it.speaker in config.speaker_map:
                        it.speaker_dst = config.speaker_map[it.speaker]
                    elif it.proofread_by != "editor":
                        it.speaker_dst = ""
                    if config.use_post_dict_in_name and post_dict.rules:
                        base_name = it.speaker_dst or it.speaker
                        if base_name:
                            it.speaker_dst = post_dict.apply(base_name, it, preserver=preserver)

                if not ignore_post_src and it.cache_miss_reason == MISS_POST_SRC_CHANGED:
                    # Keep post_src_changed miss intact so subsequent `batch-pull --only-miss` retranslates it
                    if it.problems:
                        after_problem_items += 1
                        for p in it.problems:
                            type_counter[p.split("：", 1)[0]] += 1
                    continue

                if it.effective_core_dst != "":
                    normalize_and_restore_dst(it, config, post_dict=post_dict, preserver=preserver)
                    check_item_problems(
                        it,
                        config,
                        profile=profile,
                        preserver=preserver,
                        gpt_dict=gpt_dict,
                    )
                    rebuilt_translated += 1
                else:
                    it.problems = []

                if it.problems:
                    after_problem_items += 1
                    for p in it.problems:
                        type_counter[p.split("：", 1)[0]] += 1

            save_cache_file(
                project_dir,
                config,
                rel_name,
                items,
                preserve_miss_post_src=not ignore_post_src,
            )
            export_payload.append((rel_name, fmt_meta, items))

        output_files: list[str] = []
        export_errors: list[str] = []
        if refresh_output and export_payload:
            for rel_name, fmt_meta, items in export_payload:
                try:
                    output_files.extend(
                        export_project_outputs(project_dir, config, [(rel_name, fmt_meta, items)])
                    )
                except UnicodeEncodeError as exc:
                    export_errors.append(f"{rel_name}: encoding error ({exc})")

    return {
        "files_rebuilt": len(loaded),
        "total_items": total_items,
        "cache_hits": total_hits,
        "cache_misses": total_misses,
        "miss_breakdown": dict(miss_reasons),
        "miss_details": miss_details,
        "rebuilt_translated_items": rebuilt_translated,
        "problems_before": before_problem_items,
        "problems_after": after_problem_items,
        "delta": after_problem_items - before_problem_items,
        "by_type": [{"type": k, "count": v} for k, v in type_counter.most_common()],
        "output_files": output_files,
        "export_errors": export_errors,
    }


def revert_task(
    project_dir: Path,
    task_id: str,
    *,
    refresh_output: bool = True,
) -> dict[str, Any]:
    """Revert all cache mutations recorded under `task_id` in `cache_history.jsonl`."""
    config = ProjectConfig.load(project_dir)
    events = read_history_events(project_dir, config)
    target_event = next((ev for ev in reversed(events) if ev.get("task_id") == task_id), None)
    if target_event is None:
        return {
            "status": "not_found",
            "task_id": task_id,
            "reverted_count": 0,
            "error": f"task_id not found in {config.history_file}: {task_id}",
        }

    profile = config.load_profile_data(project_dir)
    pre_dict, gpt_dict, post_dict = load_project_dicts(config, project_dir)
    preserver = TextPreserver.from_project(config, project_dir)

    with project_ledger_lock(project_dir):
        changes_by_file: dict[str, list[dict[str, Any]]] = {}
        for ch in target_event.get("changes", []):
            fname = str(ch.get("file", "")).replace("\\", "/")
            if fname:
                changes_by_file.setdefault(fname, []).append(ch)

        loaded = load_project_files(
            project_dir,
            config,
            pre_dict=pre_dict,
            file_patterns=list(changes_by_file.keys()),
            ignore_post_src=True,
        )

        reverted_records: list[dict[str, Any]] = []
        export_payload: list[tuple[str, dict[str, Any], list[Any]]] = []

        for rel_name, fmt_meta, items, _hits, _misses in loaded:
            file_changes = changes_by_file.get(rel_name)
            if not file_changes:
                continue
            by_idx = {it.index: it for it in items}
            # Reverse order ensures duplicate index edits within a single task unwind to the true original state
            for ch in reversed(file_changes):
                idx = int(ch["index"])
                it = by_idx.get(idx)
                if it is None:
                    continue
                before_now = snapshot_item_fields(it)
                prev_snap = ch.get("before", {})
                it.pre_dst = str(prev_snap.get("pre_dst", ""))
                it.proofread_dst = str(prev_snap.get("proofread_dst", ""))
                it.speaker_dst = str(prev_snap.get("speaker_dst", ""))
                it.trans_by = str(prev_snap.get("trans_by", ""))
                it.proofread_by = str(prev_snap.get("proofread_by", ""))
                it.proofread_comment = str(prev_snap.get("proofread_comment", ""))
                it.skip_check = bool(prev_snap.get("skip_check", False))

                if it.effective_core_dst != "":
                    it.status = ItemStatus.PROOFREAD.value if it.proofread_dst else ItemStatus.PROCESSED.value
                    normalize_and_restore_dst(it, config, post_dict=post_dict, preserver=preserver)
                    check_item_problems(it, config, profile=profile, preserver=preserver, gpt_dict=gpt_dict)
                else:
                    it.status = ItemStatus.NONE.value
                    it.post_dst = ""
                    it.problems = []

                reverted_records.append(
                    {
                        "file": rel_name,
                        "index": idx,
                        "before": before_now,
                        "after": snapshot_item_fields(it),
                    }
                )

            save_cache_file(project_dir, config, rel_name, items)
            export_payload.append((rel_name, fmt_meta, items))

        revert_tid = append_history_event(
            project_dir,
            config,
            action="revert",
            reason=f"revert {task_id}",
            changes=reverted_records,
        )

        output_files: list[str] = []
        if refresh_output and export_payload:
            for rel_name, fmt_meta, items in export_payload:
                try:
                    output_files.extend(
                        export_project_outputs(project_dir, config, [(rel_name, fmt_meta, items)])
                    )
                except UnicodeEncodeError:
                    pass

    return {
        "status": "reverted",
        "reverted_task_id": task_id,
        "revert_history_id": revert_tid,
        "reverted_count": len(reverted_records),
        "touched_files": sorted(changes_by_file.keys()),
        "output_files": output_files,
    }
