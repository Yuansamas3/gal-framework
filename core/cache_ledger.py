"""Three-separation atomic cache ledger (`gt_input` / `transl_cache` / `gt_output` + `cache_history.jsonl`)."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from adapters import gt_json
from .dictionary import NormalDict
from .models import ItemStatus, ProjectConfig, SentenceItem
from .normalizer import preprocess_items
from .problems import is_whitelisted

CACHE_TEMP_SUFFIX = ".json.tmp"

# Legacy GalTransl key compatibility map: modern_key -> legacy_key
CACHE_KEY_COMPAT: dict[str, str] = {
    "pre_src": "pre_jp",
    "post_src": "post_jp",
    "pre_dst": "pre_zh",
    "proofread_dst": "proofread_zh",
    "post_dst_preview": "post_zh_preview",
    "proofread_comment": "doub_content",
}

# Cache miss reason codes (aligned with GalTransl Cache.py)
MISS_KEY_NOT_FOUND = "key_not_found"
MISS_POST_SRC_CHANGED = "post_src_changed"
MISS_PRE_DST_EMPTY = "pre_dst_empty"
MISS_TRANSLATE_FAILED = "translate_failed"
MISS_RETRAN_KEY = "retran_key"
MISS_RETRAN_PROBLEM = "retran_problem"
MISS_PROOFREAD_MISSING = "proofread_missing"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_task_id(prefix: str = "task") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def cache_get(cache_obj: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a cache entry field, preferring modern key and falling back to legacy GalTransl key."""
    if key in cache_obj:
        return cache_obj[key]
    old_key = CACHE_KEY_COMPAT.get(key)
    if old_key and old_key in cache_obj:
        return cache_obj[old_key]
    return default


def cache_has(cache_obj: dict[str, Any], key: str) -> bool:
    if key in cache_obj:
        return True
    old_key = CACHE_KEY_COMPAT.get(key)
    return bool(old_key and old_key in cache_obj)


def atomic_write_json(target_path: Path, data: Any) -> Path:
    """Windows-safe atomic JSON write: `<file>.json.tmp` -> `os.replace()`."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(str(target_path) + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
    return target_path


def cleanup_stale_cache_temp_files(cache_dir: Path) -> int:
    """Remove leftover `*.json.tmp` files from interrupted writes at task startup."""
    if not cache_dir.exists() or not cache_dir.is_dir():
        return 0
    removed = 0
    for p in cache_dir.rglob(f"*{CACHE_TEMP_SUFFIX}"):
        if not p.is_file():
            continue
        try:
            p.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def build_context_cache_key(items: Sequence[SentenceItem], pos: int) -> str:
    """Build GalTransl-style 3-line context key: prev_non_empty + cur + next_non_empty."""
    cur = items[pos]
    line_now = f"{cur.speaker}{cur.pre_src}"

    line_prev = "None"
    p = pos - 1
    while p >= 0 and not items[p].post_src:
        p -= 1
    if p >= 0:
        line_prev = f"{items[p].speaker}{items[p].pre_src}" or "None"

    line_next = "None"
    n = pos + 1
    while n < len(items) and not items[n].post_src:
        n += 1
    if n < len(items):
        line_next = f"{items[n].speaker}{items[n].pre_src}" or "None"

    return line_prev + line_now + line_next


def _build_snapshot_lookup(cache_list: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    n = len(cache_list)
    for i, obj in enumerate(cache_list):
        if not isinstance(obj, dict):
            continue
        if "index" in obj:
            try:
                by_index[int(obj["index"])] = obj
            except (TypeError, ValueError):
                pass
        explicit_key = str(obj.get("__cache_key", ""))
        if explicit_key:
            by_key[explicit_key] = obj
            continue
        line_now = f"{obj.get('name', '')}{cache_get(obj, 'pre_src', '')}"
        prev_s = f"{cache_list[i - 1].get('name', '')}{cache_get(cache_list[i - 1], 'pre_src', '')}" if i > 0 else "None"
        next_s = f"{cache_list[i + 1].get('name', '')}{cache_get(cache_list[i + 1], 'pre_src', '')}" if i + 1 < n else "None"
        by_key[(prev_s or "None") + line_now + (next_s or "None")] = obj
    return by_key, by_index


def _filter_problem_str(problem_str: str, filter_regexes: list[str]) -> str:
    if not problem_str or not filter_regexes:
        return problem_str
    parts = [p.strip() for p in re.split(r",\s*", problem_str) if p.strip()]
    compiled = []
    for pat in filter_regexes:
        try:
            compiled.append(re.compile(pat))
        except re.error:
            continue
    kept = [p for p in parts if not any(rx.search(p) for rx in compiled)]
    return ", ".join(kept)


def _matches_retran_key(retran_keys: Sequence[str], target: str) -> bool:
    if not target or not retran_keys:
        return False
    return any(bool(k) and k in target for k in retran_keys)


def input_file_rel_list(
    project_dir: Path,
    config: ProjectConfig,
    file_patterns: Sequence[str] | None = None,
) -> list[str]:
    """List relative JSON paths under `gt_input/` matching optional glob/name patterns."""
    in_dir = config.input_path(project_dir)
    if not in_dir.exists():
        return []
    patterns = [p.strip().replace("\\", "/") for p in (file_patterns or []) if p.strip()]
    rels: list[str] = []
    for p in sorted(in_dir.rglob("*.json")):
        if not p.is_file() or p.name.endswith(".meta.json"):
            continue
        rel = p.relative_to(in_dir).as_posix()
        if patterns:
            matched = any(
                rel == pat
                or Path(rel).name == pat
                or fnmatch.fnmatch(rel, pat)
                or fnmatch.fnmatch(Path(rel).name, pat)
                for pat in patterns
            )
            if not matched:
                continue
        rels.append(rel)
    return rels


def meta_sidecar_path(project_dir: Path, config: ProjectConfig, rel_name: str) -> Path:
    return config.input_path(project_dir) / f"{rel_name}.meta.json"


def cache_file_path_for(project_dir: Path, config: ProjectConfig, rel_name: str) -> Path:
    rel = Path(rel_name)
    out = config.cache_path(project_dir) / rel
    return out if str(out).endswith(".json") else Path(str(out) + ".json")


def load_cache_entries(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        return []
    raw = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    return raw if isinstance(raw, list) else []


def load_format_meta(project_dir: Path, config: ProjectConfig, rel_name: str) -> dict[str, Any]:
    sidecar = meta_sidecar_path(project_dir, config, rel_name)
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8-sig"))
    return {"adapter": "gt_json"}


def save_format_meta(project_dir: Path, config: ProjectConfig, rel_name: str, format_meta: dict[str, Any]) -> None:
    if format_meta and format_meta.get("adapter") != "gt_json":
        atomic_write_json(meta_sidecar_path(project_dir, config, rel_name), format_meta)


def save_input_file(
    project_dir: Path,
    config: ProjectConfig,
    rel_name: str,
    items: Sequence[SentenceItem],
    format_meta: dict[str, Any] | None = None,
) -> Path:
    """Write normalized `gt_input/<rel_name>` preserving `_meta` and speaker structure."""
    rows: list[dict[str, Any]] = []
    for it in items:
        row: dict[str, Any] = {}
        name_key = str(it.extra.get("_name_key", ""))
        if name_key == "names" and isinstance(it.speakers_raw, list):
            row["names"] = list(it.speakers_raw)
        elif name_key == "name":
            row["name"] = str(it.speakers_raw if it.speakers_raw is not None else it.speaker)
        elif it.speaker:
            row["name"] = it.speaker

        row["message"] = it.pre_src

        meta_extra = {k: v for k, v in it.extra.items() if k != "_name_key"}
        if it.raw_id:
            meta_extra["raw_id"] = it.raw_id
        if it.speaker_style:
            meta_extra["speaker_style"] = it.speaker_style
        if meta_extra:
            row["_meta"] = meta_extra
        rows.append(row)

    target = config.input_path(project_dir) / rel_name
    atomic_write_json(target, rows)
    if format_meta:
        save_format_meta(project_dir, config, rel_name, format_meta)
    return target


def load_input_items(project_dir: Path, config: ProjectConfig, rel_name: str) -> tuple[dict[str, Any], list[SentenceItem]]:
    """Read `gt_input/<rel_name>` into raw SentenceItems (unpacking `_meta`)."""
    path = config.input_path(project_dir) / rel_name
    _r, _m, items = gt_json.import_file(path, rel_name)
    for it in items:
        meta = it.extra.pop("_meta", None)
        if isinstance(meta, dict):
            if "raw_id" in meta:
                it.raw_id = str(meta.pop("raw_id"))
            if "speaker_style" in meta:
                it.speaker_style = str(meta.pop("speaker_style"))
            for k, v in meta.items():
                it.extra[k] = v
    fmt_meta = load_format_meta(project_dir, config, rel_name)
    return fmt_meta, items


def save_cache_file(
    project_dir: Path,
    config: ProjectConfig,
    rel_name: str,
    items: Sequence[SentenceItem],
) -> Path:
    """Atomically write `transl_cache/<rel_name>` using canonical modern keys."""
    records: list[dict[str, Any]] = []
    for pos, it in enumerate(items):
        # Persist entries that have translation, proofread, skip_check, or comments
        if (
            it.pre_dst == ""
            and it.proofread_dst == ""
            and not it.skip_check
            and not it.proofread_comment
        ):
            continue
        rec: dict[str, Any] = {
            "index": it.index,
            "name": it.speaker,
            "name_preview": it.speaker_dst or config.speaker_map.get(it.speaker, ""),
            "pre_src": it.pre_src,
            "post_src": it.post_src,
            "pre_dst": it.pre_dst,
            "proofread_dst": it.proofread_dst,
            "post_dst_preview": it.post_dst,
            "trans_by": it.trans_by,
            "proofread_by": it.proofread_by,
            "problem": ", ".join(it.problems),
            "problems": list(it.problems),
            "proofread_comment": it.proofread_comment,
            "skip_check": it.skip_check,
            "__cache_key": build_context_cache_key(items, pos),
        }
        records.append(rec)

    cache_path = cache_file_path_for(project_dir, config, rel_name)
    return atomic_write_json(cache_path, records)


def join_input_with_cache(
    items: Sequence[SentenceItem],
    cache_entries: list[dict[str, Any]],
    config: ProjectConfig,
    *,
    ignore_post_src: bool = False,
    proofread_mode: bool = False,
    retran_keys: Sequence[str] | None = None,
    retry_failed: bool | None = None,
    is_rebuild: bool = False,
) -> tuple[list[SentenceItem], list[SentenceItem]]:
    """Join preprocessed `gt_input` items with `transl_cache`, setting exact cache_miss_reason codes."""
    by_key, by_index = _build_snapshot_lookup(cache_entries)
    active_retran = config.retransl_keys if retran_keys is None else list(retran_keys)
    active_retry_failed = config.retransl_fail if retry_failed is None else retry_failed

    hits: list[SentenceItem] = []
    misses: list[SentenceItem] = []

    for pos, it in enumerate(items):
        it.cache_miss_reason = ""
        if it.pre_src == "" or it.post_src == "":
            it.status = ItemStatus.RULE_SKIPPED.value
            it.pre_dst = ""
            it.post_dst = ""
            hits.append(it)
            continue

        ctx_key = build_context_cache_key(items, pos)
        cache_obj = by_key.get(ctx_key)
        if cache_obj is None:
            cand = by_index.get(it.index)
            if cand is not None and cache_get(cand, "pre_src", "") == it.pre_src:
                cache_obj = cand

        if cache_obj is None:
            it.cache_miss_reason = MISS_KEY_NOT_FOUND
            it.status = ItemStatus.NONE.value
            misses.append(it)
            continue

        # Always hydrate metadata from cache before deciding hit/miss
        cached_pre_dst = str(cache_get(cache_obj, "pre_dst", "") or "")
        cached_proof = str(cache_get(cache_obj, "proofread_dst", "") or "")
        cached_post_src = str(cache_get(cache_obj, "post_src", "") or "")
        cached_preview = str(cache_get(cache_obj, "post_dst_preview", "") or "")
        cached_probs = cache_obj.get("problems")
        if isinstance(cached_probs, list):
            prob_list = [str(x) for x in cached_probs if str(x).strip()]
        else:
            raw_prob = str(cache_obj.get("problem", "") or "")
            prob_list = [p.strip() for p in re.split(r",\s*", raw_prob) if p.strip()]

        it.pre_dst = cached_pre_dst
        it.proofread_dst = cached_proof
        it.post_dst = cached_preview
        it.speaker_dst = str(cache_obj.get("name_preview", "") or config.speaker_map.get(it.speaker, ""))
        it.trans_by = str(cache_obj.get("trans_by", "") or "")
        it.proofread_by = str(cache_obj.get("proofread_by", "") or "")
        it.proofread_comment = str(cache_get(cache_obj, "proofread_comment", "") or "")
        it.skip_check = bool(cache_obj.get("skip_check", False))
        it.problems = prob_list

        no_proofread = (cached_proof == "")
        if no_proofread:
            if not ignore_post_src and cached_post_src != it.post_src:
                it.cache_miss_reason = MISS_POST_SRC_CHANGED
                it.status = ItemStatus.NONE.value
                misses.append(it)
                continue

            if not cache_has(cache_obj, "pre_dst") or cached_pre_dst == "":
                it.cache_miss_reason = MISS_PRE_DST_EMPTY
                it.status = ItemStatus.NONE.value
                misses.append(it)
                continue

            if active_retry_failed and "(Failed)" in cached_pre_dst:
                it.cache_miss_reason = MISS_TRANSLATE_FAILED
                it.status = ItemStatus.NONE.value
                misses.append(it)
                continue

            if not is_rebuild and active_retran:
                if _matches_retran_key(active_retran, it.pre_src):
                    it.cache_miss_reason = MISS_RETRAN_KEY
                    it.status = ItemStatus.NONE.value
                    misses.append(it)
                    continue
                if prob_list and not is_whitelisted(it, config.problem_whitelist):
                    filtered_prob_str = _filter_problem_str(
                        ", ".join(prob_list),
                        config.problem_filter_regex,
                    )
                    if _matches_retran_key(active_retran, filtered_prob_str):
                        it.cache_miss_reason = MISS_RETRAN_PROBLEM
                        it.status = ItemStatus.NONE.value
                        misses.append(it)
                        continue

        if proofread_mode and cached_proof == "":
            it.cache_miss_reason = MISS_PROOFREAD_MISSING
            it.status = ItemStatus.PROCESSED.value
            misses.append(it)
            continue

        it.status = ItemStatus.PROOFREAD.value if cached_proof != "" else ItemStatus.PROCESSED.value
        hits.append(it)

    return hits, misses


def load_project_files(
    project_dir: Path,
    config: ProjectConfig,
    pre_dict: NormalDict | None = None,
    *,
    file_patterns: Sequence[str] | None = None,
    ignore_post_src: bool = False,
    proofread_mode: bool = False,
    is_rebuild: bool = False,
) -> list[tuple[str, dict[str, Any], list[SentenceItem], list[SentenceItem], list[SentenceItem]]]:
    """Sweep stale temp files, load `gt_input`, preprocess, and join with `transl_cache`.

    Returns list of `(rel_name, format_meta, all_items, hit_items, miss_items)`.
    """
    cleanup_stale_cache_temp_files(config.cache_path(project_dir))
    results = []
    for rel_name in input_file_rel_list(project_dir, config, file_patterns):
        fmt_meta, items = load_input_items(project_dir, config, rel_name)
        preprocess_items(items, config, pre_dict=pre_dict, force=True)
        cache_entries = load_cache_entries(cache_file_path_for(project_dir, config, rel_name))
        hits, misses = join_input_with_cache(
            items,
            cache_entries,
            config,
            ignore_post_src=ignore_post_src,
            proofread_mode=proofread_mode,
            is_rebuild=is_rebuild,
        )
        results.append((rel_name, fmt_meta, items, hits, misses))
    return results


def snapshot_item_fields(it: SentenceItem) -> dict[str, Any]:
    """Capture mutable cache fields for history diff & revert."""
    return {
        "pre_dst": it.pre_dst,
        "proofread_dst": it.proofread_dst,
        "post_dst": it.post_dst,
        "speaker_dst": it.speaker_dst,
        "trans_by": it.trans_by,
        "proofread_by": it.proofread_by,
        "proofread_comment": it.proofread_comment,
        "skip_check": it.skip_check,
        "status": it.status,
    }


def append_history_event(
    project_dir: Path,
    config: ProjectConfig,
    *,
    action: str,
    reason: str,
    changes: list[dict[str, Any]],
    task_id: str | None = None,
) -> str:
    """Append an auditable mutation event to `cache_history.jsonl`."""
    tid = task_id or generate_task_id(action)
    if not changes:
        return tid
    hist_path = config.history_path(project_dir)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "task_id": tid,
        "timestamp": _utc_now(),
        "action": action,
        "reason": reason,
        "change_count": len(changes),
        "changes": changes,
    }
    with hist_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, ensure_ascii=False) + "\n")
    return tid


def read_history_events(project_dir: Path, config: ProjectConfig) -> list[dict[str, Any]]:
    hist_path = config.history_path(project_dir)
    if not hist_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw_ln in hist_path.read_text(encoding="utf-8-sig").splitlines():
        s = raw_ln.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return events
