"""Lossless bidirectional adapter for standard GalTransl JSON (`[{name, message}]`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from core.models import SentenceItem


def import_file(file_path: Path, rel_name: str) -> tuple[str, dict[str, Any], list[SentenceItem]]:
    """Import a single GalTransl JSON file into SentenceItem list."""
    raw = json.loads(file_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError(f"GalTransl JSON must be a list of objects: {file_path}")

    items: list[SentenceItem] = []
    for idx, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            continue
        message = str(row.get("message", ""))
        extra = {k: v for k, v in row.items() if k not in ("name", "names", "message")}

        if "names" in row:
            raw_names = row["names"]
            if isinstance(raw_names, list):
                speakers_raw: Any = [str(n) for n in raw_names]
                speaker = "、".join(n for n in speakers_raw if n)
            else:
                speakers_raw = str(raw_names)
                speaker = speakers_raw
            extra["_name_key"] = "names"
        elif "name" in row:
            speakers_raw = str(row["name"])
            speaker = speakers_raw
            extra["_name_key"] = "name"
        else:
            speakers_raw = ""
            speaker = ""
            extra["_name_key"] = ""

        item = SentenceItem(
            file_name=rel_name,
            index=idx,
            pre_src=message,
            post_src=message,
            speaker=speaker,
            speakers_raw=speakers_raw,
            extra=extra,
        )
        items.append(item)

    return rel_name, {"adapter": "gt_json"}, items


def import_path(input_path: Path) -> list[tuple[str, dict[str, Any], list[SentenceItem]]]:
    """Import a single JSON file or all JSON files under a directory."""
    if input_path.is_file():
        return [import_file(input_path, input_path.name)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    results: list[tuple[str, dict[str, Any], list[SentenceItem]]] = []
    for p in sorted(input_path.rglob("*.json")):
        if p.is_file():
            rel = p.relative_to(input_path).as_posix()
            results.append(import_file(p, rel))
    return results


def _resolve_speaker_output(
    item: SentenceItem,
    speaker_map: dict[str, str],
) -> Any:
    raw = item.speakers_raw
    if isinstance(raw, list):
        if item.speaker_dst and len(raw) == 1:
            return [item.speaker_dst]
        return [speaker_map.get(n, n) for n in raw]
    if isinstance(raw, str) and raw:
        return item.speaker_dst or speaker_map.get(raw, raw)
    return raw


def export_file(
    output_path: Path,
    items: Sequence[SentenceItem],
    format_meta: dict[str, Any],
    *,
    speaker_map: dict[str, str] | None = None,
    encoding: str = "utf-8",
    with_src: bool = False,
) -> Path:
    """Export SentenceItems back to standard GalTransl JSON format."""
    spk_map = speaker_map or {}
    out_rows: list[dict[str, Any]] = []

    for it in items:
        row: dict[str, Any] = {}
        name_key = str(it.extra.get("_name_key", ""))
        if name_key == "names":
            row["names"] = _resolve_speaker_output(it, spk_map)
        elif name_key == "name":
            row["name"] = _resolve_speaker_output(it, spk_map)

        row["message"] = it.post_dst if it.post_dst != "" else it.pre_src
        if with_src:
            row["ori_message"] = it.pre_src

        for k, v in it.extra.items():
            if k == "_name_key":
                continue
            if k not in row:
                row[k] = v
        out_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc = "gbk" if encoding.lower() in ("gbk", "gb2312", "cp936") else "utf-8"
    output_path.write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n",
        encoding=enc,
        errors="strict",
    )
    return output_path
