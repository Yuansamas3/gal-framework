"""Bidirectional lossless source/target format adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from core.models import SentenceItem

from . import compact_pipe, gt_json

ADAPTERS = {
    "gt_json": gt_json,
    "compact_pipe": compact_pipe,
}


def get_adapter(fmt: str):
    if fmt not in ADAPTERS:
        raise ValueError(f"Unsupported adapter format: {fmt!r}. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[fmt]


def import_path(
    fmt: str,
    input_path: Path,
    *,
    pipe_style: str = "auto",
) -> list[tuple[str, dict[str, Any], list[SentenceItem]]]:
    adapter = get_adapter(fmt)
    if fmt == "compact_pipe":
        return adapter.import_path(input_path, pipe_style=pipe_style)
    return adapter.import_path(input_path)


def export_file(
    fmt: str,
    output_path: Path,
    items: Sequence[SentenceItem],
    format_meta: dict[str, Any],
    *,
    speaker_map: dict[str, str] | None = None,
    encoding: str = "utf-8",
    with_src: bool = False,
) -> Path:
    return get_adapter(fmt).export_file(
        output_path,
        items,
        format_meta,
        speaker_map=speaker_map,
        encoding=encoding,
        with_src=with_src,
    )
