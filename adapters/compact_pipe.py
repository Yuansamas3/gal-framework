"""Lossless bidirectional adapter for compact pipe scripts (`ID|spk|text` and `ID|[spk]text`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from core.models import SentenceItem, atomic_write_text

_KNOWN_CONTROL_BRACKET_TAGS: frozenset[str] = frozenset(
    {
        "l",
        "r",
        "p",
        "cm",
        "er",
        "ct",
        "lr",
        "np",
        "k",
        "x",
        "・",
        "resetfont",
        "resetstyle",
        "endruby",
        "endindent",
        "indent",
    }
)


def _is_bracket_speaker_tag(spk: str) -> bool:
    if not spk or len(spk) > 32:
        return False
    if any(ch in spk for ch in ("|", ",", "=", " ", "\t")):
        return False
    if spk.lower() in _KNOWN_CONTROL_BRACKET_TAGS:
        return False
    return True


def _parse_compact_line(
    line: str,
    *,
    pipe_style: str = "auto",
) -> tuple[str, str, str, str] | None:
    """Parse a non-comment compact-pipe line into (raw_id, speaker, text, speaker_style).

    Supported formats (`pipe_style` = 'auto' | 'pipe' | 'bracket' | 'plain'):
      - `ID|spk|text`    -> speaker_style = "pipe"
      - `ID||text`       -> speaker_style = "pipe" (empty speaker)
      - `ID|[spk]text`   -> speaker_style = "bracket" (when `[spk]` precedes a dialogue quote, not inline furigana/control)
      - `ID|text`        -> speaker_style = "plain"
    """
    if "|" not in line:
        return None
    raw_id, rest = line.split("|", 1)
    if not raw_id.strip():
        return None

    style_mode = (pipe_style or "auto").lower()

    if style_mode == "plain":
        return raw_id, "", rest, "plain"

    if style_mode == "pipe":
        if "|" in rest:
            spk_cand, text_cand = rest.split("|", 1)
            return raw_id, spk_cand, text_cand, "pipe"
        return raw_id, "", rest, "plain"

    if style_mode == "bracket":
        if rest.startswith("["):
            close_idx = rest.find("]")
            if close_idx > 0:
                spk = rest[1:close_idx]
                after = rest[close_idx + 1 :]
                if _is_bracket_speaker_tag(spk):
                    return raw_id, spk, after, "bracket"
        return raw_id, "", rest, "plain"

    # Auto mode:
    # 1) Check Format A (`ID|[spk]text`) BEFORE Format B so `[遥]"text\|more"` is never sliced on `\|`
    #    (distinguished from inline KAG/M2 furigana `[・]text`, `[l]「...」`, or `[ルビ]漢字` by tag filter & dialogue quote)
    if rest.startswith("["):
        close_idx = rest.find("]")
        if close_idx > 0:
            spk = rest[1:close_idx]
            after = rest[close_idx + 1 :]
            if _is_bracket_speaker_tag(spk) and after.lstrip().startswith(
                ("「", "『", "“", "\"", "（", "(")
            ):
                return raw_id, spk, after, "bracket"

    # 2) Format B: `ID|spk|text` or `ID||text` (guard against escaped `\|` control codes and prose sentences)
    if "|" in rest:
        spk_cand, text_cand = rest.split("|", 1)
        if (
            len(spk_cand) <= 32
            and not spk_cand.endswith("\\")
            and "\\" not in spk_cand
            and not any(q in spk_cand for q in ("「", "『", "“", "\"", "（", "(", "。", "！", "？", "、", "，"))
        ):
            return raw_id, spk_cand, text_cand, "pipe"

    # 3) Format C: `ID|text`
    return raw_id, "", rest, "plain"


def import_file(
    file_path: Path,
    rel_name: str,
    *,
    pipe_style: str = "auto",
) -> tuple[str, dict[str, Any], list[SentenceItem]]:
    """Import a compact-pipe text file while recording full line layout for lossless export."""
    raw_content = file_path.read_text(encoding="utf-8-sig")
    trailing_newline = raw_content.endswith("\n")
    lines = raw_content.splitlines()

    items: list[SentenceItem] = []
    layout: list[dict[str, Any]] = []
    seq_idx = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            layout.append({"kind": "raw", "line": raw_line})
            continue

        parsed = _parse_compact_line(raw_line, pipe_style=pipe_style)
        if parsed is None:
            layout.append({"kind": "raw", "line": raw_line})
            continue

        raw_id, spk, text, style = parsed
        seq_idx += 1
        item = SentenceItem(
            file_name=rel_name,
            index=seq_idx,
            raw_id=raw_id,
            pre_src=text,
            post_src=text,
            speaker=spk,
            speakers_raw=spk,
            speaker_style=style,
        )
        items.append(item)
        layout.append({"kind": "item", "index": seq_idx})

    meta = {
        "adapter": "compact_pipe",
        "trailing_newline": trailing_newline,
        "layout": layout,
    }
    return rel_name, meta, items


def import_path(
    input_path: Path,
    *,
    pipe_style: str = "auto",
) -> list[tuple[str, dict[str, Any], list[SentenceItem]]]:
    """Import a single compact-pipe file or all `.txt`/`.tsv`/`.pipe` files under a directory."""
    if input_path.is_file():
        return [import_file(input_path, input_path.name, pipe_style=pipe_style)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    exts = {".txt", ".pipe", ".tsv", ".dat"}
    results: list[tuple[str, dict[str, Any], list[SentenceItem]]] = []
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(input_path).as_posix()
            results.append(import_file(p, rel, pipe_style=pipe_style))
    return results


def _format_compact_item(item: SentenceItem, speaker_map: dict[str, str]) -> str:
    orig_spk = item.speakers_raw if isinstance(item.speakers_raw, str) else ""
    if orig_spk:
        out_spk = item.speaker_dst or speaker_map.get(orig_spk, orig_spk)
    else:
        out_spk = ""

    out_text = item.post_dst if item.post_dst != "" else item.pre_src
    # Keep single-line pipe structure intact
    out_text = out_text.replace("\r\n", "\\n").replace("\n", "\\n")
    raw_id = item.raw_id or str(item.index)
    style = item.speaker_style or ("pipe" if orig_spk else "plain")

    if style == "bracket":
        return f"{raw_id}|[{out_spk}]{out_text}"
    if style == "pipe":
        return f"{raw_id}|{out_spk}|{out_text}"
    return f"{raw_id}|{out_text}"


def export_file(
    output_path: Path,
    items: Sequence[SentenceItem],
    format_meta: dict[str, Any],
    *,
    speaker_map: dict[str, str] | None = None,
    encoding: str = "utf-8",
    with_src: bool = False,
) -> Path:
    """Losslessly reconstruct compact-pipe file preserving comments, blank lines, and ID/speaker styles."""
    spk_map = speaker_map or {}
    by_idx = {it.index: it for it in items}
    layout = format_meta.get("layout")
    out_lines: list[str] = []

    if isinstance(layout, list) and layout:
        for slot in layout:
            if slot.get("kind") == "raw":
                out_lines.append(str(slot.get("line", "")))
            elif slot.get("kind") == "item":
                it = by_idx.get(int(slot["index"]))
                if it is None:
                    continue
                if with_src:
                    out_lines.append(f"# SRC({it.raw_id or it.index}): {it.pre_src}")
                out_lines.append(_format_compact_item(it, spk_map))
    else:
        for it in items:
            if with_src:
                out_lines.append(f"# SRC({it.raw_id or it.index}): {it.pre_src}")
            out_lines.append(_format_compact_item(it, spk_map))

    trailing_nl = bool(format_meta.get("trailing_newline", True))
    content = "\n".join(out_lines) + ("\n" if trailing_nl and out_lines else "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc = "gbk" if encoding.lower() in ("gbk", "gb2312", "cp936") else "utf-8"
    atomic_write_text(output_path, content, encoding=enc, errors="strict")
    return output_path
