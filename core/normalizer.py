"""Deterministic pre/post text normalizer: idempotent outer-shell strip/restore and unified control protection."""

from __future__ import annotations

import re
from typing import Sequence

from .dictionary import NormalDict
from .models import ProjectConfig, SentenceItem
from .text_preserve import PLACEHOLDER_RE, TextPreserver

_TRAILING_PLACEHOLDERS_RE = re.compile(r"(?:\ue000[\ue010-\ue01f]+\ue001)+$")

_QUOTE_PAIRS: dict[str, str] = {
    "「": "」",
    "『": "』",
}


def detect_linebreak_symbol(text: str, configured: str = "auto") -> str:
    """Detect the primary inline linebreak token present in text."""
    if configured and configured != "auto":
        return configured
    if "\\r\\n" in text:
        return "\\r\\n"
    if "\r\n" in text:
        return "\r\n"
    if "\\n" in text:
        return "\\n"
    if "\n" in text:
        return "\n"
    return ""


def _outer_pair_wraps_whole(text: str, open_q: str, close_q: str) -> bool:
    """Return True iff `open_q` at index 0 closes precisely at the last character of `text`."""
    if len(text) < 2 or text[0] != open_q or text[-1] != close_q:
        return False
    depth = 0
    last_idx = len(text) - 1
    for i, ch in enumerate(text):
        if ch == open_q:
            depth += 1
        elif ch == close_q:
            depth -= 1
            if depth == 0:
                return i == last_idx
            if depth < 0:
                return False
    return False


def _trace_quote_depth(text: str, open_q: str, close_q: str, initial_depth: int = 0) -> tuple[int, int]:
    """Trace nesting depth for a quote pair: (end_depth, min_depth_before_last_char)."""
    depth = initial_depth
    min_before_last = initial_depth if len(text) <= 1 else 10**9
    last_idx = len(text) - 1
    for i, ch in enumerate(text):
        if ch == open_q:
            depth += 1
        elif ch == close_q:
            depth -= 1
        if i < last_idx and depth < min_before_last:
            min_before_last = depth
    return depth, min_before_last


def _strip_outer_ws_single(item: SentenceItem, configured_lb: str = "auto") -> None:
    """Strip leading/trailing spaces, fullwidth spaces, and newlines into left_ws / right_ws."""
    item.post_src = item.pre_src
    item.left_ws = ""
    item.right_ws = ""
    item.left_quote = ""
    item.right_quote = ""
    item.has_diag_symbol = False
    item.is_dialogue = bool(item.speaker)

    s = item.post_src
    while s:
        if s.startswith("　"):
            s = s[1:]
            item.left_ws += "　"
        elif s.endswith("　"):
            s = s[:-1]
            item.right_ws = "　" + item.right_ws
        elif s.startswith(" "):
            s = s[1:]
            item.left_ws += " "
        elif s.endswith(" "):
            s = s[:-1]
            item.right_ws = " " + item.right_ws
        elif s.startswith("\\r\\n"):
            s = s[4:]
            item.left_ws += "\\r\\n"
        elif s.endswith("\\r\\n"):
            s = s[:-4]
            item.right_ws = "\\r\\n" + item.right_ws
        elif s.startswith("\r\n"):
            s = s[2:]
            item.left_ws += "\r\n"
        elif s.endswith("\r\n"):
            s = s[:-2]
            item.right_ws = "\r\n" + item.right_ws
        elif s.startswith("\\n"):
            s = s[2:]
            item.left_ws += "\\n"
        elif s.endswith("\\n"):
            s = s[:-2]
            item.right_ws = "\\n" + item.right_ws
        elif s.startswith("\n"):
            s = s[1:]
            item.left_ws += "\n"
        elif s.endswith("\n"):
            s = s[:-1]
            item.right_ws = "\n" + item.right_ws
        else:
            break
    item.post_src = s
    item.linebreak_symbol = detect_linebreak_symbol(item.post_src, configured_lb)


def strip_outer_shell(
    items: Sequence[SentenceItem],
    *,
    strip_outer_quotes: bool = True,
    configured_lb: str = "auto",
    force: bool = False,
) -> None:
    """Idempotently strip outer whitespace and dialogue quotes (single, nested, and 2-3 line spans).

    Items with `shell_stripped=True` are skipped unless `force=True`, ensuring accidental
    repeated calls never silently wipe `pre_dict` transformations on `post_src`.
    """
    active_indices = [i for i, it in enumerate(items) if force or not it.shell_stripped]
    if not active_indices:
        return
    active_set = set(active_indices)

    # Phase 1: strip outer whitespace/newlines on active items
    for idx in active_indices:
        _strip_outer_ws_single(items[idx], configured_lb=configured_lb)

    if strip_outer_quotes:
        # Phase 2: single-item matched outer quotes (supports nesting, guards against `「A」B「C」`)
        for idx in active_indices:
            item = items[idx]
            while item.post_src:
                first = item.post_src[0]
                close_q = _QUOTE_PAIRS.get(first)
                if close_q and _outer_pair_wraps_whole(item.post_src, first, close_q):
                    item.is_dialogue = True
                    item.has_diag_symbol = True
                    item.left_quote += first
                    item.right_quote = close_q + item.right_quote
                    item.post_src = item.post_src[1:-1]
                else:
                    break

        # Phase 3: multi-item dialogue spans (2 consecutive items and 3 consecutive items)
        n = len(items)
        for i in active_indices:
            cur = items[i]
            if not cur.post_src:
                continue
            open_q = cur.post_src[0]
            close_q = _QUOTE_PAIRS.get(open_q)
            if not close_q:
                continue
            d0, min0 = _trace_quote_depth(cur.post_src, open_q, close_q, initial_depth=0)
            if d0 != 1 or min0 <= 0:
                continue

            # Case 2: 2-line dialogue (`「...` + `...」`)
            if i + 1 < n and (i + 1) in active_set:
                nxt = items[i + 1]
                if nxt.post_src and nxt.post_src[0] != open_q and nxt.post_src[-1] == close_q:
                    d1, min1 = _trace_quote_depth(nxt.post_src, open_q, close_q, initial_depth=d0)
                    if d1 == 0 and min1 >= 1:
                        cur.is_dialogue = True
                        nxt.is_dialogue = True
                        cur.has_diag_symbol = True
                        nxt.has_diag_symbol = True
                        if not nxt.speaker and cur.speaker:
                            nxt.speaker = cur.speaker
                        cur.left_quote += open_q
                        nxt.right_quote = close_q + nxt.right_quote
                        cur.post_src = cur.post_src[1:]
                        nxt.post_src = nxt.post_src[:-1]
                        continue

            # Case 3: 3-line dialogue (`「...` + `...` + `...」`)
            if i + 2 < n and (i + 1) in active_set and (i + 2) in active_set:
                nxt = items[i + 1]
                nxt2 = items[i + 2]
                if (
                    nxt.post_src
                    and nxt2.post_src
                    and nxt.post_src[0] != open_q
                    and nxt.post_src[-1] != close_q
                    and nxt2.post_src[0] != open_q
                    and nxt2.post_src[-1] == close_q
                ):
                    d1, min1 = _trace_quote_depth(nxt.post_src, open_q, close_q, initial_depth=d0)
                    if d1 == 1 and min1 >= 1:
                        d2, min2 = _trace_quote_depth(nxt2.post_src, open_q, close_q, initial_depth=d1)
                        if d2 == 0 and min2 >= 1:
                            cur.is_dialogue = True
                            nxt.is_dialogue = True
                            nxt2.is_dialogue = True
                            cur.has_diag_symbol = True
                            nxt.has_diag_symbol = False
                            nxt2.has_diag_symbol = True
                            if cur.speaker:
                                if not nxt.speaker:
                                    nxt.speaker = cur.speaker
                                if not nxt2.speaker:
                                    nxt2.speaker = cur.speaker
                            cur.left_quote += open_q
                            nxt2.right_quote = close_q + nxt2.right_quote
                            cur.post_src = cur.post_src[1:]
                            nxt2.post_src = nxt2.post_src[:-1]

    for idx in active_indices:
        items[idx].shell_stripped = True
        items[idx].linebreak_symbol = detect_linebreak_symbol(items[idx].post_src, configured_lb)


def preprocess_items(
    items: Sequence[SentenceItem],
    config: ProjectConfig,
    pre_dict: NormalDict | None = None,
    *,
    force: bool = True,
) -> None:
    """Run outer-shell stripping followed by pre_dict replacement cleanly from immutable pre_src."""
    strip_outer_shell(
        items,
        strip_outer_quotes=config.strip_outer_quotes,
        configured_lb=config.linebreak_symbol,
        force=force,
    )
    if pre_dict is not None and pre_dict.rules:
        for item in items:
            if item.post_src:
                item.post_src = pre_dict.apply(item.post_src, item)
                item.linebreak_symbol = detect_linebreak_symbol(item.post_src, config.linebreak_symbol)
            if config.use_pre_dict_in_name and item.speaker:
                item.speaker = pre_dict.apply(item.speaker, item)


def extract_inner_dst(item: SentenceItem, post_dst: str | None = None) -> str:
    """Strip the known outer shell (left_ws/left_quote/right_quote/right_ws) from post_dst."""
    inner = item.post_dst if post_dst is None else post_dst
    if item.left_ws and inner.startswith(item.left_ws):
        inner = inner[len(item.left_ws) :]
    if item.left_quote and inner.startswith(item.left_quote):
        inner = inner[len(item.left_quote) :]
    if item.right_ws and inner.endswith(item.right_ws):
        inner = inner[: -len(item.right_ws)]
    if item.right_quote and inner.endswith(item.right_quote):
        inner = inner[: -len(item.right_quote)]
    return inner


def _strip_accidental_outer_quotes(dst_core: str, item: SentenceItem) -> str:
    """Guard against double-wrapping if the agent accidentally included stripped outer quotes."""
    if not dst_core:
        return dst_core
    s = dst_core
    if item.left_quote and item.right_quote:
        if (
            len(s) > len(item.left_quote) + len(item.right_quote)
            and s.startswith(item.left_quote)
            and s.endswith(item.right_quote)
        ):
            if len(item.left_quote) == 1 and len(item.right_quote) == 1:
                if _outer_pair_wraps_whole(s, item.left_quote, item.right_quote):
                    s = s[1:-1]
            else:
                s = s[len(item.left_quote) : -len(item.right_quote)]
    elif item.left_quote and not item.right_quote:
        if s.startswith(item.left_quote):
            close_q = _QUOTE_PAIRS.get(item.left_quote, "")
            if not close_q or close_q not in s:
                s = s[len(item.left_quote) :]
    elif item.right_quote and not item.left_quote:
        if s.endswith(item.right_quote):
            open_q = next((k for k, v in _QUOTE_PAIRS.items() if v == item.right_quote), "")
            if not open_q or open_q not in s:
                s = s[: -len(item.right_quote)]
    return s


def _remove_first_symbol(dst: str, post_src: str) -> str:
    """Remove spurious leading comma, period, or unintended leading line break."""
    if not dst:
        return dst
    out = dst
    if out[:1] in ("，", "。") and post_src[:1] not in ("、", "，", "。"):
        out = out[1:]
    for nl in ("\\r\\n", "\r\n", "\\n", "\n"):
        if out.startswith(nl) and not post_src.startswith(nl):
            out = out[len(nl) :]
            break
    return out


def _fix_last_symbol(
    dst: str,
    post_src: str,
    *,
    align_trailing_period: bool = True,
    protected_post_src: str | None = None,
) -> str:
    """Deterministically align trailing punctuation against post_src (even before trailing control codes)."""
    if not dst:
        return dst

    # Temporarily detach trailing control placeholders so e.g. `早上好。[l]` aligns cleanly against `おはよう[l]`
    tail = ""
    out = dst
    m_dst = _TRAILING_PLACEHOLDERS_RE.search(out)
    if m_dst:
        tail = m_dst.group(0)
        out = out[: m_dst.start()]
        if not out:
            return dst

    src_ref = protected_post_src if protected_post_src is not None else post_src
    m_src = _TRAILING_PLACEHOLDERS_RE.search(src_ref)
    if m_src:
        src_ref = src_ref[: m_src.start()]

    for nl in ("\r\n", "\n", "\\r\\n", "\\n"):
        if out.endswith(nl) and not src_ref.endswith(nl):
            out = out[: -len(nl)]

    if src_ref[-1:] not in ("、", "，") and out.endswith("，"):
        out = out[:-1]

    if align_trailing_period and out.endswith("。") and src_ref[-1:] not in (".", "。"):
        out = out[:-1]

    if src_ref.endswith("♪"):
        if out.endswith("。♪") and not src_ref.endswith("。♪"):
            out = out[:-2] + "♪"
        elif not out.endswith("♪"):
            out += "♪"

    if src_ref.endswith("！？"):
        if out.endswith("！"):
            out += "？"
        elif out.endswith("!"):
            out = out[:-1] + "！？"

    return out + tail


def _fix_inner_punctuation_and_quotes(dst: str, post_src: str) -> str:
    """Fix malformed ellipsis-period combos, doubled periods, and single-pair same-direction quotes."""
    if not dst:
        return dst
    out = dst

    if "…。" not in post_src:
        while "…。" in out:
            out = out.replace("…。", "…")
    if "。。" not in post_src:
        while "。。" in out:
            out = out.replace("。。", "。")
    if "。）" not in post_src:
        out = out.replace("。）", "）")

    if out.count("”") == 2 and out.count("“") == 0:
        out = out.replace("”", "“", 1)
    elif out.count("“") == 2 and out.count("”") == 0:
        idx = out.rfind("“")
        out = out[:idx] + "”" + out[idx + 1 :]

    if out.count("』") == 2 and out.count("『") == 0:
        out = out.replace("』", "『", 1)
    elif out.count("『") == 2 and out.count("』") == 0:
        idx = out.rfind("『")
        out = out[:idx] + "』" + out[idx + 1 :]

    if out.count("」") == 2 and out.count("「") == 0:
        out = out.replace("」", "「", 1)
    elif out.count("「") == 2 and out.count("」") == 0:
        idx = out.rfind("「")
        out = out[:idx] + "」" + out[idx + 1 :]

    return out


def normalize_and_restore_dst(
    item: SentenceItem,
    config: ProjectConfig,
    post_dict: NormalDict | None = None,
    preserver: TextPreserver | None = None,
) -> str:
    """Run unified control-protected post-normalization, post_dict, and outer-shell restoration.

    Protection order (prevents post_dict or punctuation fixes from ever mutating control or ruby tags):
      protect_text(include_ruby=True) -> _remove_first_symbol -> post_dict.apply ->
      _fix_inner_punctuation_and_quotes -> _fix_last_symbol -> restore_text -> linebreak policy -> shell restore.
    """
    raw_core = item.effective_core_dst
    if raw_core == "":
        item.post_dst = ""
        return ""

    post_src = item.post_src
    dst = _strip_accidental_outer_quotes(raw_core, item)

    # Align line break encoding style against post_src before shielding
    if "\r\n" in post_src and "\r\n" not in dst and "\n" in dst:
        dst = dst.replace("\n", "\r\n")
    elif "\\n" in post_src and "\\n" not in dst and "\n" in dst:
        dst = dst.replace("\n", "\\n")

    # Single unbroken shield over control tokens AND ruby tags across post_dict + punctuation pipeline
    token_map: dict[str, str] = {}
    protected_post_src: str | None = None
    if preserver is not None:
        dst, token_map = preserver.protect_text(dst, include_ruby=True)
        protected_post_src, _ = preserver.protect_text(post_src, include_ruby=True)

    dst = _remove_first_symbol(dst, post_src)
    dst = _fix_last_symbol(
        dst,
        post_src,
        align_trailing_period=config.align_trailing_period,
        protected_post_src=protected_post_src,
    )

    if post_dict is not None and post_dict.rules:
        dst = post_dict.apply(
            dst,
            item,
            post_dst=dst,
            preserver=None,
            external_token_map=token_map,
        )

    dst = _fix_inner_punctuation_and_quotes(dst, post_src)
    dst = _fix_last_symbol(
        dst,
        post_src,
        align_trailing_period=config.align_trailing_period,
        protected_post_src=protected_post_src,
    )

    if token_map:
        dst, _ = TextPreserver.restore_text(dst, token_map)

    if config.use_post_dict_in_name and post_dict is not None and post_dict.rules and item.speaker_dst:
        item.speaker_dst = post_dict.apply(item.speaker_dst, item, preserver=preserver)

    # Inline linebreak count alignment using post_src as the single authoritative baseline
    if config.extra_linebreak_policy == "trim":
        lb = detect_linebreak_symbol(post_src, config.linebreak_symbol)
        if lb:
            expected = post_src.count(lb)
            while dst.count(lb) > expected:
                dst = dst.replace(lb, "", 1)
        else:
            for extra_lb in ("\\r\\n", "\r\n", "\\n", "\n"):
                if extra_lb in dst and extra_lb not in post_src:
                    dst = dst.replace(extra_lb, "")

    item.post_dst = f"{item.left_ws}{item.left_quote}{dst}{item.right_quote}{item.right_ws}"
    return item.post_dst
