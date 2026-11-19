"""Three-tier dictionary engine (pre_dict / gpt_dict / post_dict) & dynamic batch glossary injector."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import ProjectConfig, SentenceItem
from .text_preserve import PLACEHOLDER_RE, TextPreserver

_PLACEHOLDER_SPLIT_RE = re.compile(r"(\ue000[\ue010-\ue01f]+\ue001)")

_VALID_COND_TYPES: dict[str, str] = {
    "pre_jp": "pre_src",
    "post_jp": "post_src",
    "pre_src": "pre_src",
    "post_src": "post_src",
    "pre_dst": "pre_dst",
    "pre_zh": "pre_dst",
    "post_dst": "post_dst",
    "post_zh": "post_dst",
    "speaker": "speaker",
}

_SECTION_DIVIDER_RE = re.compile(r"^[-=]{3,}.*[-=]{3,}$")
_CONDITIONAL_NOTE_RE = re.compile(r"^(?:\[条件\]|条件[:：]|仅当|若指)")
_EXCLUDE_TAG_RE = re.compile(r"\[exclude[:：]([^\]]+)\]", re.IGNORECASE)

_ALLOWED_CJK_NAME_PREFIXES: tuple[str, ...] = ("御", "聖", "故")
# Kanji suffixes that validly follow a 1-Kanji character name (kana honorifics like さん/くん/ちゃん
# are naturally permitted because non-Kanji characters fall through to the okurigana check below).
_ALLOWED_CJK_NAME_SUFFIXES: tuple[str, ...] = (
    "様",
    "氏",
    "殿",
    "先生",
    "先輩",
    "後輩",
    "姉",
    "兄",
    "弟",
    "妹",
    "父",
    "母",
    "伯母",
    "叔母",
    "伯父",
    "叔父",
    "嬢",
    "姫",
    "王",
    "君",
    "達",
)

# Verbal/adjectival okurigana stems that mark lexical words rather than 1-Kanji character names.
# Carefully excludes noun particles/copula/quotatives (は/が/を/に/で/と/も/の/か/だ/って/すら/しか/かな).
_SINGLE_KANJI_NON_NAME_OKURIGANA: dict[str, tuple[str, ...]] = {
    "潤": (
        "う",
        "い",
        "った",
        "っている",
        "っていた",
        "ってる",
        "ってた",
        "ってきた",
        "ってくる",
        "ってしまう",
        "ってしまっ",
        "わない",
        "わず",
        "えば",
        "おう",
        "む",
        "み",
        "んだ",
        "んで",
        "まない",
        "めば",
        "す",
        "し",
        "した",
        "して",
        "さない",
        "さず",
        "せば",
    ),
    "奏": (
        "でる",
        "でて",
        "でた",
        "でれば",
        "でよう",
        "でます",
        "でまし",
        "でたい",
        "でられ",
        "でさせ",
        "でながら",
    ),
    "薫": ("る", "り", "った", "っている", "っていた", "ってる"),
    "光": ("る", "り", "った", "って", "らない", "らず", "れば", "ろう", "らせる", "らせて"),
    "舞": ("う", "い", "った", "わない", "わず", "えば", "おう", "っている", "っていた", "ってる"),
    "響": ("く", "き", "いた", "いて", "かない", "かず", "けば", "こう", "かせる", "かせて"),
    "望": ("む", "み", "んだ", "んで", "まない", "まず", "めば", "もう", "まれる", "まれ"),
    "願": ("う", "い", "った", "わない", "わず", "えば", "おう", "っている", "っていた", "ってる"),
    "遥": ("かな", "かに", "か彼", "か昔", "か遠", "か上", "か下", "か先", "か向", "か高"),
}


def _is_cjk_ideograph(ch: str) -> bool:
    return ("\u4e00" <= ch <= "\u9fff") or ch == "々"


def _parse_search_modifiers(raw_search: str) -> tuple[str, bool, bool]:
    """Parse GalTransl `^^` (start-of-line) and `1^` (replace-once) prefixes."""
    match_start = False
    replace_once = False
    s = raw_search
    while True:
        if s.startswith("^^"):
            match_start = True
            s = s[2:]
        elif s.startswith("1^"):
            replace_once = True
            s = s[2:]
        else:
            break
    return s, match_start, replace_once


def evaluate_condition(cond_expr: str, target_text: str) -> bool:
    """Evaluate a GalTransl conditional expression supporting [or], [and], and ! negation."""
    if not cond_expr:
        return True
    or_branches = re.split(r"\s*\[or\]\s*", cond_expr)
    for branch in or_branches:
        and_clauses = re.split(r"\s*\[and\]\s*", branch)
        branch_ok = True
        for clause in and_clauses:
            if clause == "":
                continue
            if clause.startswith("!"):
                needle = clause[1:]
                if needle and needle in target_text:
                    branch_ok = False
                    break
            else:
                if clause not in target_text:
                    branch_ok = False
                    break
        if branch_ok:
            return True
    return False


@dataclass
class NormalDictRule:
    """Single pre_dict or post_dict replacement rule."""

    search: str
    replace: str
    match_start: bool = False
    replace_once: bool = False
    cond_type: str = ""
    cond_expr: str = ""
    is_regex: bool = False
    case_sensitive: bool = True
    comment: str = ""
    source_priority: int = 1
    _compiled: re.Pattern[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.is_regex and self.search:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._compiled = re.compile(self.search, flags)


class NormalDict:
    """Pre-translation and post-translation dictionary with full GalTransl conditional syntax."""

    def __init__(self, rules: list[NormalDictRule] | None = None) -> None:
        self.rules: list[NormalDictRule] = list(rules or [])

    def load_file(
        self,
        path: str | Path,
        *,
        priority: int = 1,
        override_lower_priority: bool = False,
    ) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        raw_text = p.read_text(encoding="utf-8-sig")
        before_len = len(self.rules)
        if p.suffix.lower() == ".json":
            added = self._load_json(raw_text, priority=priority)
        else:
            added = self.load_tsv_text(raw_text, priority=priority)
        if override_lower_priority and added > 0:
            new_rules = self.rules[before_len:]
            uncond_searches = {r.search for r in new_rules if not r.cond_type and not r.is_regex}
            exact_keys = {(r.search, r.cond_type, r.cond_expr) for r in new_rules}
            filtered_old: list[NormalDictRule] = []
            for old in self.rules[:before_len]:
                if old.source_priority > priority:
                    if old.search in uncond_searches or (old.search, old.cond_type, old.cond_expr) in exact_keys:
                        continue
                filtered_old.append(old)
            self.rules = filtered_old + new_rules
        return added

    def _load_json(self, raw_text: str, *, priority: int = 1) -> int:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return 0
        added = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            raw_search = str(item.get("src", ""))
            if not raw_search:
                continue
            is_regex = bool(item.get("regex", False))
            if is_regex:
                search, match_start, replace_once = raw_search, False, False
            else:
                search, match_start, replace_once = _parse_search_modifiers(raw_search)
            self.rules.append(
                NormalDictRule(
                    search=search,
                    replace=str(item.get("dst", "")),
                    match_start=match_start,
                    replace_once=replace_once,
                    cond_type=_VALID_COND_TYPES.get(str(item.get("cond_type", "")), ""),
                    cond_expr=str(item.get("cond_expr", "")),
                    is_regex=is_regex,
                    case_sensitive=bool(item.get("case_sensitive", True)),
                    comment=str(item.get("info", "")),
                    source_priority=priority,
                )
            )
            added += 1
        return added

    def load_tsv_text(self, raw_text: str, *, priority: int = 1) -> int:
        """Parse TSV dictionary lines with comments, prefixes, conditions, and (同上)."""
        added = 0
        prev_cond_type = ""
        prev_cond_expr = ""

        for raw_line in raw_text.splitlines():
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue
            if _SECTION_DIVIDER_RE.match(stripped):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            col0 = parts[0].strip()
            if col0 in _VALID_COND_TYPES and len(parts) >= 4:
                cond_type = _VALID_COND_TYPES[col0]
                cond_expr = parts[1]
                if cond_expr.strip() in ("(同上)", "（同上）"):
                    cond_type = prev_cond_type or cond_type
                    cond_expr = prev_cond_expr
                else:
                    prev_cond_type = cond_type
                    prev_cond_expr = cond_expr

                raw_search = parts[2]
                replace = parts[3]
                comment = parts[4].strip() if len(parts) >= 5 else ""
                if replace.strip().startswith("//") and len(parts) == 4:
                    comment = replace.strip()
                    replace = ""
                if not raw_search:
                    continue
                search, match_start, replace_once = _parse_search_modifiers(raw_search)
                if not search:
                    continue
                self.rules.append(
                    NormalDictRule(
                        search=search,
                        replace=replace,
                        match_start=match_start,
                        replace_once=replace_once,
                        cond_type=cond_type,
                        cond_expr=cond_expr,
                        comment=comment,
                        source_priority=priority,
                    )
                )
                added += 1
            else:
                raw_search = parts[0]
                replace = parts[1]
                comment = parts[2].strip() if len(parts) >= 3 else ""
                if not raw_search:
                    continue
                search, match_start, replace_once = _parse_search_modifiers(raw_search)
                if not search:
                    continue
                self.rules.append(
                    NormalDictRule(
                        search=search,
                        replace=replace,
                        match_start=match_start,
                        replace_once=replace_once,
                        comment=comment,
                        source_priority=priority,
                    )
                )
                added += 1

        return added

    def sort_by_length(self) -> None:
        """Stable-sort rules by search length descending, with project priority ahead of builtin."""
        self.rules.sort(key=lambda r: (-len(r.search), r.source_priority))

    @staticmethod
    def _apply_rule_safe(working: str, rule: NormalDictRule, has_placeholders: bool) -> str:
        """Apply a single rule while never mutating PUA control/ruby placeholders."""
        if not has_placeholders:
            if rule.is_regex and rule._compiled is not None:
                return rule._compiled.sub(rule.replace, working, count=1 if rule.replace_once else 0)
            if rule.match_start:
                return (rule.replace + working[len(rule.search) :]) if working.startswith(rule.search) else working
            if rule.replace_once:
                return working.replace(rule.search, rule.replace, 1) if rule.search in working else working
            return working.replace(rule.search, rule.replace) if rule.search in working else working

        parts = _PLACEHOLDER_SPLIT_RE.split(working)
        replaced_any = False
        for i in range(0, len(parts), 2):
            seg = parts[i]
            if not seg:
                continue
            if rule.match_start:
                if i == 0 and seg.startswith(rule.search):
                    parts[0] = rule.replace + seg[len(rule.search) :]
                break
            if rule.is_regex and rule._compiled is not None:
                if rule.replace_once:
                    new_seg, cnt = rule._compiled.subn(rule.replace, seg, count=1)
                    if cnt > 0:
                        parts[i] = new_seg
                        break
                else:
                    parts[i] = rule._compiled.sub(rule.replace, seg)
            elif rule.replace_once:
                if rule.search in seg:
                    parts[i] = seg.replace(rule.search, rule.replace, 1)
                    replaced_any = True
                    break
            else:
                if rule.search in seg:
                    parts[i] = seg.replace(rule.search, rule.replace)
                    replaced_any = True
        _ = replaced_any
        return "".join(parts)

    def apply(
        self,
        text: str,
        item: SentenceItem | None = None,
        *,
        pre_src: str = "",
        post_src: str = "",
        pre_dst: str = "",
        post_dst: str = "",
        speaker: str = "",
        preserver: TextPreserver | None = None,
        external_token_map: dict[str, str] | None = None,
    ) -> str:
        """Apply all matching rules in order, shielding control and ruby spans."""
        if not text or not self.rules:
            return text

        ctx = {
            "pre_src": item.pre_src if item is not None else pre_src,
            "post_src": item.post_src if item is not None else post_src,
            "pre_dst": item.effective_core_dst if item is not None else pre_dst,
            "post_dst": post_dst if post_dst else (item.post_dst if item is not None else text),
            "speaker": item.speaker if item is not None else speaker,
        }

        local_token_map: dict[str, str] = {}
        working = text
        if preserver is not None:
            working, local_token_map = preserver.protect_text(working, include_ruby=True)

        active_map = local_token_map or (external_token_map or {})
        has_placeholders = bool(active_map) or bool(PLACEHOLDER_RE.search(working))

        for rule in self.rules:
            if rule.cond_type:
                target_field = ctx.get(rule.cond_type, "")
                if rule.cond_type == "post_dst":
                    target_field = working
                    for ph, orig in active_map.items():
                        target_field = target_field.replace(ph, orig)
                if not evaluate_condition(rule.cond_expr, target_field):
                    continue

            working = self._apply_rule_safe(working, rule, has_placeholders)

        if local_token_map:
            working, _ = TextPreserver.restore_text(working, local_token_map)
        return working


@dataclass
class GptDictEntry:
    """Single glossary entry for dynamic prompt injection and post-translation verification."""

    src: str
    dst: str
    info: str = ""
    case_sensitive: bool = False
    conditional: bool = False
    exclude_if: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.conditional and self.info and _CONDITIONAL_NOTE_RE.match(self.info.strip()):
            self.conditional = True
        if self.info:
            for m in _EXCLUDE_TAG_RE.finditer(self.info):
                for tok in re.split(r"[,，、/／;；]+", m.group(1)):
                    ex = tok.strip()
                    if ex and ex not in self.exclude_if:
                        self.exclude_if.append(ex)

    def matches_text(self, text: str) -> bool:
        if not text or not self.src:
            return False
        if self.case_sensitive:
            return self.src in text
        return self.src.lower() in text.lower()

    def dst_variants(self) -> list[str]:
        variants = [v.strip() for v in re.split(r"[/／]", self.dst) if v.strip()]
        if self.dst.strip() and self.dst.strip() not in variants:
            variants.insert(0, self.dst.strip())
        return variants

    def format_prompt_line(self) -> str:
        if self.info:
            return f"{self.src} -> {self.dst} #{self.info}"
        return f"{self.src} -> {self.dst}"


class GptDict:
    """Dynamic glossary supporting batch-local activation and usage QA verification."""

    def __init__(self, entries: list[GptDictEntry] | None = None) -> None:
        self.entries: list[GptDictEntry] = list(entries or [])
        self._index_by_src: dict[str, GptDictEntry] = {e.src: e for e in self.entries}

    def add_entry(self, entry: GptDictEntry, *, overwrite: bool = True) -> None:
        if not entry.src or not entry.dst:
            return
        if entry.src in self._index_by_src:
            if not overwrite:
                return
            self.entries = [e for e in self.entries if e.src != entry.src]
        self.entries.append(entry)
        self._index_by_src[entry.src] = entry

    def load_file(self, path: str | Path) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        raw_text = p.read_text(encoding="utf-8-sig")
        if p.suffix.lower() == ".json":
            return self._load_json(raw_text)
        return self.load_tsv_text(raw_text)

    def _load_json(self, raw_text: str) -> int:
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return 0
        added = 0
        for row in data:
            if not isinstance(row, dict):
                continue
            src = str(row.get("src", "")).strip()
            dst = str(row.get("dst", "")).strip()
            if not src or not dst:
                continue
            self.add_entry(
                GptDictEntry(
                    src=src,
                    dst=dst,
                    info=str(row.get("info", "")).strip(),
                    case_sensitive=bool(row.get("case_sensitive", False)),
                    conditional=bool(row.get("conditional", False)),
                    exclude_if=list(row.get("exclude_if", [])),
                ),
                overwrite=True,
            )
            added += 1
        return added

    def load_tsv_text(self, raw_text: str) -> int:
        added = 0
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if _SECTION_DIVIDER_RE.match(line):
                continue
            parts = raw_line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            src = parts[0].strip()
            dst = parts[1].strip()
            info = parts[2].strip() if len(parts) >= 3 else ""
            if info.startswith("//"):
                info = info[2:].strip()
            if not src or not dst:
                continue
            self.add_entry(GptDictEntry(src=src, dst=dst, info=info), overwrite=True)
            added += 1
        return added

    def sort_by_length(self) -> None:
        self.entries.sort(key=lambda e: len(e.src), reverse=True)

    def get(self, src: str) -> GptDictEntry | None:
        return self._index_by_src.get(src)

    def _is_standalone_single_cjk_at(self, src_text: str, pos: int, ch: str) -> bool:
        """Return False if 1-char CJK `ch` at `src_text[pos]` is inside a non-name compound/stem."""
        if pos > 0:
            prev_ch = src_text[pos - 1]
            if _is_cjk_ideograph(prev_ch):
                prefix_text = src_text[:pos]
                allowed_prev = prev_ch in _ALLOWED_CJK_NAME_PREFIXES or any(
                    other.src != ch and len(other.src) >= 1 and prefix_text.endswith(other.src)
                    for other in self.entries
                )
                if not allowed_prev:
                    return False

        if pos + 1 < len(src_text):
            after = src_text[pos + 1 :]
            next_ch = after[0]
            if _is_cjk_ideograph(next_ch):
                if not after.startswith(_ALLOWED_CJK_NAME_SUFFIXES):
                    return False
            else:
                bad_okuri = _SINGLE_KANJI_NON_NAME_OKURIGANA.get(ch, ())
                if bad_okuri and after.startswith(bad_okuri):
                    return False

        return True

    def _entry_hits_text(self, entry: GptDictEntry, orig_text: str, masked_text: str | None = None) -> tuple[bool, str]:
        """Check whether `entry` genuinely matches `orig_text` (respecting `exclude_if` & 1-CJK guard).

        Returns `(matched, updated_masked_text)`.
        """
        working_mask = orig_text if masked_text is None else masked_text
        if not orig_text or not entry.src:
            return False, working_mask

        # Context-level exclusions (respect case_sensitive and upstream working_mask)
        local_mask = working_mask
        for ex in entry.exclude_if:
            if not ex:
                continue
            contains_src = (entry.src in ex) if entry.case_sensitive else (entry.src.lower() in ex.lower())
            if contains_src:
                if entry.case_sensitive:
                    local_mask = local_mask.replace(ex, "\x00" * len(ex))
                else:
                    local_mask = re.compile(re.escape(ex), re.IGNORECASE).sub("\x00" * len(ex), local_mask)
            else:
                ex_found = (ex in working_mask) if entry.case_sensitive else (ex.lower() in working_mask.lower())
                if ex_found:
                    return False, working_mask

        is_single_cjk = len(entry.src) == 1 and _is_cjk_ideograph(entry.src)
        if is_single_cjk:
            ch = entry.src
            if ch not in local_mask:
                return False, working_mask
            valid_positions: list[int] = []
            start = 0
            while True:
                idx = local_mask.find(ch, start)
                if idx == -1:
                    break
                if self._is_standalone_single_cjk_at(orig_text, idx, ch):
                    valid_positions.append(idx)
                start = idx + 1
            if not valid_positions:
                return False, working_mask
            chars = list(working_mask)
            for idx in valid_positions:
                chars[idx] = "\x00"
            return True, "".join(chars)

        flags = 0 if entry.case_sensitive else re.IGNORECASE
        pat = re.compile(re.escape(entry.src), flags)
        matched_spans = [m.span() for m in pat.finditer(local_mask)]
        if not matched_spans:
            return False, working_mask

        # Mask ONLY positions matched in local_mask so excluded compounds remain intact for shorter terms
        chars = list(working_mask)
        for s_pos, e_pos in matched_spans:
            for k in range(s_pos, e_pos):
                chars[k] = "\x00"
        return True, "".join(chars)

    def match_batch(
        self,
        items: Iterable[SentenceItem],
        *,
        include_speaker: bool = True,
    ) -> list[GptDictEntry]:
        """Dynamically select only glossary terms literally hit in the current batch."""
        corpus_parts: list[str] = []
        for item in items:
            if item.post_src:
                corpus_parts.append(item.post_src)
            if item.pre_src and item.pre_src != item.post_src:
                corpus_parts.append(item.pre_src)
            if include_speaker and item.speaker:
                corpus_parts.append(item.speaker)
        if not corpus_parts:
            return []

        corpus = "\n".join(corpus_parts)
        corpus_lower = corpus.lower()

        matched: list[GptDictEntry] = []
        for entry in self.entries:
            if entry.case_sensitive:
                if entry.src not in corpus:
                    continue
            else:
                if entry.src.lower() not in corpus_lower:
                    continue
            is_single_cjk = len(entry.src) == 1 and _is_cjk_ideograph(entry.src)
            if is_single_cjk or entry.exclude_if:
                if any(self._entry_hits_text(entry, part)[0] for part in corpus_parts):
                    matched.append(entry)
            else:
                matched.append(entry)
        return matched

    @staticmethod
    def format_prompt_glossary(entries: Iterable[GptDictEntry]) -> str:
        lines = [e.format_prompt_line() for e in entries]
        return "\n".join(lines) if lines else "(none)"

    def check_usage(
        self,
        item: SentenceItem,
        dst_text: str,
        *,
        check_speaker: bool = False,
    ) -> list[str]:
        """Check whether unconditional glossary terms hit in post_src were used in dst_text."""
        src_text = item.post_src or item.pre_src
        if not src_text or not dst_text or not self.entries:
            return []

        ordered = sorted(self.entries, key=lambda e: len(e.src), reverse=True)
        masked_src = src_text
        dst_lower = dst_text.lower()
        misses: list[str] = []

        for entry in ordered:
            hit, next_mask = self._entry_hits_text(entry, src_text, masked_src)
            if not hit:
                continue
            if entry.conditional:
                # Conditional (unchecked) terms must NOT shadow shorter unconditional terms
                continue
            masked_src = next_mask

            variants = entry.dst_variants()
            used = any(
                (v in dst_text) if entry.case_sensitive else (v.lower() in dst_lower)
                for v in variants
            )
            if not used:
                misses.append(f"字典未使用：{entry.src}->{entry.dst}")

        if check_speaker and item.speaker and item.speaker_dst:
            spk_entry = self.get(item.speaker)
            if spk_entry and not spk_entry.conditional:
                variants = spk_entry.dst_variants()
                if not any(v in item.speaker_dst for v in variants):
                    misses.append(f"字典未使用(说话人)：{spk_entry.src}->{spk_entry.dst}")

        return misses


def inject_glossary_for_batch(
    items: Sequence[SentenceItem],
    gpt_dict: GptDict,
    *,
    include_speaker: bool = True,
) -> tuple[list[GptDictEntry], str]:
    """Blueprint helper: match batch-local glossary entries and return (entries, prompt_text)."""
    matched = gpt_dict.match_batch(items, include_speaker=include_speaker)
    return matched, GptDict.format_prompt_glossary(matched)


def load_project_dicts(
    config: ProjectConfig,
    project_dir: Path,
) -> tuple[NormalDict, GptDict, NormalDict]:
    """Load and optionally length-sort pre_dict, gpt_dict, and post_dict (project overrides builtin)."""
    pre_dict = NormalDict()
    for ref in config.pre_dicts:
        is_proj = not ref.startswith("builtin:")
        pre_dict.load_file(
            config.resolve_resource_path(ref, project_dir),
            priority=0 if is_proj else 1,
            override_lower_priority=is_proj,
        )
    if config.sort_dicts:
        pre_dict.sort_by_length()

    gpt_dict = GptDict()
    for ref in config.gpt_dicts:
        gpt_dict.load_file(config.resolve_resource_path(ref, project_dir))
    if config.sort_dicts:
        gpt_dict.sort_by_length()

    post_dict = NormalDict()
    for ref in config.post_dicts:
        is_proj = not ref.startswith("builtin:")
        post_dict.load_file(
            config.resolve_resource_path(ref, project_dir),
            priority=0 if is_proj else 1,
            override_lower_priority=is_proj,
        )
    if config.sort_dicts:
        post_dict.sort_by_length()

    return pre_dict, gpt_dict, post_dict
