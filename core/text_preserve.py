"""Control-code & structural text preservation engine with nested-token and ruby auditing."""

from __future__ import annotations

import json
import re
import secrets
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .models import ProjectConfig, SentenceItem, get_framework_root

_JS_UNICODE_SCRIPT_MAP: dict[str, str] = {
    r"\p{Script=Han}": r"\u4e00-\u9fff\u3400-\u4dbf",
    r"\p{Script=Hiragana}": r"\u3040-\u309f",
    r"\p{Script=Katakana}": r"\u30a0-\u30ff",
    r"\p{Script=Hangul}": r"\uac00-\ud7af\u1100-\u11ff\u3130-\u318f",
}

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_CJK_PROSE_PUNCT_RE = re.compile(r"[。！？、，；：]")

# Unicode permanent noncharacters (U+FDD0..U+FDEF, Unicode Standard §23.7) + per-process
# random 4-nibble key: immune to both post_dict text rules AND Japanese engine PUA Gaiji (U+E000..U+F8FF).
_NC_START = "\ufdd0"
_NC_END = "\ufdd1"
_NC_HEX = tuple(chr(0xFDE0 + i) for i in range(16))
_SENTINEL_KEY = "".join(_NC_HEX[int(ch, 16)] for ch in secrets.token_hex(2))
PLACEHOLDER_RE = re.compile(rf"\ufdd0{re.escape(_SENTINEL_KEY)}[\ufde0-\ufdef]+\ufdd1")
_ZERO_LEN_PROBES: tuple[str, ...] = ("", "a", "あ", "测试")


def _make_sentinel_placeholder(idx: int) -> str:
    """Encode a token index using U+FDD0..U+FDEF noncharacters gated by a per-run random key."""
    digits = "".join(_NC_HEX[int(ch, 16)] for ch in f"{idx:x}")
    return f"{_NC_START}{_SENTINEL_KEY}{digits}{_NC_END}"


def normalize_regex_pattern(pattern: str) -> str:
    """Adapt common JS Unicode script escapes so Python stdlib `re` compiles them safely."""
    out: list[str] = []
    in_class = False
    i = 0
    n = len(pattern)
    while i < n:
        matched_js = False
        if pattern[i] == "\\":
            for js_tok, py_range in _JS_UNICODE_SCRIPT_MAP.items():
                if pattern.startswith(js_tok, i):
                    out.append(py_range if in_class else f"[{py_range}]")
                    i += len(js_tok)
                    matched_js = True
                    break
            if matched_js:
                continue
            if i + 1 < n:
                out.append(pattern[i : i + 2])
                i += 2
                continue
        ch = pattern[i]
        if ch == "[" and not in_class:
            in_class = True
            out.append(ch)
            i += 1
            if i < n and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < n and pattern[i] == "]":
                out.append("]")
                i += 1
            continue
        if ch == "]" and in_class:
            in_class = False
        out.append(ch)
        i += 1
    return "".join(out)


@dataclass
class PreserveRule:
    """Single compiled text-preservation rule."""

    src: str
    info: str = ""
    is_ruby: bool = False
    case_sensitive: bool = False
    ignore_in_verify: bool = False
    compiled: re.Pattern[str] | None = None
    compile_error: str | None = None

    def __post_init__(self) -> None:
        normalized = normalize_regex_pattern(self.src)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        try:
            self.compiled = re.compile(normalized, flags)
        except re.error as exc:
            self.compiled = None
            self.compile_error = str(exc)


@dataclass
class PreserveSpan:
    """Matched control-code span inside a text string."""

    start: int
    end: int
    text: str
    is_ruby: bool = False
    info: str = ""
    rule_idx: int = -1


@dataclass
class PreserveVerifyResult:
    """Detailed control-token & ruby verification report."""

    ok: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    order_mismatch: bool = False
    src_sequence: list[str] = field(default_factory=list)
    dst_sequence: list[str] = field(default_factory=list)
    ruby_missing: list[str] = field(default_factory=list)
    ruby_extra: list[str] = field(default_factory=list)


@dataclass
class RuleAuditResult:
    """Safety validation report for a single text-preserve regex rule."""

    src: str
    info: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    match_count: int = 0
    zero_length_count: int = 0
    sample_matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "info": self.info,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "match_count": self.match_count,
            "zero_length_count": self.zero_length_count,
            "sample_matches": list(self.sample_matches),
        }


def validate_rule(
    pattern: str,
    info: str = "",
    sample_texts: Iterable[str] = (),
    *,
    is_ruby: bool = False,
    max_samples: int = 5,
) -> RuleAuditResult:
    """Audit a regex rule for syntax validity, zero-length matches, and body swallowing."""
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[str] = []
    match_count = 0
    zero_len = 0

    rule = PreserveRule(src=pattern, info=info, is_ruby=is_ruby)
    if rule.compiled is None:
        return RuleAuditResult(
            src=pattern,
            info=info,
            valid=False,
            errors=[f"regex compile error: {rule.compile_error}"],
        )

    probe_lines = ["", "これはテストの文章です。", "这是一段普通的中文测试文本。"]
    for probe in probe_lines:
        for m in rule.compiled.finditer(probe):
            if m.end() == m.start():
                zero_len += 1
                break

    body_swallow_samples: list[str] = []
    for text in sample_texts:
        for line in text.splitlines() or [text]:
            for m in rule.compiled.finditer(line):
                match_count += 1
                matched = m.group(0)
                if len(matched) == 0:
                    zero_len += 1
                    continue
                if len(samples) < max_samples and matched not in samples:
                    samples.append(matched)
                if not is_ruby:
                    cjk_hits = len(_CJK_CHAR_RE.findall(matched))
                    has_prose_punct = bool(_CJK_PROSE_PUNCT_RE.search(matched))
                    if cjk_hits >= 8 or (cjk_hits >= 4 and has_prose_punct):
                        if len(body_swallow_samples) < 3:
                            body_swallow_samples.append(matched[:40])

    if zero_len > 0:
        errors.append(f"zero-length match detected ({zero_len} occurrences)")
    if body_swallow_samples:
        errors.append(
            "possible prose body swallowing detected: "
            + ", ".join(repr(s) for s in body_swallow_samples)
        )

    return RuleAuditResult(
        src=pattern,
        info=info,
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        match_count=match_count,
        zero_length_count=zero_len,
        sample_matches=samples,
    )


def _ruby_envelope_kind(tag: str) -> str:
    """Classify the structural envelope of a ruby tag so tag kind parity can be checked in translate mode."""
    t = tag.strip()
    tl = t.lower()
    if tl.startswith("</"):
        return tl
    if tl.startswith("<r") and not tl.startswith(("<ruby", "<rt", "<rb")):
        return "<r...>"
    if tl.startswith("<ruby"):
        return "<ruby>"
    if tl.startswith("<rt"):
        return "<rt>"
    if tl.startswith("<rb"):
        return "<rb>"
    if tl.startswith("[ruby"):
        return "[ruby ...]"
    if tl.startswith(r"\r["):
        return r"\r[...]"
    return "ruby_tag"


class TextPreserver:
    """Manages control-code protection, stripping, placeholder wrapping, and full fidelity verification."""

    def __init__(self, rules: list[PreserveRule] | None = None, *, ruby_mode: str = "translate") -> None:
        self.rules: list[PreserveRule] = []
        self.rejected_rules: list[RuleAuditResult] = []
        self.ruby_mode = ruby_mode  # "translate" | "verbatim"
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: PreserveRule, *, enforce_safe: bool = True) -> bool:
        if rule.compiled is None:
            self.rejected_rules.append(
                RuleAuditResult(
                    src=rule.src,
                    info=rule.info,
                    valid=False,
                    errors=[f"regex compile error: {rule.compile_error}"],
                )
            )
            return False
        if enforce_safe:
            zero_hits = 0
            for probe in _ZERO_LEN_PROBES:
                for m in rule.compiled.finditer(probe):
                    if m.end() == m.start():
                        zero_hits += 1
                        break
            if zero_hits > 0:
                self.rejected_rules.append(
                    RuleAuditResult(
                        src=rule.src,
                        info=rule.info,
                        valid=False,
                        errors=[f"zero-length match detected ({zero_hits} probe hit(s))"],
                        zero_length_count=zero_hits,
                    )
                )
                return False
        self.rules.append(rule)
        return True

    @classmethod
    def from_project(cls, config: ProjectConfig, project_dir: Path | None = None) -> "TextPreserver":
        """Load configured built-in presets and custom project rules."""
        preserver = cls(ruby_mode=config.ruby_preserve_mode)
        presets_dir = get_framework_root() / "text_preserve_presets"
        seen_src: set[str] = set()

        def _load_entries(entries: list[dict[str, Any]]) -> None:
            for entry in entries:
                src = str(entry.get("src", ""))
                if not src or src in seen_src:
                    continue
                seen_src.add(src)
                preserver.add_rule(
                    PreserveRule(
                        src=src,
                        info=str(entry.get("info", "")),
                        is_ruby=bool(entry.get("is_ruby", False)),
                        case_sensitive=bool(entry.get("case_sensitive", False)),
                        ignore_in_verify=bool(entry.get("ignore_in_verify", False)),
                    )
                )

        for preset_name in config.preserve_presets:
            fname = preset_name if preset_name.endswith(".json") else f"{preset_name}.json"
            candidates = []
            if project_dir is not None:
                candidates.append(project_dir / "text_preserve_presets" / fname)
            candidates.append(presets_dir / fname)
            for path in candidates:
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                    if isinstance(data, list):
                        _load_entries(data)
                    break

        if config.custom_preserve_rules:
            _load_entries(config.custom_preserve_rules)

        return preserver

    def _raw_matches(
        self,
        text: str,
        *,
        include_ruby: bool = True,
        include_verify_ignored: bool = False,
    ) -> list[PreserveSpan]:
        if not text or not self.rules:
            return []
        raw_spans: list[PreserveSpan] = []
        for idx, rule in enumerate(self.rules):
            if rule.compiled is None:
                continue
            if rule.is_ruby and not include_ruby:
                continue
            if rule.ignore_in_verify and not include_verify_ignored:
                continue
            for m in rule.compiled.finditer(text):
                if m.end() <= m.start():
                    continue
                raw_spans.append(
                    PreserveSpan(
                        start=m.start(),
                        end=m.end(),
                        text=m.group(0),
                        is_ruby=rule.is_ruby,
                        info=rule.info,
                        rule_idx=idx,
                    )
                )
        return raw_spans

    def extract_spans(
        self,
        text: str,
        *,
        include_ruby: bool = True,
        include_verify_ignored: bool = False,
    ) -> list[PreserveSpan]:
        """Extract disjoint non-overlapping spans for safe text slicing (strip/protect)."""
        raw_spans = self._raw_matches(
            text,
            include_ruby=include_ruby,
            include_verify_ignored=include_verify_ignored,
        )
        if not raw_spans:
            return []
        raw_spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        merged: list[PreserveSpan] = []
        cursor = -1
        for sp in raw_spans:
            if sp.start >= cursor:
                merged.append(sp)
                cursor = sp.end
        return merged

    def extract_control_tokens(self, text: str) -> list[str]:
        """Extract verbatim (non-ruby) control tokens, preserving strictly nested inner tokens.

        Example: if an outer wrapper `[emb exp="f.x %s"]` or ruby tag `[ruby text="%s"]`
        encloses `%s`, both outer (if non-ruby) and strictly nested inner non-ruby tokens
        are registered so nested tokens can never be silently dropped.
        """
        raw = self._raw_matches(text, include_ruby=False, include_verify_ignored=False)
        if not raw:
            return []

        # Deduplicate exact same (start, end) matched by overlapping preset rules
        by_range: dict[tuple[int, int], PreserveSpan] = {}
        for sp in raw:
            key = (sp.start, sp.end)
            if key not in by_range:
                by_range[key] = sp

        unique = sorted(by_range.values(), key=lambda s: (s.start, -(s.end - s.start)))
        kept: list[PreserveSpan] = []
        for sp in unique:
            # Skip partial cross-overlaps; KEEP disjoint spans AND strictly nested spans!
            conflict = False
            for existing in kept:
                # Disjoint
                if sp.end <= existing.start or sp.start >= existing.end:
                    continue
                # Strictly nested inside `existing` -> keep inner token too!
                if existing.start <= sp.start and sp.end <= existing.end:
                    continue
                # Partial cross-overlap -> drop the shorter crossing match
                conflict = True
                break
            if not conflict:
                kept.append(sp)

        kept.sort(key=lambda s: (s.start, s.end))
        return [sp.text for sp in kept]

    def extract_ruby_spans(self, text: str) -> list[PreserveSpan]:
        """Extract ruby annotation spans only (disjoint)."""
        raw = [sp for sp in self._raw_matches(text, include_ruby=True) if sp.is_ruby]
        if not raw:
            return []
        raw.sort(key=lambda s: (s.start, -(s.end - s.start)))
        merged: list[PreserveSpan] = []
        cursor = -1
        for sp in raw:
            if sp.start >= cursor:
                merged.append(sp)
                cursor = sp.end
        return merged

    def strip_preserved(self, text: str, *, include_ruby: bool = False) -> str:
        """Remove preserved control spans from text for clean linguistic/length checks."""
        spans = self.extract_spans(text, include_ruby=include_ruby, include_verify_ignored=False)
        if not spans:
            return text
        parts: list[str] = []
        last = 0
        for sp in spans:
            if sp.start > last:
                parts.append(text[last : sp.start])
            last = sp.end
        if last < len(text):
            parts.append(text[last:])
        return "".join(parts)

    def protect_text(self, text: str, *, include_ruby: bool = True) -> tuple[str, dict[str, str]]:
        """Replace control spans (and ruby spans by default) with sentinel placeholders.

        Shielding ruby spans as well guarantees post_dict and punctuation rules cannot
        corrupt characters inside `[ruby text=...]` or `<r...>`.
        """
        spans = self.extract_spans(text, include_ruby=include_ruby, include_verify_ignored=False)
        if not spans:
            return text, {}
        token_map: dict[str, str] = {}
        parts: list[str] = []
        last = 0
        idx = 0
        for sp in spans:
            if sp.start > last:
                parts.append(text[last : sp.start])
            while True:
                placeholder = _make_sentinel_placeholder(idx)
                idx += 1
                if placeholder not in text and placeholder not in token_map:
                    break
            token_map[placeholder] = sp.text
            parts.append(placeholder)
            last = sp.end
        if last < len(text):
            parts.append(text[last:])
        return "".join(parts), token_map

    @staticmethod
    def restore_text(text: str, token_map: dict[str, str]) -> tuple[str, list[str]]:
        """Restore placeholders back to original control tokens; return missing originals."""
        if not token_map:
            return text, []
        out = text
        missing: list[str] = []
        for placeholder, original in token_map.items():
            if placeholder in out:
                out = out.replace(placeholder, original)
            else:
                missing.append(original)
        return out, missing

    def verify_preserved_detailed(
        self,
        src: str,
        dst: str,
        *,
        ruby_mode: str | None = None,
    ) -> PreserveVerifyResult:
        """Verify 100% fidelity of control tokens (missing, extra, order) and ruby tags."""
        mode = ruby_mode or self.ruby_mode
        src_tokens = self.extract_control_tokens(src)
        dst_tokens = self.extract_control_tokens(dst)

        src_counter = Counter(src_tokens)
        dst_counter = Counter(dst_tokens)

        missing: list[str] = []
        for tok, cnt in src_counter.items():
            diff = cnt - dst_counter.get(tok, 0)
            if diff > 0:
                missing.extend([tok] * diff)

        extra: list[str] = []
        for tok, cnt in dst_counter.items():
            diff = cnt - src_counter.get(tok, 0)
            if diff > 0:
                extra.extend([tok] * diff)

        order_mismatch = (not missing and not extra and src_tokens != dst_tokens)

        src_ruby = self.extract_ruby_spans(src)
        dst_ruby = self.extract_ruby_spans(dst)
        ruby_missing: list[str] = []
        ruby_extra: list[str] = []

        if mode == "verbatim":
            # Exact ruby content preservation mode
            sr_c = Counter(sp.text for sp in src_ruby)
            dr_c = Counter(sp.text for sp in dst_ruby)
            for tok, cnt in sr_c.items():
                diff = cnt - dr_c.get(tok, 0)
                if diff > 0:
                    ruby_missing.extend([tok] * diff)
            for tok, cnt in dr_c.items():
                diff = cnt - sr_c.get(tok, 0)
                if diff > 0:
                    ruby_extra.extend([tok] * diff)
        elif mode == "flexible":
            # Flexible ruby mode (e.g. M2 PSB `[・]` emphasis dots or per-char `[ほ]星[し]空` furigana):
            # allows natural dropping/merging of source ruby in Chinese while forbidding newly invented ruby when src had none.
            if not src_ruby and dst_ruby:
                ruby_extra.extend(_ruby_envelope_kind(sp.text) for sp in dst_ruby)
        else:
            # Default "translate" mode: ruby reading may be localized, but envelope kind & count must match
            sr_k = Counter(_ruby_envelope_kind(sp.text) for sp in src_ruby)
            dr_k = Counter(_ruby_envelope_kind(sp.text) for sp in dst_ruby)
            for kind, cnt in sr_k.items():
                diff = cnt - dr_k.get(kind, 0)
                if diff > 0:
                    ruby_missing.extend([kind] * diff)
            for kind, cnt in dr_k.items():
                diff = cnt - sr_k.get(kind, 0)
                if diff > 0:
                    ruby_extra.extend([kind] * diff)

        ok = not (missing or extra or order_mismatch or ruby_missing or ruby_extra)
        return PreserveVerifyResult(
            ok=ok,
            missing=missing,
            extra=extra,
            order_mismatch=order_mismatch,
            src_sequence=src_tokens,
            dst_sequence=dst_tokens,
            ruby_missing=ruby_missing,
            ruby_extra=ruby_extra,
        )

    def verify_preserved(
        self,
        src: str,
        dst: str,
        *,
        ruby_mode: str | None = None,
    ) -> tuple[bool, list[str], list[str]]:
        """Backward-compatible wrapper returning (ok, missing_all, extra_all)."""
        res = self.verify_preserved_detailed(src, dst, ruby_mode=ruby_mode)
        return res.ok, res.missing + res.ruby_missing, res.extra + res.ruby_extra

    def scan_samples(
        self,
        items: Iterable[SentenceItem],
        *,
        max_samples_per_rule: int = 5,
    ) -> dict[str, Any]:
        """Scan items to report active control-code hits and rule safety audits."""
        texts = [item.pre_src for item in items if item.pre_src]
        rule_reports: list[dict[str, Any]] = []
        unique_tokens: Counter[str] = Counter()

        for rule in self.rules:
            audit = validate_rule(
                rule.src,
                info=rule.info,
                sample_texts=texts,
                is_ruby=rule.is_ruby,
                max_samples=max_samples_per_rule,
            )
            if audit.match_count > 0 or not audit.valid:
                rule_reports.append(audit.to_dict())

        for text in texts:
            for tok in self.extract_control_tokens(text):
                unique_tokens[tok] += 1

        return {
            "scanned_items": len(texts),
            "active_rules": len(self.rules),
            "rejected_rules": [r.to_dict() for r in self.rejected_rules],
            "matched_rules": rule_reports,
            "top_tokens": [{"token": k, "count": v} for k, v in unique_tokens.most_common(30)],
        }
