"""Automated QA & problem detection engine (16 core dimensions / 22 granular checks)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

from .dictionary import GptDict
from .models import ProjectConfig, SentenceItem
from .normalizer import detect_linebreak_symbol, extract_inner_dst
from .text_preserve import TextPreserver

# Complete, authoritative registry of every check label governed by config.enabled_problems
ALL_PROBLEM_TYPES: tuple[str, ...] = (
    "缺控制符",
    "多控制符",
    "控制符错序",
    "丢失换行",
    "多加换行",
    "残留日文",
    "附着记号残留",
    "本无韩文",
    "本无括号",
    "本有括号",
    "本无引号",
    "本有引号",
    "引入英文",
    "字典未使用",
    "比日文长",
    "单句过长",
    "词频过高",
    "独白男他",
    "非GBK字符",
    "无依据器官词强补",
    "语域违和词",
    "翻译失败",
)

HARD_ERROR_TYPES: tuple[str, ...] = (
    "缺控制符",
    "多控制符",
    "控制符错序",
    "丢失换行",
    "多加换行",
    "残留日文",
    "原文输出",
    "附着记号残留",
    "本无韩文",
    "本无括号",
    "引入英文",
    "非GBK字符",
    "无依据器官词强补",
    "语域违和词",
    "翻译失败",
)

SOFT_WARN_TYPES: tuple[str, ...] = (
    "单句过长",
    "比日文长",
    "字典未使用",
    "本有括号",
    "本无引号",
    "本有引号",
    "独白男他",
    "词频过高",
)

_HARD_ERROR_SET: frozenset[str] = frozenset(HARD_ERROR_TYPES)
_SOFT_WARN_SET: frozenset[str] = frozenset(SOFT_WARN_TYPES)


def problem_severity(prob_str: str) -> str:
    """Classify a QA problem string into `'error'` (hard defect) or `'warn'` (advisory hint)."""
    s = (prob_str or "").strip()
    if not s:
        return "warn"
    head = s.split("：", 1)[0].split(":", 1)[0].strip()
    if head in _SOFT_WARN_SET or head.startswith("字典未使用"):
        return "warn"
    if head in _HARD_ERROR_SET:
        return "error"
    if head.startswith(("本无", "本有")):
        return "error"
    return "error"


def split_problems_by_severity(problems: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split problem strings into `(hard_errors, soft_warnings)`."""
    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    for p in problems:
        if problem_severity(p) == "error":
            hard_errors.append(p)
        else:
            soft_warnings.append(p)
    return hard_errors, soft_warnings

_ATTACHED_MARK_RE = re.compile(r"[\u3099-\u309Cっッー・]")
_KANA_CORE_RE = re.compile(
    r"[\u3040-\u3062\u3064-\u3098\u309D-\u309F\u30A0-\u30C2\u30C4-\u30FA\u30FD-\u30FF\uFF66-\uFF9F]"
)
_HANGUL_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]+")
_CLAUSE_SPLIT_RE = re.compile(r"(?:\\r\\n|\r\n|\\n|\n|……|——|[，、。！？；!?;])+")
_QUOTE_CHARS: tuple[str, ...] = ("「", "」", "『", "』", "“", "”", "‘", "’")
_BRACKET_CHARS: tuple[str, ...] = ("（", "）", "(", ")")

# Lexical compounds containing '他' that are NOT the third-person male pronoun
MONOLOGUE_MALE_HE_EXCLUDES: tuple[str, ...] = (
    "电吉他",
    "木吉他",
    "吉他",
    "维他命",
    "马耳他",
    "犹他",
    "利他",
    "排他",
    "其他",
    "他们",
    "他人",
    "他乡",
    "他国",
    "他日",
    "他山",
    "他处",
    "他年",
    "他俩",
    "他妈",
    "他娘",
    "无他",
    "别无他法",
    "他者",
    "他界",
    "他杀",
    "他校",
    "他班",
    "他院",
    "他队",
)

DEFAULT_SRC_EXPLICIT_ORGAN_TERMS: tuple[str, ...] = (
    "ちんぽ", "チンポ", "おちんぽ", "オチンポ", "ちんちん", "チンチン", "おちんちん", "ちんこ", "チンコ",
    "ペニス", "陰茎", "男根", "肉棒", "亀頭", "睾丸", "金玉", "キンタマ",
    "まんこ", "マンコ", "おまんこ", "オマンコ", "膣", "陰唇", "陰核", "クリトリス", "秘裂", "秘部", "肉壺",
    "アナル", "アヌス", "肛門", "菊穴", "直腸", "子宮", "子宮口", "尿道",
    "乳首", "乳房", "おっぱい", "オッパイ", "精液", "ザーメン", "性器", "局部", "股間", "あそこ", "アソコ",
)

DEFAULT_DST_EXPLICIT_ORGAN_TERMS: tuple[str, ...] = (
    "阴茎", "阴道", "阴唇", "阴蒂", "阳具", "男根", "肉棒", "生殖器", "外阴",
    "肛门", "直肠", "子宫", "宫颈", "龟头", "睾丸", "小穴", "蜜穴", "肉穴",
    "花穴", "菊穴", "乳头", "乳房", "阴户", "耻骨", "盆腔", "会阴",
)


def is_whitelisted(item: SentenceItem, whitelist: list[str]) -> bool:
    """Check whether `file:index` or range `file:start-end` exempts an item."""
    if item.skip_check:
        return True
    if not whitelist:
        return False
    for entry in whitelist:
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        f_part, idx_part = entry.rsplit(":", 1)
        if f_part != item.file_name and f_part != "*":
            continue
        if "-" in idx_part:
            lo_s, hi_s = idx_part.split("-", 1)
            try:
                if int(lo_s) <= item.index <= int(hi_s):
                    return True
            except ValueError:
                continue
        else:
            if idx_part == str(item.index) or (item.raw_id and idx_part == item.raw_id):
                return True
    return False


def _most_common_char(text: str) -> tuple[str, int]:
    ignore = {".", "，", "…", "—", " ", "　", "\n", "\r", "\t"}
    counts = Counter(ch for ch in text if ch not in ignore)
    if not counts:
        return "", 0
    return counts.most_common(1)[0]


def _non_gbk_chars(text: str, whitelist: list[str]) -> str:
    allowed = set(whitelist)
    bad: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch in allowed or ch in seen:
            continue
        try:
            ch.encode("gbk")
        except UnicodeEncodeError:
            seen.add(ch)
            bad.append(ch)
    return "".join(bad)


def _strip_monologue_excludes(text: str) -> str:
    cleaned = text
    for word in MONOLOGUE_MALE_HE_EXCLUDES:
        cleaned = cleaned.replace(word, "")
    return cleaned


def _normalize_fullwidth_ascii(word: str) -> str:
    out: list[str] = []
    for ch in word:
        cp = ord(ch)
        if 0xFF21 <= cp <= 0xFF3A:
            out.append(chr(cp - 0xFF21 + ord("A")))
        elif 0xFF41 <= cp <= 0xFF5A:
            out.append(chr(cp - 0xFF41 + ord("a")))
        else:
            out.append(ch)
    return "".join(out)


def check_item_problems(
    item: SentenceItem,
    config: ProjectConfig,
    profile: dict[str, Any] | None = None,
    preserver: TextPreserver | None = None,
    gpt_dict: GptDict | None = None,
) -> list[str]:
    """Run all enabled QA checks on an item, updating `item.problems` and returning the list."""
    if is_whitelisted(item, config.problem_whitelist):
        item.problems = []
        return []

    dst_core = item.effective_core_dst
    post_dst = item.post_dst or dst_core
    if not dst_core and not post_dst:
        item.problems = []
        return []

    pre_src = item.pre_src
    post_src = item.post_src
    inner_post_dst = extract_inner_dst(item, post_dst)
    enabled = set(config.enabled_problems) if config.enabled_problems is not None else set(ALL_PROBLEM_TYPES)
    prof = profile or {}
    found: list[str] = []

    if "翻译失败" in enabled and "(Failed)" in post_dst:
        found.append("翻译失败")

    if preserver is not None:
        clean_pre_src = preserver.strip_preserved(pre_src, include_ruby=True)
        clean_post_src = preserver.strip_preserved(post_src, include_ruby=True)
        clean_post_dst = preserver.strip_preserved(post_dst, include_ruby=True)
        clean_dst_core = preserver.strip_preserved(dst_core, include_ruby=True)
        clean_inner_post_dst = preserver.strip_preserved(inner_post_dst, include_ruby=True)
    else:
        clean_pre_src = pre_src
        clean_post_src = post_src
        clean_post_dst = post_dst
        clean_dst_core = dst_core
        clean_inner_post_dst = inner_post_dst

    # 1. 控制符完整性：缺控制符 + 多控制符 + 控制符错序（含嵌套 token 与 ruby 模式）
    if preserver is not None and ({"缺控制符", "多控制符", "控制符错序"} & enabled):
        vres = preserver.verify_preserved_detailed(
            post_src,
            inner_post_dst,
            ruby_mode=config.ruby_preserve_mode,
        )
        missing_all = vres.missing + vres.ruby_missing
        extra_all = vres.extra + vres.ruby_extra
        if "缺控制符" in enabled and missing_all:
            found.append(f"缺控制符：{' '.join(missing_all)}")
        if "多控制符" in enabled and extra_all:
            found.append(f"多控制符：{' '.join(extra_all)}")
        if "控制符错序" in enabled and vres.order_mismatch:
            found.append(
                f"控制符错序：期望[{' '.join(vres.src_sequence)}]实际[{' '.join(vres.dst_sequence)}]"
            )

    # 2. 丢失换行 / 多加换行 — 统一以 post_src vs inner_post_dst 为唯一基准，与 normalizer 严格一致
    lb = detect_linebreak_symbol(post_src, config.linebreak_symbol)
    if lb:
        src_lb_cnt = post_src.count(lb)
        dst_lb_cnt = inner_post_dst.count(lb)
        if "丢失换行" in enabled and src_lb_cnt > dst_lb_cnt:
            found.append(f"丢失换行：原{src_lb_cnt}译{dst_lb_cnt}")
        if "多加换行" in enabled and src_lb_cnt < dst_lb_cnt:
            found.append(f"多加换行：原{src_lb_cnt}译{dst_lb_cnt}")
    else:
        if "多加换行" in enabled:
            for extra_lb in ("\\r\\n", "\r\n", "\\n", "\n"):
                if extra_lb in inner_post_dst and extra_lb not in post_src:
                    found.append(f"多加换行：原0译{inner_post_dst.count(extra_lb)}")
                    break

    # 3. 残留日文 / 原文输出 & 4. 附着记号残留
    attached_hits = "".join(dict.fromkeys(_ATTACHED_MARK_RE.findall(clean_post_dst)))
    kana_hits = "".join(dict.fromkeys(_KANA_CORE_RE.findall(clean_post_dst)))
    if "残留日文" in enabled:
        if pre_src and pre_src == post_dst and (_KANA_CORE_RE.search(pre_src) or _ATTACHED_MARK_RE.search(pre_src)):
            found.append("原文输出")
        elif kana_hits:
            found.append(f"残留日文：{kana_hits}")
    if "附着记号残留" in enabled and attached_hits and "原文输出" not in found:
        found.append(f"附着记号残留：{attached_hits}")

    # 5. 本无韩文（已纳入 ALL_PROBLEM_TYPES，受 enabled_problems 统一控制）
    if "本无韩文" in enabled:
        if _HANGUL_RE.search(clean_post_dst) and not _HANGUL_RE.search(clean_pre_src):
            found.append("本无韩文")

    # 6. 本无括号 / 本有括号
    src_has_bracket = any(ch in clean_post_src for ch in _BRACKET_CHARS)
    dst_has_bracket = any(ch in clean_inner_post_dst for ch in _BRACKET_CHARS)
    if "本无括号" in enabled and not src_has_bracket and dst_has_bracket:
        found.append("本无括号")
    if "本有括号" in enabled and src_has_bracket and not dst_has_bracket:
        found.append("本有括号")

    # 7. 本无引号 / 本有引号（针对剥壳后的内层文本）
    src_has_quote = any(ch in clean_post_src for ch in _QUOTE_CHARS)
    dst_has_quote = any(ch in clean_inner_post_dst for ch in _QUOTE_CHARS)
    if "本无引号" in enabled and not src_has_quote and dst_has_quote:
        found.append("本无引号")
    if "本有引号" in enabled and src_has_quote and not dst_has_quote:
        found.append("本有引号")

    # 8. 引入英文
    if "引入英文" in enabled:
        src_latin = [_normalize_fullwidth_ascii(w) for w in _LATIN_WORD_RE.findall(clean_post_src)]
        dst_latin = [_normalize_fullwidth_ascii(w) for w in _LATIN_WORD_RE.findall(clean_inner_post_dst)]
        if not src_latin and dst_latin:
            wl_lower = {w.lower() for w in config.english_whitelist}
            if gpt_dict is not None:
                for entry in gpt_dict.entries:
                    if entry.matches_text(post_src):
                        for w in _LATIN_WORD_RE.findall(entry.dst):
                            wl_lower.add(_normalize_fullwidth_ascii(w).lower())
            leaked = [w for w in dst_latin if w.lower() not in wl_lower]
            if leaked:
                found.append(f"引入英文：{' '.join(dict.fromkeys(leaked))}")

    # 9. 字典未使用
    if "字典未使用" in enabled and gpt_dict is not None:
        misses = gpt_dict.check_usage(
            item,
            post_dst,
            check_speaker=config.use_gpt_dict_in_name,
        )
        found.extend(misses)

    # 10. 比日文长
    if "比日文长" in enabled:
        src_len = len(clean_post_src.strip())
        dst_len = len(clean_inner_post_dst.strip())
        if src_len > 0:
            ratio = dst_len / src_len
            diff = dst_len - src_len
            if config.strict_length_check and dst_len > src_len:
                found.append(f"比日文长：{ratio:.1f}倍(+{diff}字)")
            elif ratio > config.length_ratio_threshold and diff >= config.length_diff_threshold:
                found.append(f"比日文长：{ratio:.1f}倍(+{diff}字)")

    # 11. 单句过长
    if "单句过长" in enabled:
        threshold = int(prof.get("avg_sentence_length_threshold", config.avg_sentence_length_threshold))
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(clean_inner_post_dst) if c.strip()]
        if clauses:
            avg_len = sum(len(c) for c in clauses) / len(clauses)
            if avg_len > threshold:
                found.append(f"单句过长：平均分句{avg_len:.1f}字(>{threshold})")

    # 12. 词频过高
    if "词频过高" in enabled:
        dst_ch, dst_cnt = _most_common_char(clean_dst_core)
        _, src_cnt = _most_common_char(clean_post_src)
        if dst_cnt > 20 and dst_cnt > src_cnt * 2:
            found.append(f"词频过高：'{dst_ch}'{dst_cnt}次")

    # 13. 独白男他（含「吉他/维他命/利他/排他」等防误报排除表）
    if "独白男他" in enabled:
        if not item.speaker and not item.is_dialogue and "他" in clean_post_dst:
            if "他" in _strip_monologue_excludes(clean_post_dst):
                found.append("独白男他")

    # 14. 非GBK字符
    if "非GBK字符" in enabled and config.target_encoding.lower() in ("gbk", "gb2312", "cp936"):
        bad_gbk = _non_gbk_chars(post_dst, config.gbk_whitelist)
        if bad_gbk:
            found.append(f"非GBK字符：{bad_gbk}")

    # 15. 无依据器官词强补
    organ_enabled = config.enable_organ_injection_check and bool(
        prof.get("enable_organ_injection_check", True)
    )
    if "无依据器官词强补" in enabled and organ_enabled:
        src_terms = prof.get("src_explicit_organ_terms") or DEFAULT_SRC_EXPLICIT_ORGAN_TERMS
        dst_terms = prof.get("dst_explicit_organ_terms") or DEFAULT_DST_EXPLICIT_ORGAN_TERMS
        src_combined = f"{pre_src}\n{post_src}"
        if not any(term in src_combined for term in src_terms):
            injected = [term for term in dst_terms if term in clean_post_dst]
            if injected:
                found.append(f"无依据器官词强补：{'、'.join(dict.fromkeys(injected))}")

    # 16. 语域违和词
    register_enabled = config.enable_register_check and bool(prof.get("enable_register_check", True))
    if "语域违和词" in enabled and register_enabled:
        reg_hits: list[str] = []
        for word in prof.get("forbidden_words", []):
            if word and word in clean_post_dst:
                reg_hits.append(word)
        if item.is_dialogue:
            for word in prof.get("dialogue_forbidden_words", []):
                if word and word in clean_post_dst and word not in reg_hits:
                    reg_hits.append(word)
            high_markers = prof.get("high_register_markers", [])
            max_cnt = int(prof.get("high_register_max_count", 2))
            if high_markers and max_cnt > 0:
                hit_markers = [m for m in high_markers if m and m in clean_inner_post_dst]
                if len(hit_markers) >= max_cnt:
                    reg_hits.append(f"高语域过密({'/'.join(hit_markers[:4])})")
        if reg_hits:
            found.append(f"语域违和词：{'、'.join(dict.fromkeys(reg_hits))}")

    # Custom arinashi presence/absence rules
    for key, val in config.arinashi_dict.items():
        if key not in pre_src and val in post_dst:
            found.append(f"本无{key}译有{val}")
        elif key in pre_src and val not in post_dst:
            found.append(f"本有{key}译无{val}")

    # Fine-grained project-level regex filter
    if config.problem_filter_regex and found:
        compiled_filters = []
        for pat in config.problem_filter_regex:
            try:
                compiled_filters.append(re.compile(pat))
            except re.error:
                continue
        found = [p for p in found if not any(rx.search(p) for rx in compiled_filters)]

    item.problems = found
    return found
