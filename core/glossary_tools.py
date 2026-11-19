"""Pre-translation scanning toolkit: speakers, repeats, term candidates, and pruned consistency audits."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from .dictionary import GptDict
from .models import SentenceItem

_KATAKANA_SEQ_RE = re.compile(r"[ァ-ヶー・]{2,}")
_BRACKETED_TERM_RE = re.compile(r"[『《【]([^『』《》【】\r\n]{2,12})[』》】]")
_HONORIFIC_COMPOUND_RE = re.compile(
    r"([\u4e00-\u9fffァ-ヶ]{1,6})(?:学園|学校|高校|大学|帝国|王国|共和国|公国|騎士団|商会|神社|教会|病院|通り|広場|公園|市|町|村|城|島|山|川|部|委員会|隊|団|組|家)"
)
_SPEAKER_JOINER_RE = re.compile(r"[・&＆+＋、,，/／]")

_KATAKANA_STOPWORDS: set[str] = {
    "ダメ", "マジ", "ヤバ", "ヤバい", "ホント", "ホンマ", "ウソ", "バカ", "アホ",
    "クソ", "チクショウ", "エッチ", "キス", "ハグ", "デート", "ラブ", "ベッド",
    "ドア", "テーブル", "ソファー", "ソファ", "トイレ", "シャワー", "スマホ", "テレビ",
    "メール", "ゲーム", "アニメ", "マンガ", "カフェ", "レストラン", "ホテル", "マンション",
    "アパート", "クラス", "テスト", "バイト", "アルバイト", "パート", "プレゼント",
    "ケーキ", "コーヒー", "ジュース", "ミルク", "ビール", "ワイン", "パン", "ランチ",
    "ディナー", "メニュー", "チャンス", "ピンチ", "ショック", "ストレス", "テンション",
    "イメージ", "タイミング", "バランス", "パターン", "レベル", "ポイント", "タイプ",
    "スタイル", "モード", "システム", "データ", "メッセージ", "ボタン", "スタート",
    "ストップ", "クリア", "オーバー", "エラー", "チェック", "セット", "カット",
    "ニュース", "ドラマ", "イベント", "パーティ", "パーティー", "チケット", "カード",
    "バッグ", "スカート", "シャツ", "ドレス", "パンツ", "ブーツ", "リボン", "メガネ",
    "カメラ", "パソコン", "ネット", "サイト", "ページ", "タイトル", "シーン", "ルート",
    "ピンク", "ブラック", "ホワイト", "ブルー", "レッド", "グリーン", "イエロー",
    "ゴールド", "シルバー", "ゼロ", "ワン", "ツー", "スリー", "ハイ", "ロー",
    "プラス", "マイナス", "ダブル", "シングル", "スーパー", "ハイパー", "ウルトラ",
    "ドキドキ", "ワクワク", "イライラ", "ニコニコ", "ニヤニヤ", "ボロボロ", "ギリギリ",
    "フラフラ", "クラクラ", "ビクビク", "ブルブル", "ズルズル", "ヌルヌル", "トロトロ",
    "ビチョビチョ", "グチョグチョ", "ズボズボ", "パンパン", "ゴクゴク", "ハァハァ",
    "アッ", "ンッ", "フッ", "ハッ", "クッ", "ギクッ", "ビクッ", "ピクッ",
}


def _snippet(item: SentenceItem, max_len: int = 44) -> str:
    text = (item.post_src or item.pre_src).replace("\r\n", " ").replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    prefix = f"{item.speaker}：" if item.speaker else ""
    return f"[{item.ref_key}] {prefix}{text}"


def dump_names(
    items: Iterable[SentenceItem],
    gpt_dict: GptDict | None = None,
    speaker_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate speaker frequencies, composite-speaker coverage, and example lines."""
    spk_map = speaker_map or {}
    counts: Counter[str] = Counter()
    examples_by_spk: dict[str, list[SentenceItem]] = defaultdict(list)

    for item in items:
        spk = item.speaker.strip()
        if not spk:
            continue
        counts[spk] += 1
        examples_by_spk[spk].append(item)

    rows: list[dict[str, Any]] = []
    covered_count = 0
    uncovered_composites: list[dict[str, Any]] = []

    for spk, cnt in counts.most_common():
        mapped = spk_map.get(spk, "")
        dict_entry = gpt_dict.get(spk) if gpt_dict is not None else None
        dict_dst = dict_entry.dst if dict_entry else ""
        parts = [p.strip() for p in _SPEAKER_JOINER_RE.split(spk) if p.strip()]
        missing_parts = [
            p
            for p in parts
            if not (spk_map.get(p) or (gpt_dict.get(p) if gpt_dict else None))
        ]
        part_covered = len(parts) >= 2 and not missing_parts
        is_covered = bool(mapped or dict_dst or part_covered)
        if is_covered:
            covered_count += 1
        elif len(parts) >= 2:
            uncovered_composites.append(
                {"speaker": spk, "parts": parts, "missing_parts": missing_parts, "count": cnt}
            )

        occ = examples_by_spk[spk]
        sample_items = [occ[0]]
        if len(occ) >= 3:
            sample_items.append(occ[len(occ) // 2])
        elif len(occ) == 2:
            sample_items.append(occ[1])

        rows.append(
            {
                "speaker": spk,
                "count": cnt,
                "covered": is_covered,
                "mapped_dst": mapped or dict_dst,
                "examples": [_snippet(it) for it in sample_items],
            }
        )

    total_unique = len(rows)
    uncovered = [r for r in rows if not r["covered"]]
    return {
        "total_unique_speakers": total_unique,
        "covered_speakers": covered_count,
        "uncovered_speakers": len(uncovered),
        "coverage_ratio": round(covered_count / total_unique, 4) if total_unique else 1.0,
        "speakers": rows,
        "uncovered_list": [r["speaker"] for r in uncovered],
        "uncovered_composites": uncovered_composites,
    }


def extract_repeats(
    items: Iterable[SentenceItem],
    *,
    min_count: int = 3,
    min_len: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Extract high-frequency repeated source sentences for consistency & UI pre-translation."""
    grouped: dict[str, list[SentenceItem]] = defaultdict(list)
    for item in items:
        key = (item.post_src or item.pre_src).strip()
        if len(key) < min_len:
            continue
        grouped[key].append(item)

    ranked = sorted(
        ((k, v) for k, v in grouped.items() if len(v) >= min_count),
        key=lambda kv: (-len(kv[1]), -len(kv[0]), kv[0]),
    )
    return [
        {
            "text": text,
            "count": len(occ),
            "sample_refs": [it.ref_key for it in occ[:5]],
            "speakers": list(dict.fromkeys(it.speaker for it in occ if it.speaker))[:5],
        }
        for text, occ in ranked[:limit]
    ]


def extract_term_candidates(
    items: Iterable[SentenceItem],
    gpt_dict: GptDict | None = None,
    *,
    min_count: int = 2,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Heuristically extract uncovered proper-noun / glossary candidates with sample lines."""
    item_list = list(items)
    known = {e.src for e in gpt_dict.entries} if gpt_dict is not None else set()

    term_counts: Counter[str] = Counter()
    term_sources: dict[str, str] = {}
    term_items: dict[str, list[SentenceItem]] = defaultdict(list)

    def _record(term: str, source_kind: str, item: SentenceItem) -> None:
        t = term.strip("・ー ")
        if len(t) < 2 or len(t) > 20:
            return
        if t in known or t in _KATAKANA_STOPWORDS:
            return
        if len(set(t)) <= 1:
            return
        term_counts[t] += 1
        term_sources.setdefault(t, source_kind)
        if len(term_items[t]) < 4:
            term_items[t].append(item)

    for item in item_list:
        text = item.post_src or item.pre_src
        if not text:
            continue
        for m in _KATAKANA_SEQ_RE.finditer(text):
            _record(m.group(0), "katakana", item)
        for m in _BRACKETED_TERM_RE.finditer(text):
            _record(m.group(1), "bracketed", item)
        for m in _HONORIFIC_COMPOUND_RE.finditer(text):
            _record(m.group(0), "compound", item)

    ranked = sorted(
        ((t, cnt) for t, cnt in term_counts.items() if cnt >= min_count),
        key=lambda kv: (-kv[1], -len(kv[0]), kv[0]),
    )
    results: list[dict[str, Any]] = []
    for term, cnt in ranked[:limit]:
        occ = term_items[term]
        samples = [_snippet(occ[0])]
        if len(occ) >= 2:
            samples.append(_snippet(occ[-1]))
        results.append(
            {
                "term": term,
                "count": cnt,
                "source": term_sources.get(term, "heuristic"),
                "examples": samples,
            }
        )
    return results


def audit_variant_inconsistency(
    items: Iterable[SentenceItem],
    gpt_dict: GptDict | None = None,
) -> dict[str, Any]:
    """Audit speaker drift, same-src different-dst drift, and bigram-pruned glossary root consistency."""
    item_list = list(items)

    # 1. Speaker name consistency across translated items
    spk_dsts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in item_list:
        if item.speaker and item.speaker_dst:
            spk_dsts[item.speaker][item.speaker_dst] += 1

    speaker_conflicts: list[dict[str, Any]] = []
    for spk, dst_counter in sorted(spk_dsts.items()):
        if len(dst_counter) > 1:
            speaker_conflicts.append(
                {
                    "speaker": spk,
                    "variants": [
                        {"dst": dst, "count": cnt} for dst, cnt in dst_counter.most_common()
                    ],
                }
            )

    # 2. Same post_src mapped to multiple distinct post_dst
    src_to_dsts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in item_list:
        src = (item.post_src or "").strip()
        dst = (item.effective_core_dst or "").strip()
        if not src or not dst or len(src) > 30:
            continue
        src_to_dsts[src][dst].append(item.ref_key)

    same_src_conflicts: list[dict[str, Any]] = []
    for src, dst_map in src_to_dsts.items():
        if len(dst_map) > 1:
            variants = sorted(
                (
                    {"dst": dst, "count": len(refs), "sample_refs": refs[:3]}
                    for dst, refs in dst_map.items()
                ),
                key=lambda x: -x["count"],
            )
            same_src_conflicts.append({"src": src, "variants": variants})
    same_src_conflicts.sort(key=lambda x: -sum(v["count"] for v in x["variants"]))

    # 3. Related glossary root clusters using bigram inverted index (avoids O(n^2) blow-up)
    related_groups: list[dict[str, Any]] = []
    if gpt_dict is not None and len(gpt_dict.entries) >= 2:
        entries = [e for e in gpt_dict.entries if len(e.src) >= 2]
        # Index every entry by all its 2-character bigrams
        bigram_index: dict[str, set[int]] = defaultdict(set)
        for idx, entry in enumerate(entries):
            s = entry.src
            for pos in range(len(s) - 1):
                bigram_index[s[pos : pos + 2]].add(idx)

        seen_pairs: set[tuple[str, str]] = set()
        for i, a in enumerate(entries):
            head_bg = a.src[:2]
            tail_bg = a.src[-2:]
            cands = bigram_index.get(head_bg)
            if not cands:
                continue
            if tail_bg != head_bg:
                cands = cands & bigram_index.get(tail_bg, set())
            for j in cands:
                if i == j:
                    continue
                b = entries[j]
                if len(b.src) <= len(a.src):
                    continue
                if a.src in b.src and a.dst not in b.dst:
                    pair = (a.src, b.src)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        related_groups.append(
                            {
                                "root_src": a.src,
                                "root_dst": a.dst,
                                "compound_src": b.src,
                                "compound_dst": b.dst,
                                "note": "compound translation does not contain root translation",
                            }
                        )
                        if len(related_groups) >= 40:
                            break
            if len(related_groups) >= 40:
                break

    return {
        "speaker_conflicts": speaker_conflicts,
        "same_src_diff_dst": same_src_conflicts[:40],
        "glossary_root_warnings": related_groups[:40],
    }


_NON_PRONOUN_COMPOUNDS: tuple[str, ...] = (
    "私立",
    "私服",
    "私物",
    "私生活",
    "私室",
    "私語",
    "私情",
    "私欲",
    "私見",
    "私文書",
    "私鉄",
    "私有",
    "私設",
)

_FIRST_PERSON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("わたくし", re.compile(r"わたくし")),
    ("わたし", re.compile(r"(?<!わたく)わたし")),
    ("あたし", re.compile(r"あたし")),
    ("あーし", re.compile(r"あーし")),
    ("あっし", re.compile(r"あっし")),
    ("僕", re.compile(r"僕")),
    ("俺", re.compile(r"俺")),
    ("私", re.compile(r"私")),
    ("ウチ", re.compile(r"(?<![ァ-ヶー])ウチ(?![ァ-ヶー])")),
    ("オレ", re.compile(r"(?<![ァ-ヶー])オレ(?![ァ-ヶー])")),
    ("ボク", re.compile(r"(?<![ァ-ヶー])ボク(?![ァ-ヶー])")),
)

_END_BOUNDARY = r"(?:[。！？!?…♪～ー、\s」』）)]|$)"
_SENTENCE_ENDING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("っす", re.compile(r"っす(?:ね|よ|か|よね|な)?" + _END_BOUNDARY)),
    ("ですわ", re.compile(r"ですわ(?:ね|よ)?" + _END_BOUNDARY)),
    ("かしら", re.compile(r"かしら(?:ね)?" + _END_BOUNDARY)),
    ("わね", re.compile(r"わね" + _END_BOUNDARY)),
    ("だぜ", re.compile(r"だぜ" + _END_BOUNDARY)),
    ("なの", re.compile(r"なの(?:よ|ね|か|かな)?" + _END_BOUNDARY)),
    ("じゃ", re.compile(r"(?<!ん)じゃ(?:な|よ|ぞ|ろ|わい|のう)?" + _END_BOUNDARY)),
)


def _normalize_loaded_files_for_debuts(loaded_files: Iterable[Any]) -> list[tuple[str, list[SentenceItem]]]:
    raw_list = list(loaded_files)
    if not raw_list:
        return []
    if isinstance(raw_list[0], SentenceItem):
        grouped: dict[str, list[SentenceItem]] = defaultdict(list)
        for it in raw_list:
            grouped[it.file_name].append(it)
        return sorted(grouped.items(), key=lambda kv: kv[0])

    pairs: list[tuple[str, list[SentenceItem]]] = []
    for entry in raw_list:
        if isinstance(entry, tuple):
            if len(entry) >= 3 and isinstance(entry[2], list):
                pairs.append((str(entry[0]), list(entry[2])))
            elif len(entry) >= 2 and isinstance(entry[1], list):
                pairs.append((str(entry[0]), list(entry[1])))
    pairs.sort(key=lambda kv: kv[0])
    return pairs


def sample_character_debuts(
    loaded_files: Iterable[Any],
    *,
    top_n: int = 25,
    context_radius: int = 6,
) -> dict[str, Any]:
    """Mine each unique speaker's first script appearance (±window) plus pronoun & sentence-ending stats."""
    files_sorted = _normalize_loaded_files_for_debuts(loaded_files)
    speaker_counts: Counter[str] = Counter()
    debut_order: dict[str, int] = {}
    first_seen: dict[str, tuple[str, int, list[SentenceItem]]] = {}
    pronoun_tallies: dict[str, Counter[str]] = defaultdict(Counter)
    ending_tallies: dict[str, Counter[str]] = defaultdict(Counter)

    for rel_name, items in files_sorted:
        for pos, it in enumerate(items):
            spk = it.speaker.strip()
            if not spk:
                continue
            speaker_counts[spk] += 1
            if spk not in first_seen:
                debut_order[spk] = len(debut_order)
                first_seen[spk] = (rel_name, pos, items)

            text = it.post_src or it.pre_src
            if not text:
                continue

            cleaned_for_pronoun = text
            for comp in _NON_PRONOUN_COMPOUNDS:
                if comp in cleaned_for_pronoun:
                    cleaned_for_pronoun = cleaned_for_pronoun.replace(comp, "　" * len(comp))

            for label, rx in _FIRST_PERSON_PATTERNS:
                hits = len(rx.findall(cleaned_for_pronoun))
                if hits:
                    pronoun_tallies[spk][label] += hits

            for label, rx in _SENTENCE_ENDING_PATTERNS:
                hits = len(rx.findall(text))
                if hits:
                    ending_tallies[spk][label] += hits

    ranked_speakers = sorted(
        speaker_counts.keys(),
        key=lambda s: (-speaker_counts[s], debut_order[s], s),
    )
    if top_n > 0:
        ranked_speakers = ranked_speakers[:top_n]

    radius = max(0, context_radius)
    characters: list[dict[str, Any]] = []
    md_lines: list[str] = [
        f"# Character Debut & Voice Evidence ({len(ranked_speakers)}/{len(speaker_counts)} speakers | ±{radius} lines)",
    ]

    for rank_idx, spk in enumerate(ranked_speakers, start=1):
        rel_name, pos, file_items = first_seen[spk]
        debut_it = file_items[pos]
        lo = max(0, pos - radius)
        hi = min(len(file_items), pos + radius + 1)

        window_rows: list[dict[str, Any]] = []
        win_text_lines: list[str] = []
        for w_pos in range(lo, hi):
            wit = file_items[w_pos]
            is_debut = w_pos == pos
            raw_line = wit.pre_src.replace("\r\n", "\\n").replace("\n", "\\n")
            window_rows.append(
                {
                    "index": wit.index,
                    "speaker": wit.speaker,
                    "pre_src": wit.pre_src,
                    "is_debut": is_debut,
                }
            )
            marker = ">" if is_debut else " "
            win_text_lines.append(f"{marker} {wit.index} | {wit.speaker or '-'} | {raw_line}")

        p_dict = dict(pronoun_tallies[spk].most_common())
        e_dict = dict(ending_tallies[spk].most_common())
        characters.append(
            {
                "speaker": spk,
                "count": speaker_counts[spk],
                "debut_file": rel_name,
                "debut_index": debut_it.index,
                "debut_ref": debut_it.ref_key,
                "pronouns": p_dict,
                "sentence_endings": e_dict,
                "debut_window": window_rows,
            }
        )

        p_str = ", ".join(f"`{k}`({v})" for k, v in p_dict.items()) or "-"
        e_str = ", ".join(f"`{k}`({v})" for k, v in e_dict.items()) or "-"
        md_lines.extend(
            [
                "",
                f"## {rank_idx}. `{spk}` (lines: {speaker_counts[spk]} | debut: `{debut_it.ref_key}`)",
                f"- First-Person Pronouns: {p_str} | Sentence Endings: {e_str}",
                "```text",
                *win_text_lines,
                "```",
            ]
        )

    return {
        "files_scanned": len(files_sorted),
        "total_unique_speakers": len(speaker_counts),
        "selected_count": len(characters),
        "context_radius": radius,
        "characters": characters,
        "markdown": "\n".join(md_lines),
    }
