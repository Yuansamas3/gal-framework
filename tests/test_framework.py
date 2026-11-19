"""Comprehensive unit & sampleProject end-to-end tests for gal_framework."""

from __future__ import annotations

import concurrent.futures
import io
import json
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from unittest.mock import patch

import gal_cli
from adapters import compact_pipe, gt_json
from core import batcher as batcher_mod
from core.batcher import (
    _render_src_for_prompt,
    edit_cache_entries,
    parse_patch_text,
    pull_batch,
    push_batch,
    search_replace_project,
)
from core.cache_ledger import (
    MISS_KEY_NOT_FOUND,
    MISS_POST_SRC_CHANGED,
    cache_file_path_for,
    cleanup_stale_cache_temp_files,
    join_input_with_cache,
    load_cache_entries,
    load_project_files,
    save_input_file,
)
from core.dictionary import (
    GptDict,
    GptDictEntry,
    NormalDict,
    inject_glossary_for_batch,
    load_project_dicts,
)
from core.glossary_tools import audit_variant_inconsistency, sample_character_debuts
from core.models import ItemStatus, ProjectConfig, SentenceItem
from core.normalizer import normalize_and_restore_dst, preprocess_items, strip_outer_shell
from core.problems import (
    ALL_PROBLEM_TYPES,
    HARD_ERROR_TYPES,
    SOFT_WARN_TYPES,
    check_item_problems,
    problem_severity,
    split_problems_by_severity,
)
from core.rebuilder import export_project_outputs, rebuild_project, revert_task
from core.seed_ledger import SeedLedger
from core.text_preserve import PreserveRule, TextPreserver, normalize_regex_pattern, validate_rule


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = gal_cli.main(argv)
    return rc, buf.getvalue()


class TestNormalizerAndProtectionReviewFixes(unittest.TestCase):
    def test_outer_ws_single_nested_mixed_and_cross_multiline(self) -> None:
        items = [
            SentenceItem("s.json", 1, pre_src="　「おはよう」\\n", speaker="遥"),
            SentenceItem("s.json", 2, pre_src="「『秘密だよ』」", speaker="遥"),
            SentenceItem("s.json", 3, pre_src="「你好」と彼女は言った、「またね」", speaker=""),
            SentenceItem("s.json", 4, pre_src="「今日はいい天気だね", speaker="遥"),
            SentenceItem("s.json", 5, pre_src="屋上で食べようよ」", speaker=""),
            SentenceItem("s.json", 6, pre_src="『ねえ、", speaker="琴音"),
            SentenceItem("s.json", 7, pre_src="放課後に、", speaker=""),
            SentenceItem("s.json", 8, pre_src="寄っていかない？』", speaker=""),
        ]
        strip_outer_shell(items, strip_outer_quotes=True)

        self.assertEqual((items[0].left_ws, items[0].right_ws, items[0].left_quote, items[0].right_quote, items[0].post_src),
                         ("　", "\\n", "「", "」", "おはよう"))
        self.assertEqual((items[1].left_quote, items[1].right_quote, items[1].post_src), ("「『", "』」", "秘密だよ"))
        # Mixed narration must not be corrupted
        self.assertEqual((items[2].left_quote, items[2].right_quote, items[2].post_src), ("", "", "「你好」と彼女は言った、「またね」"))
        # 2-line cross dialogue
        self.assertEqual((items[3].left_quote, items[3].post_src), ("「", "今日はいい天気だね"))
        self.assertEqual((items[4].right_quote, items[4].post_src, items[4].speaker), ("」", "屋上で食べようよ", "遥"))
        # 3-line cross dialogue
        self.assertEqual((items[5].left_quote, items[5].post_src), ("『", "ねえ、"))
        self.assertEqual((items[6].left_quote, items[6].right_quote, items[6].speaker), ("", "", "琴音"))
        self.assertEqual((items[7].right_quote, items[7].post_src, items[7].speaker), ("』", "寄っていかない？", "琴音"))

    def test_strip_outer_shell_idempotent_preserves_pre_dict(self) -> None:
        """Review #8: repeated strip_outer_shell(force=False) must NOT wipe pre_dict results."""
        cfg = ProjectConfig()
        pre_dict = NormalDict()
        pre_dict.load_tsv_text("ごしゅじんひゃま\tご主人様\n")
        items = [SentenceItem("s.json", 1, pre_src="「ごしゅじんひゃま」", speaker="遥")]
        preprocess_items(items, cfg, pre_dict=pre_dict)
        self.assertEqual(items[0].post_src, "ご主人様")
        self.assertTrue(items[0].shell_stripped)

        # Calling strip_outer_shell again without force=True must keep pre_dict output intact!
        strip_outer_shell(items, strip_outer_quotes=True)
        self.assertEqual(items[0].post_src, "ご主人様")

    def test_unified_protection_shields_post_dict_and_ruby_tags(self) -> None:
        """Review #1: protect_text must wrap post_dict + punctuation cleanup AND ruby tags."""
        cfg = ProjectConfig(preserve_presets=["base", "kag", "rpgmaker"])
        tp = TextPreserver.from_project(cfg)
        post_dict = NormalDict()
        post_dict.load_tsv_text(
            "\n".join(
                [
                    "...\t…",
                    ",\t，",
                    "!\t！",
                    "~\t～",
                ]
            )
        )
        post_dict.sort_by_length()

        # Control tag `en(!s[123])` and ruby tag `[ruby text=foo...bar]` must NOT be mutated by post_dict!
        it = SentenceItem(
            "s.json",
            1,
            pre_src="en(!s[123])[ruby text=a...b]テスト...!",
            post_src="en(!s[123])[ruby text=a...b]テスト...!",
            pre_dst="en(!s[123])[ruby text=a...b]测试...!",
        )
        out = normalize_and_restore_dst(it, cfg, post_dict=post_dict, preserver=tp)
        self.assertEqual(out, "en(!s[123])[ruby text=a...b]测试…！")

    def test_post_punctuation_alignment_and_quote_repair(self) -> None:
        cfg = ProjectConfig()
        post_dict = NormalDict()
        post_dict.load_file(FRAMEWORK_ROOT / "dicts" / "post_symbol_common.tsv")
        post_dict.sort_by_length()

        # Trailing period trimmed + accidental outer quote deduplicated
        it1 = SentenceItem("s.json", 1, pre_src="「おはよう」", speaker="遥")
        strip_outer_shell([it1])
        it1.pre_dst = "「早上好。」"
        self.assertEqual(normalize_and_restore_dst(it1, cfg, post_dict), "「早上好」")

        # ♪ and ！？ auto-completed
        it2 = SentenceItem("s.json", 2, pre_src="「行こう♪」", speaker="遥")
        strip_outer_shell([it2])
        it2.pre_dst = "走吧"
        self.assertEqual(normalize_and_restore_dst(it2, cfg, post_dict), "「走吧♪」")

        it3 = SentenceItem("s.json", 3, pre_src="「本当！？」", speaker="蓮")
        strip_outer_shell([it3])
        it3.pre_dst = "真的吗！"
        self.assertEqual(normalize_and_restore_dst(it3, cfg, post_dict), "「真的吗！？」")

        # …。 -> …, leading comma trimmed, same-direction quotes repaired
        it4 = SentenceItem("s.json", 4, pre_src="彼は『秘密』と言った（笑）")
        strip_outer_shell([it4])
        it4.pre_dst = "，他说”秘密”（笑。）……。"
        self.assertEqual(normalize_and_restore_dst(it4, cfg, post_dict), "他说『秘密』（笑）……")


class TestDictionaryGear2(unittest.TestCase):
    def test_conditional_syntax_prefixes_and_builtin_h_dict(self) -> None:
        nd = NormalDict()
        nd.load_tsv_text(
            "\n".join(
                [
                    "^^啊\t嗯",
                    "1^测试\t样板",
                    "pre_jp\t人妻[or]ひとづま\t有夫之妇\t人妻",
                    "pre_jp\t(同上)\t已婚妇女\t人妻",
                    "pre_jp\t!サタン\t撒旦\t魔王",
                    'post_jp\t「 [and] !"\t1^"\t「',
                    "短い\t短",
                    "短い言葉\t长词优先",
                ]
            )
        )
        nd.sort_by_length()
        self.assertEqual(nd.apply("短い言葉と短い"), "长词优先と短")
        self.assertEqual(nd.apply("啊啊，测试与测试"), "嗯啊，样板与测试")

        it_wife = SentenceItem("a.json", 1, pre_src="彼女はひとづまだ")
        self.assertEqual(nd.apply("她是有夫之妇，也是已婚妇女", it_wife), "她是人妻，也是人妻")

        it_demon = SentenceItem("a.json", 2, pre_src="魔王が来た")
        self.assertEqual(nd.apply("撒旦来了", it_demon), "魔王来了")
        it_satan = SentenceItem("a.json", 3, pre_src="サタンが来た")
        self.assertEqual(nd.apply("撒旦来了", it_satan), "撒旦来了")

        it_q = SentenceItem("a.json", 4, pre_src="彼は「待って」と言った")
        self.assertEqual(nd.apply('他说"等等"', it_q), '他说「等等"')

        # Built-in H phonetic dict: >=200 entries, safe on normal words
        h_dict = NormalDict()
        self.assertGreaterEqual(h_dict.load_file(FRAMEWORK_ROOT / "dicts" / "pre_h_phonetic.tsv"), 200)
        h_dict.sort_by_length()
        it_h = SentenceItem("h.json", 1, pre_src="ごしゅじんひゃま、しゅごひ、チ○ポ")
        self.assertEqual(h_dict.apply(it_h.pre_src, it_h), "ご主人様、すごい、チンポ")
        it_norm = SentenceItem("h.json", 2, pre_src="ひょっとして、にゃあ、りゅう")
        self.assertEqual(h_dict.apply(it_norm.pre_src, it_norm), "ひょっとして、にゃあ、りゅう")

    def test_dynamic_batch_glossary_injector(self) -> None:
        gd = GptDict()
        gd.load_file(FRAMEWORK_ROOT / "dicts" / "gpt_common_acg.tsv")
        gd.load_file(FRAMEWORK_ROOT / "dicts" / "gpt_common_menu.tsv")
        gd.add_entry(GptDictEntry("桜井遥", "樱井遥", "女主角"))
        gd.sort_by_length()

        items = [SentenceItem("c.json", 1, pre_src="「先輩、ツッコミが鋭いよ」", speaker="桜井遥")]
        strip_outer_shell(items)
        matched, block = inject_glossary_for_batch(items, gd)
        srcs = {e.src for e in matched}
        self.assertIn("先輩", srcs)
        self.assertIn("ツッコミ", srcs)
        self.assertIn("桜井遥", srcs)
        self.assertNotIn("ゴブリン", srcs)
        self.assertIn("桜井遥 -> 樱井遥 #女主角", block)


class TestProblemsAndTextPreserveReviewFixes(unittest.TestCase):
    def test_presets_nested_tokens_extra_order_and_ruby_modes(self) -> None:
        for preset in ("base", "kag", "renpy", "rpgmaker", "wolf"):
            tp = TextPreserver.from_project(ProjectConfig(preserve_presets=[preset]))
            self.assertEqual(tp.rejected_rules, [], f"Preset {preset} rejected rules")
            self.assertGreater(len(tp.rules), 0)

        self.assertFalse(validate_rule(r".*?", "zero-len", ["テスト"]).valid)
        self.assertFalse(validate_rule(r"\{.*\}", "swallow", ["{这是一段很长的正文内容，不能被吞掉。}"]).valid)

        # Review #2: missing, extra, and order_mismatch ALL detected
        cfg = ProjectConfig(preserve_presets=["base", "kag"])
        tp = TextPreserver.from_project(cfg)
        prof = json.loads((FRAMEWORK_ROOT / "profiles" / "modern_acg.json").read_text(encoding="utf-8"))

        # Extra hallucinated control token `<br>` -> MUST trigger 多控制符!
        it_extra = SentenceItem("p.json", 1, pre_src="おはよう", post_src="おはよう", pre_dst="早上好<br>", post_dst="早上好<br>")
        probs_extra = check_item_problems(it_extra, cfg, profile=prof, preserver=tp)
        self.assertTrue(any(p.startswith("多控制符") and "<br>" in p for p in probs_extra))

        # Scrambled token order -> MUST trigger 控制符错序!
        it_order = SentenceItem("p.json", 2, pre_src="<size=30>注目</size>", post_src="<size=30>注目</size>", pre_dst="</size>注目<size=30>", post_dst="</size>注目<size=30>")
        probs_order = check_item_problems(it_order, cfg, profile=prof, preserver=tp)
        self.assertTrue(any(p.startswith("控制符错序") for p in probs_order))

        # Review #6: nested `%s` inside ruby tag `[ruby text="%s"]` MUST be tracked!
        it_nested = SentenceItem(
            "p.json",
            3,
            pre_src='[ruby text="%s"]名前',
            post_src='[ruby text="%s"]名前',
            pre_dst='[ruby text="名"]名字',
            post_dst='[ruby text="名"]名字',
        )
        probs_nested = check_item_problems(it_nested, cfg, profile=prof, preserver=tp)
        self.assertTrue(any(p.startswith("缺控制符") and "%s" in p for p in probs_nested))

        # Review #11: ruby_preserve_mode="translate" allows localized ruby, "verbatim" catches changed ruby!
        it_ruby = SentenceItem(
            "p.json",
            4,
            pre_src="<rウェアウルフ>人狼</r>",
            post_src="<rウェアウルフ>人狼</r>",
            pre_dst="<r沃尔夫>狼人</r>",
            post_dst="<r沃尔夫>狼人</r>",
        )
        self.assertEqual(check_item_problems(it_ruby, cfg, profile=prof, preserver=tp), [])

        cfg_verbatim = ProjectConfig(preserve_presets=["base"], ruby_preserve_mode="verbatim")
        tp_verb = TextPreserver.from_project(cfg_verbatim)
        probs_verb = check_item_problems(it_ruby, cfg_verbatim, profile=prof, preserver=tp_verb)
        self.assertTrue(any(p.startswith("缺控制符") for p in probs_verb))
        self.assertTrue(any(p.startswith("多控制符") for p in probs_verb))

    def test_linebreak_baseline_guitar_exclusion_and_registry(self) -> None:
        """Review #3 (post_src linebreak baseline), #7 (吉他), #9 (翻译失败/本无韩文 in ALL_PROBLEM_TYPES)."""
        self.assertIn("翻译失败", ALL_PROBLEM_TYPES)
        self.assertIn("本无韩文", ALL_PROBLEM_TYPES)
        self.assertIn("多控制符", ALL_PROBLEM_TYPES)
        self.assertIn("控制符错序", ALL_PROBLEM_TYPES)

        cfg = ProjectConfig(extra_linebreak_policy="trim")
        prof = json.loads((FRAMEWORK_ROOT / "profiles" / "modern_acg.json").read_text(encoding="utf-8"))
        tp = TextPreserver.from_project(cfg)

        # Review #3: outer trailing `\n` in shell (`right_ws`), zero inline `\n` in `post_src`
        it_lb = SentenceItem("p.json", 1, pre_src="「おはよう」\\n", speaker="遥")
        strip_outer_shell([it_lb])
        it_lb.pre_dst = "早上好\\n今天不错"  # Extra inline \n trimmed by normalizer against post_src
        normalize_and_restore_dst(it_lb, cfg, preserver=tp)
        self.assertEqual(it_lb.post_dst, "「早上好今天不错」\\n")
        # Both normalizer and problems use post_src vs inner_post_dst -> 0 linebreak mismatch!
        probs_lb = check_item_problems(it_lb, cfg, profile=prof, preserver=tp)
        self.assertFalse(any("换行" in p for p in probs_lb))

        # Review #7: 吉他 in speakerless monologue must NOT trigger 独白男他, while 他抱着吉他 MUST!
        it_guitar_ok = SentenceItem("p.json", 2, pre_src="ギターの音色が響く", post_src="ギターの音色が響く", pre_dst="吉他的音色回荡着", post_dst="吉他的音色回荡着", speaker="")
        self.assertNotIn("独白男他", check_item_problems(it_guitar_ok, cfg, profile=prof, preserver=tp))
        it_guitar_he = SentenceItem("p.json", 3, pre_src="ギターを抱えた", post_src="ギターを抱えた", pre_dst="他抱着吉他", post_dst="他抱着吉他", speaker="")
        self.assertIn("独白男他", check_item_problems(it_guitar_he, cfg, profile=prof, preserver=tp))

        # Review #9: enabled_problems governs 翻译失败 and 本无韩文 cleanly
        cfg_no_fail = ProjectConfig(enabled_problems=["残留日文"])
        it_ko_fail = SentenceItem("p.json", 4, pre_src="テスト", post_src="テスト", pre_dst="(Failed) 안녕", post_dst="(Failed) 안녕")
        self.assertEqual(check_item_problems(it_ko_fail, cfg_no_fail, profile=prof, preserver=tp), [])
        probs_all = check_item_problems(it_ko_fail, cfg, profile=prof, preserver=tp)
        self.assertIn("翻译失败", probs_all)
        self.assertIn("本无韩文", probs_all)

    def test_all_remaining_qa_dimensions_and_three_profiles(self) -> None:
        prof_mod = json.loads((FRAMEWORK_ROOT / "profiles" / "modern_acg.json").read_text(encoding="utf-8"))
        prof_dark = json.loads((FRAMEWORK_ROOT / "profiles" / "dark_mystery.json").read_text(encoding="utf-8"))
        prof_cls = json.loads((FRAMEWORK_ROOT / "profiles" / "classical_fantasy.json").read_text(encoding="utf-8"))
        cfg_mod = ProjectConfig(profile="modern_acg", target_encoding="utf8")
        tp = TextPreserver.from_project(cfg_mod)
        gd = GptDict([GptDictEntry("幼馴染", "青梅竹马", "关系")])

        # 残留日文 & 原文输出
        it_jp = SentenceItem("q.json", 1, pre_src="晴れだね", post_src="晴れだね", pre_dst="晴天ね", post_dst="晴天ね")
        self.assertTrue(any(p.startswith("残留日文") for p in check_item_problems(it_jp, cfg_mod, prof_mod, tp, gd)))
        it_same = SentenceItem("q.json", 2, pre_src="おはよう", post_src="おはよう", pre_dst="おはよう", post_dst="おはよう")
        self.assertIn("原文输出", check_item_problems(it_same, cfg_mod, prof_mod, tp, gd))

        # 本有括号 / 本无引号 / 本有引号 / 字典未使用 / 比日文长 / 单句过长 / 词频过高
        it_b = SentenceItem("q.json", 3, pre_src="（笑）", post_src="（笑）", pre_dst="笑", post_dst="笑")
        self.assertIn("本有括号", check_item_problems(it_b, cfg_mod, prof_mod, tp, gd))
        it_q1 = SentenceItem("q.json", 4, pre_src="違う", post_src="違う", pre_dst="“不对”", post_dst="“不对”")
        self.assertIn("本无引号", check_item_problems(it_q1, cfg_mod, prof_mod, tp, gd))
        it_q2 = SentenceItem("q.json", 5, pre_src="『秘密』だ", post_src="『秘密』だ", pre_dst="是秘密", post_dst="是秘密")
        self.assertIn("本有引号", check_item_problems(it_q2, cfg_mod, prof_mod, tp, gd))
        it_d = SentenceItem("q.json", 6, pre_src="幼馴染", post_src="幼馴染", pre_dst="发小", post_dst="发小")
        self.assertTrue(any(p.startswith("字典未使用") for p in check_item_problems(it_d, cfg_mod, prof_mod, tp, gd)))
        it_len = SentenceItem("q.json", 7, pre_src="そうか", post_src="そうか", pre_dst="原来如此啊我彻底明白了", post_dst="原来如此啊我彻底明白了")
        self.assertTrue(any(p.startswith("比日文长") for p in check_item_problems(it_len, cfg_mod, prof_mod, tp, gd)))
        it_clause = SentenceItem(
            "q.json", 8,
            pre_src="昨日駅の前で偶然出会った赤い帽子の少女が今日もそこにいた",
            post_src="昨日駅の前で偶然出会った赤い帽子の少女が今日もそこにいた",
            pre_dst="昨天在车站前面偶然遇到的那个戴着红帽子的少女今天还在那里",
            post_dst="昨天在车站前面偶然遇到的那个戴着红帽子的少女今天还在那里",
        )
        self.assertTrue(any(p.startswith("单句过长") for p in check_item_problems(it_clause, cfg_mod, prof_mod, tp, gd)))
        it_rep = SentenceItem("q.json", 9, pre_src="ああ", post_src="ああ", pre_dst="啊" * 24, post_dst="啊" * 24)
        self.assertTrue(any(p.startswith("词频过高") for p in check_item_problems(it_rep, cfg_mod, prof_mod, tp, gd)))

        # 非GBK字符
        cfg_gbk = ProjectConfig(profile="modern_acg", target_encoding="gbk")
        it_gbk = SentenceItem("q.json", 10, pre_src="嬉しい♪", post_src="嬉しい♪", pre_dst="开心♪", post_dst="开心♪")
        self.assertTrue(any(p.startswith("非GBK字符") for p in check_item_problems(it_gbk, cfg_gbk, prof_mod, tp, gd)))

        # 3-profile register differentiation
        it_cls_ok = SentenceItem("q.json", 11, pre_src="吾が剣", post_src="吾が剣", pre_dst="吾之剑锋亦乃天命", post_dst="吾之剑锋亦乃天命", speaker="剑圣", is_dialogue=True)
        self.assertFalse(any(p.startswith("语域违和词") for p in check_item_problems(it_cls_ok, ProjectConfig(profile="classical_fantasy"), prof_cls, tp)))
        self.assertTrue(any(p.startswith("语域违和词") for p in check_item_problems(it_cls_ok, cfg_mod, prof_mod, tp)))

        it_cls_bad = SentenceItem("q.json", 12, pre_src="恥ずかしい", post_src="恥ずかしい", pre_dst="太社死了", post_dst="太社死了", speaker="公主", is_dialogue=True)
        self.assertTrue(any(p.startswith("语域违和词") for p in check_item_problems(it_cls_bad, ProjectConfig(profile="classical_fantasy"), prof_cls, tp)))

        it_dark_bad = SentenceItem("q.json", 13, pre_src="兄さん", post_src="兄さん", pre_dst="欧尼酱", post_dst="欧尼酱", speaker="妹", is_dialogue=True)
        self.assertTrue(any(p.startswith("语域违和词") for p in check_item_problems(it_dark_bad, ProjectConfig(profile="dark_mystery"), prof_dark, tp)))

        # Whitelist & regex filter
        cfg_wl = ProjectConfig(problem_whitelist=["q.json:3-5"], problem_filter_regex=[r"^比日文长"])
        self.assertEqual(check_item_problems(it_q1, cfg_wl, prof_mod, tp, gd), [])
        self.assertFalse(any(p.startswith("比日文长") for p in check_item_problems(it_len, cfg_wl, prof_mod, tp, gd)))


class TestSeedLedgerAndGlossaryPerformanceReviewFixes(unittest.TestCase):
    def test_seed_gate_requires_real_sampling_and_forbids_cross_round_consume(self) -> None:
        """Review #4 (true sampling) and Review #5 (no cross-round consumption)."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SeedLedger(Path(tmp) / "seeds.json", min_required_rounds=2)

            # Calling complete_round twice with ZERO samples must STILL FAIL check_gate!
            ok1, r1, _ = ledger.complete_round()
            ok2, r2, _ = ledger.complete_round()
            self.assertTrue(ok1 and ok2 and r2 == 2)
            passed, reasons = ledger.check_gate()
            self.assertFalse(passed)
            self.assertTrue(any("insufficient real dispersed samples" in r for r in reasons))

            # Fresh ledger for round-isolation check
            ledger2 = SeedLedger(Path(tmp) / "seeds2.json", min_required_rounds=2)
            s1 = ledger2.add_seed("Round 1 seed A", round_num=1)
            s1b = ledger2.add_seed("Round 1 seed B", round_num=1)
            s2 = ledger2.add_seed("Future Round 2 seed", round_num=2)

            # Attempting to consume future Round 2 seed while in Round 1 must be skipped!
            res_fut = ledger2.consume_seed(s2["id"], resolution="illegal early consume")
            self.assertEqual(res_fut["consumed_ids"], [])
            self.assertEqual(res_fut["skipped_future_ids"], [s2["id"]])

            # Consuming only s1 with complete_round=True must NOT advance round because s1b is still open!
            res_part = ledger2.consume_seed(s1["id"], resolution="done A", complete_round=True)
            self.assertFalse(res_part["round_advanced"])
            self.assertEqual(ledger2.completed_rounds, 0)
            self.assertTrue(res_part["round_block_reason"])

            # Consuming s1b allows advancing Round 1
            res_done1 = ledger2.consume_seed(s1b["id"], resolution="done B", complete_round=True)
            self.assertTrue(res_done1["round_advanced"])
            self.assertEqual(ledger2.completed_rounds, 1)

            # Now Round 2 is active -> consuming s2 succeeds!
            res_r2 = ledger2.consume_seed(s2["id"], resolution="done R2", complete_round=True)
            self.assertTrue(res_r2["round_advanced"])
            self.assertEqual(ledger2.completed_rounds, 2)

    def test_glossary_audit_pruned_scale_2000_terms(self) -> None:
        """Review #10: 2,000-entry glossary root audit must finish rapidly via bigram index."""
        entries = [GptDictEntry(f"キャラ名{i:04d}", f"角色名{i:04d}") for i in range(1998)]
        entries.append(GptDictEntry("星見坂", "星见坂"))
        entries.append(GptDictEntry("星見坂学園", "星见坡学园"))  # Deliberate root mismatch!
        gd = GptDict(entries)

        t0 = time.perf_counter()
        audit = audit_variant_inconsistency([], gpt_dict=gd)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.5, f"audit_variant_inconsistency too slow: {elapsed:.3f}s")
        self.assertTrue(
            any(w["root_src"] == "星見坂" and w["compound_src"] == "星見坂学園" for w in audit["glossary_root_warnings"])
        )


class TestCacheLedgerAndAdapters(unittest.TestCase):
    def test_legacy_cache_keys_miss_codes_and_temp_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "transl_cache"
            cache_dir.mkdir()
            stale = cache_dir / "ch01.json.tmp"
            stale.write_text("{corrupt", encoding="utf-8")
            self.assertEqual(cleanup_stale_cache_temp_files(cache_dir), 1)
            self.assertFalse(stale.exists())

            cfg = ProjectConfig()
            items = [
                SentenceItem("c.json", 1, pre_src="「おはよう」", post_src="おはよう", speaker="遥"),
                SentenceItem("c.json", 2, pre_src="「ごしゅじんひゃま」", post_src="旦那様", speaker="遥"),
                SentenceItem("c.json", 3, pre_src="「校对锁定句」", post_src="已改译前词", speaker="遥"),
                SentenceItem("c.json", 4, pre_src="「新句」", post_src="新句", speaker="遥"),
            ]
            # Simulate legacy GalTransl cache entries (`pre_jp`, `post_jp`, `pre_zh`, `proofread_zh`, `doub_content`)
            legacy_cache = [
                {"index": 1, "name": "遥", "pre_jp": "「おはよう」", "post_jp": "おはよう", "pre_zh": "早上好", "post_zh_preview": "「早上好」"},
                {"index": 2, "name": "遥", "pre_jp": "「ごしゅじんひゃま」", "post_jp": "ご主人様", "pre_zh": "主人大人"},
                {"index": 3, "name": "遥", "pre_jp": "「校对锁定句」", "post_jp": "旧译前词", "pre_zh": "旧译", "proofread_zh": "精修译文", "doub_content": "人工确认"},
            ]
            hits, misses = join_input_with_cache(items, legacy_cache, cfg)
            self.assertEqual({it.index for it in hits}, {1, 3})
            self.assertEqual(items[2].proofread_comment, "人工确认")
            self.assertEqual(items[2].status, ItemStatus.PROOFREAD.value)
            miss_map = {it.index: it.cache_miss_reason for it in misses}
            self.assertEqual(miss_map[2], MISS_POST_SRC_CHANGED)
            self.assertEqual(miss_map[4], MISS_KEY_NOT_FOUND)

    def test_compact_pipe_cli_import_push_export_lossless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = root / "pipe_proj"
            src_file = root / "ep01.txt"
            src_file.write_text(
                "# Episode 1\n\n0001|Alice|「おはよう」\n0002||静かな朝だった。\n0003|[Bob]「元気？」\n0004|風が吹いた。\n",
                encoding="utf-8",
            )
            rc, _ = _run_cli(["init", "--project", str(proj), "--profile", "modern_acg"])
            self.assertEqual(rc, 0)
            rc, _ = _run_cli(["import", "--project", str(proj), "--format", "compact_pipe", "--input", str(src_file)])
            self.assertEqual(rc, 0)

            patch = (
                "@@ ep01.txt.json|1\n@name: 爱丽丝\n+早上好。\n"
                "@@ ep01.txt.json|2\n+那是一个安静的清晨。\n"
                "@@ ep01.txt.json|3\n@name: 鲍勃\n+精神还好吗？\n"
                "@@ ep01.txt.json|4\n+风吹了过来。\n"
            )
            pfile = root / "p.txt"
            pfile.write_text(patch, encoding="utf-8")
            rc, _ = _run_cli(["batch-push", "--project", str(proj), "--patch-file", str(pfile)])
            self.assertEqual(rc, 0)

            out_dir = root / "exported_pipe"
            rc, _ = _run_cli(["export", "--project", str(proj), "--format", "compact_pipe", "--output", str(out_dir)])
            self.assertEqual(rc, 0)
            expected = (
                "# Episode 1\n\n"
                "0001|爱丽丝|「早上好」\n"
                "0002||那是一个安静的清晨。\n"
                "0003|[鲍勃]「精神还好吗？」\n"
                "0004|风吹了过来。\n"
            )
            self.assertEqual((out_dir / "ep01.txt").read_text(encoding="utf-8"), expected)


class TestSampleProjectFullLifecycle(unittest.TestCase):
    def test_sample_project_end_to_end_lifecycle(self) -> None:
        """Drive sampleProject/ through prep-scan -> pull -> push -> edit -> rebuild -> search-replace -> revert -> seed -> export."""
        sample_src = FRAMEWORK_ROOT / "sampleProject"
        self.assertTrue((sample_src / "gt_input" / "chapter_01.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "sampleProject"
            shutil.copytree(sample_src, proj)

            # 1. prep-scan
            rc, out = _run_cli(["prep-scan", "--project", str(proj), "--out-format", "json"])
            self.assertEqual(rc, 0)
            scan = json.loads(out)
            self.assertEqual(scan["total_items"], 26)
            self.assertEqual(scan["speakers"]["coverage_ratio"], 1.0)
            self.assertTrue(any(r["text"] == "セーブしますか？" and r["count"] == 3 for r in scan["repeats"]))
            self.assertTrue(any(t["token"] == "<size=30>" for t in scan["control_scan"]["top_tokens"]))

            # 2. dict & guideline CLI helpers
            rc, out = _run_cli(["dict", "--project", str(proj), "--kind", "gpt", "--action", "list", "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(json.loads(out)["count"], 10)
            rc, _ = _run_cli(["guideline", "--project", str(proj), "--action", "append", "--text", "- 测试追加条目。"])
            self.assertEqual(rc, 0)

            # 3. make-trial & batch-pull (--only-miss)
            rc, out = _run_cli(["make-trial", "--project", str(proj), "--limit", "10", "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["trial_count"], 10)

            rc, out = _run_cli(["batch-pull", "--project", str(proj), "--batch-size", "30", "--out-format", "json"])
            self.assertEqual(rc, 0)
            pulled = json.loads(out)
            self.assertEqual(pulled["selected_count"], 26)
            # Verify pre_dict ran: line 8 had 星見坂高校 -> 星見坂学園, line 20 had あなひゃ/しゅごひ -> あなた/すごい
            by_id_pull = {r["id"]: r for r in pulled["items"]}
            self.assertIn("星見坂学園", by_id_pull[8]["post_src"])
            self.assertIn("あなた", by_id_pull[20]["post_src"])
            self.assertIn("すごい", by_id_pull[20]["post_src"])
            glossary_text = "\n".join(pulled["active_glossary"])
            self.assertIn("星見坂学園 -> 星见坂学园", glossary_text)
            self.assertIn("銀狼 -> 银狼", glossary_text)
            self.assertNotIn("ゴブリン -> 哥布林", glossary_text)

            # 4. batch-push with intentional defects to verify instant QA feedback
            a1 = by_id_pull[1]["anchor"]
            patch_lines = [
                "*** Begin Patch",
                f"@@ {a1}|1", "+莲，早上好。",  # Trailing period trimmed + ♪ restored!
                "@@ chapter_01.json|2", "+喂喂，快迟到了啊（Late）。",  # Triggers 本无括号 + 引入英文
                "@@ chapter_01.json|3", "+因为昨晚一直在看『银狼』的新刊嘛",
                "@@ chapter_01.json|4", "+今天天气这么好",
                "@@ chapter_01.json|5", "+去天台吃午饭吧゛",  # Triggers 附着记号残留：゛
                "@@ chapter_01.json|6", "+哎呀，你们俩关系真好呢",
                "@@ chapter_01.json|7", "+我说，",
                "@@ chapter_01.json|8", "+星见坡学园的文学部，",  # post_dict turns 星见坡->星见坂 & 文学部->文艺部!
                "@@ chapter_01.json|9", "+要不要顺路去一趟？",
                "@@ chapter_01.json|10", "+真的吗！",  # Auto-completed to 真的吗！？
                "@@ chapter_01.json|11", "+在等吐槽吗！",  # Missing <size=30> </size> -> 缺控制符
                "@@ chapter_01.json|12", "+放学后的教室里，吉他的音色静静回荡着。",  # 吉他 in monologue MUST PASS!
                "@@ chapter_01.json|13", "+听说过<r沃尔夫>狼人</r>的传说吗？",
                "@@ chapter_01.json|14", "+[l]等一下，\\n我喘不过气了……。<br>",  # Extra hallucinated <br> -> 多控制符!
                "@@ chapter_01.json|15", "+一、二！",
                "@@ chapter_01.json|16", "+要保存吗？",
                "@@ chapter_01.json|17", "+要保存吗？",
                "@@ chapter_01.json|18", "+要保存吗？",
                "@@ chapter_01.json|19", "+阴道好烫……",  # Triggers 无依据器官词强补!
                "@@ chapter_01.json|20", "+啊、你……好厉害……",
                "@@ chapter_01.json|21", "+她是有夫之妇。",  # Conditional post_dict -> 她是人妻。
                "@@ chapter_01.json|22", "+传闻撒旦出现在了星见坡。",  # Conditional !サタン -> 魔王, 星见坡 -> 星见坂!
                "@@ chapter_01.json|23", "+青梅竹马拜托的话那也没办法",
                "@@ chapter_01.json|24", "+作为前辈，我会支持你们的",  # ♪ restored + outer 　「...」\n restored!
                "@@ chapter_01.json|25", "+风吹动了窗边的绝绝子窗帘。",  # Triggers 语域违和词：绝绝子
                "@@ chapter_01.json|26", "+明天见，就在星见坂碰头吧",
                "*** End Patch",
            ]
            p1 = proj / "work" / "patch1.txt"
            p1.parent.mkdir(parents=True, exist_ok=True)
            p1.write_text("\n".join(patch_lines), encoding="utf-8")
            rc, out = _run_cli(["batch-push", "--project", str(proj), "--patch-file", str(p1), "--out-format", "json"])
            self.assertEqual(rc, 0)
            res1 = json.loads(out)
            self.assertEqual(res1["applied_count"], 26)
            flagged = {r["id"]: r["problems"] for r in res1["problem_items"]}

            self.assertIn(2, flagged)
            self.assertIn("本无括号", flagged[2])
            self.assertTrue(any(p.startswith("引入英文") for p in flagged[2]))
            self.assertIn(5, flagged)
            self.assertTrue(any(p.startswith("附着记号残留") for p in flagged[5]))
            self.assertIn(11, flagged)
            self.assertTrue(any(p.startswith("缺控制符") for p in flagged[11]))
            self.assertNotIn(12, flagged)  # 吉他 passed cleanly!
            self.assertNotIn(13, flagged)  # <r沃尔夫>狼人</r> passed cleanly!
            self.assertIn(14, flagged)
            self.assertTrue(any(p.startswith("多控制符") and "<br>" in p for p in flagged[14]))
            self.assertIn(19, flagged)
            self.assertTrue(any(p.startswith("无依据器官词强补") for p in flagged[19]))
            self.assertIn(25, flagged)
            self.assertTrue(any(p.startswith("语域违和词") for p in flagged[25]))

            # 5. Corrective patch clears #2, #5, #11, #14, #19
            fix_patch = (
                "@@ chapter_01.json|2\n+喂喂，快迟到了啊\n"
                "@@ chapter_01.json|5\n+去天台吃午饭吧\n"
                "@@ chapter_01.json|11\n+在等<size=30>吐槽</size>吗！\n"
                "@@ chapter_01.json|14\n+[l]等一下，\\n我喘不过气了……\n"
                "@@ chapter_01.json|19\n+那里好烫……\n"
            )
            p2 = proj / "work" / "patch2.txt"
            p2.write_text(fix_patch, encoding="utf-8")
            rc, out = _run_cli(["batch-push", "--project", str(proj), "--patch-file", str(p2), "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["problem_item_count"], 0)

            # 6. Fix #25 by adding post_dict rule & running zero-cost `rebuild`!
            rc, _ = _run_cli(
                ["dict", "--project", str(proj), "--kind", "post", "--action", "add", "--src", "绝绝子窗帘", "--dst", "窗帘"]
            )
            self.assertEqual(rc, 0)
            rc, out = _run_cli(["rebuild", "--project", str(proj), "--out-format", "json"])
            self.assertEqual(rc, 0)
            reb = json.loads(out)
            self.assertEqual(reb["problems_before"], 1)
            self.assertEqual(reb["problems_after"], 0)
            self.assertEqual(reb["delta"], -1)
            self.assertEqual(reb["cache_misses"], 0)

            # 7. Test `edit-cache`, `search-replace`, and one-click `revert`!
            rc, out = _run_cli(
                [
                    "search-replace",
                    "--project", str(proj),
                    "--find", "要保存吗？",
                    "--replace", "是否保存进度？",
                    "--dry-run",
                    "--out-format", "json",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["matched_count"], 3)

            rc, out = _run_cli(
                [
                    "search-replace",
                    "--project", str(proj),
                    "--find", "要保存吗？",
                    "--replace", "是否保存进度？",
                    "--reason", "test-sr",
                    "--out-format", "json",
                ]
            )
            self.assertEqual(rc, 0)
            sr_task = json.loads(out)["task_id"]
            self.assertTrue(sr_task)
            out_after_sr = json.loads((proj / "gt_output" / "chapter_01.json").read_text(encoding="utf-8"))
            self.assertEqual(out_after_sr[15]["message"], "是否保存进度？")

            # Revert the search-replace task via cache_history.jsonl!
            rc, out = _run_cli(["revert", "--project", str(proj), "--task-id", sr_task, "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["reverted_count"], 3)
            out_after_rev = json.loads((proj / "gt_output" / "chapter_01.json").read_text(encoding="utf-8"))
            self.assertEqual(out_after_rev[15]["message"], "要保存吗？")

            # Test edit-cache on line 26 with proofread_dst
            rc, out = _run_cli(
                [
                    "edit-cache",
                    "--project", str(proj),
                    "--file", "chapter_01.json",
                    "--index", "26",
                    "--proofread-dst", "明天见，星见坂不见不散",
                    "--comment", "润色口语",
                    "--out-format", "json",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["edited_count"], 1)

            # 8. Seed ledger 2-round true-sampling gate
            rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "check-gate", "--out-format", "json"])
            self.assertEqual(rc, 1)

            # Round 1
            rc, out = _run_cli(["seed", "--project", str(proj), "--action", "sample", "--regions", "3", "--per-region", "3", "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["regions_sampled"], 3)
            rc, out = _run_cli(["seed", "--project", str(proj), "--action", "add", "--pattern", "核查星见坂译名一致", "--query", "星見坂", "--out-format", "json"])
            self.assertEqual(rc, 0)
            sid = json.loads(out)["id"]
            rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "consume", "--seed-id", sid, "--resolution", "已核查", "--complete-round", "--out-format", "json"])
            self.assertEqual(rc, 0)
            rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "check-gate", "--out-format", "json"])
            self.assertEqual(rc, 1)

            # Round 2
            rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "sample", "--regions", "3", "--per-region", "3", "--out-format", "json"])
            self.assertEqual(rc, 0)
            rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "complete-round", "--out-format", "json"])
            self.assertEqual(rc, 0)
            rc, out = _run_cli(["seed", "--project", str(proj), "--action", "check-gate", "--out-format", "json"])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(out)["gatePassed"])

            # 9. Final export & lossless verification on every special case!
            rc, out = _run_cli(["export", "--project", str(proj), "--out-format", "json"])
            self.assertEqual(rc, 0)
            final_rows = json.loads((proj / "gt_output" / "chapter_01.json").read_text(encoding="utf-8"))
            self.assertEqual(final_rows[0], {"name": "遥", "message": "「莲，早上好♪」", "voice": "ch01_001.ogg"})
            self.assertEqual(final_rows[3]["message"], "「今天天气这么好")
            self.assertEqual(final_rows[4], {"message": "去天台吃午饭吧」"})
            self.assertEqual(final_rows[6]["message"], "『我说，")
            self.assertEqual(final_rows[7], {"message": "星见坂学园的文艺部，"})
            self.assertEqual(final_rows[8], {"message": "要不要顺路去一趟？』"})
            self.assertEqual(final_rows[9]["message"], "「真的吗！？」")
            self.assertEqual(final_rows[10]["message"], "「在等<size=30>吐槽</size>吗！」")
            self.assertEqual(final_rows[11]["message"], "放学后的教室里，吉他的音色静静回荡着。")
            self.assertEqual(final_rows[12]["message"], "「听说过<r沃尔夫>狼人</r>的传说吗？」")
            self.assertEqual(final_rows[13]["message"], "「[l]等一下，\\n我喘不过气了……」")
            self.assertEqual(final_rows[20]["message"], "她是人妻。")
            self.assertEqual(final_rows[21]["message"], "传闻魔王出现在了星见坂。")
            self.assertEqual(final_rows[23]["message"], "　「作为前辈，我会支持你们的♪」\\n")
            self.assertEqual(final_rows[24]["message"], "风吹动了窗边的窗帘。")
            self.assertEqual(final_rows[25]["message"], "「明天见，星见坂不见不散」")


class TestV2EfficiencyAndTwoTierQA(unittest.TestCase):
    def _init_temp_project(self, root: Path, files_payload: dict[str, list[dict[str, str]]]) -> Path:
        proj = root / "proj_v2"
        src_dir = root / "raw_in"
        src_dir.mkdir(parents=True, exist_ok=True)
        for fname, rows in files_payload.items():
            (src_dir / fname).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        rc, _ = _run_cli(["init", "--project", str(proj), "--profile", "modern_acg", "--preserve", "base,kag"])
        self.assertEqual(rc, 0)
        rc, _ = _run_cli(["import", "--project", str(proj), "--format", "gt_json", "--input", str(src_dir)])
        self.assertEqual(rc, 0)
        return proj

    def test_v2_default_batch_size_120_and_warn_budget_scaling(self) -> None:
        cfg = ProjectConfig()
        self.assertEqual(cfg.batch_size, 120)
        self.assertAlmostEqual(cfg.max_warn_ratio, 0.02)
        self.assertEqual(cfg.compute_warn_budget(26), 5)  # min cap 5 for tiny projects
        self.assertEqual(cfg.compute_warn_budget(1000), 20)  # 2% of 1,000

    def test_v2_session_bible_deduplication_and_hash_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._init_temp_project(
                Path(tmp),
                {"ch01.json": [{"name": "遥", "message": "「おはよう」"}, {"name": "蓮", "message": "「ああ」"}]},
            )
            (proj / "project_guideline.md").write_text(
                "# 专属人物圣经\n- 遥：元气少女，句尾轻快。\n" * 15,
                encoding="utf-8",
            )

            # Without session_id -> always full guideline_anchor, classic table format
            legacy = pull_batch(proj, batch_size=1)
            self.assertFalse(legacy["bible_deduplicated"])
            self.assertFalse(legacy["compact"])
            self.assertIn("<project_bible>", legacy["markdown"])
            self.assertIn("| ANCHOR\\|ID | FILE | SPEAKER | POST_SRC |", legacy["markdown"])

            # Turn 1 with session_id -> full bible written to .sessions/worker_a.json + compact mode auto-enabled
            b1 = pull_batch(proj, batch_size=1, session_id="worker_a")
            self.assertFalse(b1["bible_deduplicated"])
            self.assertTrue(b1["compact"])
            self.assertIn("<project_bible>", b1["markdown"])
            self.assertTrue((proj / "work" / ".sessions" / "worker_a.json").exists())

            # Turn 2 with same session_id & unchanged bible -> 2-line reminder ONLY!
            b2 = pull_batch(proj, batch_size=1, offset=1, session_id="worker_a")
            self.assertTrue(b2["bible_deduplicated"])
            self.assertNotIn("<project_bible>", b2["markdown"])
            self.assertIn(f"Session: worker_a | BibleHash: {b1['bible_hash'][:8]}", b2["markdown"])
            self.assertLess(len(b2["markdown"]), len(b1["markdown"]) // 2)

            # Modify project_guideline.md -> hash changes -> full bible automatically re-emitted!
            (proj / "project_guideline.md").write_text("# 新版人物圣经\n- 新增设定补充。\n", encoding="utf-8")
            b3 = pull_batch(proj, batch_size=1, session_id="worker_a")
            self.assertFalse(b3["bible_deduplicated"])
            self.assertIn("<project_bible>", b3["markdown"])
            self.assertNotEqual(b3["bible_hash"], b1["bible_hash"])

            # --no-compact disables compact grouping even when --session is active
            rc, out = _run_cli(
                ["batch-pull", "--project", str(proj), "--session", "worker_a", "--no-compact", "--out-format", "json"]
            )
            self.assertEqual(rc, 0)
            res_nc = json.loads(out)
            self.assertTrue(res_nc["bible_deduplicated"])
            self.assertFalse(res_nc["compact"])
            self.assertIn("| ANCHOR\\|ID | FILE | SPEAKER | POST_SRC |", res_nc["markdown"])

    def test_v2_compact_grouped_rendering_and_patch_parser_pipe_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._init_temp_project(
                Path(tmp),
                {
                    "sc01.json": [
                        {"name": "遥", "message": "「一文目」"},
                        {"name": "蓮", "message": "「二文目」"},
                    ],
                    "sc02.json": [
                        {"message": "三文目の地の文。"},
                    ],
                },
            )
            # Translate sc01 line 1 first so context appears when pulling line 2
            push_batch(proj, "@@ file: sc01.json\n1|第一句")
            pulled = pull_batch(proj, compact=True, context_num=2)
            md = pulled["markdown"]
            self.assertIn("## @file: sc01.json", md)
            self.assertIn("## @file: sc02.json", md)
            self.assertIn("@@ file: <file_name>\n<id>|<纯净译文内核>", md)

            # Test parse_patch_text across stateful @@ file:, @file:, id|dst (with | inside dst!),
            # 3-digit hex anchor|id|dst (e.g. 402|12|...), and @@ id + +dst under current_file!
            raw_patch = "\n".join(
                [
                    "@@ file: sc01.json",
                    "2|第二句|带竖线分支A|分支B",
                    "120|第一百二十句|同样保留竖线",
                    "@file: sc02.json",
                    "402|1|全数字十六进制锚点|竖线保留",
                    "a1f|2|字母十六进制锚点",
                    "@@ 3",
                    "+多行传统兼容格式",
                ]
            )
            entries, errs = parse_patch_text(raw_patch)
            self.assertEqual(errs, [])
            self.assertEqual(len(entries), 5)
            self.assertEqual((entries[0].file_name, entries[0].index, entries[0].anchor, entries[0].dst),
                             ("sc01.json", 2, "", "第二句|带竖线分支A|分支B"))
            self.assertEqual((entries[1].file_name, entries[1].index, entries[1].anchor, entries[1].dst),
                             ("sc01.json", 120, "", "第一百二十句|同样保留竖线"))
            self.assertEqual((entries[2].file_name, entries[2].index, entries[2].anchor, entries[2].dst),
                             ("sc02.json", 1, "402", "全数字十六进制锚点|竖线保留"))
            self.assertEqual((entries[3].file_name, entries[3].index, entries[3].anchor, entries[3].dst),
                             ("sc02.json", 2, "a1f", "字母十六进制锚点"))
            self.assertEqual((entries[4].file_name, entries[4].index, entries[4].dst),
                             ("sc02.json", 3, "多行传统兼容格式"))

    def test_v2_pull_next_circuit_breaker_blocks_on_hard_error_and_advances_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = self._init_temp_project(
                root,
                {
                    "ch01.json": [
                        {"name": "遥", "message": "「<size=30>注目</size>して！」"},
                        {"message": "昨日駅の前で偶然出会った赤い帽子の少女が今日もそこにいた。"},
                        {"name": "蓮", "message": "「次のバッチの台詞だ」"},
                        {"name": "遥", "message": "「最後の台詞だよ」"},
                    ]
                },
            )
            # Pull batch 1 (size 2)
            rc, out0 = _run_cli(
                ["batch-pull", "--project", str(proj), "--session", "w1", "--batch-size", "2", "--out-format", "json"]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out0)["selected_count"], 2)

            # Push batch 1 with:
            # - item 1: HARD ERROR (missing <size=30></size> -> 缺控制符)
            # - item 2: SOFT WARN ONLY (单句过长 > 18 chars)
            bad_patch = root / "bad_patch.txt"
            bad_patch.write_text(
                "@@ file: ch01.json\n"
                "1|快看这边！\n"
                "2|昨天在车站前面偶然遇到的那个戴着红帽子的少女今天也依然站在那里。\n",
                encoding="utf-8",
            )
            rc, out_blocked = _run_cli(
                [
                    "batch-push",
                    "--project", str(proj),
                    "--patch-file", str(bad_patch),
                    "--session", "w1",
                    "--batch-size", "2",
                    "--no-refresh-output",
                    "--pull-next",
                    "--out-format", "json",
                ]
            )
            self.assertEqual(rc, 0)
            res_blocked = json.loads(out_blocked)
            self.assertEqual(res_blocked["hard_error_count"], 1)
            self.assertEqual(res_blocked["soft_warn_count"], 1)
            # Circuit-Breaker MUST block advancing to items 3 & 4 and emit ONLY failed hard-error item 1!
            self.assertEqual(res_blocked["pull_next_status"], "blocked_by_errors")
            repair_unit = res_blocked["next_batch"]
            self.assertEqual(repair_unit["mode"], "delta_repair")
            self.assertTrue(repair_unit["blocked"])
            self.assertEqual([it["id"] for it in repair_unit["items"]], [1])

            # Now submit repair for item 1 ONLY with --pull-next (item 2 still has soft warn 单句过长!)
            fix_patch = root / "fix_patch.txt"
            fix_patch.write_text("@@ file: ch01.json\n1|快<size=30>看这边</size>！\n", encoding="utf-8")
            rc, out_adv = _run_cli(
                [
                    "batch-push",
                    "--project", str(proj),
                    "--patch-file", str(fix_patch),
                    "--session", "w1",
                    "--batch-size", "2",
                    "--no-refresh-output",
                    "--pull-next",
                    "--out-format", "json",
                ]
            )
            self.assertEqual(rc, 0)
            res_adv = json.loads(out_adv)
            self.assertEqual(res_adv["hard_error_count"], 0)
            self.assertEqual(res_adv["pull_next_status"], "advanced")
            next_unit = res_adv["next_batch"]
            self.assertEqual(next_unit["mode"], "only_miss")
            self.assertTrue(next_unit["bible_deduplicated"])
            self.assertEqual([it["id"] for it in next_unit["items"]], [3, 4])

    def test_v2_o1_selective_file_loading_and_true_delta_relinting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files_payload = {
                f"ch_{i:02d}.json": [
                    {"name": "遥", "message": f"「第{i}章の一行目」"},
                    {"name": "蓮", "message": f"「第{i}章の二行目」"},
                ]
                for i in range(1, 11)
            }
            proj = self._init_temp_project(Path(tmp), files_payload)

            from core import cache_ledger as cl_mod

            loaded_rels: list[str] = []
            orig_load_input = cl_mod.load_input_items
            orig_check_probs = batcher_mod.check_item_problems
            linted_keys: list[str] = []

            def spy_load_input(p_dir: Path, cfg: ProjectConfig, rel_name: str):
                loaded_rels.append(rel_name)
                return orig_load_input(p_dir, cfg, rel_name)

            def spy_check_probs(item: SentenceItem, *args, **kwargs):
                linted_keys.append(item.ref_key)
                return orig_check_probs(item, *args, **kwargs)

            with patch.object(cl_mod, "load_input_items", side_effect=spy_load_input), patch.object(
                batcher_mod, "check_item_problems", side_effect=spy_check_probs
            ):
                res = push_batch(proj, "@@ file: ch_07.json\n1|第七章第一句", refresh_output=False)
                self.assertEqual(res["applied_count"], 1)
                self.assertEqual(loaded_rels, ["ch_07.json"])
                self.assertEqual(linted_keys, ["ch_07.json:1"])

                loaded_rels.clear()
                linted_keys.clear()
                res_ec = edit_cache_entries(
                    proj,
                    [{"file": "ch_03.json", "index": 2, "pre_dst": "第三章第二句"}],
                    refresh_output=False,
                )
                self.assertEqual(res_ec["edited_count"], 1)
                self.assertEqual(loaded_rels, ["ch_03.json"])
                self.assertEqual(linted_keys, ["ch_03.json:2"])

            # Fallback verification: bare @@ anchor|id without filename still loads all files and resolves!
            pulled_09 = pull_batch(proj, file_patterns=["ch_09.json"], batch_size=1)
            anc_09 = pulled_09["items"][0]["anchor"]
            res_bare = push_batch(proj, f"@@ {anc_09}|1\n+第九章锚点直推", refresh_output=False)
            self.assertEqual(res_bare["applied_count"], 1)
            self.assertEqual(res_bare["error_count"], 0)
            self.assertEqual(res_bare["touched_files"], ["ch_09.json"])

    def test_v2_single_kanji_name_compound_boundary_and_exclude_tag(self) -> None:
        gd = GptDict(
            [
                GptDictEntry("潤", "润", "男主角"),
                GptDictEntry("奏", "奏", "女主角"),
                GptDictEntry("光", "小光", "配角 [exclude:光線,栄光]"),
            ]
        )
        self.assertEqual(gd.get("光").exclude_if, ["光線", "栄光"])

        cfg = ProjectConfig()
        tp = TextPreserver.from_project(cfg)

        # 1. Explicit production false-positive traps MUST NOT trigger 字典未使用!
        non_name_cases = [
            ("潤沢な資金源はそこだろう。", "充裕的资金来源大概就在那里吧。"),
            ("湿潤な空気と莫大な利潤。", "潮湿的空气与巨额的利润。"),
            ("瞳が潤んだ。喉が潤う。", "眼眶湿了。喉咙得到了滋润。"),
            ("美しい旋律を奏でる。", "弹起优美的旋律。"),
            ("ピアノを演奏する。", "弹钢琴。"),
            ("眩しい光線が差し込む。", "耀眼的光线照了进来。"),
        ]
        for idx, (src, dst) in enumerate(non_name_cases, start=1):
            it = SentenceItem("k.json", idx, pre_src=src, post_src=src, pre_dst=dst, post_dst=dst)
            probs = check_item_problems(it, cfg, preserver=tp, gpt_dict=gd)
            self.assertFalse(
                any(p.startswith("字典未使用") for p in probs),
                f"Unexpected glossary false positive on {src!r}: {probs}",
            )
        # Also verify match_batch does NOT inject 潤 or 奏 when only 潤沢 / 奏でる appear
        batch_non_names = [
            SentenceItem("k.json", 1, pre_src="潤沢な資金源はそこだろう。", post_src="潤沢な資金源はそこだろう。"),
            SentenceItem("k.json", 2, pre_src="美しい旋律を奏でる。", post_src="美しい旋律を奏でる。"),
        ]
        matched_entries, _ = inject_glossary_for_batch(batch_non_names, gd)
        self.assertEqual(matched_entries, [])

        # 2. Standalone character occurrences & honorifics MUST still match and verify!
        it_jun1 = SentenceItem(
            "k.json", 10,
            pre_src="潤は静かに頷いた。", post_src="潤は静かに頷いた。",
            pre_dst="少年静静地点了点头。", post_dst="少年静静地点了点头。",
        )
        self.assertIn("字典未使用：潤->润", check_item_problems(it_jun1, cfg, preserver=tp, gpt_dict=gd))
        it_jun1_ok = SentenceItem(
            "k.json", 11,
            pre_src="潤は静かに頷いた。", post_src="潤は静かに頷いた。",
            pre_dst="润静静地点了点头。", post_dst="润静静地点了点头。",
        )
        self.assertEqual(check_item_problems(it_jun1_ok, cfg, preserver=tp, gpt_dict=gd), [])

        it_jun2 = SentenceItem(
            "k.json", 12,
            pre_src="「潤さん、おはよう。」", post_src="潤さん、おはよう。",
            pre_dst="先生，早上好。", post_dst="「先生，早上好。」",
        )
        self.assertIn("字典未使用：潤->润", check_item_problems(it_jun2, cfg, preserver=tp, gpt_dict=gd))

        it_jun_senpai = SentenceItem(
            "k.json", 13,
            pre_src="「潤先輩、お疲れ様！」", post_src="潤先輩、お疲れ様！",
            pre_dst="前辈，辛苦啦！", post_dst="「前辈，辛苦啦！」",
        )
        self.assertIn("字典未使用：潤->润", check_item_problems(it_jun_senpai, cfg, preserver=tp, gpt_dict=gd))

        # Mixed sentence containing BOTH excluded compound 光線 AND standalone 光は -> MUST still check 光!
        it_mixed = SentenceItem(
            "k.json", 14,
            pre_src="光線の中で光は笑った。", post_src="光線の中で光は笑った。",
            pre_dst="在光线之中少女笑了。", post_dst="在光线之中少女笑了。",
        )
        self.assertIn("字典未使用：光->小光", check_item_problems(it_mixed, cfg, preserver=tp, gpt_dict=gd))

    def test_v2_two_tier_qa_severity_and_cli_problems_filter(self) -> None:
        for t in HARD_ERROR_TYPES:
            self.assertEqual(problem_severity(f"{t}：detail"), "error")
        for t in SOFT_WARN_TYPES:
            self.assertEqual(problem_severity(f"{t}：detail"), "warn")
        self.assertEqual(problem_severity("字典未使用(说话人)：遥->遥"), "warn")
        self.assertEqual(problem_severity("本无♥译有♥"), "error")
        self.assertEqual(problem_severity("本有♪译无♪"), "error")

        errs, warns = split_problems_by_severity(
            ["缺控制符：<br>", "单句过长：平均分句22.0字(>18)", "本无括号", "比日文长：1.5倍(+5字)"]
        )
        self.assertEqual(errs, ["缺控制符：<br>", "本无括号"])
        self.assertEqual(warns, ["单句过长：平均分句22.0字(>18)", "比日文长：1.5倍(+5字)"])

        with tempfile.TemporaryDirectory() as tmp:
            proj = self._init_temp_project(
                Path(tmp),
                {
                    "c1.json": [
                        {"name": "遥", "message": "「おはよう」"},
                        {"message": "昨日駅の前で偶然出会った赤い帽子の少女が今日もそこにいた。"},
                    ]
                },
            )
            push_batch(
                proj,
                "@@ file: c1.json\n"
                "1|早上好（备注）\n"
                "2|昨天在车站前面偶然遇到的那个戴着红帽子的少女今天也依然站在那里。\n",
            )
            rc, out_err = _run_cli(["problems", "--project", str(proj), "--severity", "error", "--out-format", "json"])
            self.assertEqual(rc, 0)
            d_err = json.loads(out_err)
            self.assertEqual(d_err["problem_item_count"], 1)
            self.assertEqual(d_err["hard_error_items"], 1)
            self.assertEqual(d_err["soft_warn_items"], 1)
            self.assertEqual([r["id"] for r in d_err["items"]], [1])

            rc, out_warn = _run_cli(["problems", "--project", str(proj), "--severity", "warn", "--out-format", "json"])
            self.assertEqual(rc, 0)
            d_warn = json.loads(out_warn)
            self.assertEqual(d_warn["problem_item_count"], 1)
            self.assertEqual([r["id"] for r in d_warn["items"]], [2])

            rc, out_st = _run_cli(["status", "--project", str(proj), "--out-format", "json"])
            self.assertEqual(rc, 0)
            st = json.loads(out_st)
            self.assertEqual(st["hard_error_items"], 1)
            self.assertEqual(st["soft_warn_items"], 1)

    def test_v2_seed_gate_allows_minor_warns_and_blocks_when_warn_budget_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 10-item project -> default warn budget = max(5, int(10 * 0.02)) = 5
            rows = [
                {"message": f"第{idx}日、昨日駅の前で偶然出会った赤い帽子の少女が今日もそこにいた。"}
                for idx in range(1, 11)
            ]
            proj = self._init_temp_project(Path(tmp), {"ch01.json": rows})

            # 3 items with 单句过长 (warn), 7 items punctuated into short clauses (clean) -> 3 <= 5 budget!
            patch_lines = ["@@ file: ch01.json"]
            for idx in range(1, 11):
                if idx <= 3:
                    patch_lines.append(f"{idx}|第{idx}天在车站前面偶然遇到的那个戴着红帽子的少女今天也依然站在那里。")
                else:
                    patch_lines.append(f"{idx}|第{idx}天，在站前偶遇了红帽少女，她今天也在。")
            push_batch(proj, "\n".join(patch_lines))

            # Complete 2 real sampling rounds
            for _ in range(2):
                rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "sample", "--regions", "3", "--per-region", "1", "--out-format", "json"])
                self.assertEqual(rc, 0)
                rc, _ = _run_cli(["seed", "--project", str(proj), "--action", "complete-round", "--out-format", "json"])
                self.assertEqual(rc, 0)

            # 3 soft warnings <= budget 5 & 0 hard errors -> Gate MUST PASS!
            rc, out_pass = _run_cli(["seed", "--project", str(proj), "--action", "check-gate", "--out-format", "json"])
            self.assertEqual(rc, 0)
            gate_ok = json.loads(out_pass)
            self.assertTrue(gate_ok["gatePassed"])
            self.assertFalse(gate_ok["warnBudgetExceeded"])
            self.assertEqual(gate_ok["openErrorCount"], 0)
            self.assertEqual(gate_ok["openWarnCount"], 3)

            # Escalate soft warnings to 6 items (> budget 5) -> Gate MUST BLOCK with warnBudgetExceeded!
            over_patch = "\n".join(
                ["@@ file: ch01.json"]
                + [
                    f"{idx}|第{idx}天在车站前面偶然遇到的那个戴着红帽子的少女今天也依然站在那里。"
                    for idx in range(4, 7)
                ]
            )
            push_batch(proj, over_patch)
            rc_block, out_block = _run_cli(["seed", "--project", str(proj), "--action", "check-gate", "--out-format", "json"])
            self.assertEqual(rc_block, 1)
            gate_blocked = json.loads(out_block)
            self.assertFalse(gate_blocked["gatePassed"])
            self.assertTrue(gate_blocked["warnBudgetExceeded"])
            self.assertTrue(any("exceed warn budget" in r for r in gate_blocked["blockingReasons"]))

    def test_v2_sample_character_debuts_cli_and_pronoun_ending_miner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = self._init_temp_project(
                Path(tmp),
                {
                    "01_prologue.json": [
                        {"message": "礼拝堂の奥に、若いシスターが静かに立っていた。"},
                        {"name": "マリア", "message": "「わたくしはマリア。神の導きがあらんことをですわ。」"},
                        {"name": "潤", "message": "「俺は潤だ。よろしく頼むぜ、っと……いや、よろしくだぜ！」"},
                        {"name": "マリア", "message": "「まあ、元気な方ですわね。お茶はいかがかしら？」"},
                    ],
                    "02_school.json": [
                        {"message": "私立学園の廊下で後輩がオレンジジュースを片手に走ってきた。"},
                        {"name": "健太", "message": "「先輩！　ボクシング部の練習、マジきついっす！」"},
                        {"name": "凛", "message": "「あーしは平気だけどね！　ウチらの気合い見せてやるなの！」"},
                        {"name": "長老", "message": "「うむ、若いというのは良いことじゃ。」"},
                    ],
                },
            )
            rc, out_json = _run_cli(
                ["sample-debuts", "--project", str(proj), "--top", "10", "--window", "1", "--out-format", "json"]
            )
            self.assertEqual(rc, 0)
            report = json.loads(out_json)
            self.assertEqual(report["files_scanned"], 2)
            self.assertEqual(report["total_unique_speakers"], 5)
            by_spk = {c["speaker"]: c for c in report["characters"]}

            # Verify Maria's debut window includes the preceding narration ("若いシスター") & pronouns/endings
            maria = by_spk["マリア"]
            self.assertEqual(maria["debut_ref"], "01_prologue.json:2")
            self.assertEqual(maria["pronouns"], {"わたくし": 1})
            self.assertEqual(maria["sentence_endings"]["ですわ"], 2)
            self.assertEqual(maria["sentence_endings"]["かしら"], 1)
            self.assertTrue(any("シスター" in w["pre_src"] for w in maria["debut_window"]))

            # Verify Kenta: ボクシング must NOT trigger false-positive ボク, while っす IS counted!
            kenta = by_spk["健太"]
            self.assertNotIn("ボク", kenta["pronouns"])
            self.assertEqual(kenta["sentence_endings"], {"っす": 1})

            # Verify Rin & Elder
            self.assertEqual(by_spk["凛"]["pronouns"], {"あーし": 1, "ウチ": 1})
            self.assertEqual(by_spk["凛"]["sentence_endings"], {"なの": 1})
            self.assertEqual(by_spk["長老"]["sentence_endings"], {"じゃ": 1})

            # Verify Markdown output contains `index | speaker | pre_src`
            rc_md, out_md = _run_cli(["sample-debuts", "--project", str(proj), "--top", "5", "--window", "1"])
            self.assertEqual(rc_md, 0)
            self.assertIn("1 | - | 礼拝堂の奥に、若いシスターが静かに立っていた。", out_md)
            self.assertIn("> 2 | マリア |", out_md)

    def test_v2_cli_markdown_combo_output_and_status_formatting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = self._init_temp_project(
                root,
                {
                    "ep1.json": [
                        {"name": "遥", "message": "「一文目だよ」"},
                        {"name": "蓮", "message": "「二文目だな」"},
                    ]
                },
            )
            pfile = root / "p.txt"
            pfile.write_text("@@ file: ep1.json\n1|第一句哦\n", encoding="utf-8")
            rc, md_combo = _run_cli(
                [
                    "batch-push",
                    "--project", str(proj),
                    "--patch-file", str(pfile),
                    "--session", "s_md",
                    "--batch-size", "1",
                    "--pull-next",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertIn("# Batch Push Instant Lint Report", md_combo)
            self.assertIn("Pull-Next Status: `advanced`", md_combo)
            self.assertIn("# Work-Unit (1 items | remaining=0)", md_combo)
            self.assertIn("## @file: ep1.json", md_combo)

            rc_st, md_st = _run_cli(["status", "--project", str(proj)])
            self.assertEqual(rc_st, 0)
            self.assertIn("Open QA: **0** item(s) (**0** hard errors, **0** warnings", md_st)


class TestSecurityAndEdgeCaseAuditFixes(unittest.TestCase):
    """Regression suite pinning all C-1..C-4, H-1..H-7, M-1..M-5, and L-1..L-3 audit fixes."""

    def test_c1_windows_concurrent_cross_file_and_same_file_locking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Cross-file concurrent push_batch + pull_batch must never delete active temps or raise WinError
            proj = Path(tmp) / "conc_cross"
            cfg = ProjectConfig(project_name="conc_cross")
            cfg.save(proj)
            for fidx in range(4):
                fname = f"sc0{fidx}.json"
                save_input_file(
                    proj,
                    cfg,
                    fname,
                    [SentenceItem(fname, i, pre_src=f"「テスト{fidx}_{i}」", speaker=f"S{fidx}") for i in range(1, 41)],
                )

            errors: list[tuple[str, str, str]] = []

            def worker_push(fidx: int) -> None:
                fname = f"sc0{fidx}.json"
                for batch in range(4):
                    lines = [f"@@ file: {fname}"]
                    for i in range(batch * 10 + 1, (batch + 1) * 10 + 1):
                        lines.append(f"{i}|译文{fidx}_{i}")
                    try:
                        push_batch(proj, "\n".join(lines), refresh_output=False, session_id=f"w{fidx}")
                    except Exception as exc:
                        errors.append((f"push_{fidx}_{batch}", type(exc).__name__, str(exc)))

            def worker_pull(fidx: int) -> None:
                fname = f"sc0{fidx}.json"
                for _ in range(12):
                    try:
                        pull_batch(proj, file_patterns=[fname], session_id=f"p{fidx}")
                    except Exception as exc:
                        errors.append((f"pull_{fidx}", type(exc).__name__, str(exc)))

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(worker_push, i) for i in range(4)] + [ex.submit(worker_pull, i) for i in range(4)]
                concurrent.futures.wait(futs)

            self.assertEqual(errors, [])
            hist_lines = cfg.history_path(proj).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(hist_lines), 16)
            for ln in hist_lines:
                json.loads(ln)

            # 2) Same-file concurrent push_batch (disjoint index ranges) must suffer ZERO lost updates
            proj2 = Path(tmp) / "conc_same"
            cfg.save(proj2)
            save_input_file(
                proj2,
                cfg,
                "sc00.json",
                [SentenceItem("sc00.json", i, pre_src=f"「テスト_{i}」", speaker="遥") for i in range(1, 61)],
            )
            errors2: list[tuple[str, str, str]] = []

            def push_range(lo: int, hi: int, tag: str) -> None:
                for start in range(lo, hi, 5):
                    lines = ["@@ file: sc00.json"] + [
                        f"{i}|译文_{tag}_{i}" for i in range(start, min(start + 5, hi))
                    ]
                    try:
                        push_batch(proj2, "\n".join(lines), refresh_output=False)
                    except Exception as exc:
                        errors2.append((tag, type(exc).__name__, str(exc)))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(push_range, 1, 31, "A")
                f2 = ex.submit(push_range, 31, 61, "B")
                concurrent.futures.wait([f1, f2])

            self.assertEqual(errors2, [])
            loaded2 = load_project_files(proj2, cfg)[0][2]
            self.assertEqual(sum(1 for it in loaded2 if it.pre_dst), 60)

    def test_c2_rebuild_preserves_post_src_changed_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "p"
            cfg = ProjectConfig(project_name="c2")
            cfg.save(proj)
            save_input_file(proj, cfg, "ch01.json", [SentenceItem("ch01.json", 1, pre_src="「旧単語だ」", speaker="遥")])
            push_batch(proj, "@@ file: ch01.json\n1|旧词译文\n")

            (proj / "项目字典_译前.tsv").write_text("旧単語\t新単語\n", encoding="utf-8")
            reb = rebuild_project(proj, ignore_post_src=False)
            self.assertEqual(reb["cache_misses"], 1)
            self.assertEqual(reb["miss_breakdown"].get(MISS_POST_SRC_CHANGED), 1)

            # Subsequent batch-pull --only-miss MUST still see post_src_changed and pull the line!
            after_pull = pull_batch(proj, only_miss=True)
            self.assertEqual(after_pull["selected_count"], 1)
            self.assertEqual(after_pull["miss_breakdown"].get(MISS_POST_SRC_CHANGED), 1)

    def test_c3_consecutive_identical_lines_no_cache_key_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "p"
            cfg = ProjectConfig(project_name="c3")
            cfg.save(proj)
            items = [
                SentenceItem("ch01.json", i, pre_src="「……」", speaker="遥")
                for i in range(1, 5)
            ]
            save_input_file(proj, cfg, "ch01.json", items)
            push_batch(proj, "@@ file: ch01.json\n1|一\n2|二\n3|三\n4|四\n")
            loaded = load_project_files(proj, cfg)[0][2]
            self.assertEqual([(it.index, it.pre_dst) for it in loaded], [(1, "一"), (2, "二"), (3, "三"), (4, "四")])

    def test_c4_gbk_push_no_crash_speaker_gbk_check_and_clean_export_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "gbk_p"
            cfg = ProjectConfig(project_name="gbk_p", target_encoding="gbk", speaker_map={"遥・琴音": "遥・琴音"})
            cfg.save(proj)
            save_input_file(
                proj,
                cfg,
                "ch01.json",
                [
                    SentenceItem(
                        "ch01.json",
                        1,
                        pre_src="「おはよう♪」",
                        speaker="遥・琴音",
                        speakers_raw="遥・琴音",
                        extra={"_name_key": "name"},
                    )
                ],
            )
            # Must NOT crash with UnicodeEncodeError before returning Delta Repair!
            res = push_batch(proj, "@@ file: ch01.json\n1|早上好♪\n", pull_next=True)
            self.assertEqual(res["pull_next_status"], "blocked_by_errors")
            self.assertEqual(res["hard_error_count"], 1)
            probs = res["hard_error_items"][0]["problems"]
            self.assertTrue(any(p.startswith("非GBK字符：") for p in probs))
            self.assertTrue(any(p.startswith("非GBK字符(说话人)：") for p in probs))

            # Even with gbk_whitelist + --allow-errors, CLI export must return clean rc=1 instead of crashing
            cfg.gbk_whitelist = ["♪", "・"]
            cfg.save(proj)
            rc, out = _run_cli(["export", "--project", str(proj), "--allow-errors", "--out-format", "json"])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(out)["status"], "encoding_error")

    def test_h1_modern_acg_no_single_char_false_hard_errors(self) -> None:
        cfg = ProjectConfig(profile="modern_acg")
        prof = json.loads((FRAMEWORK_ROOT / "profiles" / "modern_acg.json").read_text(encoding="utf-8"))
        tp = TextPreserver.from_project(cfg)

        for sent in ("之前我也去过", "你也太在乎了吧", "我也不是故意的", "或者这样也行", "甚至连他也来了"):
            it = SentenceItem(
                "s.json", 1, pre_src="「テスト」", post_src="テスト",
                pre_dst=sent, post_dst=f"「{sent}」", speaker="遥", is_dialogue=True,
            )
            probs = check_item_problems(it, cfg, profile=prof, preserver=tp)
            self.assertFalse(any(p.startswith("语域违和词") for p in probs), f"False register flag on {sent!r}: {probs}")

        # Truly archaic dialogue still flags AND its severity is soft warn (not circuit-breaker hard error)
        it_arch = SentenceItem(
            "s.json", 2, pre_src="「吾が剣」", post_src="吾が剣",
            pre_dst="吾之剑锋亦乃天命", post_dst="「吾之剑锋亦乃天命」", speaker="剑圣", is_dialogue=True,
        )
        arch_probs = [p for p in check_item_problems(it_arch, cfg, profile=prof, preserver=tp) if p.startswith("语域违和词")]
        self.assertTrue(arch_probs)
        self.assertEqual(problem_severity(arch_probs[0]), "warn")
        self.assertEqual(problem_severity("语域违和词：绝绝子"), "error")

    def test_h2_compact_patch_braces_numeric_id_missing_plus_and_empty_dst(self) -> None:
        # 1) Brace-starting & brace-containing dst in grouped compact and bare compact
        e1, err1 = parse_patch_text("@@ file: ch01.json\n1|{player_name}，早上好\n2|{color=#fff}警告{/color}\n")
        self.assertEqual(err1, [])
        self.assertEqual([(e.index, e.dst) for e in e1], [(1, "{player_name}，早上好"), (2, "{color=#fff}警告{/color}")])

        e2, err2 = parse_patch_text("1|你好，{player_name}\n")
        self.assertEqual(err2, [])
        self.assertEqual([(e.index, e.dst) for e in e2], [(1, "你好，{player_name}")])

        # JSONL with numeric 3-char anchor prefix must still parse as JSONL
        e_jl, err_jl = parse_patch_text('123|{"file": "ch01.json", "id": 5, "dst": "你好"}\n')
        self.assertEqual(err_jl, [])
        self.assertEqual((e_jl[0].anchor, e_jl[0].index, e_jl[0].dst), ("123", 5, "你好"))

        # 2) Indented `+` accepted; missing `+` inside `@@` block produces explicit parse error
        e3, err3 = parse_patch_text("@@ ch01.json|1\n  +缩进译文\n@@ ch01.json|2\n忘了加号\n")
        self.assertEqual(e3[0].dst, "缩进译文")
        self.assertTrue(any("missing '+'" in err for err in err3))

        # 3) Empty dst `1|` rejected as error (not Clean!) & 3-digit ID `123|1|第一选项` disambiguated
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "h2"
            cfg = ProjectConfig(project_name="h2")
            cfg.save(proj)
            items = [SentenceItem("ch01.json", i, pre_src=f"「台詞{i}」", speaker="遥") for i in range(1, 125)]
            save_input_file(proj, cfg, "ch01.json", items)

            res_empty = push_batch(proj, "@@ file: ch01.json\n1|\n")
            self.assertEqual(res_empty["applied_count"], 0)
            self.assertEqual(res_empty["error_count"], 1)
            self.assertTrue(any("empty translation" in e for e in res_empty["errors"]))

            res_amb = push_batch(proj, "@@ file: ch01.json\n123|1|第一选项\n")
            self.assertEqual(res_amb["error_count"], 0)
            self.assertEqual(res_amb["applied_count"], 1)
            loaded = load_project_files(proj, cfg)[0][2]
            self.assertEqual(loaded[0].pre_dst, "")
            self.assertEqual(loaded[122].pre_dst, "1|第一选项")

    def test_h3_circuit_breaker_blocks_partial_repair_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "h3"
            cfg = ProjectConfig(project_name="h3", preserve_presets=["base", "kag"])
            cfg.save(proj)
            save_input_file(
                proj,
                cfg,
                "ch01.json",
                [
                    SentenceItem("ch01.json", 1, pre_src="「[l]一」", speaker="遥"),
                    SentenceItem("ch01.json", 2, pre_src="「[l]二」", speaker="遥"),
                    SentenceItem("ch01.json", 3, pre_src="「三」", speaker="遥"),
                ],
            )
            r1 = push_batch(proj, "@@ file: ch01.json\n1|一\n2|二\n", pull_next=True, session_id="w_cb", batch_size=2)
            self.assertEqual(r1["pull_next_status"], "blocked_by_errors")
            self.assertEqual([x["id"] for x in r1["next_batch"]["items"]], [1, 2])

            # Fixing ONLY #1 (omitting #2) MUST stay blocked and point to #2!
            r2 = push_batch(proj, "@@ file: ch01.json\n1|[l]一\n", pull_next=True, session_id="w_cb", batch_size=2)
            self.assertEqual(r2["pull_next_status"], "blocked_by_errors")
            self.assertEqual([x["id"] for x in r2["next_batch"]["items"]], [2])

            # Fixing #2 clears all blocking hard errors and advances to #3!
            r3 = push_batch(proj, "@@ file: ch01.json\n2|[l]二\n", pull_next=True, session_id="w_cb", batch_size=2)
            self.assertEqual(r3["pull_next_status"], "advanced")
            self.assertEqual([x["id"] for x in r3["next_batch"]["items"]], [3])

    def test_h4_builtin_dicts_safe_and_project_normal_dict_overrides_builtin(self) -> None:
        # 1) pre_h_phonetic.tsv is opt-in by default and safe on normal Japanese even when loaded
        self.assertNotIn("builtin:pre_h_phonetic.tsv", ProjectConfig().pre_dicts)
        h_dict = NormalDict()
        self.assertGreaterEqual(h_dict.load_file(FRAMEWORK_ROOT / "dicts" / "pre_h_phonetic.tsv"), 200)
        h_dict.sort_by_length()
        for jp in ("助けろ！", "開けろ！", "もらって嬉しい", "あきらめ、立ち上がる", "あおぐ", "こんにゃく", "ひょっこり"):
            it = SentenceItem("s.json", 1, pre_src=jp, post_src=jp)
            self.assertEqual(h_dict.apply(jp, it), jp)

        # 2) post_symbol_common.tsv preserves Chinese quotes, does NOT rewrite 撒旦->魔王, handles 《》+『』 together
        post_common = NormalDict()
        post_common.load_file(FRAMEWORK_ROOT / "dicts" / "post_symbol_common.tsv")
        post_common.sort_by_length()
        cfg = ProjectConfig()
        it_q = SentenceItem("s.json", 1, pre_src="「彼は天才だ」", pre_dst="他就是个“天才”", speaker="遥")
        strip_outer_shell([it_q])
        self.assertEqual(normalize_and_restore_dst(it_q, cfg, post_dict=post_common), "「他就是个“天才”」")
        it_sat = SentenceItem("s.json", 2, pre_src="「悪魔が来た」", pre_dst="撒旦来了", speaker="遥")
        strip_outer_shell([it_sat])
        self.assertEqual(normalize_and_restore_dst(it_sat, cfg, post_dict=post_common), "「撒旦来了」")
        it_both = SentenceItem("s.json", 3, pre_src="《章》と『名』", post_src="《章》と『名』", pre_dst="《章》和『名』")
        self.assertEqual(normalize_and_restore_dst(it_both, cfg, post_dict=post_common), "《章》和『名』")

        # 3) gpt_common_acg.tsv allows 学长/学姐/学弟/学妹
        gd = GptDict()
        gd.load_file(FRAMEWORK_ROOT / "dicts" / "gpt_common_acg.tsv")
        it_sp = SentenceItem("s.json", 4, pre_src="「先輩、おはよう」", post_src="先輩、おはよう", pre_dst="学长，早上好", post_dst="「学长，早上好」", speaker="遥")
        self.assertFalse(any(p.startswith("字典未使用") for p in check_item_problems(it_sp, cfg, gpt_dict=gd)))

        # 4) Project NormalDict overrides lower-priority builtin rule of equal search key
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "nd_ov"
            cfg_ov = ProjectConfig(project_name="nd_ov")
            cfg_ov.save(proj)
            (proj / "项目字典_译后.tsv").write_text("~\t~\n", encoding="utf-8")
            _, _, loaded_post = load_project_dicts(cfg_ov, proj)
            it_ov = SentenceItem("s.json", 1, pre_src="a~b", post_src="a~b", pre_dst="a~b")
            self.assertEqual(normalize_and_restore_dst(it_ov, cfg_ov, post_dict=loaded_post), "a~b")

    def test_h5_noncharacter_randomized_sentinels_immune_to_post_dict_and_exhibit_pua_gaiji(self) -> None:
        from core.text_preserve import PLACEHOLDER_RE, _SENTINEL_KEY, _make_sentinel_placeholder

        # Sentinel must strictly reside in Unicode permanent noncharacters U+FDD0..U+FDEF with 4-nibble key
        sample_ph = _make_sentinel_placeholder(0)
        self.assertTrue(all(0xFDD0 <= ord(ch) <= 0xFDEF for ch in sample_ph))
        self.assertEqual(len(_SENTINEL_KEY), 4)
        self.assertIsNotNone(PLACEHOLDER_RE.fullmatch(sample_ph))
        # Forged sentinel with wrong key or BMP PUA Gaiji (e.g. ExHIBIT U+E08A, U+E000..U+E01F) must NOT match
        forged_nc = "\ufdd0" + ("\ufde0" if _SENTINEL_KEY[0] != "\ufde0" else "\ufde1") + _SENTINEL_KEY[1:] + "\ufde0\ufdd1"
        self.assertIsNone(PLACEHOLDER_RE.search(forged_nc))
        self.assertIsNone(PLACEHOLDER_RE.search("\ue000\ue010\ue001\ue08a"))

        cfg = ProjectConfig(preserve_presets=["base", "kag"])
        tp = TextPreserver.from_project(cfg)
        pd = NormalDict()
        pd.load_tsv_text("TP\t技力\n")
        # ExHIBIT PUA Gaiji U+E08A + legacy PUA U+E000 sequence inside prose must survive intact alongside tags
        it = SentenceItem(
            "s.json",
            1,
            pre_src="「<color=#fff>テスト</color>\ue08a\ue000\ue010\ue001[l]」",
            pre_dst="<color=#fff>TP回复</color>\ue08a\ue000\ue010\ue001。[l]",
            speaker="遥",
        )
        strip_outer_shell([it])
        self.assertEqual(
            normalize_and_restore_dst(it, cfg, post_dict=pd, preserver=tp),
            "「<color=#fff>技力回复</color>\ue08a\ue000\ue010\ue001[l]」",
        )

    def test_h6_presets_and_ruby_kana_leak_detection(self) -> None:
        tp_base = TextPreserver.from_project(ProjectConfig(preserve_presets=["base"]))
        self.assertTrue(tp_base.verify_preserved_detailed("50%OFF", "半价").ok)
        self.assertTrue(tp_base.verify_preserved_detailed("100%Clear", "全通关").ok)
        self.assertFalse(tp_base.verify_preserved_detailed("%sさん", "先生").ok)

        tp_kag = TextPreserver.from_project(ProjectConfig(preserve_presets=["base", "kag"], ruby_preserve_mode="translate"))
        self.assertTrue(tp_kag.verify_preserved_detailed('[ruby text="magic"]魔法', '[ruby text="魔法"]魔法').ok)

        tp_rm = TextPreserver.from_project(ProjectConfig(preserve_presets=["base", "rpgmaker"]))
        self.assertTrue(tp_rm.verify_preserved_detailed("hello\\nworld", "你好\\n世界").ok)
        tp_wolf = TextPreserver.from_project(ProjectConfig(preserve_presets=["base", "wolf"]))
        self.assertTrue(tp_wolf.verify_preserved_detailed("hello\\nworld", "你好\\n世界").ok)

        # Ruby kana leaks caught in translate mode; M2 emphasis dot `[・]` allowed
        cfg_all = ProjectConfig(preserve_presets=["base", "kag", "rpgmaker", "scn_m2"], ruby_preserve_mode="translate")
        tp_all = TextPreserver.from_project(cfg_all)
        for src, dst in (
            ("<rしんじつ>真実</r>を知る", "知晓<rしんじつ>真实</r>"),
            ('[ruby text="しんじつ"]真実', '[ruby text="しんじつ"]真实'),
            ("これは\\r[真実,しんじつ]だ", "这就是\\r[真実,しんじつ]"),
            ("「[ルビ,1]の力だ」", "「[ルビ,1]的力量」"),
        ):
            it = SentenceItem("s.json", 1, pre_src=src, post_src=src, pre_dst=dst, post_dst=dst)
            probs = check_item_problems(it, cfg_all, preserver=tp_all)
            self.assertTrue(any(p.startswith("残留日文") for p in probs), f"Missed ruby kana leak on {dst!r}: {probs}")

        it_dot = SentenceItem("s.json", 2, pre_src="「[・]重点」", post_src="[・]重点", pre_dst="[・]重点", post_dst="「[・]重点」")
        self.assertFalse(any("残留" in p for p in check_item_problems(it_dot, cfg_all, preserver=tp_all)))

    def test_h7_compact_pipe_edge_cases_and_br_literal_newline_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sc.txt"
            p.write_text(
                "0001|ちょっと待って\\|すぐ行く\n"
                "0002|[遥]\"待って\\|すぐ行く\"\n"
                "0003|[l]「おはよう」\n",
                encoding="utf-8",
            )
            _, _, its = compact_pipe.import_file(p, "sc.txt")
            self.assertEqual((its[0].speaker, its[0].pre_src, its[0].speaker_style), ("", "ちょっと待って\\|すぐ行く", "plain"))
            self.assertEqual((its[1].speaker, its[1].pre_src, its[1].speaker_style), ("遥", "\"待って\\|すぐ行く\"", "bracket"))
            self.assertEqual((its[2].speaker, its[2].pre_src, its[2].speaker_style), ("", "[l]「おはよう」", "plain"))

            p_brk = Path(tmp) / "brk.txt"
            p_brk.write_text("0004|[遥]……\n", encoding="utf-8")
            _, _, its_b = compact_pipe.import_file(p_brk, "brk.txt", pipe_style="bracket")
            self.assertEqual((its_b[0].speaker, its_b[0].pre_src, its_b[0].speaker_style), ("遥", "……", "bracket"))

            # <br> converts to literal \\n when post_src uses \\n and dst has no \\n
            proj = Path(tmp) / "p_br"
            cfg = ProjectConfig(project_name="p_br", preserve_presets=["base", "kag"])
            cfg.save(proj)
            save_input_file(proj, cfg, "ch01.json", [SentenceItem("ch01.json", 1, pre_src="「[l]ちょっと待って、\\n息が切れた……」", speaker="遥")])
            r_ok = push_batch(proj, "@@ file: ch01.json\n1|[l]等一下，<br>喘不过气了……\n")
            self.assertEqual(r_ok["hard_error_count"], 0)
            self.assertEqual(load_project_files(proj, cfg)[0][2][0].post_dst, "「[l]等一下，\\n喘不过气了……」")

        # Prompt rendering fixes mixed CRLF+LF and avoids corrupting `|` in compact mode
        it_mix = SentenceItem("s.json", 1, pre_src="一行目\r\n二行目\n三行目", post_src="一行目\r\n二行目\n三行目")
        self.assertEqual(_render_src_for_prompt(it_mix), "一行目<br>二行目<br>三行目")
        it_pipe = SentenceItem("s.json", 2, pre_src="待って\\|すぐ行く", post_src="待って\\|すぐ行く")
        self.assertEqual(_render_src_for_prompt(it_pipe, escape_pipe=False), "待って\\|すぐ行く")

    def test_m1_seed_gate_requires_per_round_sampling_and_export_blocks_hard_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sl = SeedLedger(Path(tmp) / "seeds.json", min_required_rounds=2)
            dummy = [
                SentenceItem("c.json", i, pre_src=f"s{i}", post_src=f"s{i}", pre_dst=f"d{i}", post_dst=f"d{i}", status="PROCESSED")
                for i in range(1, 31)
            ]
            # 2 samples BOTH in Round 1 + 0 samples in Round 2 -> MUST FAIL check_gate!
            sl.sample_dispersed(dummy)
            sl.sample_dispersed(dummy)
            sl.complete_round()
            sl.complete_round()
            passed, reasons = sl.check_gate(open_error_count=0, open_warn_count=0, max_warn_budget=5)
            self.assertFalse(passed)
            self.assertTrue(any("round(s): [2]" in r for r in reasons))

            # CLI export blocks hard errors unless --allow-errors is supplied
            proj = Path(tmp) / "p_exp"
            cfg = ProjectConfig(project_name="p_exp", preserve_presets=["base", "kag"])
            cfg.save(proj)
            save_input_file(proj, cfg, "ch01.json", [SentenceItem("ch01.json", 1, pre_src="「[l]おはよう」", speaker="遥")])
            push_batch(proj, "@@ file: ch01.json\n1|早上好\n", refresh_output=False)
            rc_blk, out_blk = _run_cli(["export", "--project", str(proj), "--out-format", "json"])
            self.assertEqual(rc_blk, 1)
            self.assertEqual(json.loads(out_blk)["status"], "blocked")
            rc_ok, _ = _run_cli(["export", "--project", str(proj), "--allow-errors", "--out-format", "json"])
            self.assertEqual(rc_ok, 0)

    def test_m2_dictionary_conditional_exclude_and_single_kanji_boundaries(self) -> None:
        gd = GptDict(
            [
                GptDictEntry("潤", "润"),
                GptDictEntry("遥", "遥"),
                GptDictEntry("光", "光"),
                GptDictEntry("星見坂駅前", "星见坂站前", info="[条件]仅指车站"),
                GptDictEntry("星見坂", "星见坂"),
                GptDictEntry("星見坂学園", "星见坂学园", info="[exclude:星見坂学園前]"),
                GptDictEntry("学園前", "学园前"),
            ]
        )
        # Non-name okurigana stems must NOT false-positive
        for src, dst in (
            ("瞳が潤っている", "眼眶泛起水雾"),
            ("遥かな空を見上げた", "望着远方的天空"),
            ("星が光らない夜", "星辰黯淡的夜晚"),
        ):
            it = SentenceItem("s.json", 1, pre_src=src, post_src=src, pre_dst=dst, post_dst=dst)
            self.assertEqual(gd.check_usage(it, dst), [])

        # Quotative `潤って優しい` and question `遥か？` MUST still verify character name!
        it_q1 = SentenceItem("s.json", 2, pre_src="潤って優しいよね", post_src="潤って優しいよね", pre_dst="他真温柔呢", post_dst="他真温柔呢")
        self.assertIn("字典未使用：潤->润", gd.check_usage(it_q1, it_q1.post_dst))

        # Conditional longer term must NOT shadow unconditional shorter term 星見坂
        it_cond = SentenceItem("s.json", 3, pre_src="星見坂駅前で待つ", post_src="星見坂駅前で待つ", pre_dst="在车站前等你", post_dst="在车站前等你")
        self.assertIn("字典未使用：星見坂->星见坂", gd.check_usage(it_cond, it_cond.post_dst))

        # Excluded span of longer term must NOT be masked away from shorter term 学園前
        it_ex = SentenceItem("s.json", 4, pre_src="星見坂学園前と星見坂学園", post_src="星見坂学園前と星見坂学園", pre_dst="在站前和星见坂学园", post_dst="在站前和星见坂学园")
        self.assertIn("字典未使用：学園前->学园前", gd.check_usage(it_ex, it_ex.post_dst))

    def test_m3_m4_m5_revert_order_edit_cache_search_replace_and_gt_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # M-3: duplicate index in same batch reverts cleanly in reverse order
            proj = Path(tmp) / "m34"
            cfg = ProjectConfig(project_name="m34")
            cfg.save(proj)
            save_input_file(proj, cfg, "sub/ch01.json", [SentenceItem("sub/ch01.json", 1, pre_src="「一」", speaker="遥")])
            push_batch(proj, "@@ file: sub/ch01.json\n1|原始初译\n")
            r_dup = push_batch(proj, "@@ file: sub/ch01.json\n1|错误A\n1|错误B\n")
            revert_task(proj, r_dup["task_id"])
            self.assertEqual(load_project_files(proj, cfg)[0][2][0].pre_dst, "原始初译")

            # M-4: Windows backslash path in edit_cache + status reconciliation + strict field in search_replace
            push_batch(proj, "@@ file: sub/ch01.json\n1|旧词精修\n", proofread=True)
            res_win = edit_cache_entries(proj, [{"file": r"sub\ch01.json", "index": 1, "pre_dst": "旧词初译"}])
            self.assertEqual(res_win["error_count"], 0)
            it_mid = load_project_files(proj, cfg)[0][2][0]
            self.assertEqual(it_mid.status, ItemStatus.PROOFREAD.value)
            self.assertEqual(it_mid.pre_dst, "旧词初译")
            self.assertEqual(it_mid.proofread_dst, "旧词精修")

            search_replace_project(proj, "旧词", "新词", field="pre_dst")
            it_sr = load_project_files(proj, cfg)[0][2][0]
            self.assertEqual(it_sr.pre_dst, "新词初译")
            self.assertEqual(it_sr.proofread_dst, "旧词精修")

            # Clearing proofread_dst drops status cleanly back to PROCESSED
            edit_cache_entries(proj, [{"file": "sub/ch01.json", "index": 1, "proofread_dst": ""}])
            it_clr = load_project_files(proj, cfg)[0][2][0]
            self.assertEqual(it_clr.status, ItemStatus.PROCESSED.value)
            self.assertEqual(it_clr.post_dst, "「新词初译」")

            # M-5: gt_json lossless null, _meta/raw_id/speaker_style preservation, and multi-speaker override
            gj = Path(tmp) / "g.json"
            gj.write_text(
                json.dumps(
                    [
                        {"name": None, "message": None, "raw_id": "ORIG_1", "speaker_style": "custom", "_meta": {"keep": 1}},
                        {"names": ["遥", "琴音"], "message": "「せーの」"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            _, _, gitems = gt_json.import_file(gj, "g.json")
            self.assertEqual((gitems[0].speaker, gitems[0].speakers_raw), ("", None))
            proj_g = Path(tmp) / "pg"
            cfg.save(proj_g)
            save_input_file(proj_g, cfg, "g.json", gitems)
            loaded_g = load_project_files(proj_g, cfg)[0][2]
            loaded_g[1].speaker_dst = "遥、琴音"
            out_g = Path(tmp) / "out_g.json"
            gt_json.export_file(out_g, loaded_g, {}, speaker_map={})
            exported = json.loads(out_g.read_text(encoding="utf-8"))
            self.assertIsNone(exported[0]["name"])
            self.assertIsNone(exported[0]["message"])
            self.assertEqual(exported[0]["raw_id"], "ORIG_1")
            self.assertEqual(exported[0]["speaker_style"], "custom")
            self.assertEqual(exported[0]["_meta"], {"keep": 1})
            self.assertEqual(exported[1]["names"], ["遥", "琴音"])

    def test_l1_l2_l3_regex_trailing_control_and_speaker_post_dict(self) -> None:
        # L-1: standalone \p{Script=Han}, uppercase <R...> envelope, non-empty zero-len rejection
        self.assertEqual(normalize_regex_pattern(r"\p{Script=Han}+"), r"[\u4e00-\u9fff\u3400-\u4dbf]+")
        self.assertEqual(normalize_regex_pattern(r"[^\p{Script=Han}]"), r"[^\u4e00-\u9fff\u3400-\u4dbf]")
        tp = TextPreserver.from_project(ProjectConfig(preserve_presets=["base"]))
        self.assertTrue(tp.verify_preserved_detailed("<Rしんじつ>真実</r>", "<r真实>真实</r>", ruby_mode="translate").ok)
        self.assertFalse(tp.add_rule(PreserveRule(src=r"(?=a)", info="lookahead")))
        self.assertFalse(tp.add_rule(PreserveRule(src=r"\b", info="word-boundary")))

        # L-2: trailing control code still aligns trailing period and ♪
        cfg_kag = ProjectConfig(preserve_presets=["base", "kag"])
        tp_kag = TextPreserver.from_project(cfg_kag)
        it_l = SentenceItem("s.json", 1, pre_src="「おはよう[l]」", speaker="遥")
        strip_outer_shell([it_l])
        it_l.pre_dst = "早上好。[l]"
        self.assertEqual(normalize_and_restore_dst(it_l, cfg_kag, preserver=tp_kag), "「早上好[l]」")
        it_n = SentenceItem("s.json", 2, pre_src="「行こう♪[l]」", speaker="遥")
        strip_outer_shell([it_n])
        it_n.pre_dst = "走吧[l]"
        self.assertEqual(normalize_and_restore_dst(it_n, cfg_kag, preserver=tp_kag), "「走吧♪[l]」")

        # L-3: use_post_dict_in_name active & rebuild clears stale speaker_dst removed from speaker_map
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "l3"
            cfg = ProjectConfig(project_name="l3", use_post_dict_in_name=True, speaker_map={"遥": "旧遥"})
            cfg.save(proj)
            (proj / "项目字典_译后.tsv").write_text("旧遥\t新遥\n", encoding="utf-8")
            save_input_file(proj, cfg, "ch01.json", [SentenceItem("ch01.json", 1, pre_src="「一」", speaker="遥")])
            push_batch(proj, "@@ file: ch01.json\n1|一\n")
            self.assertEqual(load_project_files(proj, cfg)[0][2][0].speaker_dst, "新遥")

            # Remove from speaker_map & disable use_post_dict_in_name -> rebuild clears stale speaker_dst
            cfg.speaker_map = {}
            cfg.use_post_dict_in_name = False
            cfg.save(proj)
            rebuild_project(proj)
            self.assertEqual(load_project_files(proj, cfg)[0][2][0].speaker_dst, "")


if __name__ == "__main__":
    unittest.main()
