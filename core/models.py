"""Universal Intermediate Representation (CSentense / SentenceItem) and ProjectConfig."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def get_framework_root() -> Path:
    """Return the repository root directory of gal_framework."""
    return Path(__file__).resolve().parent.parent


def compute_anchor(file_name: str, index: int, pre_src: str) -> str:
    """Compute a deterministic 3-character lowercase hex anchor for an item."""
    payload = f"{file_name}|{index}|{pre_src}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:3]


class ItemStatus(str, Enum):
    """Lifecycle status of a localization entry."""

    NONE = "NONE"
    PROCESSED = "PROCESSED"
    PROOFREAD = "PROOFREAD"
    EXCLUDED = "EXCLUDED"
    RULE_SKIPPED = "RULE_SKIPPED"
    UNDECIDED = "UNDECIDED"


@dataclass
class SentenceItem:
    """Unified sentence/entry IR (CSentense) supporting three-separation caching and QA."""

    file_name: str
    index: int
    pre_src: str = ""
    post_src: str = ""
    pre_dst: str = ""
    proofread_dst: str = ""
    post_dst: str = ""

    speaker: str = ""
    speakers_raw: Any = None
    speaker_dst: str = ""
    speaker_style: str = ""
    raw_id: str = ""
    anchor: str = ""

    is_dialogue: bool = False
    has_diag_symbol: bool = False
    shell_stripped: bool = False
    left_ws: str = ""
    right_ws: str = ""
    left_quote: str = ""
    right_quote: str = ""
    linebreak_symbol: str = ""

    status: str = ItemStatus.NONE.value
    problems: list[str] = field(default_factory=list)
    skip_check: bool = False
    trans_by: str = ""
    proofread_by: str = ""
    proofread_comment: str = ""
    cache_miss_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.post_src and self.pre_src:
            self.post_src = self.pre_src
        if not self.anchor:
            self.anchor = compute_anchor(self.file_name, self.index, self.pre_src)
        if self.speaker and not self.is_dialogue:
            self.is_dialogue = True

    # Compatibility aliases with GalTransl CSentense naming
    @property
    def pre_jp(self) -> str:
        return self.pre_src

    @property
    def post_jp(self) -> str:
        return self.post_src

    @post_jp.setter
    def post_jp(self, value: str) -> None:
        self.post_src = value

    @property
    def pre_zh(self) -> str:
        return self.pre_dst

    @pre_zh.setter
    def pre_zh(self, value: str) -> None:
        self.pre_dst = value

    @property
    def proofread_zh(self) -> str:
        return self.proofread_dst

    @proofread_zh.setter
    def proofread_zh(self, value: str) -> None:
        self.proofread_dst = value

    @property
    def post_zh(self) -> str:
        return self.post_dst

    @post_zh.setter
    def post_zh(self, value: str) -> None:
        self.post_dst = value

    @property
    def effective_core_dst(self) -> str:
        """Return the active bare translation (proofread takes priority over initial draft)."""
        return self.proofread_dst if self.proofread_dst != "" else self.pre_dst

    @property
    def ref_key(self) -> str:
        """Canonical file:index reference key used in whitelists and reports."""
        return f"{self.file_name}:{self.index}"

    @property
    def patch_key(self) -> str:
        """Canonical file|index patch key."""
        return f"{self.file_name}|{self.index}"

    @property
    def anchor_id(self) -> str:
        """Compact anchor|id token used in work units."""
        return f"{self.anchor}|{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "index": self.index,
            "raw_id": self.raw_id,
            "anchor": self.anchor,
            "pre_src": self.pre_src,
            "post_src": self.post_src,
            "pre_dst": self.pre_dst,
            "proofread_dst": self.proofread_dst,
            "post_dst": self.post_dst,
            "speaker": self.speaker,
            "speakers_raw": self.speakers_raw,
            "speaker_dst": self.speaker_dst,
            "speaker_style": self.speaker_style,
            "is_dialogue": self.is_dialogue,
            "has_diag_symbol": self.has_diag_symbol,
            "shell_stripped": self.shell_stripped,
            "left_ws": self.left_ws,
            "right_ws": self.right_ws,
            "left_quote": self.left_quote,
            "right_quote": self.right_quote,
            "linebreak_symbol": self.linebreak_symbol,
            "status": self.status,
            "problems": list(self.problems),
            "skip_check": self.skip_check,
            "trans_by": self.trans_by,
            "proofread_by": self.proofread_by,
            "proofread_comment": self.proofread_comment,
            "cache_miss_reason": self.cache_miss_reason,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceItem":
        return cls(
            file_name=str(data.get("file_name", "")),
            index=int(data.get("index", 0)),
            raw_id=str(data.get("raw_id", "")),
            anchor=str(data.get("anchor", "")),
            pre_src=str(data.get("pre_src", data.get("pre_jp", ""))),
            post_src=str(data.get("post_src", data.get("post_jp", ""))),
            pre_dst=str(data.get("pre_dst", data.get("pre_zh", ""))),
            proofread_dst=str(data.get("proofread_dst", data.get("proofread_zh", ""))),
            post_dst=str(data.get("post_dst", data.get("post_dst_preview", data.get("post_zh_preview", "")))),
            speaker=str(data.get("speaker", data.get("name", ""))),
            speakers_raw=data.get("speakers_raw"),
            speaker_dst=str(data.get("speaker_dst", data.get("name_preview", ""))),
            speaker_style=str(data.get("speaker_style", "")),
            is_dialogue=bool(data.get("is_dialogue", False)),
            has_diag_symbol=bool(data.get("has_diag_symbol", False)),
            shell_stripped=bool(data.get("shell_stripped", False)),
            left_ws=str(data.get("left_ws", "")),
            right_ws=str(data.get("right_ws", "")),
            left_quote=str(data.get("left_quote", "")),
            right_quote=str(data.get("right_quote", "")),
            linebreak_symbol=str(data.get("linebreak_symbol", "")),
            status=str(data.get("status", ItemStatus.NONE.value)),
            problems=list(data.get("problems", [])),
            skip_check=bool(data.get("skip_check", False)),
            trans_by=str(data.get("trans_by", "")),
            proofread_by=str(data.get("proofread_by", "")),
            proofread_comment=str(data.get("proofread_comment", data.get("doub_content", ""))),
            cache_miss_reason=str(data.get("cache_miss_reason", "")),
            extra=dict(data.get("extra", {})),
        )


# Explicit GalTransl-style alias
CSentense = SentenceItem

DEFAULT_ENGLISH_WHITELIST: list[str] = [
    "HP", "MP", "SP", "EXP", "LV", "Lv", "AP", "EP", "TP", "LP", "BP",
    "OK", "NG", "CG", "BGM", "SE", "UI", "UX", "NPC", "RPG", "AVG", "SLG",
    "FPS", "TPS", "PVP", "PVE", "PV", "OP", "ED", "CM", "TV", "PC", "CD",
    "DVD", "BD", "SNS", "APP", "URL", "ID", "VIP", "MVP", "SOS", "DNA",
    "AI", "VR", "AR", "SM", "VS", "NO", "YES", "ON", "OFF", "SAVE", "LOAD",
    "AUTO", "SKIP", "LOG", "END", "FIN", "NEXT", "BACK", "MENU", "HELP",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


@dataclass
class ProjectConfig:
    """Configuration stored in <project_dir>/config.json (with project.json read compat)."""

    project_name: str = "untitled"
    profile: str = "modern_acg"
    guideline: str = "zh_light_novel_v2.md"
    project_guideline: str = "project_guideline.md"
    charter: str = "00_agent_charter.md"
    target_lang: str = "zh-cn"
    target_encoding: str = "utf8"
    source_format: str = "gt_json"

    input_dir: str = "gt_input"
    cache_dir: str = "transl_cache"
    output_dir: str = "gt_output"
    history_file: str = "cache_history.jsonl"

    preserve_presets: list[str] = field(default_factory=lambda: ["base"])
    custom_preserve_rules: list[dict[str, Any]] = field(default_factory=list)
    ruby_preserve_mode: str = "translate"  # "translate" | "verbatim"
    use_placeholder_protection: bool = False

    strip_outer_quotes: bool = True
    align_trailing_period: bool = True
    extra_linebreak_policy: str = "warn"  # "warn" | "trim"
    linebreak_symbol: str = "auto"

    sort_dicts: bool = True
    use_pre_dict_in_name: bool = False
    use_gpt_dict_in_name: bool = True
    use_post_dict_in_name: bool = False
    pre_dicts: list[str] = field(
        default_factory=lambda: [
            "builtin:pre_h_phonetic.tsv",
            "builtin:pre_common.tsv",
            "project:项目字典_译前.tsv",
        ]
    )
    gpt_dicts: list[str] = field(
        default_factory=lambda: [
            "builtin:gpt_common_acg.tsv",
            "builtin:gpt_common_menu.tsv",
            "project:项目GPT字典.tsv",
        ]
    )
    post_dicts: list[str] = field(
        default_factory=lambda: [
            "builtin:post_symbol_common.tsv",
            "project:项目字典_译后.tsv",
        ]
    )

    batch_size: int = 120
    context_num: int = 8
    max_warn_ratio: float = 0.02
    max_warn_count: int = 50
    max_batch_warn_ratio: float = 0.08
    avg_sentence_length_threshold: int = 18
    length_ratio_threshold: float = 1.3
    length_diff_threshold: int = 4
    strict_length_check: bool = False

    retransl_fail: bool = False
    retransl_keys: list[str] = field(default_factory=lambda: ["翻译失败"])

    enable_organ_injection_check: bool = True
    enable_register_check: bool = True
    enabled_problems: list[str] | None = None
    problem_filter_regex: list[str] = field(default_factory=list)
    problem_whitelist: list[str] = field(default_factory=list)
    english_whitelist: list[str] = field(default_factory=lambda: list(DEFAULT_ENGLISH_WHITELIST))
    gbk_whitelist: list[str] = field(default_factory=list)
    arinashi_dict: dict[str, str] = field(default_factory=dict)
    speaker_map: dict[str, str] = field(default_factory=dict)

    def resolve_resource_path(self, ref: str, project_dir: Path) -> Path:
        """Resolve a builtin: or project: resource reference to a filesystem Path."""
        fw_root = get_framework_root()
        if ref.startswith("builtin:"):
            rel = ref[len("builtin:") :].lstrip("/\\")
            if "/" not in rel and "\\" not in rel:
                if rel.endswith((".tsv", ".txt")):
                    return fw_root / "dicts" / rel
                if rel.endswith(".md"):
                    return fw_root / "guidelines" / rel
                if rel.endswith(".json"):
                    return fw_root / "profiles" / rel
            return fw_root / rel
        if ref.startswith("project:"):
            rel = ref[len("project:") :].lstrip("/\\")
            return project_dir / rel
        candidate = Path(ref)
        if candidate.is_absolute():
            return candidate
        proj_candidate = project_dir / ref
        if proj_candidate.exists():
            return proj_candidate
        return fw_root / ref

    def resolve_guideline_path(self, name_or_ref: str, project_dir: Path) -> Path:
        if name_or_ref.startswith(("builtin:", "project:")):
            return self.resolve_resource_path(name_or_ref, project_dir)
        for cand in (project_dir / name_or_ref, project_dir / "guidelines" / name_or_ref):
            if cand.exists():
                return cand
        return get_framework_root() / "guidelines" / name_or_ref

    def resolve_profile_path(self, project_dir: Path) -> Path:
        name = self.profile if self.profile.endswith(".json") else f"{self.profile}.json"
        if name.startswith(("builtin:", "project:")):
            return self.resolve_resource_path(name, project_dir)
        proj_path = project_dir / "profiles" / name
        if proj_path.exists():
            return proj_path
        return get_framework_root() / "profiles" / name

    def load_profile_data(self, project_dir: Path) -> dict[str, Any]:
        path = self.resolve_profile_path(project_dir)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def input_path(self, project_dir: Path) -> Path:
        return project_dir / self.input_dir

    def cache_path(self, project_dir: Path) -> Path:
        return project_dir / self.cache_dir

    def output_path(self, project_dir: Path) -> Path:
        return project_dir / self.output_dir

    def history_path(self, project_dir: Path) -> Path:
        return project_dir / self.history_file

    def compute_warn_budget(self, total_items: int) -> int:
        """Compute global soft-warning cap: min(2% of corpus [floor 5], absolute cap max_warn_count=50)."""
        if self.max_warn_ratio <= 0 or self.max_warn_count <= 0:
            return 0
        scaled = max(5, int(total_items * self.max_warn_ratio))
        return min(self.max_warn_count, scaled)

    def compute_batch_warn_budget(self, batch_items: int) -> int:
        """Compute per-batch soft-warning circuit-breaker cap (default 8% of batch, floor 3)."""
        if self.max_batch_warn_ratio <= 0:
            return 0
        return max(3, int(batch_items * self.max_batch_warn_ratio))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "profile": self.profile,
            "guideline": self.guideline,
            "project_guideline": self.project_guideline,
            "charter": self.charter,
            "target_lang": self.target_lang,
            "target_encoding": self.target_encoding,
            "source_format": self.source_format,
            "input_dir": self.input_dir,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir,
            "history_file": self.history_file,
            "preserve_presets": list(self.preserve_presets),
            "custom_preserve_rules": list(self.custom_preserve_rules),
            "ruby_preserve_mode": self.ruby_preserve_mode,
            "use_placeholder_protection": self.use_placeholder_protection,
            "strip_outer_quotes": self.strip_outer_quotes,
            "align_trailing_period": self.align_trailing_period,
            "extra_linebreak_policy": self.extra_linebreak_policy,
            "linebreak_symbol": self.linebreak_symbol,
            "sort_dicts": self.sort_dicts,
            "use_pre_dict_in_name": self.use_pre_dict_in_name,
            "use_gpt_dict_in_name": self.use_gpt_dict_in_name,
            "use_post_dict_in_name": self.use_post_dict_in_name,
            "pre_dicts": list(self.pre_dicts),
            "gpt_dicts": list(self.gpt_dicts),
            "post_dicts": list(self.post_dicts),
            "batch_size": self.batch_size,
            "context_num": self.context_num,
            "max_warn_ratio": self.max_warn_ratio,
            "max_warn_count": self.max_warn_count,
            "max_batch_warn_ratio": self.max_batch_warn_ratio,
            "avg_sentence_length_threshold": self.avg_sentence_length_threshold,
            "length_ratio_threshold": self.length_ratio_threshold,
            "length_diff_threshold": self.length_diff_threshold,
            "strict_length_check": self.strict_length_check,
            "retransl_fail": self.retransl_fail,
            "retransl_keys": list(self.retransl_keys),
            "enable_organ_injection_check": self.enable_organ_injection_check,
            "enable_register_check": self.enable_register_check,
            "enabled_problems": list(self.enabled_problems) if self.enabled_problems is not None else None,
            "problem_filter_regex": list(self.problem_filter_regex),
            "problem_whitelist": list(self.problem_whitelist),
            "english_whitelist": list(self.english_whitelist),
            "gbk_whitelist": list(self.gbk_whitelist),
            "arinashi_dict": dict(self.arinashi_dict),
            "speaker_map": dict(self.speaker_map),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def config_file_path(cls, project_dir: str | Path) -> Path:
        root = Path(project_dir)
        primary = root / "config.json"
        if primary.exists():
            return primary
        legacy = root / "project.json"
        if legacy.exists():
            return legacy
        return primary

    @classmethod
    def load(cls, project_dir: str | Path) -> "ProjectConfig":
        path = cls.config_file_path(project_dir)
        if not path.exists():
            raise FileNotFoundError(f"Project configuration not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))

    def save(self, project_dir: str | Path) -> Path:
        root = Path(project_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = self.config_file_path(root)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path
