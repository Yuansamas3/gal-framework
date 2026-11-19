"""Self-heuristic review seed ledger (`work/seeds.json`) with true-sampling & round-isolation gating."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .models import ItemStatus, SentenceItem, atomic_write_json, file_lock


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_int(r_key: str) -> int:
    return int(r_key) if r_key.isdigit() else 999999


class SeedLedger:
    """State machine managing dispersed sampling, round-isolated review seeds, and >=2-round gate."""

    def __init__(self, path: str | Path, *, min_required_rounds: int = 2) -> None:
        self.path = Path(path)
        self.completed_rounds: int = 0
        self.min_required_rounds: int = min_required_rounds
        self.seeds: dict[str, list[dict[str, Any]]] = {}
        self.samples: list[dict[str, Any]] = []
        if self.path.exists():
            self.load()

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.completed_rounds = int(data.get("completedRounds", 0))
        self.min_required_rounds = int(data.get("minRequiredRounds", self.min_required_rounds))
        self.seeds = {str(k): list(v) for k, v in data.get("seeds", {}).items()}
        self.samples = list(data.get("samples", []))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completedRounds": self.completed_rounds,
            "minRequiredRounds": self.min_required_rounds,
            "seeds": self.seeds,
            "samples": self.samples,
        }
        with file_lock(self.path.with_suffix(".lock")):
            atomic_write_json(self.path, payload)
        return self.path

    def all_seeds(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r_key in sorted(self.seeds.keys(), key=_round_int):
            for seed in self.seeds[r_key]:
                out.append({"round": r_key, **seed})
        return out

    def unconsumed_seeds(self, *, max_round: int | None = None) -> list[dict[str, Any]]:
        return [
            s
            for s in self.all_seeds()
            if not s.get("consumed", False)
            and (max_round is None or _round_int(str(s["round"])) <= max_round)
        ]

    def add_seed(
        self,
        pattern: str,
        *,
        query: str = "",
        category: str = "expand",
        round_num: int | None = None,
        items: Sequence[SentenceItem] | None = None,
    ) -> dict[str, Any]:
        """Register a new unconsumed review seed and optionally preview corpus hits."""
        target_round = round_num if round_num is not None else self.completed_rounds + 1
        r_key = str(max(1, target_round))
        bucket = self.seeds.setdefault(r_key, [])
        seed_id = f"S{r_key}-{len(bucket) + 1:03d}"

        hit_refs: list[str] = []
        hit_count = 0
        if query and items is not None:
            try:
                rx = re.compile(query)
                use_rx = True
            except re.error:
                rx = None
                use_rx = False
            for it in items:
                blob = f"{it.pre_src}\n{it.post_src}\n{it.post_dst}"
                matched = bool(rx.search(blob)) if use_rx and rx else (query in blob)
                if matched:
                    hit_count += 1
                    if len(hit_refs) < 10:
                        hit_refs.append(it.ref_key)

        record: dict[str, Any] = {
            "id": seed_id,
            "pattern": pattern.strip(),
            "query": query,
            "category": category,
            "consumed": False,
            "resolution": "",
            "createdAt": _utc_now(),
            "consumedAt": "",
        }
        if query and items is not None:
            record["previewHitCount"] = hit_count
            record["previewSampleRefs"] = hit_refs

        bucket.append(record)
        self.save()
        return {"round": r_key, **record}

    def consume_seed(
        self,
        seed_id: str,
        *,
        resolution: str = "",
        complete_round: bool = False,
    ) -> dict[str, Any]:
        """Consume seed(s) ONLY in active/past rounds (`<= completedRounds + 1`).

        Prevents cross-round future consumption and refuses to advance `completedRounds`
        if any seed in the current round remains unconsumed.
        """
        current_round = self.completed_rounds + 1
        targets = {s.strip() for s in seed_id.split(",") if s.strip()}
        consume_all = "all" in targets or "*" in targets
        updated: list[str] = []
        skipped_future: list[str] = []

        for r_key, bucket in self.seeds.items():
            r_num = _round_int(r_key)
            for seed in bucket:
                if seed.get("consumed", False):
                    continue
                sid = str(seed.get("id"))
                if not (consume_all or sid in targets):
                    continue
                if r_num > current_round:
                    if not consume_all:
                        skipped_future.append(sid)
                    continue
                seed["consumed"] = True
                seed["consumedAt"] = _utc_now()
                if resolution:
                    seed["resolution"] = resolution
                updated.append(sid)

        round_advanced = False
        round_block_reason = ""
        if complete_round:
            remaining_active = self.unconsumed_seeds(max_round=current_round)
            if remaining_active:
                round_block_reason = (
                    f"cannot complete round {current_round}: "
                    f"{len(remaining_active)} unconsumed seed(s) remain in active round"
                )
            else:
                self.completed_rounds += 1
                round_advanced = True

        self.save()
        return {
            "consumed_ids": updated,
            "skipped_future_ids": skipped_future,
            "round_advanced": round_advanced,
            "round_block_reason": round_block_reason,
            "completedRounds": self.completed_rounds,
            "remaining_unconsumed": len(self.unconsumed_seeds()),
        }

    def complete_round(self) -> tuple[bool, int, str]:
        """Advance completedRounds by 1 only when all active-round seeds are consumed."""
        current_round = self.completed_rounds + 1
        remaining_active = self.unconsumed_seeds(max_round=current_round)
        if remaining_active:
            return (
                False,
                self.completed_rounds,
                f"cannot complete round {current_round}: "
                f"{len(remaining_active)} unconsumed seed(s) remain in round <= {current_round}",
            )
        self.completed_rounds += 1
        self.save()
        return True, self.completed_rounds, ""

    def sample_dispersed(
        self,
        items: Sequence[SentenceItem],
        *,
        regions: int = 3,
        per_region: int = 10,
    ) -> dict[str, Any]:
        """Sample translated items across >=3 non-contiguous regions avoiding prior samples."""
        regions = max(3, regions)
        per_region = max(1, per_region)

        translated = [
            it
            for it in items
            if it.status in (ItemStatus.PROCESSED.value, ItemStatus.PROOFREAD.value)
            and (it.post_dst or it.effective_core_dst)
        ]
        if not translated:
            return {
                "round": self.completed_rounds + 1,
                "regions_sampled": 0,
                "sampled_count": 0,
                "items": [],
            }

        previously_sampled: set[str] = set()
        for batch in self.samples:
            for ref in batch.get("refs", []):
                previously_sampled.add(ref)

        total = len(translated)
        actual_regions = min(regions, total)
        picked: list[SentenceItem] = []
        region_summaries: list[dict[str, Any]] = []

        for r in range(actual_regions):
            start = (r * total) // actual_regions
            end = ((r + 1) * total) // actual_regions
            segment = translated[start:end]
            fresh = [it for it in segment if it.ref_key not in previously_sampled]
            pool = fresh if fresh else segment
            if not pool:
                continue
            take = min(per_region, len(pool))
            if take == len(pool):
                chosen = list(pool)
            else:
                step = len(pool) / take
                chosen = [pool[int(i * step)] for i in range(take)]
            picked.extend(chosen)
            region_summaries.append(
                {
                    "region": r + 1,
                    "range": f"{start}..{end - 1}",
                    "count": len(chosen),
                    "refs": [it.ref_key for it in chosen],
                }
            )

        record = {
            "round": self.completed_rounds + 1,
            "sampledAt": _utc_now(),
            "regions": len(region_summaries),
            "totalTranslatedAtSample": total,
            "sampledCount": len(picked),
            "refs": [it.ref_key for it in picked],
        }
        self.samples.append(record)
        self.save()

        return {
            "round": record["round"],
            "regions_sampled": len(region_summaries),
            "sampled_count": len(picked),
            "region_summaries": region_summaries,
            "items": [
                {
                    "file": it.file_name,
                    "id": it.index,
                    "anchor": it.anchor,
                    "speaker": it.speaker,
                    "post_src": it.post_src,
                    "post_dst": it.post_dst,
                    "problems": list(it.problems),
                }
                for it in picked
            ],
        }

    def check_gate(
        self,
        *,
        open_problem_count: int = 0,
        open_error_count: int | None = None,
        open_warn_count: int = 0,
        max_warn_budget: int | None = None,
    ) -> tuple[bool, list[str]]:
        """Enforce completion gate: >=2 rounds, 0 open seeds, 0 hard QA errors, warn <= budget, AND true >=3-region sampling."""
        reasons: list[str] = []
        unconsumed = self.unconsumed_seeds()
        blocking_qa = open_problem_count if open_error_count is None else open_error_count
        if self.completed_rounds < self.min_required_rounds:
            reasons.append(
                f"completedRounds={self.completed_rounds} < minRequiredRounds={self.min_required_rounds}"
            )
        if unconsumed:
            reasons.append(f"{len(unconsumed)} unconsumed seed(s) remain")
        if blocking_qa > 0:
            label = "QA hard error item(s)" if open_error_count is not None else "QA problem item(s)"
            reasons.append(f"{blocking_qa} {label} remain")
        if max_warn_budget is not None and open_warn_count > max_warn_budget:
            reasons.append(
                f"soft QA warnings ({open_warn_count}) exceed warn budget ({max_warn_budget})"
            )

        valid_samples = [
            s for s in self.samples if int(s.get("sampledCount", len(s.get("refs", [])))) > 0
        ]
        if len(valid_samples) < self.min_required_rounds:
            reasons.append(
                f"insufficient real dispersed samples ({len(valid_samples)} < {self.min_required_rounds})"
            )
        else:
            sampled_rounds = {int(s.get("round", 0)) for s in valid_samples}
            missing_rounds = [
                r for r in range(1, self.min_required_rounds + 1) if r not in sampled_rounds
            ]
            if missing_rounds:
                reasons.append(
                    f"missing real dispersed sample(s) for round(s): {missing_rounds}"
                )
            if valid_samples:
                last_sample = valid_samples[-1]
                last_regions = int(last_sample.get("regions", 0))
                corpus_size = int(last_sample.get("totalTranslatedAtSample", 3))
                required_regions = min(3, max(1, corpus_size))
                if last_regions < required_regions:
                    reasons.append(
                        f"latest sample covered only {last_regions} region(s) (< {required_regions} required)"
                    )

        return len(reasons) == 0, reasons

    def status(
        self,
        *,
        open_problem_count: int = 0,
        open_error_count: int | None = None,
        open_warn_count: int = 0,
        max_warn_budget: int | None = None,
    ) -> dict[str, Any]:
        all_s = self.all_seeds()
        unconsumed = self.unconsumed_seeds()
        effective_errors = open_problem_count if open_error_count is None else open_error_count
        total_open = open_problem_count if open_problem_count else (effective_errors + open_warn_count)
        warn_budget_exceeded = bool(
            max_warn_budget is not None and open_warn_count > max_warn_budget
        )
        gate_passed, reasons = self.check_gate(
            open_problem_count=open_problem_count,
            open_error_count=open_error_count,
            open_warn_count=open_warn_count,
            max_warn_budget=max_warn_budget,
        )

        if gate_passed:
            next_action = "Review gate satisfied. Ready for final export."
        elif effective_errors > 0:
            next_action = "Inspect and fix remaining hard QA errors first, then distill seeds."
        elif warn_budget_exceeded:
            next_action = (
                f"Reduce soft QA warnings ({open_warn_count} > budget {max_warn_budget}) "
                "before completing review gate."
            )
        elif unconsumed:
            next_action = (
                f"Investigate {len(unconsumed)} unconsumed seed(s), apply fixes, "
                "register follow-up seeds if any, and consume."
            )
        else:
            next_action = (
                f"Run dispersed sampling across >=3 regions (round {self.completed_rounds + 1}"
                f"/{self.min_required_rounds}) and complete the round."
            )

        return {
            "ledgerPath": str(self.path),
            "completedRounds": self.completed_rounds,
            "minRequiredRounds": self.min_required_rounds,
            "totalSeeds": len(all_s),
            "consumedSeeds": len(all_s) - len(unconsumed),
            "unconsumedCount": len(unconsumed),
            "unconsumedSeeds": unconsumed,
            "sampleBatches": len(self.samples),
            "openProblemCount": total_open,
            "openErrorCount": effective_errors,
            "openWarnCount": open_warn_count,
            "maxWarnBudget": max_warn_budget,
            "warnBudgetExceeded": warn_budget_exceeded,
            "gatePassed": gate_passed,
            "blockingReasons": reasons,
            "nextAction": next_action,
        }
