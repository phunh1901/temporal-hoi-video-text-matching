"""Wall-clock stages and label-backed retrieval diagnostics.

Timers are for synchronous CPU work. GPU callers must synchronize before the
timed boundaries; this module never interprets a rank error as an HOI error.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.seconds = {}
        self.calls = {}

    @contextmanager
    def measure(self, name):
        start = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - start
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Timer clock must be finite and monotonic")
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
            self.calls[name] = self.calls.get(name, 0) + 1

    def summary(self):
        total = sum(self.seconds.values())
        return {
            name: {
                "seconds": seconds,
                "calls": self.calls[name],
                "mean_seconds_per_call": seconds / self.calls[name],
                "fraction_of_measured_stages": seconds / total if total else None,
            }
            for name, seconds in self.seconds.items()
        }


def retrieval_error_groups(predictions, clip_groups=None):
    """Count observed ranking mistakes using explicit positives only.

    Group names may describe candidate scene/action labels; the returned
    confusions are observations and do not establish their causal mechanism.
    """
    clip_groups = clip_groups or {}
    groups, errors, excluded = {}, [], []
    for prediction in predictions:
        direction = prediction["direction"]
        if direction not in {"video_to_text", "text_to_video"}:
            raise ValueError("Unknown retrieval direction")
        candidates = prediction["ranked_candidates"]
        if [c["rank"] for c in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError("Candidate ranks must be consecutive in stored order")
        if any(type(c["is_positive"]) is not bool for c in candidates):
            raise ValueError("Only explicit boolean labels can diagnose rank errors")
        positive = [c for c in candidates if c["is_positive"]]
        if not positive:
            excluded.append({"direction": direction, "query_id": prediction["query_id"]})
            continue
        group_name = (
            clip_groups.get(prediction["query_id"], "unclassified")
            if direction == "video_to_text"
            else "all_text_queries"
        )
        key = (direction, prediction.get("split", "all_pilot"), group_name)
        group = groups.setdefault(
            key, {"queries": 0, "top1_errors": 0, "positive_negative_pairs": 0,
                  "positive_not_strictly_above_negative_pairs": 0}
        )
        group["queries"] += 1
        negative = [c for c in candidates if not c["is_positive"]]
        group["positive_negative_pairs"] += len(positive) * len(negative)
        group["positive_not_strictly_above_negative_pairs"] += sum(
            p["score"] <= n["score"] for p in positive for n in negative
        )
        if not candidates[0]["is_positive"]:
            group["top1_errors"] += 1
            errors.append({
                "direction": direction,
                "query_id": prediction["query_id"],
                "split": prediction.get("split", "all_pilot"),
                "group": group_name,
                "top1_candidate": candidates[0]["candidate_id"],
                "top1_score": candidates[0]["score"],
                "first_positive_candidate": positive[0]["candidate_id"],
                "first_positive_rank": positive[0]["rank"],
                "first_positive_score": positive[0]["score"],
                "top1_minus_first_positive": candidates[0]["score"] - positive[0]["score"],
            })
    return {
        "label_basis": "locked explicit relevance labels; no causal or HOI localization labels",
        "groups": [
            {"direction": key[0], "split": key[1], "candidate_group": key[2], **value}
            for key, value in sorted(groups.items())
        ],
        "top1_errors": errors,
        "excluded_unlabelled_queries": excluded,
        "hoi_localization_errors": {
            "status": "not_run",
            "reason": "No verified person/object boxes or interaction ground truth in pilot",
            "count": None,
        },
    }
