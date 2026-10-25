"""Development-only experiment summaries and explicit zero-shot usage audits.

No dataset is inferred, no missing label becomes a negative, and a usage audit
only establishes consistency of the supplied manifest, not completeness of
the original dataset or absence of concepts in pretraining.
"""

import math
import statistics


def summarize_seeds(records):
    """Summarize one scalar metric on identical inputs/config across distinct seeds.

    Call separately per model, direction and metric; values retain their input
    units. Sample standard deviation is undefined for a single run. Synthetic
    probes retain their execution level and cannot be mixed with benchmark runs.
    """
    if not records:
        raise ValueError("At least one completed run is required")
    keys = ("comparison_sha256", "input_sha256", "execution_level", "metric", "units")
    seeds, values = [], []
    for record in records:
        if any(not isinstance(record.get(key), str) or not record[key] for key in keys):
            raise ValueError("Every run requires comparison/input/execution/metric/units provenance")
        if any(record[key] != records[0][key] for key in keys):
            raise ValueError("Cannot combine different inputs, configurations, levels or metrics")
        seed, value = record.get("seed"), record.get("value")
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError("Seeds must be distinct nonnegative integers")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Metrics must be finite numeric values")
        seeds.append(seed)
        values.append(value)
    return {
        **{key: records[0][key] for key in keys},
        "seeds": seeds, "values": values, "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "uncertainty_scope": "between_optimization_seeds_on_fixed_inputs",
        "interpretation": "one-seed exploratory" if len(values) == 1 else "multi-seed descriptive",
        "significance_claim": False,
    }


def _ids(value, field):
    if (not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{field} must contain unique nonempty string IDs")
    return set(value)


def audit_zero_shot_usage(manifest):
    """Audit materialized window/context/cache/prompt/checkpoint usage by ID.

    Each sample includes labels of its *entire* window, and context positives
    outside the center frame. Unknown classes must be explicitly listed and
    cannot overlap positive/negative targets. A cache used for training is a
    training use even when its backbone was frozen. Final split inspection for
    dataset integrity is outside this selection-use log.
    """
    if manifest.get("schema_version") != "zero_shot_usage_v1":
        raise ValueError("Require zero_shot_usage_v1")
    groups = {key: _ids(manifest.get(key), key)
              for key in ("seen_classes", "pseudo_unseen_classes", "final_unseen_classes")}
    if any(not group for group in groups.values()):
        raise ValueError("Seen, pseudo-unseen and final-unseen groups must be materialized")
    seen, pseudo, final = groups.values()
    if seen & pseudo or seen & final or pseudo & final:
        raise ValueError("Class partitions must be disjoint")
    universe = seen | pseudo | final
    samples, videos, violations = {}, {}, []
    for row in manifest.get("samples", []):
        sid, vid, split = row.get("sample_id"), row.get("video_id"), row.get("split")
        if (not isinstance(sid, str) or not sid or sid in samples
                or not isinstance(vid, str) or not vid
                or split not in {"train", "selection", "final"}):
            raise ValueError("Require unique sample IDs, video IDs and explicit split roles")
        if vid in videos and videos[vid] != split:
            violations.append({"sample_id": sid, "reason": "video crosses split roles"})
        videos[vid] = split
        labels = {key: _ids(row.get(key), key) for key in (
            "positive_classes", "negative_classes", "unknown_classes", "context_positive_classes")}
        if any(not group <= universe for group in labels.values()):
            raise ValueError("Sample labels must use the declared class universe")
        positive, negative, unknown, context = labels.values()
        if positive & negative or unknown & (positive | negative | context) or context & negative:
            violations.append({"sample_id": sid, "reason": "contradictory or unknown target"})
        samples[sid] = (split, positive | context, negative)
    if not samples or not manifest.get("uses"):
        raise ValueError("Require materialized samples and an explicit usage log")
    allowed = {"supervised_training", "cache_training", "validation_selection",
               "prompt_selection", "checkpoint_selection"}
    for use in manifest["uses"]:
        sid, purpose = use.get("sample_id"), use.get("purpose")
        if sid not in samples or purpose not in allowed:
            raise ValueError("Every use must resolve to a sample and declared purpose")
        split, positives, negatives = samples[sid]
        reason = None
        if split == "final" or (positives | negatives) & final:
            reason = "final split/class used during development"
        elif purpose in {"supervised_training", "cache_training"}:
            if split != "train" or (positives | negatives) & pseudo:
                reason = "held-out selection sample/class used for training or training cache"
        elif split != "selection":
            reason = "selection must use held-out development samples"
        if reason:
            violations.append({"sample_id": sid, "purpose": purpose, "reason": reason})
    return {
        "status": "failed" if violations else "manifest_consistent",
        "sample_count": len(samples), "use_count": len(manifest["uses"]),
        "violations": violations, "benchmark_evaluated": False,
        "scope": "declared_manifest_only",
        "limitations": ["Original annotation/cache/prompt/checkpoint coverage needs independent hashes",
                        "Pretraining concept exposure is not established by this audit"],
    }
