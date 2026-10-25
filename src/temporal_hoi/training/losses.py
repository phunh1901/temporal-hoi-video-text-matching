"""Numerically stable symmetric contrastive learning with explicit unknowns."""

import math

import torch


def symmetric_multi_positive_loss(similarities, relation_labels, *, temperature=0.07):
    """Average video→text and text→video masked log-sum-exp losses.

    Both arrays are [videos, texts]. Labels are 1=known positive, 0=known
    negative, -1=unknown. Every positive contributes to the numerator; unknowns
    contribute to neither numerator nor denominator. An anchor needs a known
    positive AND at least one known negative to provide a learning signal.
    Counts of excluded anchors are returned for auditing, never hidden.
    """
    if (similarities.ndim != 2 or not all(similarities.shape)
            or relation_labels.shape != similarities.shape
            or not torch.isfinite(similarities).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Loss requires finite nonempty matching matrices and positive temperature")
    labels = relation_labels.to(device=similarities.device)
    if not torch.isin(labels, labels.new_tensor([-1, 0, 1])).all():
        raise ValueError("Relation labels must be -1 (unknown), 0 (negative), or 1 (positive)")
    logits = similarities / temperature

    def direction(values, relations):
        positive, negative = relations == 1, relations == 0
        eligible = positive.any(dim=1) & negative.any(dim=1)
        count = int(eligible.sum().item())
        if count == 0:
            raise ValueError("Each contrastive direction needs positive and negative anchors")
        values, positive = values[eligible], positive[eligible]
        valid = relations[eligible] >= 0
        numerator = values.masked_fill(~positive, -torch.inf).logsumexp(dim=1)
        denominator = values.masked_fill(~valid, -torch.inf).logsumexp(dim=1)
        return (denominator - numerator).mean(), count

    video_loss, video_count = direction(logits, labels)
    text_loss, text_count = direction(logits.T, labels.T)
    loss = (video_loss + text_loss) / 2
    if not torch.isfinite(loss):
        raise ValueError("Contrastive loss is nonfinite")
    return loss, {
        "video_to_text_loss": float(video_loss.detach()),
        "text_to_video_loss": float(text_loss.detach()),
        "video_anchors": video_count,
        "text_anchors": text_count,
        "excluded_video_anchors": similarities.shape[0] - video_count,
        "excluded_text_anchors": similarities.shape[1] - text_count,
        "positive_pairs": int((labels == 1).sum()),
        "negative_pairs": int((labels == 0).sum()),
        "unknown_pairs": int((labels == -1).sum()),
    }


bidirectional_multi_positive_loss = symmetric_multi_positive_loss
