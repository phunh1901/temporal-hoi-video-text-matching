"""PyTorch Dataset implementation for Temporal HOI video-text matching."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    import torch

from temporal_hoi.data.video_reader import VideoReader


class TemporalHOIDataset:
    """PyTorch Dataset loading video clips and their associated text prompts based on manifests."""

    def __init__(
        self,
        clips_csv: str | Path = "data/manifests/clips.csv",
        texts_csv: str | Path = "data/manifests/texts.csv",
        splits_csv: str | Path = "data/manifests/splits.csv",
        split: str | None = None,  # 'dev', 'val', 'test', or None for all
        num_frames: int = 8,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        relevance_csv: str | Path | None = None,
        max_frame_side: int | None = 320,
    ) -> None:
        self.num_frames = num_frames
        self.transform = transform
        self.max_frame_side = max_frame_side

        # Load manifests
        clips_df = pd.read_csv(clips_csv)
        texts_df = pd.read_csv(texts_csv)
        splits_df = pd.read_csv(splits_csv)

        # Merge clips with splits
        if clips_df["clip_id"].duplicated().any() or splits_df["clip_id"].duplicated().any():
            raise ValueError("Duplicate clip IDs in manifests")
        if set(clips_df["clip_id"]) != set(splits_df["clip_id"]):
            raise ValueError("Every clip must have exactly one split")
        if split is not None and split not in {"dev", "val", "test", "extension"}:
            raise ValueError("Unknown split")
        if not splits_df["split"].isin(["dev", "val", "test", "extension"]).all():
            raise ValueError("Unknown split in manifest")
        merged_df = clips_df.merge(
            splits_df, on=["clip_id", "session_id"], how="left", validate="one_to_one"
        )
        if merged_df["split"].isna().any():
            raise ValueError("Session IDs disagree between clips and splits")
        for key in ["session_id", "video_path"] + (["video_id"] if "video_id" in clips_df else []):
            if (merged_df.groupby(key)["split"].nunique() > 1).any():
                raise ValueError(f"Split leakage by {key}")
        if texts_df["text_id"].duplicated().any():
            raise ValueError("Duplicate text IDs")
        self.relevance = None
        if relevance_csv is not None:
            self.relevance = pd.read_csv(relevance_csv)
            rel = self.relevance
            if rel.duplicated(["clip_id", "text_id"]).any():
                raise ValueError("Duplicate relevance annotation")
            if (
                not set(rel["clip_id"]) <= set(clips_df["clip_id"])
                or not set(rel["text_id"]) <= set(texts_df["text_id"])
                or not rel["is_match"].isin([0, 1]).all()
            ):
                raise ValueError("Invalid clip-text relevance annotation")
        if split is not None:
            merged_df = merged_df[merged_df["split"] == split].reset_index(drop=True)

        self.clips = merged_df
        self.texts = texts_df

    def __len__(self) -> int:
        return len(self.clips)

    def get_prompts_for_behavior(self, behavior_id: str) -> dict[str, list[dict[str, str]]]:
        """Get positive and negative prompts for a behavior."""
        subset = self.texts[self.texts["behavior_id"] == behavior_id]
        positives = subset[subset["prompt_type"] == "positive"].to_dict(orient="records")
        negatives = subset[subset["prompt_type"].str.startswith("hard_negative")].to_dict(
            orient="records"
        )
        return {"positives": positives, "negatives": negatives}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.clips.iloc[idx]
        video_path = Path(row["video_path"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])

        # Extract frames
        with VideoReader(video_path) as reader:
            pil_frames, timestamps = reader.read_interval_frames(
                start_s=start_s,
                end_s=end_s,
                num_frames=self.num_frames,
                max_frame_side=self.max_frame_side,
            )

        # Apply transforms if provided
        if self.transform is not None:
            import torch

            tensor_frames = torch.stack([self.transform(img) for img in pil_frames])
        else:
            tensor_frames = None

        # Behavior-level prompt templates are NOT clip-level ground truth.
        prompts = {"positives": [], "negatives": []}
        if self.relevance is not None:
            annotations = self.relevance[self.relevance["clip_id"] == row["clip_id"]]
            labeled = annotations.merge(self.texts, on="text_id", validate="many_to_one")
            prompts = {
                "positives": labeled[labeled["is_match"] == 1].to_dict("records"),
                "negatives": labeled[labeled["is_match"] == 0].to_dict("records"),
            }

        return {
            "clip_id": row["clip_id"],
            "video_path": str(video_path),
            "session_id": row["session_id"],
            "split": row["split"],
            "start_s": start_s,
            "end_s": end_s,
            "behavior_id": row["behavior_id"],
            "rule_id": row["rule_id"],
            "has_violation": int(row["has_violation"]) if pd.notna(row["has_violation"]) else None,
            "zone_polygon": row["zone_polygon"],
            "pil_frames": pil_frames,
            "tensor_frames": tensor_frames,
            "timestamps": timestamps,
            "positive_prompts": prompts["positives"],
            "negative_prompts": prompts["negatives"],
            "matching_labels_available": bool(prompts["positives"]),
            "label_status": row.get("label_status", "unverified"),
            "evaluation_eligible": str(row.get("evaluation_eligible", "false")).lower() == "true",
        }
