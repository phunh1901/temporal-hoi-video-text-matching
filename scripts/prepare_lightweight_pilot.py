"""Apply the explicitly reviewed small pilot annotations; preserve prior manifests."""

import csv
import json
import shutil
from pathlib import Path

ROOT = Path("data/manifests")


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive = ROOT / "archive/pre_lightweight"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["clips.csv", "texts.csv", "splits.csv"]:
        if not (archive / name).exists():
            shutil.copy2(ROOT / name, archive / name)
    source_specs = {
        "apartment_parking": (
            "bhupender0415/CarParkingDetection",
            "master/carPark.mp4",
            "MIT_claim_in_readme_media_scope_unclear",
            "https://github.com/bhupender0415/CarParkingDetection",
        ),
        "trash_dumping": (
            "M4D-AI/Highway-Incident-Detection",
            "main/data/trash.mp4",
            "Apache-2.0_repository_media_provenance_unclear",
            "https://github.com/M4D-AI/Highway-Incident-Detection/blob/main/LICENSE",
        ),
        "parking_violation": (
            "M4D-AI/Highway-Incident-Detection",
            "main/data/parking.mp4",
            "Apache-2.0_repository_media_provenance_unclear",
            "https://github.com/M4D-AI/Highway-Incident-Detection/blob/main/LICENSE",
        ),
        "normal_surveillance": (
            "intel-iot-devkit/sample-videos",
            "master/people-detection.mp4",
            "CC-BY-4.0_repository",
            "https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE",
        ),
        "bicycle_pedestrian": (
            "intel-iot-devkit/sample-videos",
            "master/person-bicycle-car-detection.mp4",
            "CC-BY-4.0_repository",
            "https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE",
        ),
    }
    inventory = json.loads(Path("outputs/w02/review/sources.json").read_text())
    sources = {}
    for record in inventory:
        stem = Path(record["path"]).stem
        repo, remote, license_status, license_url = source_specs[stem]
        sources[stem] = dict(
            video_id=stem,
            video_path=Path(record["path"]).as_posix(),
            source_url=f"https://raw.githubusercontent.com/{repo}/{remote}",
            attribution=repo,
            license_status=license_status,
            license_url=license_url,
            source_checked_at="2026-09-30",
            **{k: record[k] for k in ["sha256", "bytes", "duration_s", "fps"]},
        )
    write_csv("sources.csv", list(sources.values()))
    # Label only observable scene/action; no inferred legal zone or event boundary.
    specs = [
        (
            "parking_lot_01",
            "apartment_parking",
            0,
            7,
            "parking",
            "RULE_01_PARKING",
            "lot",
            "dev",
            "Nhiều ô tô đỗ theo hàng; có xe di chuyển trong bãi. Không đủ 30 giây.",
        ),
        (
            "parking_lot_02",
            "apartment_parking",
            8,
            26,
            "parking",
            "RULE_01_PARKING",
            "lot",
            "dev",
            "Xe đỗ và xe di chuyển cùng xuất hiện; nhãn vi phạm cũ bị rút lại.",
        ),
        (
            "highway_traffic_01",
            "trash_dumping",
            1,
            11,
            "traffic",
            "RULE_02_TRASH",
            "road",
            "dev",
            "Xe chạy trên cao tốc; không có bằng chứng người đổ rác. Tên tệp không phải nhãn.",
        ),
        (
            "indoor_walk_01",
            "normal_surveillance",
            0,
            24,
            "walking",
            "NONE",
            "room",
            "dev",
            "Người đi qua phòng sàn gỗ, không phải sân chung cư.",
        ),
        (
            "indoor_walk_02",
            "normal_surveillance",
            25,
            48,
            "walking",
            "NONE",
            "room",
            "dev",
            "Người đi qua phòng, có lúc không có người trong hình.",
        ),
        (
            "highway_stop_01",
            "parking_violation",
            2,
            13,
            "traffic",
            "RULE_01_PARKING",
            "road",
            "dev",
            "Xe trên đường và xe cạnh vùng kẻ chéo; không đủ 30 giây xác nhận vi phạm.",
        ),
        (
            "cycling_demo_01",
            "bicycle_pedestrian",
            26,
            32,
            "cycling",
            "RULE_03_BIKE",
            "cycle",
            "extension",
            "Người đi xe đạp qua bãi ngoài trời. Không xác nhận đây là vùng cấm.",
        ),
        (
            "walking_demo_01",
            "bicycle_pedestrian",
            2,
            6,
            "walking",
            "RULE_03_BIKE",
            "walk",
            "extension",
            "Người đi bộ qua bãi ngoài trời; thay khoảng cũ 33–52 có cả xe đạp.",
        ),
    ]
    # Entire fixed candidate pool is annotated; positives may be shared across clips.
    groups = {
        "lot": [
            "cars are parked in rows in an outdoor parking lot",
            "many cars occupy marked parking spaces",
        ],
        "road": [
            "vehicles travel along a multilane highway",
            "cars and trucks drive on a divided road",
        ],
        "room": [
            "people walk through a room with a wooden floor",
            "pedestrians pass through an indoor room",
        ],
        "cycle": [
            "a person rides a bicycle across an outdoor paved area",
            "someone cycles through an outdoor parking area",
        ],
        "walk": [
            "a person walks across an outdoor paved area",
            "someone crosses a parking area on foot",
        ],
    }
    texts = [
        dict(
            text_id=f"{group}_{i + 1}",
            behavior_id=group,
            prompt_text=text,
            prompt_type="candidate",
            description="Observed action/scene; no rule semantics",
        )
        for group, phrases in groups.items()
        for i, text in enumerate(phrases)
    ]
    clips, splits, relevance, reviews = [], [], [], []
    for cid, stem, start, end, behavior, rule, group, split, note in specs:
        source = sources[stem]
        clips.append(
            dict(
                clip_id=cid,
                video_path=source["video_path"],
                video_id=stem,
                session_id=f"source_{stem}",
                start_s=start,
                end_s=end,
                behavior_id=behavior,
                rule_id=rule,
                has_violation="",
                zone_polygon="[]",
                zone_status="not_validated",
                evidence_status="insufficient_for_violation",
                description=note,
                label_status="ai_visual_review",
                license_status=source["license_status"],
                source_url=source["source_url"],
                reviewer="Codex_visual_review_not_human",
                reviewed_at="2026-09-30",
                review_scope="16_frame_source_contact_sheet",
                event_start_s="",
                event_end_s="",
                evaluation_eligible="false",
                matching_pilot_eligible="true",
                sample_type="technical_pilot",
                candidate_group=group,
            )
        )
        splits.append(dict(clip_id=cid, split=split, session_id=f"source_{stem}"))
        for text in texts:
            relevance.append(
                dict(
                    clip_id=cid,
                    text_id=text["text_id"],
                    is_match=int(text["behavior_id"] == group),
                    reviewer="Codex_visual_review_not_human",
                    reviewed_at="2026-09-30",
                )
            )
        reviews.append(
            dict(
                clip_id=cid,
                reviewer="Codex_visual_review_not_human",
                reviewed_at="2026-09-30",
                evidence=f"outputs/w02/review/{stem}.jpg",
                method="source contact sheet; scene/action labels only",
                correction=note,
                human_reviewed="false",
            )
        )
    write_csv("clips.csv", clips)
    write_csv("splits.csv", splits)
    write_csv("texts.csv", texts)
    write_csv("relevance.csv", relevance)
    write_csv("label_reviews.csv", reviews)
    write_csv(
        "normal_intervals.csv",
        [
            dict(
                video_id="normal_surveillance",
                session_id="source_normal_surveillance",
                split="dev",
                start_s=0,
                end_s=48,
                duration_s=48,
                status="pilot_background_not_independent_test",
                eligible_for_fa_hour="false",
            )
        ],
    )
    hashes = {
        p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.csv")
    }
    Path("outputs/w02/pilot_manifest_lock.json").write_text(
        json.dumps(
            dict(
                profile="w02_lightweight_pilot_v1",
                date="2026-09-30",
                purpose="technical_pilot_only",
                files=hashes,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Prepared {len(clips)} pilot intervals and {len(relevance)} relevance pairs; no new videos."
    )


if __name__ == "__main__":
    main()
