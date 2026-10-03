import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ANNOTATION_FIELDS = {
    "query_id",
    "video_id",
    "video_path",
    "text_query",
    "target_video_id",
    "split",
}


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load JSONL or CSV annotation rows using the shared annotation schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                _validate_annotation_row(row, path, line_number)
                rows.append(row)
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for line_number, row in enumerate(rows, start=2):
            _validate_annotation_row(row, path, line_number)
        return rows

    raise ValueError(f"Unsupported annotation file type: {path.suffix}")


def create_splits(
    raw_metadata_path: Path,
    video_root: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    """Create train, val, and test split files and return their paths."""
    raise NotImplementedError(
        "Split creation depends on the raw Something-Something V2 metadata format."
    )


def compute_dataset_stats(annotation_paths: dict[str, Path]) -> dict[str, Any]:
    """Compute counts by split, label, and missing-video status."""
    stats: dict[str, Any] = {"splits": {}, "labels": {}, "missing_videos": []}
    for split_name, path in annotation_paths.items():
        rows = load_annotations(Path(path))
        stats["splits"][split_name] = len(rows)
        for row in rows:
            label = row.get("label", "")
            if label:
                stats["labels"][label] = stats["labels"].get(label, 0) + 1
            video_path = Path(row["video_path"])
            if not video_path.exists():
                stats["missing_videos"].append(
                    {"split": split_name, "video_id": row["video_id"], "path": str(video_path)}
                )
    stats["num_missing_videos"] = len(stats["missing_videos"])
    return stats


def sample_ordered_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Return ordered frame indices with shape (num_frames,)."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    indices = np.linspace(0, total_frames - 1, num=num_frames)
    return np.rint(indices).astype(np.int64)


def load_video_frames(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    """Return RGB frames with shape (T, H, W, 3), dtype uint8."""
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    try:
        for index in frame_indices.astype(np.int64):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame_bgr = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {int(index)} from {video_path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return np.stack(frames, axis=0).astype(np.uint8)


def get_dataset_sample(row: dict[str, Any], num_frames: int) -> dict[str, Any]:
    """Return one sample using the shared dataset sample format."""
    _validate_annotation_row(row)

    video_path = Path(row["video_path"])
    total_frames = _get_total_frames(video_path)
    frame_indices = sample_ordered_frame_indices(total_frames, num_frames)
    frames = load_video_frames(video_path, frame_indices)

    return {
        "query_id": row["query_id"],
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "text_query": row["text_query"],
        "target_video_id": row["target_video_id"],
        "frame_indices": frame_indices,
        "frames": frames,
    }


def _validate_annotation_row(
    row: dict[str, Any],
    path: Path | None = None,
    line_number: int | None = None,
) -> None:
    missing = REQUIRED_ANNOTATION_FIELDS.difference(row)
    if missing:
        location = ""
        if path is not None:
            location = f" in {path}"
            if line_number is not None:
                location += f":{line_number}"
        raise ValueError(f"Missing annotation fields{location}: {sorted(missing)}")


def _get_total_frames(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if total_frames <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")
    return total_frames
