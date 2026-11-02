import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import PROJECT_ROOT


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
    """Convert official SSv2 train/validation metadata into train/val/test JSONL.

    The labeled official validation videos are divided 50/50, using the seed.
    The official hidden-label test set is never used.
    """

    metadata_dir = Path(raw_metadata_path)

    if metadata_dir.is_file():
        metadata_dir = metadata_dir.parent

    train_raw = _read_ssv2_metadata(metadata_dir, "train")
    validation_raw = _read_ssv2_metadata(metadata_dir, "validation")

    video_root = Path(video_root).resolve()
    output_dir = Path(output_dir)

    train_ids = [str(row["id"]) for row in train_raw]
    validation_ids = [str(row["id"]) for row in validation_raw]

    # Check for duplicate video IDs before creating the splits.
    if (
        len(train_ids) != len(set(train_ids))
        or len(validation_ids) != len(set(validation_ids))
    ):
        raise ValueError("Duplicate video ids found in official metadata")

    if set(train_ids) & set(validation_ids):
        raise ValueError(
            "Official training and validation sets have overlapping video ids"
        )

    # Sort before shuffling so the original file order does not affect results.
    validation_raw = sorted(
        validation_raw,
        key=lambda row: str(row["id"])
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(validation_raw))

    # Use half for validation and the remaining half for held-out testing.
    n_val = len(validation_raw) // 2

    val_raw = [
        validation_raw[i]
        for i in permutation[:n_val]
    ]

    test_raw = [
        validation_raw[i]
        for i in permutation[n_val:]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for split, entries in (
        ("train", train_raw),
        ("val", val_raw),
        ("test", test_raw),
    ):
        path = output_dir / f"{split}.jsonl"

        with path.open("w", encoding="utf-8") as handle:
            for item in entries:
                row = _convert_ssv2_row(
                    item,
                    video_root,
                    split
                )

                handle.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        paths[split] = path

    return paths


def _read_ssv2_metadata(
    folder: Path,
    split: str
) -> list[dict[str, Any]]:
    """Read the official Something-Something V2 metadata."""

    filenames = (
        f"something-something-v2-{split}.json",
        f"{split}.json",
    )

    path = next(
        (
            folder / name
            for name in filenames
            if (folder / name).is_file()
        ),
        None,
    )

    if path is None:
        raise FileNotFoundError(
            f"Cannot find SSv2 {split} metadata in {folder}: {filenames}"
        )

    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Expected a nonempty JSON list in {path}"
        )

    if any(
        not isinstance(row, dict)
        or "id" not in row
        or "template" not in row
        for row in rows
    ):
        raise ValueError(
            f"Expected 'id' and 'template' in every row of {path}"
        )

    return rows


def _convert_ssv2_row(
    item: dict[str, Any],
    video_root: Path,
    split: str
) -> dict[str, Any]:
    """Convert SSv2 metadata into the project's shared annotation format."""

    video_id = str(item["id"])
    template = str(item["template"]).strip()

    label = " ".join(
        template.replace("[", "")
        .replace("]", "")
        .lower()
        .split()
    )

    # Prefer the object-filled caption when it is available.
    text_query = str(
        item.get("label") or label
    ).strip()

    if not text_query:
        text_query = label

    video_path = (
        video_root / f"{video_id}.webm"
    ).resolve()

    try:
        path_text = video_path.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        path_text = str(video_path)

    row = {
        "query_id": f"{split}_{video_id}",
        "video_id": video_id,
        "video_path": path_text,
        "text_query": text_query,
        "target_video_id": video_id,
        "template": template,
        "label": label,
        "split": split,
    }

    if isinstance(item.get("placeholders"), list):
        row["objects"] = item["placeholders"]

    return row


def compute_dataset_stats(
    annotation_paths: dict[str, Path]
) -> dict[str, Any]:
    """Compute counts by split, label, and missing-video status."""

    stats: dict[str, Any] = {
        "splits": {},
        "labels": {},
        "labels_by_split": {},
        "missing_videos": []
    }

    for split_name, path in annotation_paths.items():

        rows = load_annotations(Path(path))

        stats["splits"][split_name] = len(rows)

        labels_for_split: dict[str, int] = {}

        for row in rows:

            label = row.get("label", "")

            if label:
                stats["labels"][label] = (
                    stats["labels"].get(label, 0) + 1
                )

                labels_for_split[label] = (
                    labels_for_split.get(label, 0) + 1
                )

            video_path = Path(row["video_path"])

            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path

            if not video_path.is_file():
                stats["missing_videos"].append(
                    {
                        "split": split_name,
                        "video_id": row["video_id"],
                        "path": str(video_path)
                    }
                )

        stats["labels_by_split"][split_name] = labels_for_split

    stats["num_missing_videos"] = len(
        stats["missing_videos"]
    )

    return stats


def sample_ordered_frame_indices(
    total_frames: int,
    num_frames: int
) -> np.ndarray:
    """Return ordered frame indices with shape (num_frames,)."""

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    indices = np.linspace(
        0,
        total_frames - 1,
        num=num_frames
    )

    return np.rint(indices).astype(np.int64)


def load_video_frames(
    video_path: Path,
    frame_indices: np.ndarray
) -> np.ndarray:
    """Return RGB frames with shape (T, H, W, 3), dtype uint8."""

    import cv2

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    frames = []

    try:
        for index in frame_indices.astype(np.int64):

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            ok, frame_bgr = capture.read()

            if not ok:
                raise ValueError(
                    f"Could not read frame {int(index)} from {video_path}"
                )

            frames.append(
                cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB
                )
            )

    finally:
        capture.release()

    return np.stack(
        frames,
        axis=0
    ).astype(np.uint8)


def get_dataset_sample(
    row: dict[str, Any],
    num_frames: int,
    transform=None
) -> dict[str, Any]:
    """Load a dataset sample with optional video transformation."""

    _validate_annotation_row(row)

    video_path = Path(row["video_path"])

    total_frames = _get_total_frames(video_path)

    frame_indices = sample_ordered_frame_indices(
        total_frames,
        num_frames
    )

    frames = load_video_frames(
        video_path,
        frame_indices
    )

    # Apply an optional transformation while keeping the shared frame format.
    if transform is not None:
        frames = transform(frames)

        if (
            not isinstance(frames, np.ndarray)
            or frames.ndim != 4
            or frames.shape[0] != num_frames
            or frames.shape[-1] != 3
            or frames.dtype != np.uint8
        ):
            raise ValueError(
                "Transform must return RGB uint8 frames "
                "with shape (T, H, W, 3)"
            )

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
    """Check whether an annotation contains all required fields."""

    missing = REQUIRED_ANNOTATION_FIELDS.difference(row)

    if missing:
        location = ""

        if path is not None:
            location = f" in {path}"

            if line_number is not None:
                location += f":{line_number}"

        raise ValueError(
            f"Missing annotation fields{location}: {sorted(missing)}"
        )


def _get_total_frames(video_path: Path) -> int:
    """Get the number of frames in a video."""

    import cv2

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    try:
        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    finally:
        capture.release()

    if total_frames <= 0:
        raise ValueError(
            f"Video has no readable frames: {video_path}"
        )

    return total_frames