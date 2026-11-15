"""Extract and cache ordered video states from annotation files (T1.4).

Videos are decoded by DataLoader workers and encoded in batches. A run can be
stopped and restarted: videos that already have a cache are skipped. To use
several GPUs, start one process per GPU with the same --num-shards and a
different --shard-index.
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    ANNOTATION_ROOT,
    DEFAULT_NUM_FRAMES,
    MAX_NUM_FRAMES,
    MIN_NUM_FRAMES,
    PROJECT_ROOT,
    VJEPA_FEATURE_ROOT,
    VJEPA_MODEL_NAME,
)
from src.data.dataset import get_dataset_sample, load_annotations
from src.models.transition_model.encoder import (
    VIDEO_ENCODER_NAMES,
    get_video_encoder,
    load_video_feature_cache,
    save_video_feature_cache,
)
from src.models.transition_model.utils import feature_path


# Config

BATCH_SIZE = 8
NUM_WORKERS = 0
FAILURE_LOG_NAME = "extract_failures.jsonl"


# Sub functions

def select_video_rows(
    annotations: Path | list[Path],
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, dict]:
    """Return one annotation row per unique video, in first-seen order.

    `limit` is applied before sharding, so the shards of one run always
    partition the same set of videos.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    paths = [annotations] if isinstance(annotations, (str, Path)) else list(annotations)
    unique = {}
    for path in paths:
        for row in load_annotations(Path(path)):
            unique.setdefault(str(row["video_id"]), row)
    selected = list(unique.items())[:limit]
    return dict(selected[shard_index::num_shards])


def has_valid_cache(path: Path, video_id: str, cache_name: str, num_frames: int) -> bool:
    """Return whether a usable cache exists; raise if it came from other settings."""
    if not path.exists():
        return False
    cached = load_video_feature_cache(path)
    if (
        str(cached["video_id"]) != video_id
        or str(cached["encoder_name"]) != cache_name
        or len(cached["frame_indices"]) != num_frames
    ):
        raise ValueError(f"cache conflicts with requested extraction: {path}; use --overwrite")
    return True


class VideoFrameDataset(Dataset):
    """Decode ordered frames for each video; unreadable videos yield an error."""

    def __init__(self, rows: dict[str, dict], num_frames: int):
        self.items = list(rows.items())
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        video_id, row = self.items[index]
        row = dict(row)
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            row["video_path"] = str(PROJECT_ROOT / video_path)
        try:
            sample = get_dataset_sample(row, self.num_frames)
        except (FileNotFoundError, ValueError) as exc:
            return {"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "video_id": video_id,
            "frame_indices": sample["frame_indices"],
            "frames": sample["frames"],
            "error": None,
        }


def keep_samples(batch: list[dict]) -> list[dict]:
    """Collate by keeping the sample list, since frame sizes differ between videos."""
    return batch


# Main function

def extract_features(
    annotations: Path | list[Path],
    output_dir: Path,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    encoder_name: str = "vjepa2",
    model_name: str = VJEPA_MODEL_NAME,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    num_shards: int = 1,
    shard_index: int = 0,
    strict: bool = False,
) -> list[Path]:
    """Cache unique videos in annotation order and return their cache paths.

    Unreadable videos are recorded in `<output_dir>/extract_failures.jsonl` and
    skipped, unless `strict` is set, in which case the first one raises.
    """
    if not MIN_NUM_FRAMES <= num_frames <= MAX_NUM_FRAMES:
        raise ValueError(f"num_frames must be between {MIN_NUM_FRAMES} and {MAX_NUM_FRAMES}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder = get_video_encoder(encoder_name, model_name, device)
    rows = select_video_rows(annotations, limit, num_shards, shard_index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = {
        video_id: row
        for video_id, row in rows.items()
        if overwrite
        or not has_valid_cache(
            feature_path(output_dir, video_id), video_id, encoder.cache_name, num_frames
        )
    }
    loader = DataLoader(
        VideoFrameDataset(pending, num_frames),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=keep_samples,
    )
    failures = []
    with tqdm(total=len(pending), desc="Extracting video features") as progress:
        for batch in loader:
            failed = [sample for sample in batch if sample["error"] is not None]
            if failed and strict:
                raise ValueError(f"{failed[0]['video_id']}: {failed[0]['error']}")
            failures.extend(
                {"video_id": sample["video_id"], "error": sample["error"]} for sample in failed
            )
            readable = [sample for sample in batch if sample["error"] is None]
            states = encoder.encode_batch([sample["frames"] for sample in readable])
            for sample, embeddings in zip(readable, states):
                save_video_feature_cache(
                    sample["video_id"],
                    sample["frame_indices"],
                    embeddings,
                    output_dir,
                    encoder.cache_name,
                )
            progress.update(len(batch))

    if failures:
        # Appending keeps the records of parallel shards and earlier runs.
        with (output_dir / FAILURE_LOG_NAME).open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure) + "\n")
        print(f"Skipped {len(failures)} unreadable videos; see {output_dir / FAILURE_LOG_NAME}")
    failed_ids = {failure["video_id"] for failure in failures}
    return [feature_path(output_dir, v) for v in rows if v not in failed_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[ANNOTATION_ROOT / "train.jsonl"],
        help="one or more annotation files (default: train.jsonl)",
    )
    parser.add_argument("--output-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--encoder", choices=VIDEO_ENCODER_NAMES, default="vjepa2")
    parser.add_argument("--model-name", default=VJEPA_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS, help="video decoding processes"
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--strict", action="store_true", help="stop at the first unreadable video"
    )
    args = parser.parse_args()
    paths = extract_features(
        args.annotations,
        args.output_dir,
        num_frames=args.num_frames,
        encoder_name=args.encoder,
        model_name=args.model_name,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        strict=args.strict,
    )
    print(f"Cached {len(paths)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
