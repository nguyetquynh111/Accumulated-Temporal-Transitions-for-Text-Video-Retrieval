"""Extract and cache ordered video states from annotation files."""

import argparse
from pathlib import Path

from tqdm import tqdm

from config import (
    ANNOTATION_ROOT,
    DEFAULT_NUM_FRAMES,
    FEATURE_ROOT,
    PROJECT_ROOT,
    VJEPA_MODEL_NAME,
)
from src.data.dataset import get_dataset_sample, load_annotations
from src.models.transition_model.encoder import (
    encode_video_frames,
    load_video_feature_cache,
    save_video_feature_cache,
)


def extract_features(
    annotations: Path,
    output_dir: Path,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    encoder_name: str = "vjepa2",
    model_name: str = VJEPA_MODEL_NAME,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Cache unique videos in annotation order and return their cache paths."""
    if not 8 <= num_frames <= 16:
        raise ValueError("num_frames must be between 8 and 16")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if encoder_name not in ("vjepa2", "mock"):
        raise ValueError(f"unknown video encoder: {encoder_name}")
    rows = load_annotations(Path(annotations))
    unique = {}
    for row in rows:
        unique.setdefault(str(row["video_id"]), row)
    selected = list(unique.items())[:limit]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = model_name if encoder_name == "vjepa2" else "mock"
    paths = []
    for video_id, row in tqdm(selected, desc="Extracting video features"):
        path = output_dir / f"{video_id}.npz"
        if path.exists() and not overwrite:
            cached = load_video_feature_cache(path)
            if (
                str(cached["video_id"]) != video_id
                or str(cached["encoder_name"]) != cache_name
                or len(cached["frame_indices"]) != num_frames
            ):
                raise ValueError(
                    f"cache conflicts with requested extraction: {path}; use --overwrite"
                )
            paths.append(path)
            continue
        sample_row = dict(row)
        video_path = Path(sample_row["video_path"])
        if not video_path.is_absolute():
            sample_row["video_path"] = str(PROJECT_ROOT / video_path)
        sample = get_dataset_sample(sample_row, num_frames)
        embeddings = encode_video_frames(
            sample["frames"],
            encoder_name=encoder_name,
            model_name=model_name,
            device=device,
        )
        paths.append(
            save_video_feature_cache(
                video_id, sample["frame_indices"], embeddings, output_dir, cache_name
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=ANNOTATION_ROOT / "train.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=FEATURE_ROOT / "vjepa")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--encoder", choices=("vjepa2", "mock"), default="vjepa2")
    parser.add_argument("--model-name", default=VJEPA_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    )
    print(f"Cached {len(paths)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
