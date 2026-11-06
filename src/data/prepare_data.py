"""Generate the project annotation splits from the official SSv2 metadata."""

import argparse
import json
from pathlib import Path

from config import (
    ANNOTATION_ROOT,
    RANDOM_SEED,
    RESULT_ROOT,
    VIDEO_ROOT,
)

from src.data.dataset import (
    compute_dataset_stats,
    create_splits,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prepare Something-Something V2 retrieval splits"
    )

    parser.add_argument(
        "--raw-metadata",
        type=Path,
        required=True,
        help="Folder containing the official train and validation JSON metadata"
    )

    parser.add_argument(
        "--video-root",
        type=Path,
        default=VIDEO_ROOT
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANNOTATION_ROOT
    )

    parser.add_argument(
        "--stats-output",
        type=Path,
        default=RESULT_ROOT / "dataset_stats.json"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    # Create the train, validation, and held-out test splits.
    paths = create_splits(
        args.raw_metadata,
        args.video_root,
        args.output_dir,
        args.seed
    )

    # Calculate statistics for the generated splits.
    stats = compute_dataset_stats(paths)

    args.stats_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.stats_output.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8"
    )

    print("Saved annotation files:")

    for split, path in paths.items():
        print(
            f"  {split}: {path} "
            f"({stats['splits'][split]} queries)"
        )

    print(
        f"Missing videos: {stats['num_missing_videos']}"
    )

    print(
        f"Dataset statistics: {args.stats_output}"
    )


if __name__ == "__main__":
    main()