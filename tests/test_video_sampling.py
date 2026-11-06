import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset import (
    get_dataset_sample,
    load_video_frames,
    sample_ordered_frame_indices,
)


def test_frame_loading_and_dataset_sample():

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "sample.avi"

        # Create a small video with four colored frames.
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (64, 48),
        )

        assert writer.isOpened(), "Could not create temporary test video"

        colors_bgr = [
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
        ]

        for color in colors_bgr:
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        # Sample eight frames from a four-frame video.
        indices = sample_ordered_frame_indices(4, 8)
        frames = load_video_frames(video_path, indices)

        assert frames.shape == (8, 48, 64, 3)
        assert frames.dtype == np.uint8

        # Check frame order and repeated indices.
        assert np.all(np.diff(indices) >= 0)
        assert len(set(indices.tolist())) < 8

        # Verify RGB conversion using red and blue frames.
        assert int(frames[0, 20, 20, 0]) > int(frames[0, 20, 20, 2]) + 80
        assert int(frames[-1, 20, 20, 2]) > int(frames[-1, 20, 20, 0]) + 80

        # Test the complete dataset sample function.
        row = {
            "query_id": "q_demo",
            "video_id": "demo",
            "video_path": str(video_path),
            "text_query": "moving an object",
            "target_video_id": "demo",
            "split": "val",
        }

        sample = get_dataset_sample(row, num_frames=8)

        assert np.array_equal(sample["frame_indices"], indices)
        assert np.array_equal(sample["frames"], frames)
        assert sample["text_query"] == row["text_query"]