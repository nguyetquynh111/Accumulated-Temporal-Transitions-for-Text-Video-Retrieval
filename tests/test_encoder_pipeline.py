import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.models.transition_model.encoder import (
    VideoEncoder,
    encode_text_queries,
    encode_video_frames,
    load_video_feature_cache,
    save_video_feature_cache,
)
from src.models.transition_model.pipelines.extract_features import (
    extract_features,
    select_video_rows,
)


def write_annotations(path: Path, video_ids: list[str]) -> Path:
    rows = [
        {
            "query_id": f"{path.stem}_q{i}",
            "video_id": vid,
            "video_path": f"data/videos/{vid}.webm",
            "text_query": "action",
            "target_video_id": vid,
            "split": path.stem,
        }
        for i, vid in enumerate(video_ids)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class EncoderPipelineTests(unittest.TestCase):
    def test_mock_video_order_and_cache_round_trip(self):
        frames = np.stack(
            [np.full((2, 3, 3), v, dtype=np.uint8) for v in (10, 60, 200)]
        )
        states = encode_video_frames(frames, encoder_name="mock")
        np.testing.assert_allclose(
            states[:, 0], np.array([10, 60, 200]) / 255, atol=1e-6
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_video_feature_cache(
                "v1", np.array([0, 2, 4]), states, Path(tmp), "mock"
            )
            cache = load_video_feature_cache(path)
            self.assertEqual(str(cache["video_id"]), "v1")
            self.assertEqual(str(cache["encoder_name"]), "mock")
            self.assertEqual(cache["frame_indices"].dtype, np.int64)
            self.assertEqual(cache["frame_embeddings"].dtype, np.float32)
            np.testing.assert_array_equal(cache["frame_embeddings"], states)
            np.testing.assert_allclose(cache["pooled_embedding"], states.mean(axis=0))

    def test_spatial_patches_are_pooled_before_temporal_interpolation(self):
        # Two temporal tubelets, each with two spatial patches and one feature.
        hidden = np.array([[0], [2], [10], [12]], dtype=np.float32)
        states = VideoEncoder(encoder_name="mock").pool_temporal_tokens(
            hidden, num_frames=4, tubelet_size=2
        )
        np.testing.assert_allclose(states[:, 0], [1, 3.5, 8.5, 11])

    def test_text_order_empty_rejection_and_mock_repeatability(self):
        texts = ["open the box", "close the box", "open the box"]
        embeddings = encode_text_queries(texts, encoder_name="mock", batch_size=2)
        self.assertEqual(embeddings.dtype, np.float32)
        np.testing.assert_array_equal(embeddings[0], embeddings[2])
        self.assertFalse(np.array_equal(embeddings[0], embeddings[1]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            encode_text_queries(["okay", "  "], encoder_name="mock")

    def test_pipeline_deduplicates_and_rejects_conflicting_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "train.jsonl"
            rows = [
                {
                    "query_id": f"q{i}",
                    "video_id": vid,
                    "video_path": f"data/videos/{vid}.webm",
                    "text_query": "action",
                    "target_video_id": vid,
                    "split": "train",
                }
                for i, vid in enumerate(("v1", "v1", "v2"))
            ]
            annotations.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            frames = np.zeros((8, 2, 2, 3), dtype=np.uint8)
            sample = {"frame_indices": np.arange(8), "frames": frames}
            with patch(
                "src.models.transition_model.pipelines.extract_features.get_dataset_sample",
                return_value=sample,
            ) as loader:
                paths = extract_features(
                    annotations, root / "features", num_frames=8, encoder_name="mock"
                )
                self.assertEqual(len(paths), 2)
                self.assertEqual(loader.call_count, 2)
                extract_features(
                    annotations, root / "features", num_frames=8, encoder_name="mock"
                )
                self.assertEqual(loader.call_count, 2)
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    extract_features(
                        annotations,
                        root / "features",
                        num_frames=16,
                        encoder_name="mock",
                    )

    def test_batch_encoding_matches_single_video_encoding(self):
        rng = np.random.default_rng(0)
        videos = [
            rng.integers(0, 256, size=(8, 4, width, 3), dtype=np.uint8) for width in (4, 6)
        ]
        encoder = VideoEncoder(encoder_name="mock")
        batch = encoder.encode_batch(videos)
        self.assertEqual(len(batch), 2)
        for frames, states in zip(videos, batch):
            np.testing.assert_array_equal(states, encoder.encode(frames))
        self.assertEqual(encoder.encode_batch([]), [])

    def test_shards_partition_the_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_annotations(root / "train.jsonl", ["v1", "v2", "v1", "v3"])
            second = write_annotations(root / "val.jsonl", ["v3", "v4", "v5"])
            everything = select_video_rows([first, second])
            self.assertEqual(list(everything), ["v1", "v2", "v3", "v4", "v5"])
            shards = [select_video_rows([first, second], None, 2, i) for i in range(2)]
            self.assertEqual(list(shards[0]), ["v1", "v3", "v5"])
            self.assertEqual(list(shards[1]), ["v2", "v4"])
            with self.assertRaisesRegex(ValueError, "shard_index"):
                select_video_rows(first, None, 2, 2)

    def test_unreadable_videos_are_logged_and_skipped(self):
        def fake_sample(row, num_frames):
            if row["video_id"] == "v2":
                raise FileNotFoundError(row["video_path"])
            return {
                "frame_indices": np.arange(num_frames),
                "frames": np.zeros((num_frames, 2, 2, 3), dtype=np.uint8),
            }

        target = "src.models.transition_model.pipelines.extract_features.get_dataset_sample"
        with tempfile.TemporaryDirectory() as tmp, patch(target, side_effect=fake_sample):
            root = Path(tmp)
            annotations = write_annotations(root / "train.jsonl", ["v1", "v2", "v3"])
            options = dict(num_frames=8, encoder_name="mock", batch_size=2)
            paths = extract_features(annotations, root / "features", **options)
            self.assertEqual([path.stem for path in paths], ["v1", "v3"])
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(
                sorted(path.name for path in (root / "features").iterdir()),
                ["extract_failures.jsonl", "v1.npz", "v3.npz"],
            )
            failures = (root / "features" / "extract_failures.jsonl").read_text()
            self.assertEqual(json.loads(failures)["video_id"], "v2")
            with self.assertRaisesRegex(ValueError, "v2"):
                extract_features(annotations, root / "features", strict=True, **options)

    def test_relative_config_env_path_uses_project_root(self):
        env = dict(os.environ, ATTVR_OUTPUT_ROOT="relative_outputs")
        command = [sys.executable, "-c", "import config; print(config.OUTPUT_ROOT)"]
        result = subprocess.run(
            command,
            env=env,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            Path(result.stdout.strip()),
            Path(__file__).resolve().parents[1] / "relative_outputs",
        )


if __name__ == "__main__":
    unittest.main()
