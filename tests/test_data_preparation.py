import json
import tempfile
import unittest
from pathlib import Path

from src.data.dataset import (
    compute_dataset_stats,
    create_splits,
    load_annotations,
)


class DataPreparationTests(unittest.TestCase):

    def test_official_metadata_split_is_deterministic_and_has_no_leakage(self):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            metadata = root / "raw"
            metadata.mkdir()

            # Small sample dataset to test our splitting logic.
            train = [
                {
                    "id": str(i),
                    "template": "Picking [something] up",
                    "label": f"Picking object {i} up",
                    "placeholders": [f"object {i}"]
                }
                for i in range(1, 5)
            ]

            validation = [
                {
                    "id": str(i),
                    "template": "Putting [something] down",
                    "label": f"Putting item {i} down"
                }
                for i in range(5, 13)
            ]

            (
                metadata / "something-something-v2-train.json"
            ).write_text(json.dumps(train))

            (
                metadata / "something-something-v2-validation.json"
            ).write_text(json.dumps(validation))

            videos = root / "videos"
            videos.mkdir()

            (videos / "1.webm").touch()

            # Generate the splits twice with the same seed.
            paths = create_splits(
                metadata,
                videos,
                root / "one",
                seed=42
            )

            paths_repeat = create_splits(
                metadata,
                videos,
                root / "two",
                seed=42
            )

            # Both runs should produce identical files.
            for split in ("train", "val", "test"):

                self.assertEqual(
                    paths[split].read_bytes(),
                    paths_repeat[split].read_bytes()
                )

            rows = {
                split: load_annotations(path)
                for split, path in paths.items()
            }

            self.assertEqual(
                [
                    len(rows[split])
                    for split in ("train", "val", "test")
                ],
                [4, 4, 4]
            )

            train_ids = {
                row["video_id"]
                for row in rows["train"]
            }

            val_ids = {
                row["video_id"]
                for row in rows["val"]
            }

            test_ids = {
                row["video_id"]
                for row in rows["test"]
            }

            # Make sure videos do not appear in multiple splits.
            self.assertFalse(
                train_ids & val_ids
                or train_ids & test_ids
                or val_ids & test_ids
            )

            # Each query should point to its matching video.
            self.assertTrue(
                all(
                    row["target_video_id"] == row["video_id"]
                    and row["split"] == split
                    for split, group in rows.items()
                    for row in group
                )
            )

            self.assertEqual(
                rows["train"][0]["text_query"],
                "Picking object 1 up"
            )

            self.assertEqual(
                rows["train"][0]["label"],
                "picking something up"
            )

            self.assertEqual(
                rows["train"][0]["objects"],
                ["object 1"]
            )

            # Verify dataset statistics.
            stats = compute_dataset_stats(paths)

            self.assertEqual(
                stats["splits"],
                {
                    "train": 4,
                    "val": 4,
                    "test": 4
                }
            )

            self.assertEqual(
                stats["num_missing_videos"],
                11
            )

            self.assertEqual(
                stats["labels_by_split"]["train"][
                    "picking something up"
                ],
                4
            )

    def test_duplicate_ids_raise_error(self):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)

            entries = [
                {
                    "id": "1",
                    "template": "Moving [something]"
                }
            ]

            (root / "train.json").write_text(
                json.dumps(entries)
            )

            (root / "validation.json").write_text(
                json.dumps(entries)
            )

            # Overlapping training and validation IDs must be rejected.
            with self.assertRaisesRegex(
                ValueError,
                "overlapping video ids"
            ):

                create_splits(
                    root,
                    root / "videos",
                    root / "out",
                    seed=42
                )


if __name__ == "__main__":
    unittest.main()