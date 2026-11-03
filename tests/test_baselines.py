import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.models.baselines.clip_baseline import (
    encode_clip_frames,
    encode_clip_text_queries,
    pool_frame_embeddings,
    run_clip_baseline,
    score_clip_retrieval,
)
from src.models.baselines.common import load_jsonl, ranking_records
from src.models.baselines.error_analysis import (
    categorize_error,
    categorize_retrieval_errors,
    compare_models,
    run_error_analysis,
)
from src.models.baselines.vjepa_baseline import (
    ABLATION_VARIANTS,
    evaluate_variant,
    run_transition_ablation,
    score_vjepa_baseline,
    train_variant_model,
    variant_sequence,
)
from src.models.transition_model.encoder import save_video_feature_cache


def write_split(root: Path, split: str, video_ids: list[str], texts: list[str]) -> Path:
    path = root / f"{split}.jsonl"
    rows = [
        {
            "query_id": f"{split}_q{i}",
            "video_id": vid,
            "video_path": f"data/videos/{vid}.webm",
            "text_query": text,
            "target_video_id": vid,
            "split": split,
        }
        for i, (vid, text) in enumerate(zip(video_ids, texts))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


class ClipBaselineTests(unittest.TestCase):
    def test_pooling_scoring_and_mock_encoders(self):
        frames = np.stack([np.full((2, 3, 3), v, dtype=np.uint8) for v in (10, 60, 200)])
        frame_embeddings = encode_clip_frames(frames, encoder_name="mock")
        text_embeddings = encode_clip_text_queries(["open box", "close box"], encoder_name="mock")
        self.assertEqual(frame_embeddings.shape, (3, 64))
        self.assertEqual(text_embeddings.shape, (2, 64))
        np.testing.assert_allclose(np.linalg.norm(frame_embeddings, axis=1), 1, atol=1e-5)

        pooled = pool_frame_embeddings(np.array([[1, 3], [3, 5]], dtype=np.float32))
        np.testing.assert_array_equal(pooled, [2, 4])
        scores = score_clip_retrieval(text_embeddings, frame_embeddings)
        self.assertEqual(scores.shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "empty"):
            encode_clip_text_queries(["ok", " "], encoder_name="mock")

    def test_run_end_to_end_with_mock_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = write_split(root, "test", ["v1", "v2", "v3"], ["open box", "close box", "push cup"])
            hard = root / "hard_negatives.jsonl"
            hard.write_text(
                json.dumps(
                    {"query_id": "test_q0", "target_video_id": "v1", "candidate_video_ids": ["v1", "v2"]}
                ),
                encoding="utf-8",
            )

            def fake_sample(row, num_frames):
                if row["video_id"] == "v3":
                    raise FileNotFoundError(row["video_path"])
                value = 50 if row["video_id"] == "v1" else 150
                return {
                    "frame_indices": np.arange(num_frames),
                    "frames": np.full((num_frames, 4, 4, 3), value, dtype=np.uint8),
                }

            with patch(
                "src.models.baselines.clip_baseline.get_dataset_sample", side_effect=fake_sample
            ):
                result = run_clip_baseline(
                    test,
                    root / "features",
                    root / "clip.json",
                    hard_negatives=hard,
                    rankings_path=root / "rankings.jsonl",
                    num_frames=8,
                    encoder_name="mock",
                )
            self.assertEqual(result["full"]["num_queries"], 2)
            self.assertEqual(result["full"]["num_candidates"], 2)
            self.assertEqual(result["skipped_missing_videos"], 1)
            self.assertEqual(result["unreadable_video_ids"], ["v3"])
            self.assertEqual(result["hard_negative"]["num_queries"], 1)
            self.assertEqual(len(load_jsonl(root / "rankings.jsonl")), 2)


class VjepaBaselineTests(unittest.TestCase):
    def test_variant_sequences(self):
        states = np.array([[0, 0], [1, 2], [4, 4]], dtype=np.float32)
        self.assertEqual(variant_sequence("pooled", states).shape, (1, 2))
        np.testing.assert_array_equal(variant_sequence("no_transition", states), states)
        np.testing.assert_array_equal(variant_sequence("full_transition", states), [[1, 2], [3, 2]])
        np.testing.assert_array_equal(variant_sequence("final_minus_initial", states), [[4, 4]])
        np.testing.assert_array_equal(variant_sequence("mean_transition", states), [[2, 2]])
        with self.assertRaises(ValueError):
            variant_sequence("unknown", states)

    def test_untrained_scoring_keeps_candidate_order(self):
        states = {"a": np.array([[0, 0], [1, 0]]), "b": np.array([[0, 0], [0, 1]])}
        texts = np.array([[1, 0], [0, 1]], dtype=np.float32)
        scores = run_transition_ablation("full_transition", texts, states, {})
        np.testing.assert_array_equal(scores, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(score_vjepa_baseline(texts, texts), np.eye(2))

    def test_train_and_evaluate_variants_end_to_end(self):
        rng = np.random.default_rng(0)
        texts = ["pushing left", "pushing right", "opening box", "closing box"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            video_ids = [f"v{i}" for i in range(12)]
            for vid in video_ids:
                save_video_feature_cache(vid, np.arange(8), rng.normal(size=(8, 16)), features, "mock")
            train = write_split(root, "train", video_ids[:8], texts * 2)
            val = write_split(root, "val", video_ids[8:], texts)
            hard = root / "hard_negatives.jsonl"
            hard.write_text(
                json.dumps(
                    {"query_id": "val_q0", "target_video_id": "v8", "candidate_video_ids": ["v8", "v9", "v0"]}
                ),
                encoding="utf-8",
            )

            for variant in ("pooled", *ABLATION_VARIANTS):
                checkpoint = train_variant_model(
                    variant,
                    train,
                    val,
                    features,
                    root / "checkpoints",
                    epochs=1,
                    batch_size=4,
                    text_encoder="mock",
                    device="cpu",
                )
                result = evaluate_variant(
                    variant,
                    checkpoint,
                    val,
                    features,
                    root / f"{variant}.json",
                    hard_negatives=hard,
                    rankings_path=root / f"rankings_{variant}.jsonl",
                    device="cpu",
                )
                self.assertEqual(result["variant"], variant)
                self.assertEqual(result["full"]["num_candidates"], 4)
                self.assertEqual(result["hard_negative"]["num_queries"], 1)
                self.assertEqual(len(load_jsonl(root / f"rankings_{variant}.jsonl")), 4)

            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluate_variant(
                    "pooled",
                    root / "checkpoints" / "ablation_no_transition_best.pt",
                    val,
                    features,
                    root / "bad.json",
                    device="cpu",
                )


class ErrorAnalysisTests(unittest.TestCase):
    def row(self, template, label=None, text=None):
        label = label or template.replace("[something]", "cup")
        return {"template": template, "label": label, "text_query": text or label}

    def test_rule_categories(self):
        cases = [
            ("Picking [something] up", "Putting [something] down", "wrong_temporal_direction"),
            (
                "Pushing [something] from left to right",
                "Pushing [something] from right to left",
                "wrong_temporal_direction",
            ),
            (
                "Putting [something] on [something]",
                "Pretending to put [something] on [something]",
                "similar_start_end_different_motion",
            ),
            ("Tearing [something] into two pieces", "Squeezing [something]", "correct_objects_wrong_action"),
        ]
        for target, retrieved, expected in cases:
            with self.subTest(target=target):
                self.assertEqual(
                    categorize_error(self.row(target), self.row(retrieved), 2, 100), expected
                )

        same_text = self.row("Opening [something]")
        self.assertEqual(categorize_error(same_text, dict(same_text), 2, 100), "text_ambiguity")
        unrelated = {"text_query": "spinning a coin"}
        self.assertEqual(
            categorize_error({"text_query": "folding paper"}, unrelated, 90, 100), "encoder_failure"
        )
        self.assertEqual(
            categorize_error({"text_query": "folding paper"}, unrelated, 3, 100), "visual_ambiguity"
        )

    def test_ranking_errors_comparison_and_report(self):
        rows = [
            {"query_id": "q1", "text_query": "picking cup up", "target_video_id": "a"},
            {"query_id": "q2", "text_query": "putting cup down", "target_video_id": "b"},
        ]
        scores = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
        baseline = ranking_records("baseline", rows, scores, ["a", "b"])
        self.assertEqual([r["target_rank"] for r in baseline], [2, 1])
        model = ranking_records("model", rows, scores[:, ::-1].copy(), ["a", "b"])

        annotations = {
            "a": {**self.row("Picking [something] up"), "video_id": "a"},
            "b": {**self.row("Putting [something] down"), "video_id": "b"},
        }
        errors = categorize_retrieval_errors(baseline, annotations)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["category"], "wrong_temporal_direction")
        self.assertEqual(errors[0]["retrieved_video_id"], "b")

        comparisons = compare_models({"model": model, "baseline": baseline})
        model_wins = next(c for c in comparisons if c["winner"] == "model")
        self.assertEqual(model_wins["num_wins"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "test.jsonl"
            split.write_text(
                "\n".join(
                    json.dumps(
                        {
                            **annotations[v],
                            "query_id": q,
                            "video_path": f"{v}.webm",
                            "target_video_id": v,
                            "split": "test",
                        }
                    )
                    for q, v in (("q1", "a"), ("q2", "b"))
                ),
                encoding="utf-8",
            )
            paths = []
            for name, records in (("baseline", baseline), ("model", model)):
                path = root / f"rankings_{name}.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
                paths.append(path)
            result = run_error_analysis(paths, split, root / "errors.json")
            self.assertEqual(result["summary"]["baseline"]["counts"]["wrong_temporal_direction"], 1)
            self.assertTrue((root / "errors.json").exists())


if __name__ == "__main__":
    unittest.main()
