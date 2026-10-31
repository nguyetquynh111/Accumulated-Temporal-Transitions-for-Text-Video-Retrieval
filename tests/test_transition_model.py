import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.models.transition_model.encoder import save_video_feature_cache
from src.models.transition_model.pipelines.evaluate import evaluate_transition_model
from src.models.transition_model.pipelines.train import (
    contrastive_loss,
    train_transition_model,
)
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    rank_candidates,
    score_transition_retrieval,
)
from src.models.transition_model.transition import (
    TransitionEncoder,
    compute_transitions,
    encode_transition_sequence,
    load_transition_feature_cache,
)


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


class TransitionModelTests(unittest.TestCase):
    def test_transitions_keep_order_and_require_two_states(self):
        states = np.array([[0, 1], [2, 1], [5, 0]], dtype=np.float32)
        np.testing.assert_array_equal(compute_transitions(states), [[2, 0], [3, -1]])
        with self.assertRaises(ValueError):
            compute_transitions(states[:1])

    def test_sequence_encoder_fallback_mask_and_padding(self):
        transitions = np.array([[1, 1], [3, 3], [100, 100]], dtype=np.float32)
        mask = np.array([True, True, False])
        np.testing.assert_array_equal(
            encode_transition_sequence(transitions, mask), [2, 2]
        )

        torch.manual_seed(0)
        encoder = TransitionEncoder(input_dim=2, model_dim=8, num_heads=2)
        padded = encode_transition_sequence(transitions, mask, encoder)
        unpadded = encode_transition_sequence(transitions[:2], encoder=encoder)
        self.assertEqual(padded.shape, (8,))
        np.testing.assert_allclose(padded, unpadded, atol=1e-5)
        np.testing.assert_array_equal(
            padded, encode_transition_sequence(transitions, mask, encoder)
        )

    def test_tiled_scoring_matches_single_pass_and_ranking(self):
        torch.manual_seed(0)
        model = TransitionRetrievalModel(video_dim=4, text_dim=6, model_dim=8, num_heads=2)
        texts = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
        videos = np.random.default_rng(1).normal(size=(7, 8)).astype(np.float32)
        full = score_transition_retrieval(texts, videos, model, chunk_size=100)
        tiled = score_transition_retrieval(texts, videos, model, chunk_size=2)
        self.assertEqual(full.shape, (5, 7))
        np.testing.assert_allclose(full, tiled, atol=1e-5)

        ranked = rank_candidates(np.array([[0.1, 0.9, 0.5]]), ["a", "b", "c"])
        self.assertEqual(ranked, [["b", "c", "a"]])

    def test_contrastive_loss_treats_same_text_as_positive(self):
        text_ids = torch.tensor([0, 0, 1])
        good = torch.tensor([[5.0, 5.0, -5.0], [5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]])
        self.assertLess(contrastive_loss(good, text_ids).item(), 0.01)

    def test_train_and_evaluate_end_to_end(self):
        rng = np.random.default_rng(0)
        texts = ["pushing left", "pushing right", "opening box", "closing box"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            video_ids = [f"v{i}" for i in range(12)]
            for vid in video_ids:
                save_video_feature_cache(
                    vid, np.arange(8), rng.normal(size=(8, 16)), features, "mock"
                )
            train = write_split(root, "train", video_ids[:8], texts * 2)
            val = write_split(root, "val", video_ids[8:], texts)
            hard = root / "hard_negatives.jsonl"
            hard.write_text(
                json.dumps(
                    {
                        "query_id": "val_q0",
                        "target_video_id": "v8",
                        "candidate_video_ids": ["v8", "v9", "v0"],
                    }
                ),
                encoding="utf-8",
            )

            checkpoint = train_transition_model(
                train,
                val,
                features,
                root / "checkpoints",
                epochs=2,
                batch_size=4,
                text_encoder="mock",
                device="cpu",
            )
            self.assertTrue(checkpoint.exists())
            log = (root / "checkpoints" / "train_log.jsonl").read_text().splitlines()
            self.assertEqual(len(log), 2)

            result = evaluate_transition_model(
                checkpoint,
                val,
                features,
                root / "result.json",
                hard_negatives=hard,
                transition_dir=root / "transitions",
                device="cpu",
            )
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["full"]["num_candidates"], 4)
            self.assertEqual(result["hard_negative"]["num_queries"], 1)
            self.assertTrue((root / "result.json").exists())

            cache = load_transition_feature_cache(root / "transitions" / "v8.npz")
            self.assertEqual(cache["state_embeddings"].shape, (8, 16))
            self.assertEqual(cache["transition_embeddings"].shape, (7, 16))
            self.assertEqual(cache["transition_representation"].shape, (256,))


if __name__ == "__main__":
    unittest.main()
