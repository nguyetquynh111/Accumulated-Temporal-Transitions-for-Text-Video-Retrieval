import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.eval.evaluate import split_rows_by_negative_type
from src.models.transition_model.encoder import save_video_feature_cache
from src.models.transition_model.pipelines.evaluate import evaluate_transition_model
from src.models.transition_model.pipelines.train import (
    EarlyStopping,
    contrastive_loss,
    train_transition_model,
)
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    load_checkpoint,
    rank_candidates,
    score_transition_retrieval,
)
from src.models.transition_model.transition import (
    TransitionEncoder,
    build_state_transition_tokens,
    build_video_tokens,
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
                        "candidate_video_ids": ["v9", "v8", "v0"],
                        "negative_types": ["same_objects", "opposite_action"],
                    }
                ),
                encoding="utf-8",
            )
            contrast = root / "action_contrast.jsonl"
            contrast.write_text(
                json.dumps(
                    {
                        "query_id": "val_q1",
                        "target_video_id": "v9",
                        "candidate_video_ids": ["v9", "v8"],
                        "negative_types": ["opposite_same_object"],
                    }
                ),
                encoding="utf-8",
            )

            checkpoint = train_transition_model(
                train,
                val,
                features,
                root / "checkpoints",
                log_dir=root / "logs",
                epochs=2,
                batch_size=4,
                text_encoder="mock",
                device="cpu",
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(checkpoint.name, "transition_model_seed42_best.pt")
            log = (root / "logs" / "transition_model_seed42_train_log.jsonl").read_text()
            self.assertEqual(len(log.splitlines()), 2)
            model, saved = load_checkpoint(checkpoint)
            self.assertEqual(model.video_input, "state_transition")
            self.assertEqual(saved["training_summary"]["epochs_run"], 2)
            self.assertTrue(saved["training_summary"]["reached_epoch_limit_while_improving"])

            result = evaluate_transition_model(
                checkpoint,
                val,
                features,
                root / "result.json",
                hard_negatives=hard,
                transition_dir=root / "transitions",
                device="cpu",
                action_contrast=contrast,
            )
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["seed"], 42)
            self.assertEqual(result["video_input"], "state_transition")
            self.assertEqual(result["full"]["num_candidates"], 4)
            self.assertEqual(result["hard_negative"]["num_queries"], 1)
            by_type = result["hard_negative"]["by_negative_type"]
            self.assertEqual(set(by_type), {"same_objects", "opposite_action"})
            self.assertEqual(by_type["opposite_action"]["num_negatives"], 1)
            self.assertIn(by_type["opposite_action"]["pairwise_accuracy"], (0.0, 0.5, 1.0))
            self.assertEqual(
                result["hard_negative"]["coverage"]["queries_with_opposite_action"], 1
            )
            self.assertEqual(result["action_contrast"]["num_queries"], 1)
            self.assertTrue((root / "result.json").exists())

            cache = load_transition_feature_cache(root / "transitions" / "v8.npz")
            self.assertEqual(cache["state_embeddings"].shape, (8, 16))
            self.assertEqual(cache["transition_embeddings"].shape, (7, 16))
            self.assertEqual(cache["transition_representation"].shape, (256,))

    def test_run_name_and_model_size_are_configurable(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            video_ids = [f"v{i}" for i in range(6)]
            for vid in video_ids:
                save_video_feature_cache(
                    vid, np.arange(8), rng.normal(size=(8, 16)), features, "mock"
                )
            train = write_split(root, "train", video_ids[:4], ["a", "b", "c", "d"])
            val = write_split(root, "val", video_ids[4:], ["a", "b"])
            options = dict(epochs=1, batch_size=4, text_encoder="mock", device="cpu")

            checkpoint = train_transition_model(
                train,
                val,
                features,
                root / "checkpoints",
                log_dir=root / "logs",
                run_name="small_no_reranker",
                query_conditioned=False,
                model_dim=8,
                num_heads=2,
                num_layers=1,
                selection_metric="recall@1",
                **options,
            )
            self.assertEqual(checkpoint.name, "small_no_reranker_seed42_best.pt")
            self.assertTrue(
                (root / "logs" / "small_no_reranker_seed42_train_log.jsonl").exists()
            )
            model, saved = load_checkpoint(checkpoint)
            self.assertIsNone(model.reranker)
            self.assertEqual(saved["model_config"]["model_dim"], 8)
            self.assertEqual(saved["selection_metric"], "recall@1")
            self.assertEqual(saved["train_config"]["run_name"], "small_no_reranker")

            with self.assertRaisesRegex(ValueError, "selection_metric"):
                train_transition_model(
                    train, val, features, root / "checkpoints", selection_metric="map", **options
                )

    def test_state_transition_tokens_pair_each_state_with_its_change(self):
        states = np.arange(12, dtype=np.float32).reshape(4, 3) ** 2
        tokens = build_state_transition_tokens(states)
        self.assertEqual(tokens.shape, (3, 6))
        np.testing.assert_allclose(tokens[:, :3], states[:-1])
        np.testing.assert_allclose(tokens[:, 3:], states[1:] - states[:-1])
        np.testing.assert_allclose(build_video_tokens(states, "transition"), tokens[:, 3:])
        with self.assertRaisesRegex(ValueError, "video_input"):
            build_video_tokens(states, "states")

    def test_state_and_transition_are_normalized_separately(self):
        model = TransitionRetrievalModel(
            video_dim=4, text_dim=6, model_dim=8, num_heads=2, video_input="state_transition"
        )
        norm = model.video_encoder.input[0]
        tokens = torch.cat((100 * torch.randn(2, 3, 4), 0.5 * torch.randn(2, 3, 4)), dim=-1)
        normalized = norm(tokens)
        # Each half has zero mean and unit variance regardless of its scale.
        for part in normalized.split(4, dim=-1):
            torch.testing.assert_close(part.mean(-1), torch.zeros(2, 3), atol=1e-5, rtol=0)
            torch.testing.assert_close(
                part.var(-1, unbiased=False), torch.ones(2, 3), atol=1e-2, rtol=0
            )
        videos = model.encode_videos(tokens, torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(videos.shape, (2, 8))

    def test_checkpoints_without_video_input_still_load(self):
        model = TransitionRetrievalModel(video_dim=4, text_dim=6, model_dim=8, num_heads=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            config = {k: v for k, v in model.config.items() if k != "video_input"}
            torch.save({"model_config": config, "state_dict": model.state_dict()}, path)
            loaded, _ = load_checkpoint(path)
        self.assertEqual(loaded.video_input, "transition")

    def test_early_stopping_tracks_best_epoch_and_limit(self):
        stopper = EarlyStopping(patience=2)
        improved = [stopper.update(e, s) for e, s in enumerate([0.1, 0.3, 0.2, 0.25], 1)]
        self.assertEqual(improved, [True, True, False, False])
        self.assertTrue(stopper.should_stop)
        summary = stopper.summary(max_epochs=10)
        self.assertEqual((summary["best_epoch"], summary["epochs_run"]), (2, 4))
        self.assertTrue(summary["stopped_early"])
        self.assertFalse(summary["reached_epoch_limit_while_improving"])

        unlimited = EarlyStopping(patience=None)
        for epoch in range(1, 4):
            unlimited.update(epoch, 0.0)
        self.assertFalse(unlimited.should_stop)
        self.assertTrue(unlimited.summary(max_epochs=3)["reached_epoch_limit_while_improving"])

    def test_split_rows_by_negative_type_keeps_target_position(self):
        row = {
            "query_id": "q",
            "target_video_id": "t",
            "candidate_video_ids": ["a", "t", "b", "c"],
            "negative_types": ["same_objects", "opposite_action", "same_objects"],
        }
        by_type = split_rows_by_negative_type([row, {**row, "negative_types": None}])
        self.assertEqual(by_type["same_objects"][0]["candidate_video_ids"], ["a", "t", "c"])
        self.assertEqual(by_type["opposite_action"][0]["candidate_video_ids"], ["t", "b"])
        with self.assertRaisesRegex(ValueError, "negative_types"):
            split_rows_by_negative_type([{**row, "negative_types": ["same_objects"]}])


if __name__ == "__main__":
    unittest.main()
