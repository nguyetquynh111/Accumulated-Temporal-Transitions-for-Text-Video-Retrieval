import json

import numpy as np
import pytest

from src.eval.evaluate import (
    compute_retrieval_metrics,
    compute_hard_negative_metrics,
    save_metrics,
)

def test_retrieval_metrics(tmp_path):

    # Three text queries and four candidate videos.
    scores = np.array([
        [0.90, 0.60, 0.20, 0.10],
        [0.90, 0.80, 0.20, 0.10],
        [0.95, 0.85, 0.15, 0.75],
    ], dtype=np.float32)

    query_ids = ["q1", "q2", "q3"]

    candidate_video_ids = ["v1", "v2", "v3", "v4"]

    target_video_ids = ["v1", "v2", "v3"]

    metrics = compute_retrieval_metrics(
        scores,
        query_ids,
        candidate_video_ids,
        target_video_ids,
    )

    # The correct videos are ranked 1st, 2nd, and 4th.
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)

    # MRR = (1/1 + 1/2 + 1/4) / 3
    assert metrics["mrr"] == pytest.approx(7 / 12)

    # Each query has one relevant video, so mAP equals MRR.
    assert metrics["map"] == pytest.approx(7 / 12)

    assert metrics["num_queries"] == 3
    assert metrics["num_candidates"] == 4

    # Verify that results can be saved and loaded as JSON.
    output_path = tmp_path / "retrieval_metrics.json"

    save_metrics(metrics, output_path)

    assert output_path.exists()

    saved_metrics = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_metrics == metrics


def test_missing_target_video():

    scores = np.array([
        [0.90, 0.70]
    ], dtype=np.float32)

    # The correct video v3 is not among the candidates.
    with pytest.raises(
        ValueError,
        match="target_video_id missing"
    ):
        compute_retrieval_metrics(
            scores,
            ["q1"],
            ["v1", "v2"],
            ["v3"],
        )


def test_hard_negative_metrics_with_different_candidates():

    # Each query has its own candidate videos.
    hard_negative_rows = [
        {
            "query_id": "q1",
            "target_video_id": "v2",
            "candidate_video_ids": ["v1", "v2", "v3"],
            "negative_types": ["same_objects", "other_action"],
        },
        {
            "query_id": "q2",
            "target_video_id": "v4",
            "candidate_video_ids": ["v6", "v4", "v5"],
            "negative_types": ["opposite_action", "same_objects"],
        },
    ]

    # Higher scores indicate better text-video matches.
    score_lookup = {
        ("q1", "v1"): 0.2,
        ("q1", "v2"): 0.9,
        ("q1", "v3"): 0.4,
        ("q2", "v6"): 0.9,
        ("q2", "v4"): 0.1,
        ("q2", "v5"): 0.6,
    }

    metrics = compute_hard_negative_metrics(
        hard_negative_rows,
        score_lookup,
        k_values=(1, 2, 3),
    )

    # The correct videos are ranked 1st and 3rd.
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["recall@3"] == pytest.approx(1.0)

    # MRR = (1/1 + 1/3) / 2
    assert metrics["mrr"] == pytest.approx(2 / 3)
    assert metrics["map"] == pytest.approx(2 / 3)

    assert metrics["num_queries"] == 2
    assert metrics["num_candidates"] == 3