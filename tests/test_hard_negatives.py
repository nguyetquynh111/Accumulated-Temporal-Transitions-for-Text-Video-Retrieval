import json

import pytest

from src.eval.evaluate import (
    build_hard_negative_set,
    compute_hard_negative_metrics,
)


def make_sample_annotations():
    # Small dataset with similar objects and different actions.
    examples = [
        ("101", "Picking a cup up", "picking something up", ["cup"]),
        ("102", "Putting a cup down", "putting something down", ["cup"]),
        ("103", "Moving a cup sideways", "moving something sideways", ["cup"]),
        ("104", "Pulling a box closer", "pulling something closer", ["box"]),
        ("105", "Pushing a box away", "pushing something away", ["box"]),
        ("106", "Shaking a bottle", "shaking something", ["bottle"]),
    ]

    annotations = []

    for video_id, text, label, objects in examples:
        annotations.append({
            "query_id": f"test_{video_id}",
            "video_id": video_id,
            "target_video_id": video_id,
            "text_query": text,
            "label": label,
            "objects": objects,
            "split": "test",
        })

    return annotations


def test_hard_negative_generation(tmp_path):

    annotations = make_sample_annotations()

    output_path = tmp_path / "hard_negatives.jsonl"

    result_path = build_hard_negative_set(
        annotations,
        output_path
    )

    assert result_path == output_path
    assert output_path.exists()

    # Read the generated hard-negative examples.
    with output_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    assert len(rows) == len(annotations)

    available_ids = {
        row["video_id"] for row in annotations
    }

    for row in rows:
        target = row["target_video_id"]
        candidates = row["candidate_video_ids"]

        # Every query should have exactly one correct video.
        assert candidates.count(target) == 1

        # Candidate videos should be unique.
        assert len(candidates) == len(set(candidates))

        # At least two negative videos should be included.
        assert 3 <= len(candidates) <= 10

        # All candidates must belong to this dataset.
        assert set(candidates).issubset(available_ids)

        # Each negative should have a corresponding type.
        assert len(row["negative_types"]) == len(candidates) - 1

        assert all(
            kind in {
                "opposite_action",
                "same_objects",
                "other_action",
            }
            for kind in row["negative_types"]
        )

    # Check whether the cup example includes difficult negatives.
    cup_query = next(
        row for row in rows
        if row["query_id"] == "test_101"
    )

    assert "102" in cup_query["candidate_video_ids"]
    assert "103" in cup_query["candidate_video_ids"]

    assert "opposite_action" in cup_query["negative_types"]
    assert "same_objects" in cup_query["negative_types"]

    # Running with the same input should give identical results.
    repeat_path = tmp_path / "hard_negatives_repeat.jsonl"

    build_hard_negative_set(
        annotations,
        repeat_path
    )

    assert output_path.read_bytes() == repeat_path.read_bytes()


def test_hard_negative_evaluation(tmp_path):

    annotations = make_sample_annotations()

    output_path = tmp_path / "hard_negatives.jsonl"

    build_hard_negative_set(
        annotations,
        output_path
    )

    with output_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # Simulate a model that always scores the correct video highest.
    score_lookup = {}

    for row in rows:
        query_id = row["query_id"]
        target_id = row["target_video_id"]

        for video_id in row["candidate_video_ids"]:
            score_lookup[(query_id, video_id)] = (
                1.0 if video_id == target_id else 0.0
            )

    metrics = compute_hard_negative_metrics(
        rows,
        score_lookup
    )

    # A perfect model should retrieve every correct video first.
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)

    assert metrics["num_queries"] == len(annotations)


def test_hard_negative_split_validation(tmp_path):

    annotations = make_sample_annotations()

    # Introduce an incorrect split.
    annotations[0]["split"] = "val"

    output_path = tmp_path / "invalid_negatives.jsonl"

    # Mixing validation and test videos should be rejected.
    with pytest.raises(ValueError):
        build_hard_negative_set(
            annotations,
            output_path
        )

    # Empty annotations should also be rejected.
    with pytest.raises(ValueError):
        build_hard_negative_set(
            [],
            output_path
        )


@pytest.mark.parametrize(
    "first_label, second_label, object_name, expected_type",
    [
        ("moving something up", "moving something down", "phone", "opposite_action"),
        ("turning something left", "turning something right", "camera", "opposite_action"),
        ("pulling something", "pushing something", "candle", "opposite_action"),
        ("opening something", "closing something", "drawer", "opposite_action"),
        ("turning something upside down", "moving something up", "glass", "same_objects"),
        ("picking something up", "putting something down", "something", "other_action"),
        ("closing something", "pretending to open something without actually opening it", "drawer", "same_objects"),
    ],
)
def test_hard_negative_action_matching(
    tmp_path, first_label, second_label, object_name, expected_type
):
    # Create a small dataset to check how negative pairs are classified.
    def make_row(video_id, label, obj):
        return {
            "query_id": f"test_{video_id}",
            "video_id": video_id,
            "target_video_id": video_id,
            "text_query": label.replace("something", obj),
            "label": label,
            "objects": [obj],
            "split": "test",
        }

    annotations = [
        make_row("201", first_label, object_name),
        make_row("202", second_label, object_name),
        make_row("203", "shaking something", "bottle"),
    ]

    output_path = tmp_path / "matching_test.jsonl"

    build_hard_negative_set(annotations, output_path)

    with output_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # Find the query for the first video.
    query_row = next(
        row for row in rows
        if row["query_id"] == "test_201"
    )

    target_id = query_row["target_video_id"]

    # Negative types follow the candidate order, excluding the target.
    negative_ids = [
        vid for vid in query_row["candidate_video_ids"]
        if vid != target_id
    ]

    negative_type_by_id = dict(
        zip(negative_ids, query_row["negative_types"], strict=True)
    )

    assert negative_type_by_id["202"] == expected_type