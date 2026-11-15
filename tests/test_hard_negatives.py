import json

import pytest

from src.eval.evaluate import (
    OPPOSITE_LABEL_PAIRS,
    build_action_contrast_set,
    build_hard_negative_set,
    compute_hard_negative_coverage,
    compute_hard_negative_metrics,
    compute_hard_negative_metrics_by_type,
    find_opposite_label_pairs,
    is_opposite_action,
    save_opposite_label_pairs,
    summarize_candidate_set,
)


def make_sample_annotations():
    # Small dataset with similar objects and different actions.
    examples = [
        ("101", "Picking a cup up", "picking something up", ["cup"]),
        ("102", "Putting a cup on a surface", "putting something on a surface", ["cup"]),
        ("103", "Moving a cup up", "moving something up", ["cup"]),
        ("104", "Moving a box closer to a mug", "moving something closer to something", ["box", "mug"]),
        ("105", "Moving a box away from a mug", "moving something away from something", ["box", "mug"]),
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
        (
            "turning the camera left while filming something",
            "turning the camera right while filming something",
            "lamp",
            "opposite_action",
        ),
        (
            "pushing something from left to right",
            "pushing something from right to left",
            "candle",
            "opposite_action",
        ),
        ("opening something", "closing something", "drawer", "opposite_action"),
        ("turning something upside down", "moving something up", "glass", "same_objects"),
        # Same direction, different verb: not opposite actions.
        (
            "pulling something from left to right",
            "pushing something from left to right",
            "candle",
            "same_objects",
        ),
        ("moving something up", "moving something across a surface until it falls down", "pen", "same_objects"),
        ("picking something up", "putting something on a surface", "something", "other_action"),
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


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_opposite_pairs_are_symmetric_and_exclude_pretending():
    for a, b in OPPOSITE_LABEL_PAIRS:
        assert is_opposite_action(a, b) and is_opposite_action(b, a)
    assert not is_opposite_action(
        "opening something", "pretending to open something without actually opening it"
    )
    assert not is_opposite_action("opening something", "opening something")


def test_opposite_label_pairs_are_written_for_review(tmp_path):
    annotations = make_sample_annotations()
    pairs = find_opposite_label_pairs(annotations)
    assert {(p["label_a"], p["label_b"]) for p in pairs} == {
        ("picking something up", "putting something on a surface"),
        ("moving something away from something", "moving something closer to something"),
    }
    path = save_opposite_label_pairs(annotations, tmp_path / "pairs.tsv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "label_a\tlabel_b\tvideos_a\tvideos_b"
    assert len(lines) == 3


def test_action_contrast_set_has_only_opposite_negatives(tmp_path):
    annotations = make_sample_annotations()
    annotations.append({
        "query_id": "test_107",
        "video_id": "107",
        "target_video_id": "107",
        "text_query": "Putting a pen on a surface",
        "label": "putting something on a surface",
        "objects": ["pen"],
        "split": "test",
    })
    path = build_action_contrast_set(annotations, tmp_path / "action_contrast.jsonl")
    rows = {row["query_id"]: row for row in read_jsonl(path)}

    # Queries whose action has no opposite in the split are left out.
    assert set(rows) == {"test_101", "test_102", "test_104", "test_105", "test_107"}
    for row in rows.values():
        assert row["candidate_video_ids"].count(row["target_video_id"]) == 1
        assert set(row["negative_types"]) == {"opposite_action"}
        assert len(row["negative_types"]) == len(row["candidate_video_ids"]) - 1

    # No filler: picking up has two opposite videos, and the same-object one comes first.
    assert sorted(rows["test_101"]["candidate_video_ids"]) == ["101", "102", "107"]
    assert rows["test_104"]["candidate_video_ids"].count("105") == 1
    assert len(rows["test_104"]["candidate_video_ids"]) == 2

    capped = build_action_contrast_set(annotations, tmp_path / "capped.jsonl", max_negatives=1)
    row = next(r for r in read_jsonl(capped) if r["query_id"] == "test_101")
    assert [v for v in row["candidate_video_ids"] if v != "101"] == ["102"]

    with pytest.raises(ValueError):
        build_action_contrast_set(annotations, tmp_path / "bad.jsonl", max_negatives=0)


def test_metrics_by_negative_type_and_coverage():
    rows = [
        {
            "query_id": "q1",
            "target_video_id": "t1",
            "candidate_video_ids": ["a", "t1", "b", "c"],
            "negative_types": ["opposite_action", "same_objects", "same_objects"],
        },
        {
            "query_id": "q2",
            "target_video_id": "t2",
            "candidate_video_ids": ["t2", "d", "e"],
            "negative_types": ["same_objects", "other_action"],
        },
    ]
    scores = {
        # q1: the opposite action beats the target; one same-object negative ties it.
        ("q1", "t1"): 0.5, ("q1", "a"): 0.9, ("q1", "b"): 0.5, ("q1", "c"): 0.1,
        # q2: the target wins against both negatives.
        ("q2", "t2"): 0.8, ("q2", "d"): 0.2, ("q2", "e"): 0.3,
    }
    by_type = compute_hard_negative_metrics_by_type(rows, scores)

    assert by_type["opposite_action"] == {
        "pairwise_accuracy": 0.0, "recall@1": 0.0, "mrr": 0.5,
        "num_queries": 1, "num_negatives": 1,
    }
    # Pairs: q1 tie (0.5), q1 win (1), q2 win (1). The q1 tie keeps the target
    # first because it comes before b in the candidate list.
    assert by_type["same_objects"]["pairwise_accuracy"] == pytest.approx(2.5 / 3)
    assert by_type["same_objects"]["recall@1"] == pytest.approx(1.0)
    assert by_type["same_objects"]["num_queries"] == 2
    assert by_type["same_objects"]["num_negatives"] == 3
    assert by_type["other_action"]["pairwise_accuracy"] == 1.0

    coverage = compute_hard_negative_coverage(rows)
    assert coverage["num_negatives"] == 5
    assert coverage["negatives_by_type"] == {
        "opposite_action": 1, "other_action": 1, "same_objects": 3,
    }
    assert coverage["negative_share_by_type"]["same_objects"] == pytest.approx(0.6)
    assert coverage["queries_with_opposite_action"] == 1
    assert coverage["queries_with_opposite_action_share"] == pytest.approx(0.5)

    summary = summarize_candidate_set(rows, scores)
    assert summary["recall@1"] == pytest.approx(0.5)
    assert summary["by_negative_type"] == by_type
    assert summary["coverage"] == coverage


def test_metrics_by_type_rejects_mismatched_types():
    rows = [{
        "query_id": "q1",
        "target_video_id": "t1",
        "candidate_video_ids": ["t1", "a", "b"],
        "negative_types": ["same_objects"],
    }]
    scores = {("q1", "t1"): 1.0, ("q1", "a"): 0.0, ("q1", "b"): 0.0}
    with pytest.raises(ValueError):
        compute_hard_negative_metrics_by_type(rows, scores)
