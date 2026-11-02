import random
import re
from collections import defaultdict

from config import RANDOM_SEED

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_retrieval_metrics(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_video_ids: list[str],
    target_video_ids: list[str],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Return Recall@K, MRR, mAP, and count metadata."""
    scores = np.asarray(scores, dtype=np.float32)
    _validate_retrieval_inputs(scores, query_ids, candidate_video_ids, target_video_ids)

    candidate_to_index = {video_id: index for index, video_id in enumerate(candidate_video_ids)}
    ranks = []
    for row, target_video_id in zip(scores, target_video_ids, strict=True):
        if target_video_id not in candidate_to_index:
            raise ValueError(f"target_video_id missing from candidates: {target_video_id}")
        target_index = candidate_to_index[target_video_id]
        order = np.argsort(-row, kind="stable")
        rank = int(np.where(order == target_index)[0][0]) + 1
        ranks.append(rank)

    ranks_array = np.asarray(ranks, dtype=np.float32)
    metrics: dict[str, float | int] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["map"] = metrics["mrr"]
    metrics["num_queries"] = len(query_ids)
    metrics["num_candidates"] = len(candidate_video_ids)
    return metrics


def compute_hard_negative_metrics(
    hard_negative_rows: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str], float],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Evaluate per-query hard-negative candidate sets."""
    ranks = []
    candidate_counts = []
    for row in hard_negative_rows:
        query_id = row["query_id"]
        target_video_id = row["target_video_id"]
        candidate_video_ids = row["candidate_video_ids"]
        if target_video_id not in candidate_video_ids:
            raise ValueError(f"target_video_id missing from hard negatives: {query_id}")

        scores = np.asarray(
            [score_lookup[(query_id, video_id)] for video_id in candidate_video_ids],
            dtype=np.float32,
        )
        target_index = candidate_video_ids.index(target_video_id)
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == target_index)[0][0]) + 1
        ranks.append(rank)
        candidate_counts.append(len(candidate_video_ids))

    if not ranks:
        raise ValueError("hard_negative_rows must not be empty")

    ranks_array = np.asarray(ranks, dtype=np.float32)
    metrics: dict[str, float | int] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["map"] = metrics["mrr"]
    metrics["num_queries"] = len(hard_negative_rows)
    metrics["num_candidates"] = int(max(candidate_counts))
    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_hard_negative_set(
    annotations: list[dict],
    output_path: Path,
) -> Path:
    """Build hard-negative candidate sets from one evaluation split."""

    if not annotations:
        raise ValueError("annotations must not be empty")

    # Use only one evaluation split to prevent data leakage.
    splits = {str(row.get("split", "")) for row in annotations}

    if len(splits) != 1 or not splits.issubset({"val", "test"}):
        raise ValueError(
            "Pass annotations from one split: val OR test"
        )

    rows = sorted(
        annotations,
        key=lambda row: str(row["query_id"])
    )

    required = {
        "query_id",
        "target_video_id",
        "video_id",
        "text_query",
    }

    for row in rows:
        if not required.issubset(row):
            raise ValueError(
                f"Missing required fields: {required - set(row)}"
            )

        if str(row["video_id"]) != str(row["target_video_id"]):
            raise ValueError(
                "Each source video must match its target_video_id"
            )

    video_by_id = {
        str(row["video_id"]): row
        for row in rows
    }

    if len(video_by_id) != len(rows):
        raise ValueError(
            "Expected one annotation per video_id"
        )

    if len({
        str(row["query_id"]) for row in rows
    }) != len(rows):
        raise ValueError("query_id values must be unique")

    def normalized(value):
        return " ".join(
            str(value)
            .lower()
            .replace("[", "")
            .replace("]", "")
            .split()
        )

    def opposite(a, b):
        # Pretending an action does not produce the actual state change.
        pretend_pattern = r"\bpretend(?:s|ed|ing)?\b"

        if re.search(pretend_pattern, a) or re.search(pretend_pattern, b):
            return False
        # Opposite action patterns commonly found in SSv2.
        pairs = (
            ("left to right", "right to left"),
            ("opening", "closing"),
            ("picking", "putting"),
            ("taking", "putting"),
            ("pulling", "pushing"),
            ("covering", "uncovering"),
            ("closer", "away"),
            ("upwards", "downwards"),
        )

        # Check the existing opposite-action patterns.
        for x, y in pairs:
            a_x = bool(re.search(r"\b" + re.escape(x) + r"\b", a))
            a_y = bool(re.search(r"\b" + re.escape(y) + r"\b", a))
            b_x = bool(re.search(r"\b" + re.escape(x) + r"\b", b))
            b_y = bool(re.search(r"\b" + re.escape(y) + r"\b", b))

            if (a_x and b_y) or (a_y and b_x):
                return True
        # Direction-based opposites should use the same type of movement.
        motion_verbs = (
            "moving",
            "turning",
            "sliding",
            "tilting",
            "rotating",
            "shifting",
            "dragging",
            "rolling",
            "pushing",
            "pulling",
        )

        same_motion = any(
            re.search(r"\b" + verb + r"\b", a)
            and re.search(r"\b" + verb + r"\b", b)
            for verb in motion_verbs
        )

        if not same_motion:
            return False
        # Also recognize individual opposite directions.
        direction_pairs = (
            ("left", "right"),
            ("up", "down"),
        )

        for x, y in direction_pairs:
            a_x = bool(re.search(r"\b" + x + r"\b", a))
            a_y = bool(re.search(r"\b" + y + r"\b", a))
            b_x = bool(re.search(r"\b" + x + r"\b", b))
            b_y = bool(re.search(r"\b" + y + r"\b", b))

            # Avoid treating phrases with both directions as opposites.
            if (
                (a_x and not a_y and b_y and not b_x)
                or
                (a_y and not a_x and b_x and not b_y)
            ):
                return True

        return False

    def objects(row):
        values = row.get("objects") or []

        if not isinstance(values, list):
            return set()

        return {
            normalized(value)
            for value in values
            if normalized(value) not in {
                "",
                "something",
                "anything",
                "object",
                "item",
            }
        }
    def primary_object(row):
        # Use the first annotated object, ignoring generic placeholders.
        values = row.get("objects") or []

        if not isinstance(values, list) or not values:
            return ""

        primary = normalized(values[0])

        generic_objects = {
            "",
            "something",
            "anything",
            "object",
            "item",
            "stuff",
        }

        if primary in generic_objects:
            return ""

        return primary

    labels = {
        vid: normalized(
            row.get("label")
            or row.get("template")
            or row["text_query"]
        )
        for vid, row in video_by_id.items()
    }

    texts = {
        vid: normalized(row["text_query"])
        for vid, row in video_by_id.items()
    }

    by_label = defaultdict(list)
    by_object = defaultdict(list)

    for video_id in sorted(video_by_id):
        by_label[labels[video_id]].append(video_id)

        for obj in sorted(objects(video_by_id[video_id])):
            by_object[obj].append(video_id)

    # Find the action labels that are potential opposites.
    opposite_labels = {
        label: [
            other
            for other in sorted(by_label)
            if other != label and opposite(label, other)
        ]
        for label in by_label
    }

    all_video_ids = sorted(video_by_id)

    rng = random.Random(RANDOM_SEED)

    output_rows = []

    for row in rows:
        target_id = str(row["target_video_id"])
        label = labels[target_id]

        opposite_ids = list(dict.fromkeys(
            vid
            for action in opposite_labels[label]
            for vid in by_label[action]
        ))

        shared_ids = list(dict.fromkeys(
            vid
            for obj in sorted(objects(row))
            for vid in by_object[obj]
        ))

        negatives = []
        negative_types = []
        seen = {target_id}

        def add(pool, kind):
            if not pool or len(negatives) == 9:
                return

            start = rng.randrange(len(pool))

            for i in range(len(pool)):
                vid = pool[(start + i) % len(pool)]

                # Do not use the target or an identical query.
                if (
                    vid in seen
                    or texts[vid] == texts[target_id]
                    or labels[vid] == label
                ):
                    continue

                negatives.append(vid)
                negative_types.append(kind)
                seen.add(vid)

                if len(negatives) == 9:
                    break

        # Priority 1: Opposite actions involving the same primary object.
        primary = primary_object(row)

        matching_opposites = [
            vid for vid in opposite_ids
            if primary
            and primary_object(video_by_id[vid]) == primary
        ]

        add(matching_opposites, "opposite_action")

        # Priority 2: Different actions involving shared objects.
        add(shared_ids, "same_objects")

        # Priority 3: Remaining different-action videos.
        add(all_video_ids, "other_action")

        if len(negatives) < 2:
            raise ValueError(
                f"Not enough distinct negative videos for query "
                f"{row['query_id']}"
            )

        # Vary the correct video's position to avoid ranking bias.
        candidates = negatives.copy()

        candidates.insert(
            rng.randrange(len(candidates) + 1),
            target_id
        )

        output_rows.append({
            "query_id": str(row["query_id"]),
            "target_video_id": target_id,
            "candidate_video_ids": candidates,
            "negative_types": negative_types,
        })

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    return output_path

def build_comparison_table(
    result_paths: list[Path],
    output_path: Path,
) -> Path:
    """Create a CSV table comparing all model metrics."""
    raise NotImplementedError("Comparison-table formatting will be added with result files.")


def categorize_retrieval_errors(
    ranked_results: list[dict],
    annotations: dict[str, dict],
) -> list[dict]:
    """Return categorized retrieval failures for qualitative analysis."""
    raise NotImplementedError(
        "Error categories require qualitative rules agreed on after first model runs."
    )


def _validate_retrieval_inputs(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_video_ids: list[str],
    target_video_ids: list[str],
) -> None:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (N_queries, N_candidates)")
    if scores.shape != (len(query_ids), len(candidate_video_ids)):
        raise ValueError("scores shape must match query and candidate ids")
    if len(query_ids) != len(target_video_ids):
        raise ValueError("query_ids and target_video_ids must have the same length")
