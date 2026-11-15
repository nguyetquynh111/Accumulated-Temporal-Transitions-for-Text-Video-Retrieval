"""Action-contrast candidate set and the opposite-label review list.

Uses the hand-checked opposite pairs from src/eval/evaluate.py:

- build_action_contrast_set: candidate sets whose negatives are all
  opposite-action videos (data/annotations/action_contrast.jsonl).
- find_opposite_label_pairs / save_opposite_label_pairs: the opposite label
  pairs found in a split, written out for manual review.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import RANDOM_SEED
from src.eval.evaluate import is_opposite_action


# Config

MAX_NEGATIVES = 9
# Placeholder words that do not name a real object (same as build_hard_negative_set).
GENERIC_OBJECTS = frozenset({"", "something", "anything", "object", "item"})
GENERIC_PRIMARY_OBJECTS = GENERIC_OBJECTS | {"stuff"}


# Sub functions

def _normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().replace("[", "").replace("]", "").split())


def _row_objects(row: dict) -> set[str]:
    values = row.get("objects") or []
    if not isinstance(values, list):
        return set()
    return {
        _normalize_text(value)
        for value in values
        if _normalize_text(value) not in GENERIC_OBJECTS
    }


def _primary_object(row: dict) -> str:
    """Return the first annotated object, or "" when it is a generic placeholder."""
    values = row.get("objects") or []
    if not isinstance(values, list) or not values:
        return ""
    primary = _normalize_text(values[0])
    return "" if primary in GENERIC_PRIMARY_OBJECTS else primary


def _row_label(row: dict) -> str:
    return _normalize_text(row.get("label") or row.get("template") or row["text_query"])


def _index_eval_annotations(annotations: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Validate one evaluation split and return rows sorted by query_id plus a video index."""
    if not annotations:
        raise ValueError("annotations must not be empty")

    # Use only one evaluation split to prevent data leakage.
    splits = {str(row.get("split", "")) for row in annotations}
    if len(splits) != 1 or not splits.issubset({"val", "test"}):
        raise ValueError("Pass annotations from one split: val OR test")

    rows = sorted(annotations, key=lambda row: str(row["query_id"]))
    required = {"query_id", "target_video_id", "video_id", "text_query"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"Missing required fields: {required - set(row)}")
        if str(row["video_id"]) != str(row["target_video_id"]):
            raise ValueError("Each source video must match its target_video_id")

    video_by_id = {str(row["video_id"]): row for row in rows}
    if len(video_by_id) != len(rows):
        raise ValueError("Expected one annotation per video_id")
    if len({str(row["query_id"]) for row in rows}) != len(rows):
        raise ValueError("query_id values must be unique")
    return rows, video_by_id


def _opposite_label_map(labels: list[str]) -> dict[str, list[str]]:
    """Map each label to the sorted labels that are its opposites."""
    unique = sorted(set(labels))
    return {
        label: [other for other in unique if other != label and is_opposite_action(label, other)]
        for label in unique
    }


def find_opposite_label_pairs(annotations: list[dict]) -> list[dict]:
    """Return every unordered pair of labels treated as opposite actions.

    Each record has both labels and how many videos carry each, so the list can
    be checked by hand before it is trusted.
    """
    counts: dict[str, int] = defaultdict(int)
    for row in annotations:
        counts[_row_label(row)] += 1
    opposites = _opposite_label_map(list(counts))
    return [
        {"label_a": a, "label_b": b, "videos_a": counts[a], "videos_b": counts[b]}
        for a in sorted(opposites)
        for b in opposites[a]
        if a < b
    ]


def save_opposite_label_pairs(annotations: list[dict], output_path: Path) -> Path:
    """Write the opposite label pairs as a TSV for manual review and return its path."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = find_opposite_label_pairs(annotations)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["label_a", "label_b", "videos_a", "videos_b"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(pairs)
    return output_path


def _write_jsonl(rows: list[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path




# Main functions

def build_action_contrast_set(
    annotations: list[dict],
    output_path: Path,
    max_negatives: int = MAX_NEGATIVES,
) -> Path:
    """Create candidate sets whose negatives are all opposite-action videos.

    There are no filler negatives: a query whose action has no opposite in the
    split is left out, and a candidate list may hold fewer than
    max_negatives + 1 videos. Opposite-action videos that share the query's
    primary object are taken first, then those sharing any object, then the
    rest, so appearance stays as close to the target as the split allows.
    """
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")
    rows, video_by_id = _index_eval_annotations(annotations)

    labels = {vid: _row_label(row) for vid, row in video_by_id.items()}
    texts = {vid: _normalize_text(row["text_query"]) for vid, row in video_by_id.items()}
    by_label = defaultdict(list)
    for video_id in sorted(video_by_id):
        by_label[labels[video_id]].append(video_id)
    opposite_labels = _opposite_label_map(list(by_label))
    rng = random.Random(RANDOM_SEED)

    output_rows = []
    for row in rows:
        target_id = str(row["target_video_id"])
        pool = [
            vid
            for action in opposite_labels[labels[target_id]]
            for vid in by_label[action]
            if texts[vid] != texts[target_id]
        ]
        if not pool:
            continue

        primary = _primary_object(row)
        objects = _row_objects(row)
        tiers: list[list[str]] = [[], [], []]
        for vid in pool:
            other = video_by_id[vid]
            if primary and _primary_object(other) == primary:
                tiers[0].append(vid)
            elif objects & _row_objects(other):
                tiers[1].append(vid)
            else:
                tiers[2].append(vid)
        negatives = []
        for tier in tiers:
            rng.shuffle(tier)
            negatives.extend(tier[: max_negatives - len(negatives)])

        candidates = negatives.copy()
        candidates.insert(rng.randrange(len(candidates) + 1), target_id)
        output_rows.append({
            "query_id": str(row["query_id"]),
            "target_video_id": target_id,
            "candidate_video_ids": candidates,
            "negative_types": ["opposite_action"] * len(negatives),
        })

    return _write_jsonl(output_rows, output_path)
