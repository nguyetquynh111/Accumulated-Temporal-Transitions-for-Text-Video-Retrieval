"""Shared retrieval evaluator, hard-negative benchmark, and comparison table."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config import RANDOM_SEED


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


# Hard-negative and action-contrast candidate sets (T2.6)

MAX_NEGATIVES = 9
# Placeholder words that do not name a real object.
GENERIC_OBJECTS = frozenset({"", "something", "anything", "object", "item"})
GENERIC_PRIMARY_OBJECTS = GENERIC_OBJECTS | {"stuff"}
# Pairs of SSv2 labels treated as opposite actions, checked by hand. In each
# pair one action is the other reversed in time or direction, with the same
# kind of motion. Labels starting with "pretending" never count, since nothing
# changes in those videos. An earlier keyword rule also paired every "pulling"
# label with every "pushing" label, including same-direction ones, so the list
# is written out explicitly instead.
OPPOSITE_LABEL_PAIRS = (
    ("closing something", "opening something"),
    ("covering something with something", "uncovering something"),
    ("folding something", "unfolding something"),
    ("putting something into something", "taking something out of something"),
    ("putting something into something", "pulling something out of something"),
    ("stuffing something into something", "taking something out of something"),
    ("stuffing something into something", "pulling something out of something"),
    ("burying something in something", "digging something out of something"),
    ("picking something up", "putting something on a surface"),
    (
        "putting something similar to other things that are already on the table",
        "taking one of many similar things on the table",
    ),
    ("pouring something into something", "pouring something out of something"),
    ("pushing something onto something", "pushing something off of something"),
    ("moving something up", "moving something down"),
    ("moving something closer to something", "moving something away from something"),
    (
        "moving something and something closer to each other",
        "moving something and something away from each other",
    ),
    ("moving something towards the camera", "moving something away from the camera"),
    ("approaching something with your camera", "moving away from something with your camera"),
    ("pushing something from left to right", "pushing something from right to left"),
    ("pulling something from left to right", "pulling something from right to left"),
    ("turning the camera left while filming something", "turning the camera right while filming something"),
    (
        "turning the camera upwards while filming something",
        "turning the camera downwards while filming something",
    ),
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().replace("[", "").replace("]", "").split())


OPPOSITE_LABELS: dict[str, frozenset[str]] = {
    label: frozenset(
        _normalize_text(other)
        for a, b in OPPOSITE_LABEL_PAIRS
        for this, other in ((a, b), (b, a))
        if _normalize_text(this) == label
    )
    for label in {_normalize_text(x) for pair in OPPOSITE_LABEL_PAIRS for x in pair}
}


def is_opposite_action(label_a: str, label_b: str) -> bool:
    """Return whether two action labels are a hand-checked opposite pair."""
    return _normalize_text(label_b) in OPPOSITE_LABELS.get(_normalize_text(label_a), ())


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


def build_hard_negative_set(
    annotations: list[dict],
    output_path: Path,
) -> Path:
    """Build hard-negative candidate sets from one evaluation split.

    Negatives are chosen by priority: opposite actions on the same primary
    object, then different actions on a shared object, then any other action.
    Lists are filled up to 9 negatives, so most of them are of the last kind;
    use the per-type metrics or the action-contrast set to isolate direction.
    """
    rows, video_by_id = _index_eval_annotations(annotations)

    labels = {vid: _row_label(row) for vid, row in video_by_id.items()}
    texts = {vid: _normalize_text(row["text_query"]) for vid, row in video_by_id.items()}

    by_label = defaultdict(list)
    by_object = defaultdict(list)
    for video_id in sorted(video_by_id):
        by_label[labels[video_id]].append(video_id)
        for obj in sorted(_row_objects(video_by_id[video_id])):
            by_object[obj].append(video_id)

    opposite_labels = _opposite_label_map(list(by_label))
    all_video_ids = sorted(video_by_id)
    rng = random.Random(RANDOM_SEED)

    output_rows = []
    for row in rows:
        target_id = str(row["target_video_id"])
        label = labels[target_id]

        opposite_ids = list(dict.fromkeys(
            vid for action in opposite_labels[label] for vid in by_label[action]
        ))
        shared_ids = list(dict.fromkeys(
            vid for obj in sorted(_row_objects(row)) for vid in by_object[obj]
        ))

        negatives = []
        negative_types = []
        seen = {target_id}

        def add(pool, kind):
            if not pool or len(negatives) == MAX_NEGATIVES:
                return
            start = rng.randrange(len(pool))
            for i in range(len(pool)):
                vid = pool[(start + i) % len(pool)]
                # Do not use the target, an identical query, or the same action.
                if vid in seen or texts[vid] == texts[target_id] or labels[vid] == label:
                    continue
                negatives.append(vid)
                negative_types.append(kind)
                seen.add(vid)
                if len(negatives) == MAX_NEGATIVES:
                    break

        # Priority 1: opposite actions involving the same primary object.
        primary = _primary_object(row)
        add(
            [
                vid
                for vid in opposite_ids
                if primary and _primary_object(video_by_id[vid]) == primary
            ],
            "opposite_action",
        )
        # Priority 2: different actions involving shared objects.
        add(shared_ids, "same_objects")
        # Priority 3: remaining different-action videos.
        add(all_video_ids, "other_action")

        if len(negatives) < 2:
            raise ValueError(
                f"Not enough distinct negative videos for query {row['query_id']}"
            )

        # Vary the correct video's position to avoid ranking bias.
        candidates = negatives.copy()
        candidates.insert(rng.randrange(len(candidates) + 1), target_id)
        output_rows.append({
            "query_id": str(row["query_id"]),
            "target_video_id": target_id,
            "candidate_video_ids": candidates,
            "negative_types": negative_types,
        })

    return _write_jsonl(output_rows, output_path)


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


def _negatives_with_types(row: dict) -> list[tuple[Any, str]]:
    """Pair each non-target candidate with its negative type, in candidate order."""
    target = row["target_video_id"]
    negatives = [v for v in row["candidate_video_ids"] if v != target]
    types = row.get("negative_types")
    if types is None or len(types) != len(negatives):
        raise ValueError(
            f"negative_types does not match the negatives of query {row['query_id']}"
        )
    return list(zip(negatives, types))


def split_rows_by_negative_type(candidate_rows: list[dict]) -> dict[str, list[dict]]:
    """Return, per negative type, rows restricted to the target and that type's negatives.

    negative_types[i] labels the i-th non-target candidate. Candidates keep
    their original order, so the target keeps its position relative to them.
    Rows without negative_types are left out.
    """
    by_type: dict[str, list[dict]] = {}
    for row in candidate_rows:
        if row.get("negative_types") is None:
            continue
        target = row["target_video_id"]
        type_of = dict(_negatives_with_types(row))
        for kind in dict.fromkeys(type_of.values()):
            by_type.setdefault(kind, []).append(
                {
                    "query_id": row["query_id"],
                    "target_video_id": target,
                    "candidate_video_ids": [
                        v
                        for v in row["candidate_video_ids"]
                        if v == target or type_of.get(v) == kind
                    ],
                }
            )
    return by_type


def compute_hard_negative_metrics_by_type(
    hard_negative_rows: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str], float],
) -> dict[str, dict]:
    """Return pairwise accuracy and restricted Recall@1 for each negative type.

    pairwise_accuracy is the share of (target, negative) pairs of that type
    where the target scores higher; a tie counts as half. The restricted
    metrics rank each query's target against only that type's negatives.
    """
    pair_scores: dict[str, list[float]] = defaultdict(list)
    for row in hard_negative_rows:
        if row.get("negative_types") is None:
            continue
        query_id = row["query_id"]
        target_score = score_lookup[(query_id, row["target_video_id"])]
        for video_id, kind in _negatives_with_types(row):
            negative_score = score_lookup[(query_id, video_id)]
            pair_scores[kind].append(
                1.0 if target_score > negative_score
                else 0.5 if target_score == negative_score
                else 0.0
            )

    by_type = {}
    for kind, type_rows in split_rows_by_negative_type(hard_negative_rows).items():
        restricted = compute_hard_negative_metrics(type_rows, score_lookup, k_values=(1,))
        by_type[kind] = {
            "pairwise_accuracy": float(np.mean(pair_scores[kind])),
            "recall@1": restricted["recall@1"],
            "mrr": restricted["mrr"],
            "num_queries": restricted["num_queries"],
            "num_negatives": len(pair_scores[kind]),
        }
    return by_type


def compute_hard_negative_coverage(hard_negative_rows: list[dict]) -> dict:
    """Report how many negatives of each type the candidate sets hold.

    Includes the share of negatives per type and the number of queries with at
    least one opposite-action negative.
    """
    counts: dict[str, int] = defaultdict(int)
    queries_by_type: dict[str, int] = defaultdict(int)
    for row in hard_negative_rows:
        types = [kind for _, kind in _negatives_with_types(row)]
        for kind in types:
            counts[kind] += 1
        for kind in set(types):
            queries_by_type[kind] += 1

    num_queries = len(hard_negative_rows)
    num_negatives = sum(counts.values())
    with_opposite = queries_by_type.get("opposite_action", 0)
    return {
        "num_queries": num_queries,
        "num_negatives": num_negatives,
        "negatives_by_type": dict(sorted(counts.items())),
        "negative_share_by_type": {
            kind: count / num_negatives for kind, count in sorted(counts.items())
        } if num_negatives else {},
        "queries_with_type": dict(sorted(queries_by_type.items())),
        "queries_with_opposite_action": with_opposite,
        "queries_with_opposite_action_share": with_opposite / num_queries if num_queries else 0.0,
    }


def summarize_candidate_set(
    candidate_rows: list[dict[str, Any]],
    score_lookup: dict[tuple[str, str], float],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Return aggregate metrics, per-type metrics, and coverage for one candidate set."""
    metrics: dict[str, Any] = compute_hard_negative_metrics(candidate_rows, score_lookup, k_values)
    typed = [row for row in candidate_rows if row.get("negative_types") is not None]
    metrics["by_negative_type"] = (
        compute_hard_negative_metrics_by_type(typed, score_lookup) if typed else None
    )
    metrics["coverage"] = compute_hard_negative_coverage(typed) if typed else None
    return metrics


COMPARISON_SETTINGS = ("full", "hard_negative")
COMPARISON_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "map")
MISSING_VALUE = "NA"


def build_comparison_table(
    result_paths: list[Path],
    output_path: Path,
) -> Path:
    """Create a CSV table comparing all model metrics.

    Writes one row per result file and evaluation setting (full or
    hard-negative). The split, query count, and candidate count are included so
    it is visible whether models were compared on the same data. Metrics a
    result file does not report are written as "NA" instead of being dropped.
    """
    import csv

    if not result_paths:
        raise ValueError("result_paths must not be empty")

    columns = [
        "model_name",
        "variant",
        "setting",
        "split",
        *COMPARISON_METRICS,
        "num_queries",
        "num_candidates",
        "result_file",
    ]
    rows = []
    for path in result_paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        for setting in COMPARISON_SETTINGS:
            metrics = result.get(setting) or {}
            row = {
                "model_name": result.get("model_name", path.stem),
                "variant": result.get("variant", MISSING_VALUE),
                "setting": setting,
                "split": result.get("split", MISSING_VALUE),
                "result_file": path.name,
            }
            for key in (*COMPARISON_METRICS, "num_queries", "num_candidates"):
                value = metrics.get(key)
                if value is None:
                    row[key] = MISSING_VALUE
                elif isinstance(value, float):
                    row[key] = f"{value:.4f}"
                else:
                    row[key] = value
            rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


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


if __name__ == "__main__":
    import argparse

    from config import RESULT_ROOT

    parser = argparse.ArgumentParser(description="Build the model comparison CSV (T3.2).")
    parser.add_argument("results", type=Path, nargs="+", help="result JSON files")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "comparison_table.csv")
    args = parser.parse_args()
    print(f"Saved {build_comparison_table(args.results, args.output)}")
