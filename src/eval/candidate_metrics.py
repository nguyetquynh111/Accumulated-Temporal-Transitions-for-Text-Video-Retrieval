"""Per-negative-type metrics and coverage of a candidate set.

Extends compute_hard_negative_metrics from evaluate.py:

- compute_hard_negative_metrics_by_type: pairwise accuracy and restricted
  Recall@1/MRR for each negative type.
- compute_hard_negative_coverage: how many negatives of each type the
  candidate sets hold.
- summarize_candidate_set: aggregate, per-type, and coverage metrics together.
"""

from collections import defaultdict
from typing import Any

import numpy as np

from src.eval.evaluate import compute_hard_negative_metrics


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
