"""Rule-based error categories, model comparisons, and qualitative figures (T4.1).

Each failed query (target not ranked first) is compared with the top-ranked
wrong video using Something-Something templates, labels, and object names. The
rules sort failures for manual review; they are heuristics, not ground truth.

Inputs are the rankings_<model>.jsonl files written by the baseline and
ablation runners.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from config import ANNOTATION_ROOT, RESULT_ROOT
from src.data.dataset import get_dataset_sample, load_annotations
from src.eval.evaluate import save_metrics
from src.models.baselines.common import load_jsonl, resolve_video_row


# Config

ERROR_CATEGORIES = (
    "correct_objects_wrong_action",
    "wrong_temporal_direction",
    "wrong_action_ordering",
    "similar_start_end_different_motion",
    "text_ambiguity",
    "visual_ambiguity",
    "encoder_failure",
)
STOPWORDS = {
    "a", "an", "and", "as", "from", "is", "it", "its", "of", "so", "something",
    "that", "the", "to", "with",
}
# Word pairs that describe the same action in opposite temporal directions.
DIRECTION_WORD_PAIRS = (
    ("up", "down"),
    ("left", "right"),
    ("closer", "farther"),
    ("closer", "further"),
    ("towards", "away"),
    ("toward", "away"),
    ("into", "out"),
    ("onto", "off"),
    ("on", "off"),
    ("opening", "closing"),
    ("picking", "putting"),
    ("taking", "putting"),
    ("pushing", "pulling"),
    ("covering", "uncovering"),
    ("folding", "unfolding"),
    ("plugging", "unplugging"),
    ("attaching", "detaching"),
    ("raising", "lowering"),
    ("front", "behind"),
)
# Words that keep the start and end states but change the motion in between.
MOTION_MODIFIER_WORDS = ("pretending", "trying", "failing", "but", "almost", "without", "not")
# Targets ranked below this fraction of candidates count as encoder failures.
ENCODER_FAILURE_FRACTION = 0.5
EXAMPLES_PER_CATEGORY = 3
FIGURE_FRAMES = 4
FIGURE_FRAME_SIZE = (128, 96)


# Sub functions

def stem(word: str) -> str:
    """Crude stemmer so that 'putting'/'put' and 'closing'/'close' match."""
    word = word.lower()
    if word.endswith("ing") and len(word) > 4:
        word = word[:-3]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    if len(word) > 2 and word[-1] == word[-2]:
        word = word[:-1]
    return word


DIRECTION_PAIRS = {frozenset((stem(a), stem(b))) for a, b in DIRECTION_WORD_PAIRS}
MOTION_MODIFIERS = {stem(word) for word in MOTION_MODIFIER_WORDS}


def tokens(text: str) -> list[str]:
    """Return stemmed content words, dropping placeholders and stopwords."""
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return [stem(word) for word in re.findall(r"[a-z]+", text) if word not in STOPWORDS]


def row_label(row: dict) -> str:
    return row.get("label") or row.get("text_query", "")


def action_tokens(row: dict) -> list[str]:
    """Return the action words of a row, preferring its template."""
    return tokens(row.get("template") or row_label(row))


def object_tokens(row: dict) -> set[str]:
    """Return object words from 'objects', or from the label words the template lacks."""
    if row.get("objects"):
        return set(tokens(" ".join(row["objects"])))
    if row.get("template"):
        return set(tokens(row_label(row))) - set(tokens(row["template"]))
    return set()


def is_direction_swap(only_target: set[str], only_retrieved: set[str]) -> bool:
    """True when every differing word has an opposite-direction partner on the other side."""
    def paired(words: set[str], others: set[str]) -> bool:
        return all(any(frozenset((w, o)) in DIRECTION_PAIRS for o in others) for w in words)

    return (
        bool(only_target)
        and bool(only_retrieved)
        and paired(only_target, only_retrieved)
        and paired(only_retrieved, only_target)
    )


def categorize_error(
    target_row: dict,
    retrieved_row: dict,
    target_rank: int,
    num_candidates: int,
) -> str:
    """Return one ERROR_CATEGORIES name for a failed query."""
    if target_row.get("text_query") and target_row.get("text_query") == retrieved_row.get(
        "text_query"
    ):
        return "text_ambiguity"
    target_actions, retrieved_actions = action_tokens(target_row), action_tokens(retrieved_row)
    only_target = set(target_actions) - set(retrieved_actions)
    only_retrieved = set(retrieved_actions) - set(target_actions)
    differing = only_target | only_retrieved
    if differing and differing <= MOTION_MODIFIERS:
        return "similar_start_end_different_motion"
    if is_direction_swap(only_target, only_retrieved):
        return "wrong_temporal_direction"
    if target_actions != retrieved_actions and sorted(target_actions) == sorted(retrieved_actions):
        # Same words in another order: "left to right" vs "right to left" is a
        # direction swap, anything else is a reordering of the action steps.
        swapped = {
            frozenset((a, b)) for a, b in zip(target_actions, retrieved_actions) if a != b
        }
        if swapped <= DIRECTION_PAIRS:
            return "wrong_temporal_direction"
        return "wrong_action_ordering"
    if differing and object_tokens(target_row) & object_tokens(retrieved_row):
        return "correct_objects_wrong_action"
    if target_rank > num_candidates * ENCODER_FAILURE_FRACTION:
        return "encoder_failure"
    return "visual_ambiguity"


# Main functions

def categorize_retrieval_errors(
    ranked_results: list[dict],
    annotations: dict[str, dict],
) -> list[dict]:
    """Return categorized retrieval failures for qualitative analysis.

    ranked_results are ranking records (see common.ranking_records);
    annotations maps video_id to that video's annotation row.
    """
    errors = []
    for record in ranked_results:
        if record["target_rank"] == 1:
            continue
        target_id = str(record["target_video_id"])
        retrieved_id = str(record["top_video_ids"][0])
        target_row = {**annotations.get(target_id, {}), "text_query": record["text_query"]}
        retrieved_row = annotations.get(retrieved_id, {})
        errors.append(
            {
                "model_name": record["model_name"],
                "query_id": record["query_id"],
                "text_query": record["text_query"],
                "target_video_id": target_id,
                "target_label": row_label(target_row),
                "target_rank": record["target_rank"],
                "retrieved_video_id": retrieved_id,
                "retrieved_label": row_label(retrieved_row),
                "top_video_ids": record["top_video_ids"],
                "category": categorize_error(
                    target_row, retrieved_row, record["target_rank"], record["num_candidates"]
                ),
            }
        )
    return errors


def summarize_errors(
    errors: list[dict],
    examples_per_category: int = EXAMPLES_PER_CATEGORY,
) -> dict:
    """Return per-model category counts and a few examples of each observed category."""
    summary = {}
    for model_name in dict.fromkeys(error["model_name"] for error in errors):
        model_errors = [error for error in errors if error["model_name"] == model_name]
        counts = Counter(error["category"] for error in model_errors)
        summary[model_name] = {
            "num_errors": len(model_errors),
            "counts": {category: counts.get(category, 0) for category in ERROR_CATEGORIES},
            "examples": {
                category: [e for e in model_errors if e["category"] == category][
                    :examples_per_category
                ]
                for category in ERROR_CATEGORIES
                if counts.get(category)
            },
        }
    return summary


def compare_models(
    ranked_by_model: dict[str, list[dict]],
    examples: int = EXAMPLES_PER_CATEGORY,
) -> list[dict]:
    """For each model pair, list queries one model ranks first and the other misses."""
    by_query = {
        name: {record["query_id"]: record for record in records}
        for name, records in ranked_by_model.items()
    }
    comparisons = []
    names = list(by_query)
    for winner in names:
        for loser in names:
            if winner == loser:
                continue
            shared = by_query[winner].keys() & by_query[loser].keys()
            wins = sorted(
                query_id
                for query_id in shared
                if by_query[winner][query_id]["target_rank"] == 1
                and by_query[loser][query_id]["target_rank"] > 1
            )
            comparisons.append(
                {
                    "winner": winner,
                    "loser": loser,
                    "num_shared_queries": len(shared),
                    "num_wins": len(wins),
                    "examples": [
                        {
                            "query_id": query_id,
                            "text_query": by_query[winner][query_id]["text_query"],
                            "target_video_id": by_query[winner][query_id]["target_video_id"],
                            f"{loser}_top_video_ids": by_query[loser][query_id]["top_video_ids"][:5],
                            f"{loser}_target_rank": by_query[loser][query_id]["target_rank"],
                        }
                        for query_id in wins[:examples]
                    ],
                }
            )
    return comparisons


def save_qualitative_figure(
    error: dict,
    annotations: dict[str, dict],
    output_path: Path,
    num_frames: int = FIGURE_FRAMES,
) -> Path:
    """Save a PNG with the query, the target's frames, and the wrongly retrieved frames."""
    import cv2

    width, height = FIGURE_FRAME_SIZE
    strips = [_text_banner(f"{error['model_name']} | {error['category']}", width * num_frames)]
    strips.append(_text_banner(f"query: {error['text_query']}", width * num_frames))
    for title, video_id in (
        (f"target {error['target_video_id']} (rank {error['target_rank']})", error["target_video_id"]),
        (f"retrieved {error['retrieved_video_id']}: {error['retrieved_label']}", error["retrieved_video_id"]),
    ):
        sample = get_dataset_sample(resolve_video_row(annotations[video_id]), num_frames)
        frames = [cv2.resize(frame, (width, height)) for frame in sample["frames"]]
        strips.append(_text_banner(title, width * num_frames))
        strips.append(np.concatenate(frames, axis=1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(np.concatenate(strips, axis=0), cv2.COLOR_RGB2BGR))
    return output_path


def _text_banner(text: str, width: int, height: int = 24) -> np.ndarray:
    import cv2

    banner = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(banner, text[:90], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return banner


def run_error_analysis(
    ranking_paths: list[Path],
    annotations: Path,
    output_path: Path,
    figure_dir: Path | None = None,
    figures_per_category: int = 1,
) -> dict:
    """Categorize every model's failures, compare models, and optionally save figures."""
    rows = load_annotations(Path(annotations))
    by_video = {}
    for row in rows:
        by_video.setdefault(str(row["video_id"]), row)

    ranked_by_model = {}
    for path in ranking_paths:
        records = load_jsonl(path)
        if records:
            ranked_by_model[records[0]["model_name"]] = records
    errors = [
        error
        for records in ranked_by_model.values()
        for error in categorize_retrieval_errors(records, by_video)
    ]
    result = {
        "annotations": str(annotations),
        "ranking_files": [str(path) for path in ranking_paths],
        "summary": summarize_errors(errors),
        "comparisons": compare_models(ranked_by_model),
    }

    if figure_dir is not None:
        figures = []
        for model_name, model_summary in result["summary"].items():
            for category, examples in model_summary["examples"].items():
                for error in examples[:figures_per_category]:
                    path = Path(figure_dir) / f"{model_name}_{category}_{error['query_id']}.png"
                    try:
                        figures.append(str(save_qualitative_figure(error, by_video, path)))
                    except (FileNotFoundError, KeyError, ValueError):
                        continue
        result["figures"] = figures

    save_metrics(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rankings",
        type=Path,
        nargs="+",
        default=sorted(RESULT_ROOT.glob("rankings_*.jsonl")),
        help="ranking files to analyze (default: every rankings_*.jsonl in outputs/results)",
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "error_analysis.json")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--figures-per-category", type=int, default=1)
    args = parser.parse_args()
    if not args.rankings:
        parser.error("no ranking files found; run a baseline or pass --rankings")
    result = run_error_analysis(
        args.rankings, args.annotations, args.output, args.figure_dir, args.figures_per_category
    )
    for model_name, model_summary in result["summary"].items():
        counts = {k: v for k, v in model_summary["counts"].items() if v}
        print(f"{model_name}: {model_summary['num_errors']} errors {counts}")
    print(f"Saved error analysis to {args.output}")


if __name__ == "__main__":
    main()
