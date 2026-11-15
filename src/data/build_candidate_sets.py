"""Build the hard-negative and action-contrast candidate sets for one split (T2.6).

Writes, by default next to the annotations:
- hard_negatives.jsonl: one positive and up to 9 negatives per query;
- action_contrast.jsonl: negatives are all opposite-action videos;
- opposite_label_pairs.tsv: the label pairs treated as opposites, for review;
and a coverage report (negatives per type, queries with an opposite-action
negative) to outputs/results/candidate_set_coverage.json.

Existing candidate files are kept unless --overwrite is given, because result
files already computed on them would no longer match.

    python -m src.data.build_candidate_sets
"""

import argparse
import json
from pathlib import Path

from config import ANNOTATION_ROOT, RESULT_ROOT
from src.data.dataset import load_annotations
from src.data.candidate_sets import build_action_contrast_set, save_opposite_label_pairs
from src.eval.candidate_metrics import compute_hard_negative_coverage
from src.eval.evaluate import build_hard_negative_set, save_metrics


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_candidate_sets(
    annotations: Path,
    hard_negatives: Path,
    action_contrast: Path,
    opposite_pairs: Path,
    coverage_output: Path,
    overwrite: bool = False,
) -> dict:
    """Build any missing candidate files, then save and return their coverage."""
    rows = load_annotations(Path(annotations))
    save_opposite_label_pairs(rows, opposite_pairs)
    for path, build in (
        (Path(hard_negatives), build_hard_negative_set),
        (Path(action_contrast), build_action_contrast_set),
    ):
        if overwrite or not path.exists():
            build(rows, path)
            print(f"Built {path}")
        else:
            print(f"Kept existing {path} (use --overwrite to rebuild)")

    coverage = {
        "annotations": str(annotations),
        "opposite_label_pairs": str(opposite_pairs),
        "hard_negative": compute_hard_negative_coverage(read_jsonl(hard_negatives)),
        "action_contrast": compute_hard_negative_coverage(read_jsonl(action_contrast)),
    }
    save_metrics(coverage, coverage_output)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / "hard_negatives.jsonl"
    )
    parser.add_argument(
        "--action-contrast", type=Path, default=ANNOTATION_ROOT / "action_contrast.jsonl"
    )
    parser.add_argument(
        "--opposite-pairs", type=Path, default=ANNOTATION_ROOT / "opposite_label_pairs.tsv"
    )
    parser.add_argument(
        "--coverage-output", type=Path, default=RESULT_ROOT / "candidate_set_coverage.json"
    )
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing files")
    args = parser.parse_args()
    coverage = build_candidate_sets(
        args.annotations,
        args.hard_negatives,
        args.action_contrast,
        args.opposite_pairs,
        args.coverage_output,
        args.overwrite,
    )
    for name in ("hard_negative", "action_contrast"):
        report = coverage[name]
        print(
            f"{name}: {report['num_queries']} queries, "
            f"{report['queries_with_opposite_action']} with an opposite-action negative, "
            f"negatives by type {report['negatives_by_type']}"
        )
    print(f"Coverage: {args.coverage_output}")


if __name__ == "__main__":
    main()
