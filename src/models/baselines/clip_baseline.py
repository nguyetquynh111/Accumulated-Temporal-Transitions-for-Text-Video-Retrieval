"""B1 baseline: CLIP frame embeddings averaged over time, matched to CLIP text.

This is zero-shot: CLIP is not fine-tuned, and frame averaging is the only
temporal pooling. Text and video come from the same CLIP model, so their
dimensions always match.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import ANNOTATION_ROOT, DEFAULT_NUM_FRAMES, FEATURE_ROOT, MODEL_CACHE_ROOT, RESULT_ROOT
from src.data.dataset import get_dataset_sample, load_annotations
from src.eval.evaluate import save_metrics
from src.models.baselines.common import (
    build_result,
    candidate_ids,
    encode_unique_texts,
    evaluate_full,
    evaluate_hard_negatives,
    hard_negative_video_ids,
    ranking_records,
    read_hard_negative_rows,
    resolve_video_row,
    save_jsonl,
    split_hard_negative_rows,
)
from src.models.transition_model.encoder import (
    TextEncoder,
    load_video_feature_cache,
    save_video_feature_cache,
)
from src.models.transition_model.utils import feature_path, get_device


# Config

CLIP_MODEL_NAME = os.environ.get("ATTVR_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
MODEL_NAME = "clip_baseline"
TEXT_BATCH_SIZE = 64
MOCK_DIM = 64


# Sub functions

def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return (embeddings / np.maximum(lengths, 1e-12)).astype(np.float32)


class ClipEncoder:
    """Encode text and RGB frames with CLIP, or with a deterministic 64-d mock."""

    def __init__(
        self,
        encoder_name: str = "clip",
        model_name: str = CLIP_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if encoder_name not in ("clip", "mock"):
            raise ValueError(f"unknown CLIP encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.device = device
        self.torch = None
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Load pretrained weights once, when first needed."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP needs torch and transformers. Install requirements.txt "
                "or use encoder_name='mock'."
            ) from exc

        self.device = get_device(self.device)
        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.processor = CLIPProcessor.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )
        self.model = (
            CLIPModel.from_pretrained(self.model_name, cache_dir=MODEL_CACHE_ROOT)
            .to(self.device)
            .eval()
        )
        self.torch = torch

    def mock_frames(self, frames: np.ndarray) -> np.ndarray:
        """Project per-frame RGB means and deviations to the mock text size."""
        pixels = frames.astype(np.float32) / 255.0
        stats = np.concatenate((pixels.mean(axis=(1, 2)), pixels.std(axis=(1, 2))), axis=1)
        projection = np.random.default_rng(0).standard_normal((stats.shape[1], MOCK_DIM))
        return normalize_rows(stats @ projection.astype(np.float32))

    def encode_texts(self, texts: list[str], batch_size: int = TEXT_BATCH_SIZE) -> np.ndarray:
        """Return L2-normalized float32 rows in the same order as texts."""
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("texts must be a list of nonempty strings")
        if self.encoder_name == "mock":
            return TextEncoder("mock").encode(texts)

        self.load_model()
        batches = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            batches.append(features.float().cpu().numpy())
        return normalize_rows(np.concatenate(batches))

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """Return one L2-normalized float32 embedding per frame, in order."""
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1:
            raise ValueError("frames must have nonempty shape (T, H, W, 3)")
        if frames.dtype != np.uint8:
            raise ValueError("frames must be RGB uint8")
        if self.encoder_name == "mock":
            return self.mock_frames(frames)

        self.load_model()
        inputs = self.processor(images=list(frames), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return normalize_rows(features.float().cpu().numpy())


clip_encoders: dict[tuple[str, str, str | None], ClipEncoder] = {}


def get_clip_encoder(
    encoder_name: str = "clip",
    model_name: str = CLIP_MODEL_NAME,
    device: str | None = None,
) -> ClipEncoder:
    key = (encoder_name, model_name, device)
    if key not in clip_encoders:
        clip_encoders[key] = ClipEncoder(encoder_name, model_name, device)
    return clip_encoders[key]


def encode_clip_text_queries(
    texts: list[str],
    *,
    encoder_name: str = "clip",
    model_name: str = CLIP_MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    """Return CLIP text embeddings with shape (N, D), dtype float32."""
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty strings")
    return get_clip_encoder(encoder_name, model_name, device).encode_texts(texts)


def encode_clip_frames(
    frames: np.ndarray,
    *,
    encoder_name: str = "clip",
    model_name: str = CLIP_MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    """Return CLIP frame embeddings with shape (T, D), dtype float32."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    return get_clip_encoder(encoder_name, model_name, device).encode_frames(frames)


def pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """Return one video embedding with shape (D,), dtype float32."""
    frame_embeddings = np.asarray(frame_embeddings, dtype=np.float32)
    if frame_embeddings.ndim != 2:
        raise ValueError("frame_embeddings must have shape (T, D)")
    if frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings must not be empty")
    return frame_embeddings.mean(axis=0).astype(np.float32)


def score_clip_retrieval(
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> np.ndarray:
    """Return retrieval scores with shape (N_queries, N_candidates)."""
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    if text_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if text_embeddings.shape[1] != video_embeddings.shape[1]:
        raise ValueError("text and video embedding dimensions must match")
    return (text_embeddings @ video_embeddings.T).astype(np.float32)


def extract_clip_features(
    rows: list[dict],
    video_ids: list[str],
    feature_dir: Path,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    encoder_name: str = "clip",
    model_name: str = CLIP_MODEL_NAME,
    device: str | None = None,
    overwrite: bool = False,
) -> tuple[list[str], list[str]]:
    """Cache CLIP frame embeddings; return (cached video ids, unreadable video ids)."""
    row_by_video = {}
    for row in rows:
        row_by_video.setdefault(str(row["video_id"]), row)
    cache_name = model_name if encoder_name == "clip" else "mock"
    cached, failed = [], []
    for video_id in tqdm(video_ids, desc="Extracting CLIP features"):
        path = feature_path(feature_dir, video_id)
        if path.exists() and not overwrite:
            if str(load_video_feature_cache(path)["encoder_name"]) != cache_name:
                raise ValueError(f"cache from another encoder: {path}; use --overwrite")
            cached.append(video_id)
            continue
        if video_id not in row_by_video:
            failed.append(video_id)
            continue
        try:
            sample = get_dataset_sample(resolve_video_row(row_by_video[video_id]), num_frames)
        except (FileNotFoundError, ValueError):
            failed.append(video_id)
            continue
        embeddings = encode_clip_frames(
            sample["frames"], encoder_name=encoder_name, model_name=model_name, device=device
        )
        save_video_feature_cache(
            video_id, sample["frame_indices"], embeddings, feature_dir, cache_name
        )
        cached.append(video_id)
    return cached, failed


# Main function

def run_clip_baseline(
    annotations: Path,
    feature_dir: Path,
    output_path: Path,
    *,
    hard_negatives: Path | None = None,
    rankings_path: Path | None = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    encoder_name: str = "clip",
    model_name: str = CLIP_MODEL_NAME,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Extract CLIP features, score the split, save metrics and rankings, return the result."""
    all_rows = load_annotations(Path(annotations))[:limit]
    hard_rows = read_hard_negative_rows(hard_negatives, all_rows)
    needed = list(dict.fromkeys(candidate_ids(all_rows) + hard_negative_video_ids(hard_rows)))
    cached, failed = extract_clip_features(
        all_rows,
        needed,
        Path(feature_dir),
        num_frames=num_frames,
        encoder_name=encoder_name,
        model_name=model_name,
        device=device,
        overwrite=overwrite,
    )
    available = set(cached)
    rows = [row for row in all_rows if str(row["target_video_id"]) in available]
    if not rows:
        raise ValueError("no annotation rows with readable target videos")

    video_ids = [video_id for video_id in needed if video_id in available]
    videos = np.stack(
        [
            pool_frame_embeddings(
                load_video_feature_cache(feature_path(feature_dir, v))["frame_embeddings"]
            )
            for v in video_ids
        ]
    )
    videos = normalize_rows(videos)
    video_index = {video_id: i for i, video_id in enumerate(video_ids)}
    candidates = candidate_ids(rows)
    texts = encode_unique_texts(
        rows,
        lambda unique: encode_clip_text_queries(
            unique, encoder_name=encoder_name, model_name=model_name, device=device
        ),
    )
    scores = score_clip_retrieval(texts, videos[[video_index[v] for v in candidates]])

    result = build_result(
        MODEL_NAME,
        rows,
        evaluate_full(scores, rows, candidates),
        encoder_name=model_name if encoder_name == "clip" else "mock",
        num_frames=num_frames,
        skipped_missing_videos=len(all_rows) - len(rows),
        unreadable_video_ids=failed,
    )
    kept, hard_skipped = split_hard_negative_rows(hard_rows, available)
    if kept:
        result["hard_negative"] = evaluate_hard_negatives(
            kept, rows, texts, video_ids, videos, score_clip_retrieval
        )
        result["hard_negative_skipped"] = hard_skipped

    save_metrics(result, output_path)
    if rankings_path is not None:
        save_jsonl(ranking_records(MODEL_NAME, rows, scores, candidates), rankings_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_ROOT / "test.jsonl")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_ROOT / "clip")
    parser.add_argument(
        "--hard-negatives", type=Path, default=ANNOTATION_ROOT / "hard_negatives.jsonl"
    )
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / f"{MODEL_NAME}.json")
    parser.add_argument(
        "--rankings", type=Path, default=RESULT_ROOT / f"rankings_{MODEL_NAME}.jsonl"
    )
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--encoder", choices=("clip", "mock"), default="clip")
    parser.add_argument("--model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_clip_baseline(
        args.annotations,
        args.feature_dir,
        args.output,
        hard_negatives=args.hard_negatives,
        rankings_path=args.rankings,
        num_frames=args.num_frames,
        encoder_name=args.encoder,
        model_name=args.model_name,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps({"full": result["full"], "hard_negative": result["hard_negative"]}, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
