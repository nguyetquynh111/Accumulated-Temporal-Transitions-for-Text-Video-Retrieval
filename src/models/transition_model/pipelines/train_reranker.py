"""Two-stage training: freeze a trained encoder, then train only the reranker.

Stage 1 is the full_transition_without_query_conditioning ablation (the main
model's [z, dz] input, trained without the reranker). This script loads its
checkpoint, copies the video encoder, text projection, and logit scale into a
query-conditioned model, freezes them, and trains the reranker alone on the
frozen embeddings. The loss, batch size, optimizer, early stopping, and
selection metric are the shared protocol from config.py.

Because the encoder is frozen, video and text embeddings are computed once,
so each epoch only runs the reranker. The result is evaluated with the same
evaluator as every other variant and written to
outputs/results/two_stage_reranker_seed<seed>.json.

    python -m src.models.transition_model.pipelines.train_reranker --seeds 42 43 44
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import ANNOTATION_ROOT, CHECKPOINT_ROOT, LOG_ROOT, RESULT_ROOT, VJEPA_FEATURE_ROOT
from src.eval.evaluate import save_metrics
from src.models.baselines.common import candidate_ids, evaluate_full
from src.models.baselines.vjepa_baseline import (
    ACTION_CONTRAST_FILE,
    encode_variant_bank,
    evaluate_variant,
    result_name,
)
from src.models.transition_model.encoder import encode_text_queries
from src.models.transition_model.pipelines.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    SELECTION_METRIC,
    WEIGHT_DECAY,
    EarlyStopping,
    checkpoint_path,
    contrastive_loss,
    log_path,
)
from src.models.transition_model.retrieval import (
    TransitionRetrievalModel,
    load_checkpoint,
    save_checkpoint,
    score_transition_retrieval,
)
from src.models.transition_model.utils import (
    encode_row_texts,
    get_device,
    load_split_rows,
    set_random_seed,
    unique_in_order,
)


# Config

RUN_NAME = "two_stage_reranker"
ENCODER_VARIANT = "full_transition_without_query_conditioning"
# The trained model reads the same [z, dz] tokens as the main model.
EVAL_VARIANT = "full_transition"


# Sub functions

def build_two_stage_model(encoder_checkpoint: Path, device: str) -> tuple[TransitionRetrievalModel, dict]:
    """Return a query-conditioned model whose encoder weights come from encoder_checkpoint, frozen."""
    encoder_model, checkpoint = load_checkpoint(encoder_checkpoint, device)
    config = dict(encoder_model.config, query_conditioned=True)
    model = TransitionRetrievalModel(**config).to(device)
    missing, unexpected = model.load_state_dict(encoder_model.state_dict(), strict=False)
    if unexpected or any(not key.startswith("reranker.") for key in missing):
        raise ValueError(f"encoder checkpoint does not match: missing={missing} unexpected={unexpected}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("reranker.")
    return model, checkpoint


def frozen_embeddings(model, rows, feature_dir, text_encoder, text_model_name, device):
    """Return per-row video embeddings, unique-text embeddings, and each row's text id."""
    video_ids = [str(row["target_video_id"]) for row in rows]
    unique_videos = unique_in_order(video_ids)
    bank = encode_variant_bank(model, EVAL_VARIANT, unique_videos, feature_dir)
    position = {video_id: i for i, video_id in enumerate(unique_videos)}
    videos = torch.from_numpy(bank[[position[v] for v in video_ids]]).to(device)

    texts = unique_in_order(row["text_query"] for row in rows)
    text_index = {text: i for i, text in enumerate(texts)}
    raw = encode_text_queries(texts, encoder_name=text_encoder, model_name=text_model_name, device=device)
    with torch.inference_mode():
        text_bank = model.encode_texts(torch.from_numpy(raw).to(device))
    text_ids = torch.tensor([text_index[row["text_query"]] for row in rows], device=device)
    return videos, text_bank.clone(), text_ids


def validate(model, val_rows, val_texts, val_videos):
    """Score every val query against the val target videos and return full metrics."""
    scores = score_transition_retrieval(val_texts, val_videos, model)
    return evaluate_full(scores, val_rows, candidate_ids(val_rows))


# Main functions

def train_reranker(
    seed: int,
    *,
    train_annotations: Path = ANNOTATION_ROOT / "train.jsonl",
    val_annotations: Path = ANNOTATION_ROOT / "val.jsonl",
    feature_dir: Path = VJEPA_FEATURE_ROOT,
    checkpoint_dir: Path = CHECKPOINT_ROOT,
    log_dir: Path = LOG_ROOT,
    epochs: int = NUM_EPOCHS,
    patience: int | None = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: str | None = None,
    limit: int | None = None,
) -> Path:
    """Train the reranker on a frozen stage-1 encoder; return the best checkpoint path."""
    set_random_seed(seed)
    device = get_device(device)
    encoder_checkpoint = checkpoint_path(checkpoint_dir, result_name(ENCODER_VARIANT), seed)
    model, stage1 = build_two_stage_model(encoder_checkpoint, device)
    text_encoder, text_model_name = stage1["text_encoder"], stage1["text_model_name"]

    train_rows, _ = load_split_rows(train_annotations, feature_dir, limit)
    val_rows, _ = load_split_rows(val_annotations, feature_dir, limit)
    print(f"[{RUN_NAME}] encoding {len(train_rows)} train rows with the frozen encoder")
    videos, text_bank, text_ids = frozen_embeddings(
        model, train_rows, feature_dir, text_encoder, text_model_name, device
    )
    val_texts = encode_row_texts(val_rows, text_encoder, text_model_name, device)
    val_videos = encode_variant_bank(model, EVAL_VARIANT, candidate_ids(val_rows), feature_dir)

    before = validate(model, val_rows, val_texts, val_videos)
    print(f"[{RUN_NAME}] val with untrained reranker: MRR={before['mrr']:.4f}")

    optimizer = torch.optim.AdamW(
        model.reranker.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    stopper = EarlyStopping(patience)
    best_path = checkpoint_path(checkpoint_dir, RUN_NAME, seed)
    train_log = log_path(log_dir, RUN_NAME, seed)
    train_log.parent.mkdir(parents=True, exist_ok=True)
    train_log.write_text("", encoding="utf-8")
    num_batches = max(len(train_rows) // batch_size, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).to(device)
        total = 0.0
        for b in range(num_batches):
            index = order[b * batch_size : (b + 1) * batch_size]
            ids = text_ids[index]
            loss = contrastive_loss(model.score(text_bank[ids], videos[index]), ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics = validate(model, val_rows, val_texts, val_videos)
        loss = total / num_batches
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": loss, "val": metrics}) + "\n")
        print(
            f"[{RUN_NAME}] Epoch {epoch}: loss={loss:.4f} "
            f"val R@1={metrics['recall@1']:.4f} MRR={metrics['mrr']:.4f}"
        )
        if stopper.update(epoch, metrics[SELECTION_METRIC]):
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                val_metrics=metrics,
                selection_metric=SELECTION_METRIC,
                variant=RUN_NAME,
                text_encoder=text_encoder,
                text_model_name=text_model_name,
                feature_dir=str(feature_dir),
                seed=seed,
                encoder_checkpoint=str(encoder_checkpoint),
                val_before_reranker=before,
            )
        if stopper.should_stop:
            break

    checkpoint = torch.load(best_path, map_location="cpu")
    checkpoint["training_summary"] = stopper.summary(epochs)
    torch.save(checkpoint, best_path)
    print(f"[{RUN_NAME}] best epoch {stopper.best_epoch} of {stopper.last_epoch}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--feature-dir", type=Path, default=VJEPA_FEATURE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for seed in args.seeds:
        best = train_reranker(
            seed,
            feature_dir=args.feature_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            limit=args.limit,
        )
        output = args.result_dir / f"{RUN_NAME}_seed{seed}.json"
        result = evaluate_variant(
            EVAL_VARIANT,
            best,
            ANNOTATION_ROOT / "test.jsonl",
            args.feature_dir,
            output,
            hard_negatives=ANNOTATION_ROOT / "hard_negatives.jsonl",
            action_contrast=ANNOTATION_ROOT / ACTION_CONTRAST_FILE,
            device=args.device,
            limit=args.limit,
        )
        result.update(model_name=RUN_NAME, variant=RUN_NAME)
        save_metrics(result, output)
        print(json.dumps({f"{RUN_NAME}_seed{seed}": result["full"]}, indent=2))
        print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
