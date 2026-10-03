from pathlib import Path


def train_transition_model(
    train_annotations: Path,
    val_annotations: Path,
    feature_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    """Train the transition-aware model and return the best checkpoint path."""
    raise NotImplementedError(
        "Training will be implemented after the transition model architecture is finalized."
    )
