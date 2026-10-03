"""Video and text encoders for the shared milestone 1 interfaces.

Text embeddings are not aligned with V-JEPA video embeddings. The retrieval
model will need a learned projection before it compares them.
"""

import hashlib
from pathlib import Path
import re

import numpy as np

from config import MODEL_CACHE_ROOT, TEXT_MODEL_NAME, VJEPA_MODEL_NAME


class VideoEncoder:
    """Encode ordered RGB frames with V-JEPA 2 or a deterministic mock."""

    def __init__(
        self,
        encoder_name: str = "vjepa2",
        model_name: str = VJEPA_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if encoder_name not in ("vjepa2", "mock"):
            raise ValueError(f"unknown video encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.device = device
        self.torch = None
        self.processor = None
        self.model = None

    def validate_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1:
            raise ValueError("frames must have nonempty shape (T, H, W, 3)")
        if frames.dtype != np.uint8:
            raise ValueError("frames must be RGB uint8")
        return frames

    def mock_states(self, frames: np.ndarray) -> np.ndarray:
        """Return RGB means and standard deviations for interface checks."""
        pixels = frames.astype(np.float32) / 255.0
        means = pixels.mean(axis=(1, 2))
        deviations = pixels.std(axis=(1, 2))
        return np.concatenate((means, deviations), axis=1).astype(np.float32)

    def load_model(self) -> None:
        """Load pretrained weights once, when first needed."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:
            raise ImportError(
                "V-JEPA 2 needs torch, torchvision, and transformers. "
                "Install requirements.txt or use encoder_name='mock'."
            ) from exc

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.processor = AutoVideoProcessor.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )
        self.model = (
            AutoModel.from_pretrained(self.model_name, cache_dir=MODEL_CACHE_ROOT)
            .to(self.device)
            .eval()
        )
        self.torch = torch

    def pool_temporal_tokens(
        self, hidden: np.ndarray, num_frames: int, tubelet_size: int
    ) -> np.ndarray:
        """Average spatial patches and map ordered tubelets to frames."""
        if hidden.ndim != 2 or tubelet_size < 1:
            raise ValueError("invalid V-JEPA hidden states or tubelet size")
        temporal_length = num_frames // tubelet_size
        if temporal_length < 1 or hidden.shape[0] % temporal_length:
            raise ValueError("V-JEPA token count does not match sampled frames")

        states = hidden.reshape(temporal_length, -1, hidden.shape[-1]).mean(axis=1)
        if temporal_length == num_frames:
            return states.astype(np.float32)

        source_times = (np.arange(temporal_length) + 0.5) * tubelet_size - 0.5
        frame_times = np.arange(num_frames)
        result = np.empty((num_frames, states.shape[1]), dtype=np.float32)
        for column in range(states.shape[1]):
            result[:, column] = np.interp(frame_times, source_times, states[:, column])
        return result

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Return one float32 state per input frame, in temporal order."""
        frames = self.validate_frames(frames)
        if self.encoder_name == "mock":
            return self.mock_states(frames)

        self.load_model()
        tubelet_size = int(self.model.config.tubelet_size)
        if len(frames) < tubelet_size:
            raise ValueError(f"V-JEPA 2 requires at least {tubelet_size} frames")

        # V-JEPA's processor expects T x C x H x W, while the dataset is RGB THWC.
        video = self.torch.from_numpy(
            np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        )
        inputs = self.processor(video, return_tensors="pt")
        processed_frames = inputs["pixel_values_videos"].shape[1]
        if processed_frames != len(frames):
            raise ValueError(
                f"V-JEPA processor changed frame count from {len(frames)} "
                f"to {processed_frames}"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs, skip_predictor=True).last_hidden_state[0]
        states = self.pool_temporal_tokens(
            hidden.float().cpu().numpy(), len(frames), tubelet_size
        )
        if not np.isfinite(states).all():
            raise ValueError("V-JEPA 2 returned non-finite states")
        return states


class TextEncoder:
    """Encode queries with MiniLM or a deterministic token-count mock."""

    def __init__(
        self,
        encoder_name: str = "minilm",
        model_name: str = TEXT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if encoder_name not in ("minilm", "mock"):
            raise ValueError(f"unknown text encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.device = device
        self.torch = None
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load pretrained weights once, when first needed."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MiniLM needs torch and transformers. Install "
                "requirements.txt or use encoder_name='mock'."
            ) from exc

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )
        self.model = (
            AutoModel.from_pretrained(self.model_name, cache_dir=MODEL_CACHE_ROOT)
            .to(self.device)
            .eval()
        )
        self.torch = torch

    def mock_embeddings(self, texts: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), 64), dtype=np.float32)
        for row, value in enumerate(texts):
            for token in re.findall(r"\w+", value.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % 64
                embeddings[row, column] += 1
        lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(lengths, 1)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return float32 rows in the same order as the input texts."""
        if not isinstance(texts, list) or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("texts must be a list of nonempty strings")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.encoder_name == "mock":
            return self.mock_embeddings(texts)

        self.load_model()
        if not texts:
            return np.empty((0, int(self.model.config.hidden_size)), dtype=np.float32)

        batches = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                tokens = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches).astype(np.float32)


video_encoders: dict[tuple[str, str, str | None], VideoEncoder] = {}
text_encoders: dict[tuple[str, str, str | None], TextEncoder] = {}


def encode_video_frames(
    frames: np.ndarray,
    *,
    encoder_name: str = "vjepa2",
    model_name: str = VJEPA_MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    """Return frame/state embeddings with shape (T, D), dtype float32."""
    key = (encoder_name, model_name, device)
    if key not in video_encoders:
        video_encoders[key] = VideoEncoder(encoder_name, model_name, device)
    return video_encoders[key].encode(frames)


def encode_text_queries(
    texts: list[str],
    *,
    encoder_name: str = "minilm",
    model_name: str = TEXT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Return text embeddings with shape (N, D_text), dtype float32."""
    key = (encoder_name, model_name, device)
    if key not in text_encoders:
        text_encoders[key] = TextEncoder(encoder_name, model_name, device)
    return text_encoders[key].encode(texts, batch_size)


def save_video_feature_cache(
    video_id: str,
    frame_indices: np.ndarray,
    frame_embeddings: np.ndarray,
    output_dir: Path,
    encoder_name: str,
) -> Path:
    """Save one video cache in the shared feature format."""
    if (
        not isinstance(video_id, str)
        or not video_id
        or video_id in (".", "..")
        or Path(video_id).name != video_id
    ):
        raise ValueError("video_id must be a nonempty filename-safe identifier")
    if not isinstance(encoder_name, str) or not encoder_name:
        raise ValueError("encoder_name must be nonempty")
    frame_indices = np.asarray(frame_indices)
    frame_embeddings = np.asarray(frame_embeddings)
    if frame_indices.ndim != 1 or frame_indices.dtype.kind not in "iu":
        raise ValueError("frame_indices must be a one-dimensional integer array")
    if (
        frame_embeddings.ndim != 2
        or frame_embeddings.shape[0] == 0
        or frame_embeddings.shape[1] == 0
    ):
        raise ValueError("frame_embeddings must have nonempty shape (T, D)")
    if len(frame_indices) != frame_embeddings.shape[0]:
        raise ValueError("frame_indices length must match frame_embeddings T")
    if np.any(np.diff(frame_indices) < 0):
        raise ValueError("frame_indices must preserve temporal order")
    if not np.isfinite(frame_embeddings).all():
        raise ValueError("frame_embeddings must be finite")

    frame_embeddings = frame_embeddings.astype(np.float32)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.npz"
    np.savez_compressed(
        output_path,
        video_id=np.array(video_id),
        frame_indices=frame_indices.astype(np.int64),
        frame_embeddings=frame_embeddings,
        pooled_embedding=frame_embeddings.mean(axis=0).astype(np.float32),
        encoder_name=np.array(encoder_name),
    )
    return output_path


def load_video_feature_cache(path: Path) -> dict:
    """Load and validate one shared-format video feature cache."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = {
            "video_id",
            "frame_indices",
            "frame_embeddings",
            "pooled_embedding",
            "encoder_name",
        }
        if not required.issubset(data.files):
            missing = sorted(required.difference(data.files))
            raise ValueError(f"feature cache missing keys: {missing}")
        cache = {key: data[key] for key in required}

    indices = cache["frame_indices"]
    frames = cache["frame_embeddings"]
    pooled = cache["pooled_embedding"]
    valid_shape = (
        indices.ndim == 1
        and indices.dtype == np.int64
        and frames.ndim == 2
        and frames.dtype == np.float32
        and frames.shape[0] == len(indices)
        and pooled.shape == (frames.shape[1],)
        and pooled.dtype == np.float32
    )
    if not valid_shape:
        raise ValueError(f"invalid feature cache shapes or dtypes: {path}")
    if not np.isfinite(frames).all() or not np.isfinite(pooled).all():
        raise ValueError(f"non-finite feature cache: {path}")
    return cache
