"""Video and text encoders plus the shared video feature cache (T1.4, T1.5).

Text embeddings are not aligned with V-JEPA video embeddings. The retrieval
model learns a projection before it compares them.

Heavy dependencies (torch, transformers) are imported only when a pretrained
model is first used, so the mock encoders and the cache helpers work without
them.
"""

import hashlib
import os
from pathlib import Path
import re

import numpy as np

from config import MODEL_CACHE_ROOT, TEXT_MODEL_NAME, VJEPA_MODEL_NAME


# Config

VIDEO_ENCODER_NAMES = ("vjepa2", "mock")
TEXT_ENCODER_NAMES = ("minilm", "mock")
MOCK_TEXT_DIM = 64
TEXT_BATCH_SIZE = 32
FEATURE_CACHE_KEYS = (
    "video_id",
    "frame_indices",
    "frame_embeddings",
    "pooled_embedding",
    "encoder_name",
)


# Sub functions

def resolve_device(device: str | None = None) -> str:
    """Return the requested device, or the best one available."""
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_video_id(video_id: str) -> str:
    """Return video_id if it can be used as a cache file name."""
    if (
        not isinstance(video_id, str)
        or not video_id
        or video_id in (".", "..")
        or Path(video_id).name != video_id
    ):
        raise ValueError("video_id must be a nonempty filename-safe identifier")
    return video_id


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> Path:
    """Write a compressed .npz so an interrupted run never leaves a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The process id keeps parallel extraction shards from sharing a temp file.
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class PretrainedEncoder:
    """Hold an encoder choice and load its pretrained weights on first use.

    Subclasses list their supported `encoder_names`, where "mock" needs no
    weights, and implement `load_pretrained`.
    """

    encoder_names: tuple[str, ...] = ()
    requirements = "torch and transformers"

    def __init__(self, encoder_name: str, model_name: str, device: str | None = None) -> None:
        if encoder_name not in self.encoder_names:
            raise ValueError(
                f"unknown encoder: {encoder_name}; expected one of {self.encoder_names}"
            )
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.device = device
        self.torch = None
        self.model = None

    @property
    def cache_name(self) -> str:
        """Name recorded in feature caches to tell encoders apart."""
        return "mock" if self.encoder_name == "mock" else self.model_name

    def load_model(self) -> None:
        """Load pretrained weights once, when first needed."""
        if self.model is not None:
            return
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(
                f"{type(self).__name__} needs {self.requirements}. "
                "Install requirements.txt or use encoder_name='mock'."
            ) from exc

        self.device = resolve_device(self.device)
        MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.model = self.load_pretrained(transformers).to(self.device).eval()

    def load_pretrained(self, transformers):
        """Load the processor or tokenizer and return the pretrained model."""
        raise NotImplementedError


class VideoEncoder(PretrainedEncoder):
    """Encode ordered RGB frames with V-JEPA 2 or a deterministic mock."""

    encoder_names = VIDEO_ENCODER_NAMES
    requirements = "torch, torchvision, and transformers"

    def __init__(
        self,
        encoder_name: str = "vjepa2",
        model_name: str = VJEPA_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        super().__init__(encoder_name, model_name, device)
        self.processor = None

    def load_pretrained(self, transformers):
        self.processor = transformers.AutoVideoProcessor.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )
        return transformers.AutoModel.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )

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

    def pool_temporal_tokens(
        self, hidden: np.ndarray, num_frames: int, tubelet_size: int
    ) -> np.ndarray:
        """Average spatial patches and map ordered tubelets to frames.

        V-JEPA emits one group of patch tokens per tubelet of `tubelet_size`
        frames. Each group is averaged into one state, then states are linearly
        interpolated in time so every sampled frame gets its own state.
        """
        if hidden.ndim != 2 or tubelet_size < 1:
            raise ValueError("invalid V-JEPA hidden states or tubelet size")
        temporal_length = num_frames // tubelet_size
        if temporal_length < 1 or hidden.shape[0] % temporal_length:
            raise ValueError("V-JEPA token count does not match sampled frames")

        states = hidden.reshape(temporal_length, -1, hidden.shape[-1]).mean(axis=1)
        if temporal_length == num_frames:
            return states.astype(np.float32)

        # Position of each frame on the tubelet axis, clamped at both ends.
        tubelet_centers = (np.arange(temporal_length) + 0.5) * tubelet_size - 0.5
        positions = np.interp(
            np.arange(num_frames), tubelet_centers, np.arange(temporal_length)
        )
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, temporal_length - 1)
        weight = (positions - lower)[:, None]
        return (states[lower] * (1 - weight) + states[upper] * weight).astype(np.float32)

    def encode_batch(self, videos: list[np.ndarray]) -> list[np.ndarray]:
        """Return one (T, D) float32 state array per video, in input order.

        Videos may differ in resolution, but for V-JEPA 2 they must share the
        same frame count so they can run as one batch.
        """
        videos = [self.validate_frames(frames) for frames in videos]
        if not videos:
            return []
        if self.encoder_name == "mock":
            return [self.mock_states(frames) for frames in videos]

        self.load_model()
        num_frames = len(videos[0])
        tubelet_size = int(self.model.config.tubelet_size)
        if any(len(frames) != num_frames for frames in videos):
            raise ValueError("videos in one batch must have the same number of frames")
        if num_frames < tubelet_size:
            raise ValueError(f"V-JEPA 2 requires at least {tubelet_size} frames")

        # V-JEPA's processor expects T x C x H x W, while the dataset is RGB THWC.
        clips = [
            self.torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))
            for frames in videos
        ]
        inputs = self.processor(clips, return_tensors="pt")
        processed_frames = inputs["pixel_values_videos"].shape[1]
        if processed_frames != num_frames:
            raise ValueError(
                f"V-JEPA processor changed frame count from {num_frames} "
                f"to {processed_frames}"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs, skip_predictor=True).last_hidden_state
        hidden = hidden.float().cpu().numpy()
        if not np.isfinite(hidden).all():
            raise ValueError("V-JEPA 2 returned non-finite states")
        return [
            self.pool_temporal_tokens(tokens, num_frames, tubelet_size) for tokens in hidden
        ]

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Return one float32 state per input frame, in temporal order."""
        return self.encode_batch([frames])[0]


class TextEncoder(PretrainedEncoder):
    """Encode queries with a mean-pooled sentence encoder or a deterministic mock.

    "minilm" loads `model_name` from Hugging Face, so any encoder-only model
    that is meant to be mean-pooled can be swapped in through that argument.
    """

    encoder_names = TEXT_ENCODER_NAMES

    def __init__(
        self,
        encoder_name: str = "minilm",
        model_name: str = TEXT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        super().__init__(encoder_name, model_name, device)
        self.tokenizer = None

    def load_pretrained(self, transformers):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )
        return transformers.AutoModel.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_ROOT
        )

    def mock_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return normalized hashed token counts, stable across runs."""
        embeddings = np.zeros((len(texts), MOCK_TEXT_DIM), dtype=np.float32)
        for row, value in enumerate(texts):
            for token in re.findall(r"\w+", value.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % MOCK_TEXT_DIM
                embeddings[row, column] += 1
        lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(lengths, 1)

    def encode(self, texts: list[str], batch_size: int = TEXT_BATCH_SIZE) -> np.ndarray:
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


# Loaded encoders are reused, so repeated calls never reload model weights.
video_encoders: dict[tuple[str, str, str | None], VideoEncoder] = {}
text_encoders: dict[tuple[str, str, str | None], TextEncoder] = {}


def get_video_encoder(
    encoder_name: str = "vjepa2",
    model_name: str = VJEPA_MODEL_NAME,
    device: str | None = None,
) -> VideoEncoder:
    """Return the shared video encoder for this configuration."""
    key = (encoder_name, model_name, device)
    if key not in video_encoders:
        video_encoders[key] = VideoEncoder(encoder_name, model_name, device)
    return video_encoders[key]


def get_text_encoder(
    encoder_name: str = "minilm",
    model_name: str = TEXT_MODEL_NAME,
    device: str | None = None,
) -> TextEncoder:
    """Return the shared text encoder for this configuration."""
    key = (encoder_name, model_name, device)
    if key not in text_encoders:
        text_encoders[key] = TextEncoder(encoder_name, model_name, device)
    return text_encoders[key]


# Main functions

def encode_video_frames(
    frames: np.ndarray,
    *,
    encoder_name: str = "vjepa2",
    model_name: str = VJEPA_MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    """Return frame/state embeddings with shape (T, D), dtype float32."""
    return get_video_encoder(encoder_name, model_name, device).encode(frames)


def encode_text_queries(
    texts: list[str],
    *,
    encoder_name: str = "minilm",
    model_name: str = TEXT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = TEXT_BATCH_SIZE,
) -> np.ndarray:
    """Return text embeddings with shape (N, D_text), dtype float32."""
    return get_text_encoder(encoder_name, model_name, device).encode(texts, batch_size)


def save_video_feature_cache(
    video_id: str,
    frame_indices: np.ndarray,
    frame_embeddings: np.ndarray,
    output_dir: Path,
    encoder_name: str,
) -> Path:
    """Save one video cache in the shared feature format and return its path."""
    validate_video_id(video_id)
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
    return save_npz_atomic(
        Path(output_dir) / f"{video_id}.npz",
        video_id=np.array(video_id),
        frame_indices=frame_indices.astype(np.int64),
        frame_embeddings=frame_embeddings,
        pooled_embedding=frame_embeddings.mean(axis=0).astype(np.float32),
        encoder_name=np.array(encoder_name),
    )


def load_video_feature_cache(path: Path) -> dict:
    """Load and validate one shared-format video feature cache."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = sorted(set(FEATURE_CACHE_KEYS).difference(data.files))
        if missing:
            raise ValueError(f"feature cache missing keys: {missing}")
        cache = {key: data[key] for key in FEATURE_CACHE_KEYS}

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
