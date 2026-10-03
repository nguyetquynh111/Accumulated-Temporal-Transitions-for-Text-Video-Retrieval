import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    """Seed standard random generators used by lightweight project code."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
