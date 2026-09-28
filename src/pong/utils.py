"""
Where the project is, and what it runs on.

Both answers have to be the same for every entry point, or a checkpoint
trained by one script cannot be read by another.
"""

import os
from pathlib import Path

import torch


# ------------------------------------------------------------
# Project root
#
# Paths must not depend on where you happen to have cd'd, and must not
# encode anyone's home directory. The root is the directory holding
# pyproject.toml, found by walking up from this file, so the repo works at
# /content/pong-pytorch and /home/you/pong-pytorch alike.
#
# The package is installed editable (uv sync), so __file__ points into the
# checkout rather than site-packages and the walk finds the marker.
# ------------------------------------------------------------

def find_project_root(start=None):
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Vendored somewhere without the marker: fall back to CWD rather than guess.
    return Path.cwd()


PROJECT_ROOT = find_project_root()

CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

def get_device(spec=None):
    """
    Resolve the torch device once, for every entry point.

    spec -- or the PONG_DEVICE environment variable -- is either an explicit
    device string ("cpu", "cuda", "mps") or "auto" to take the best available.

    The default is deliberately "cpu" rather than "auto". This network is
    1.28M parameters stepped one frame at a time, so per-step overhead
    dominates and a GPU generally loses to CPU here; and every checkpoint so
    far was trained on CPU, so staying there keeps a resumed run numerically
    consistent with what it is resuming.

        PONG_DEVICE=auto uv run scripts/train.py
        PONG_DEVICE=cuda uv run scripts/train.py
    """
    spec = spec or os.environ.get("PONG_DEVICE", "cpu")

    if spec != "auto":
        return torch.device(spec)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
