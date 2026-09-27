"""
The pieces pong.py and play.py must agree on.

A checkpoint is only meaningful next to the exact network shape, frame
preprocessing and environment settings it was trained against. When those
were written out twice -- once in pong.py, once in play.py -- they had
already drifted: play.py's Policy was missing the weight initialisation, and
pong.ipynb still uses H = 300 where both scripts use 200. So they live here,
once, and everything imports them.

The training loop stays in pong.py, deliberately flat and top-to-bottom, so
it reads in the same order as the notebooks.
"""

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

from experiment import PROJECT_ROOT


# ------------------------------------------------------------
# Shape
# ------------------------------------------------------------

# Hidden units, and the flattened 80x80 difference image.
H = 200
D = 80 * 80


# ------------------------------------------------------------
# Environment settings a checkpoint is implicitly tied to
# ------------------------------------------------------------

# These reproduce the important old Pong-v0 settings that the original
# implementation relied on. Changing them invalidates comparisons against
# policies trained before the change.
FRAMESKIP = (2, 5)
STICKY = 0.25

# Resolved against the project root, not the working directory, so play.py
# and the notebooks find the same files from wherever they are run.
CHECKPOINT = PROJECT_ROOT / "pong_policy.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
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

        PONG_DEVICE=auto uv run pong.py
        PONG_DEVICE=cuda uv run pong.py
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


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

def make_env(render_mode=None, sticky=STICKY):
    """
    sticky=0.25 reproduces training conditions (and the old Pong-v0 default).
    sticky=0.0 removes the action-repeat noise; the agent usually looks
    sharper, but it is no longer the distribution it was trained on.
    """
    gym.register_envs(ale_py)

    return gym.make(
        "ALE/Pong-v5",
        frameskip=FRAMESKIP,
        repeat_action_probability=sticky,
        full_action_space=False,
        render_mode=render_mode,
    )


# ------------------------------------------------------------
# Preprocessing
# Reference NumPy version:
#
# def prepro(I):
#     I = I[35:195]
#     I = I[::2, ::2, 0]
#     I[I == 144] = 0
#     I[I == 109] = 0
#     I[I != 0] = 1
#     return I.astype(np.float).ravel()
# ------------------------------------------------------------

def preprocess(observation, device=None):
    """
    Convert a 210x160x3 Atari RGB frame into a
    6400-element PyTorch tensor.

    No NumPy code needed explicitly.
    """

    # Gym/ALE gives us a NumPy array.
    # Turn it into a torch tensor immediately.
    x = torch.from_numpy(observation)

    # 210x160x3
    #
    # Crop away top and bottom:
    # 160x160x3
    x = x[35:195]

    # Take every second pixel in both spatial dimensions,
    # and only one colour channel:
    #
    # 160x160x3 -> 80x80
    x = x[::2, ::2, 0]

    # Remove the two background colours,
    # then turn every remaining non-zero pixel into 1.
    #
    # Background -> 0
    # Ball/paddles -> 1
    x = (
        (x != 144)
        & (x != 109)
        & (x != 0)
    ).float()

    # 80x80 -> 6400
    x = x.flatten()

    return x if device is None else x.to(device)


# ------------------------------------------------------------
# Policy network
#
# Reference NumPy version:
#
# h = np.dot(W1, x)
# h[h < 0] = 0
# logp = np.dot(W2, h)
# p = sigmoid(logp)
# ------------------------------------------------------------

class Policy(nn.Module):

    def __init__(self, inputs=D, hidden=H):
        super().__init__()

        # No biases: two bare weight matrices.
        self.fc1 = nn.Linear(
            inputs,
            hidden,
            bias=False,
        )

        self.fc2 = nn.Linear(
            hidden,
            1,
            bias=False,
        )

        # Match the reference initialization:
        #
        # W1 = randn(H, D) / sqrt(D)
        # W2 = randn(H)    / sqrt(H)

        nn.init.normal_(
            self.fc1.weight,
            mean=0.0,
            std=1.0 / math.sqrt(inputs),
        )

        nn.init.normal_(
            self.fc2.weight,
            mean=0.0,
            std=1.0 / math.sqrt(hidden),
        )

    def forward(self, x):

        h = self.fc1(x)

        h = torch.relu(h)

        logit = self.fc2(h)

        probability = torch.sigmoid(logit)

        return probability.squeeze()


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

def default_checkpoint():
    """
    Where to find weights when none were named.

    In order:

    1. pong_policy.pt in the project root -- where runs used to write, and
       where a long-lived training run may still be writing.
    2. the newest outputs/<run>/policy.pt -- the most recent experiment.
    3. the newest checkpoints/*.pt -- the archived policies, which is all a
       fresh clone has.

    All of these are gitignored except the one tracked archive: checkpoints
    stay on the machine that trained them.
    """
    if CHECKPOINT.exists():
        return CHECKPOINT

    runs = sorted(OUTPUT_DIR.glob("*/policy.pt"), key=lambda f: f.stat().st_mtime)
    if runs:
        return runs[-1]

    archived = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime)
    if archived:
        return archived[-1]

    raise FileNotFoundError(
        "No checkpoint found. Train one with `uv run pong.py`, or pass "
        "--checkpoint pointing at a .pt file."
    )


def load_policy(checkpoint=None, device=None):
    """Load a policy for inference. Weights only; the optimizer is ignored."""
    device = device or get_device()
    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    policy = Policy().to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()                       # no dropout/BN here, but state the intent

    print(
        f"loaded {checkpoint} | "
        f"episode {ckpt['episode']} | "
        f"running reward {ckpt['running_reward']:.2f}"
    )
    return policy
