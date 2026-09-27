"""
Finding and loading checkpoints.

Checkpoints are gitignored: they stay on the machine that produced them,
apart from one archived policy tracked so a fresh clone has something to
play. So nothing can hardcode a path -- it has to be resolved.
"""

from pathlib import Path

import torch

from pong.model import Policy
from pong.utils import OUTPUT_DIR, PROJECT_ROOT, get_device


# Resolved against the project root, not the working directory, so every
# entry point finds the same files from wherever it is run.
CHECKPOINT = PROJECT_ROOT / "pong_policy.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


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
        "No checkpoint found. Train one with `uv run scripts/train.py`, or pass "
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
