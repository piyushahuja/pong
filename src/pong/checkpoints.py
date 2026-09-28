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
    Where to find weights when none were named: the most recent, wherever it is.

    Three places hold checkpoints, and the newest by modification time wins:

    1. outputs/<run>/policy.pt, written by every run
    2. checkpoints/*.pt, the archived policies, which is all a fresh clone has
    3. pong_policy.pt in the project root, where runs wrote before they had run
       directories

    Newest rather than a fixed order, because a fixed order gets this wrong. It
    used to prefer the project root, on the grounds that a long-lived run was
    writing there. Nothing writes there now, so after one resumed run the root
    file was a thousand episodes stale and "resume the newest checkpoint"
    resumed the older one.

    Pass --checkpoint or --resume-from to override this entirely.
    """
    candidates = [
        *OUTPUT_DIR.glob("*/policy.pt"),
        *CHECKPOINT_DIR.glob("*.pt"),
        *([CHECKPOINT] if CHECKPOINT.exists() else []),
    ]

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Train one with `uv run scripts/train.py`, or pass "
            "--checkpoint pointing at a .pt file."
        )

    return max(candidates, key=lambda f: f.stat().st_mtime)


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
