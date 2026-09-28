"""
Pong from pixels.

The public surface is flat on purpose, so a notebook or script says

    import pong

    policy = pong.load_policy()
    env = pong.make_env("rgb_array")
    us, them, frames, probs = pong.rollout(policy, env)

rather than reaching into submodules. The submodules exist to keep each
concern in one file, not to make callers spell out a path:

    pong.model        the network
    pong.env          the environment and frame preprocessing
    pong.checkpoints  finding and loading saved policies
    pong.evaluate     playing a trained policy
    pong.experiment   configs, run directories, provenance
    pong.replay       inline animation for notebooks
    pong.utils        project root and device selection
"""

from pong.checkpoints import (
    CHECKPOINT,
    CHECKPOINT_DIR,
    default_checkpoint,
    load_policy,
)
from pong.env import FRAMESKIP, STICKY, make_env, preprocess
from pong.evaluate import rollout
from pong.model import D, H, Policy
from pong.utils import (
    CONFIG_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    get_device,
)
