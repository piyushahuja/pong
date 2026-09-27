"""
Watch a trained Pong policy play.

Two modes:

    uv run play.py --mode human          # live pygame window (needs: uv add pygame)
    uv run play.py --mode rgb            # headless; saves frames for the notebook

The policy definition here duplicates pong.py on purpose: importing pong.py
would start a training run, because its loop is at module level.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import gymnasium as gym
import ale_py


H = 200
D = 80 * 80

device = torch.device("cpu")


class Policy(nn.Module):
    """Identical to pong.py: 6400 -> 200 -> 1, no biases."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H, bias=False)
        self.fc2 = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze()


def preprocess(observation):
    """210x160x3 RGB frame -> 6400 float tensor. Same as pong.py."""
    x = torch.from_numpy(observation)
    x = x[35:195]
    x = x[::2, ::2, 0]
    x = ((x != 144) & (x != 109) & (x != 0)).float()
    return x.flatten().to(device)


CHECKPOINT_DIR = Path("checkpoints")


def default_checkpoint():
    """
    Where to find weights when none were named.

    A live training run writes pong_policy.pt into the project root, and that
    is gitignored: checkpoints stay on the machine that trained them. So prefer
    it when it exists, and otherwise fall back to the newest archived policy in
    checkpoints/, which is what a fresh clone has.
    """
    live = Path("pong_policy.pt")
    if live.exists():
        return live

    archived = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime)
    if archived:
        return archived[-1]

    raise FileNotFoundError(
        "No checkpoint found. Train one with `uv run pong.py`, or pass "
        "--checkpoint pointing at a .pt file."
    )


def load_policy(checkpoint=None):
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


def make_env(render_mode, sticky=0.25):
    """
    sticky=0.25 reproduces training conditions (and the old Pong-v0 default).
    sticky=0.0 removes the action-repeat noise; the agent usually looks
    sharper, but it is no longer the distribution it was trained on.
    """
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Pong-v5",
        frameskip=(2, 5),
        repeat_action_probability=sticky,
        full_action_space=False,
        render_mode=render_mode,
    )


@torch.no_grad()
def rollout(
    policy,
    env,
    greedy=True,
    seed=None,
    max_steps=20000,
    collect_frames=False,
    frame_stride=1,
    verbose=True,
):
    """
    Play one full episode (first to 21).

    greedy=True  -> take argmax, i.e. UP when p > 0.5. Plays visibly better.
    greedy=False -> sample from Bernoulli(p), exactly as during training.

    verbose=False silences the per-point commentary; useful when looping
    over several seeds and you only want the final scores.

    Returns (score_us, score_them, frames, probs).
    """
    observation, _ = env.reset(seed=seed)
    previous_frame = None

    frames, probs = [], []
    score_us = score_them = 0

    for step in range(max_steps):
        current_frame = preprocess(observation)

        # The network sees motion, not a still: the difference image.
        if previous_frame is None:
            x = torch.zeros(D, dtype=torch.float32, device=device)
        else:
            x = current_frame - previous_frame
        previous_frame = current_frame

        p = policy(x)

        if greedy:
            action = 2 if p.item() > 0.5 else 3
        else:
            action = 2 if torch.distributions.Bernoulli(probs=p).sample().item() == 1 else 3

        if collect_frames and step % frame_stride == 0:
            frames.append(env.render())
            probs.append(p.item())

        observation, reward, terminated, truncated, _ = env.step(action)

        if reward > 0:
            score_us += 1
            if verbose:
                print(f"  step {step:5d}  WE SCORE   {score_us:2d} - {score_them:<2d} !!!")
        elif reward < 0:
            score_them += 1
            if verbose:
                print(f"  step {step:5d}  they score {score_us:2d} - {score_them:<2d}")

        if terminated or truncated:
            break

    if verbose:
        print(f"final: {score_us} - {score_them}")
    return score_us, score_them, frames, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["human", "rgb"], default="human")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--sample", action="store_true", help="sample instead of argmax")
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    policy = load_policy(args.checkpoint)
    env = make_env("human" if args.mode == "human" else "rgb_array", args.sticky)

    try:
        for i in range(args.episodes):
            seed = None if args.seed is None else args.seed + i
            print(f"\n--- episode {i} (seed={seed}) ---")
            rollout(policy, env, greedy=not args.sample, seed=seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
