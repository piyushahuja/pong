"""
Watch a trained Pong policy play.

Two modes:

    uv run play.py --mode human          # live pygame window
    uv run play.py --mode rgb            # headless; captures frames

The network, preprocessing, environment settings and checkpoint loading all
come from agent.py, so this cannot drift from what pong.py trained. Names are
re-exported below because the notebooks and replay.py call play.load_policy(),
play.make_env() and play.rollout().
"""

import argparse
from pathlib import Path

import torch

from agent import (
    D,
    H,
    Policy,
    default_checkpoint,
    get_device,
    load_policy,
    make_env,
    preprocess,
)

device = get_device()

# Re-exported for the notebooks and replay.py, which import them from here.
__all__ = [
    "D", "H", "Policy", "default_checkpoint", "get_device",
    "load_policy", "make_env", "preprocess", "rollout",
]


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
        current_frame = preprocess(observation, device)

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
