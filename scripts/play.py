"""
Watch a trained Pong policy play.

    uv run scripts/play.py --mode human     # live pygame window
    uv run scripts/play.py --mode rgb       # headless; captures frames

Everything it needs comes from the pong package, so this cannot drift from
what training produced.
"""

import argparse
from pathlib import Path

import pong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["human", "rgb"], default="human")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--sample", action="store_true", help="sample instead of argmax")
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    policy = pong.load_policy(args.checkpoint)
    env = pong.make_env("human" if args.mode == "human" else "rgb_array", args.sticky)

    try:
        for i in range(args.episodes):
            seed = None if args.seed is None else args.seed + i
            print(f"\n--- episode {i} (seed={seed}) ---")
            pong.rollout(policy, env, greedy=not args.sample, seed=seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
