"""
Watch a trained Pong policy play.

    uv run scripts/play.py                        # a live window, best play
    uv run scripts/play.py --sample               # act as it did while training
    uv run scripts/play.py --episodes 5 --seed 1  # five reproducible games
    uv run scripts/play.py --mode rgb             # no window, scores only

The network, preprocessing and environment settings all come from the pong
package, so this cannot drift from what training produced. For an animation
inside a notebook, use pong.replay.watch() instead.
"""

import argparse
from pathlib import Path

import pong


def parse_arguments():
    parser = argparse.ArgumentParser(description="Watch a trained Pong policy play.")

    parser.add_argument(
        "--mode", choices=["human", "rgb"], default="human",
        help="human opens a window; rgb runs headless and just reports the score",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="a .pt file to load (default: the newest one found)",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="sample from the policy, as during training, instead of taking its best guess",
    )
    parser.add_argument(
        "--sticky", type=float, default=pong.STICKY,
        help=f"chance the console repeats the previous action (default {pong.STICKY}, "
             "which is what training used)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed the first episode, so a game can be replayed exactly",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="how many games to play",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    policy = pong.load_policy(arguments.checkpoint)
    render_mode = "human" if arguments.mode == "human" else "rgb_array"
    env = pong.make_env(render_mode, arguments.sticky)

    try:
        for episode in range(arguments.episodes):
            # Different seed per episode, so five games are five games rather
            # than the same one five times.
            seed = None if arguments.seed is None else arguments.seed + episode
            print(f"\n--- episode {episode} (seed={seed}) ---")
            pong.rollout(policy, env, greedy=not arguments.sample, seed=seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
