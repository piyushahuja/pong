"""
Playing a trained policy, without training it.

The same preprocessing and difference-image logic as training, in inference
mode: no gradients, and optionally argmax instead of sampling.
"""

import torch
from torch.distributions import Bernoulli

from pong.env import ACTION_DOWN, ACTION_UP, preprocess
from pong.model import D
from pong.utils import get_device


device = get_device()


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

    Returns (score_us, score_them, frames, probabilities).
    """
    observation, _ = env.reset(seed=seed)
    previous_frame = None

    frames, probabilities = [], []
    score_us = score_them = 0

    for step in range(max_steps):
        current_frame = preprocess(observation, device)

        # The network sees motion, not a still frame: feed it the difference.
        if previous_frame is None:
            difference_image = torch.zeros(D, dtype=torch.float32, device=device)
        else:
            difference_image = current_frame - previous_frame
        previous_frame = current_frame

        probability_up = policy(difference_image)

        if greedy:
            move_up = probability_up.item() > 0.5
        else:
            move_up = Bernoulli(probs=probability_up).sample().item() == 1

        action = ACTION_UP if move_up else ACTION_DOWN

        if collect_frames and step % frame_stride == 0:
            frames.append(env.render())
            probabilities.append(probability_up.item())

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
    return score_us, score_them, frames, probabilities
