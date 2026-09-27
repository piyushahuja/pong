"""
The environment, and turning its frames into network input.

The settings here are part of what a checkpoint means. A policy trained with
sticky actions at 0.25 is not the same policy evaluated at 0.0, so these are
constants rather than call-site arguments.
"""

import torch
import gymnasium as gym
import ale_py


# These reproduce the important old Pong-v0 settings that the original
# implementation relied on. Changing them invalidates comparisons against
# policies trained before the change.
FRAMESKIP = (2, 5)
STICKY = 0.25


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
