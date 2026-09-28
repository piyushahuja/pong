"""
Train a Pong policy with REINFORCE.

    uv run scripts/train.py                          # baseline settings
    uv run scripts/train.py --config gamma-090       # a config from configs/
    uv run scripts/train.py --gamma 0.95 --seed 3    # or override directly
    uv run scripts/train.py --resume                 # continue the newest checkpoint

Every run writes its own directory under outputs/ holding the settings, the
provenance (commit, seed, device, versions), per-episode metrics and the
checkpoint. Nothing is written outside it, so two runs never collide.

The training loop at the bottom runs top to bottom, in the same order as the
notebooks teach it. Everything above it is setup.
"""

import argparse
from pathlib import Path

import torch
from torch.distributions import Bernoulli

from pong import experiment
from pong.checkpoints import default_checkpoint
from pong.env import ACTION_DOWN, ACTION_UP, make_env, preprocess
from pong.model import D, Policy
from pong.utils import get_device


# Settings a config file or a command-line flag may override. The network
# shape and the environment settings are NOT here: they live in the pong
# package, because play.py and every saved checkpoint must agree with them.
DEFAULTS = {
    "hidden": 200,
    "batch_size": 10,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "decay_rate": 0.99,
    "seed": 0,
    "save_every": 100,
}

# How quickly the reported running mean forgets older episodes. Not gamma:
# gamma discounts rewards inside an episode, this only smooths the number
# printed between them.
RUNNING_REWARD_SMOOTHING = 0.99

# Guards against dividing by zero when every return in an episode is equal.
NORMALISE_EPSILON = 1e-8


def parse_arguments():
    """Every setting in DEFAULTS is also a flag, so nothing needs a config file."""
    parser = argparse.ArgumentParser(description="Train a Pong policy with REINFORCE.")

    parser.add_argument("--config", help="a TOML file in configs/, by name or path")
    parser.add_argument("--name", help="run directory label (default: the config name)")
    parser.add_argument("--render", action="store_true", help="show the game window")
    parser.add_argument("--device", help='"cpu", "cuda", "mps" or "auto"')
    parser.add_argument("--episodes", type=int, help="stop after this many episodes")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest checkpoint found")
    parser.add_argument("--resume-from", metavar="PATH",
                        help="continue from a specific checkpoint")

    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=None,            # None means "not given on the command line"
            help=f"default {default}",
        )

    return parser.parse_args()


def resolve_settings(arguments):
    """Layer the settings: defaults, then the config file, then the flags."""
    config = experiment.load_config(arguments.config) if arguments.config else {}
    overrides = {name: getattr(arguments, name) for name in DEFAULTS}
    return experiment.resolve_config(DEFAULTS, config, overrides)


def describe_settings(settings, source):
    """One line naming each setting and which layer it came from."""
    return ", ".join(
        f"{name}={value}" + ("" if source[name] == "default" else f" ({source[name]})")
        for name, value in sorted(settings.items())
    )


def discounted_returns(rewards, gamma, device):
    """
    Turn a list of rewards into a weight for every action that preceded them.

    rewards looks like [0, 0, 0, ..., -1, 0, 0, ..., +1, ...]. Walking
    backwards, each step's weight is its own reward plus the discounted
    weight of the step after it.

    The running total resets whenever a reward is nonzero, because in Pong a
    +1 or -1 ends a rally rather than the episode. Credit must not flow back
    across a scored point: the actions after it had no influence on it.
    """
    returns = [0.0] * len(rewards)
    running = 0.0

    for step in reversed(range(len(rewards))):
        if rewards[step] != 0:
            running = 0.0
        running = running * gamma + rewards[step]
        returns[step] = running

    return torch.tensor(returns, dtype=torch.float32, device=device)


def save_checkpoint(path, policy, optimizer, episode, running_reward, settings):
    """Everything needed to resume, plus the settings that produced it."""
    torch.save(
        {
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "running_reward": running_reward,
            "settings": settings,
        },
        path,
    )


def load_checkpoint(path, policy, optimizer, settings, device):
    """
    Restore a run, and return where it had got to as (episode, running_reward).

    The restore is exact rather than approximate because of where saves land:
    gradients are stepped every batch_size episodes and checkpoints written
    every save_every, and save_every is a multiple of batch_size, so a save
    always happens just after optimizer.zero_grad(). No partially accumulated
    gradient is ever in flight, which is why these four values are the whole
    picture.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Weight shapes are fixed at training time, so a different hidden size
    # cannot be continued, only compared against.
    trained_hidden = checkpoint.get("settings", {}).get("hidden", DEFAULTS["hidden"])
    if trained_hidden != settings["hidden"]:
        raise SystemExit(
            f"{path} was trained with hidden={trained_hidden}, but this run wants "
            f"hidden={settings['hidden']}. Pass --hidden {trained_hidden} to continue it."
        )

    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["episode"], checkpoint["running_reward"]


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

arguments = parse_arguments()
settings, source = resolve_settings(arguments)

device = get_device(arguments.device)
print(f"Using device: {device}")

# Only the first reset is seeded. Seeding every reset would make every episode
# identical, which is one game on a loop rather than reproducibility.
torch.manual_seed(settings["seed"])

env = make_env("human" if arguments.render else None)
observation, _ = env.reset(seed=settings["seed"])

policy = Policy(hidden=settings["hidden"]).to(device)

optimizer = torch.optim.RMSprop(
    policy.parameters(),
    lr=settings["learning_rate"],
    alpha=settings["decay_rate"],
    eps=1e-5,
)

# Written before training starts, so an interrupted run still records what it
# was trying to do.
run = experiment.start_run(
    name=arguments.name or (Path(arguments.config).stem if arguments.config else "baseline"),
    settings=settings,
    source=source,
    device=device,
    metrics_fields=["episode", "reward", "running_mean", "loss", "elapsed_seconds"],
)
print(f"Run directory: {run.dir}")
print("Settings: " + describe_settings(settings, source))

episode_number = 0
running_reward = None

# Resuming is explicit, and off by default. Silently continuing from whatever
# checkpoint happened to be lying around is how you end up reporting a
# gamma=0.90 result that was mostly trained at 0.99.
resume_from = arguments.resume_from or (default_checkpoint() if arguments.resume else None)

if resume_from is None:
    print("Starting from scratch.")
else:
    try:
        episode_number, running_reward = load_checkpoint(
            resume_from, policy, optimizer, settings, device
        )
    except SystemExit:
        run.discard()               # nothing was produced; leave no directory
        raise
    print(f"Resuming from {resume_from} at episode {episode_number}"
          + (f" | running reward {running_reward:.3f}" if running_reward is not None else ""))

# Gradients are deliberately NOT zeroed after each episode. They accumulate for
# batch_size episodes and are applied in one step, which averages over the
# noise of a single game.
optimizer.zero_grad()

previous_frame = None
episode_log_probs = []
episode_rewards = []
episode_reward_sum = 0.0


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

try:
    while True:

        # 1. The network sees motion, not a still frame. One frame cannot say
        #    which way the ball is travelling, so feed it the difference.
        current_frame = preprocess(observation, device)

        if previous_frame is None:
            difference_image = torch.zeros(D, dtype=torch.float32, device=device)
        else:
            difference_image = current_frame - previous_frame

        previous_frame = current_frame

        # 2. One number out: P(move up | this state).
        probability_up = policy(difference_image)

        # 3. Sample an actual choice from that probability, and remember how
        #    likely the choice we made was. That log-prob is the handle by
        #    which the action is later made more or less likely.
        distribution = Bernoulli(probs=probability_up)
        sampled_action = distribution.sample()
        episode_log_probs.append(distribution.log_prob(sampled_action))

        # 4. Translate the 0/1 sample into an Atari controller input.
        action = ACTION_UP if sampled_action.item() == 1 else ACTION_DOWN

        # 5. Act, and record what came back.
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_rewards.append(float(reward))
        episode_reward_sum += reward

        if reward != 0:
            marker = " !!!!!!!!" if reward == 1 else ""
            print(f"episode {episode_number}: point finished, reward {reward:+.0f}{marker}")

        if not (terminated or truncated):
            continue

        # ------------------------------------------------------------------
        # The episode ended: score every decision in it and learn from them.
        # ------------------------------------------------------------------

        episode_number += 1

        # 6. Weight each action by the discounted reward that followed it,
        #    then normalise so the weights say "better or worse than this
        #    episode's average" rather than carrying the raw scale. This is a
        #    crude advantage estimate, and it is what makes some weights
        #    negative even in an episode that was won overall.
        returns = discounted_returns(episode_rewards, settings["gamma"], device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + NORMALISE_EPSILON)

        # 7. The REINFORCE objective:
        #
        #        loss = - sum_t  return_t * log pi(a_t | s_t)
        #
        #    Minimising it raises the probability of actions with positive
        #    weight and lowers it for negative ones. Autograd does the rest.
        loss = -(torch.stack(episode_log_probs) * returns).sum()
        loss.backward()

        # 8. Apply the accumulated gradient once per batch of episodes.
        if episode_number % settings["batch_size"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"\n*** parameter update after episode {episode_number} ***\n")

        # 9. Report and record. running_reward is the number to watch: single
        #    episodes are far too noisy to judge progress by.
        if running_reward is None:
            running_reward = episode_reward_sum
        else:
            running_reward = (running_reward * RUNNING_REWARD_SMOOTHING
                              + episode_reward_sum * (1 - RUNNING_REWARD_SMOOTHING))

        print(f"episode {episode_number} finished | "
              f"reward: {episode_reward_sum:.1f} | "
              f"running mean: {running_reward:.3f} | "
              f"loss: {loss.item():.3f}")

        run.log(
            episode=episode_number,
            reward=episode_reward_sum,
            running_mean=round(running_reward, 4),
            loss=round(loss.item(), 4),
        )

        if episode_number % settings["save_every"] == 0:
            save_checkpoint(run.checkpoint, policy, optimizer,
                            episode_number, running_reward, settings)
            print(f"Saved checkpoint to {run.checkpoint}")

        if arguments.episodes is not None and episode_number >= arguments.episodes:
            print(f"Reached --episodes {arguments.episodes}; stopping.")
            break

        # 10. Start the next episode.
        episode_log_probs = []
        episode_rewards = []
        episode_reward_sum = 0.0
        previous_frame = None
        observation, _ = env.reset()

finally:
    env.close()

    # This runs on Ctrl+C too: stopping a run should not lose it.
    if episode_number > 0:
        save_checkpoint(run.checkpoint, policy, optimizer,
                        episode_number, running_reward, settings)
        print(f"\nSaved checkpoint at episode {episode_number} to {run.checkpoint}")
        run.close()
        print(f"Run directory: {run.dir}")
    else:
        # Stopped before the first episode finished, so there is no result.
        # Do not leave a directory claiming there is one.
        run.discard()
        print("\nStopped before the first episode finished; discarded the run.")
