"""
Train a Pong policy with REINFORCE.

    uv run scripts/train.py                          # baseline settings
    uv run scripts/train.py --config gamma-090       # a config from configs/
    uv run scripts/train.py --gamma 0.95 --seed 3    # or override directly
    uv run scripts/train.py --resume                 # continue the newest checkpoint

Every run writes its own directory under outputs/ holding the settings, the
provenance (commit, seed, device, versions), per-episode metrics and the
checkpoint. Nothing is written outside it, so two runs never collide.
"""

import argparse
from pathlib import Path

import torch

from pong import experiment
from pong.checkpoints import default_checkpoint
from pong.env import make_env, preprocess
from pong.model import D, Policy
from pong.utils import get_device


# ------------------------------------------------------------
# Settings
#
# Defaults are the baseline the tracked policy was trained with. A config
# file overrides them and a command-line flag overrides that, so an
# experiment is this code plus a configuration rather than a copy of it.
#
# Network shape (D) and the environment settings live in the pong
# package, because play.py and every checkpoint must agree with them.
# ------------------------------------------------------------

DEFAULTS = {
    "hidden": 200,
    "batch_size": 10,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "decay_rate": 0.99,
    "seed": 0,
    "save_every": 100,
}

parser = argparse.ArgumentParser(
    description="Train a Pong policy with REINFORCE.",
)
parser.add_argument("--config", help="a TOML file in configs/, by name or path")
parser.add_argument("--name", help="run directory label (default: the config name)")
parser.add_argument("--render", action="store_true", help="show the game window")
parser.add_argument("--device", help='"cpu", "cuda", "mps" or "auto"')
parser.add_argument("--episodes", type=int, help="stop after this many episodes")
parser.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint found")
parser.add_argument("--resume-from", metavar="PATH",
                    help="continue from a specific checkpoint")

for _key, _value in DEFAULTS.items():
    parser.add_argument(
        f"--{_key.replace('_', '-')}",
        type=type(_value),
        default=None,                       # None means "not set on the CLI"
        help=f"default {_value}",
    )

args = parser.parse_args()

config = experiment.load_config(args.config) if args.config else {}
overrides = {key: getattr(args, key) for key in DEFAULTS}
settings, source = experiment.resolve_config(DEFAULTS, config, overrides)

# The names the training loop below reads.
BATCH_SIZE = settings["batch_size"]
LEARNING_RATE = settings["learning_rate"]
GAMMA = settings["gamma"]
DECAY_RATE = settings["decay_rate"]
SAVE_EVERY = settings["save_every"]
SEED = settings["seed"]

RENDER = args.render
MAX_EPISODES = args.episodes


# ------------------------------------------------------------
# Device
#
# Defaults to CPU; --device auto or PONG_DEVICE=auto to override. See
# pong.utils.get_device() for why CPU is the default for a network
# this small.
# ------------------------------------------------------------

device = get_device(args.device)

print(f"Using device: {device}")


# ------------------------------------------------------------
# Seeding
#
# Only the first reset is seeded. Seeding every reset would make each
# episode identical, which is not reproducibility but a single game on
# a loop.
# ------------------------------------------------------------

torch.manual_seed(SEED)


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

env = make_env("human" if RENDER else None)

observation, info = env.reset(seed=SEED)


# ------------------------------------------------------------
# Run directory
#
# Written before training starts, so an interrupted run still records
# what it was trying to do.
# ------------------------------------------------------------

run = experiment.start_run(
    name=args.name or (Path(args.config).stem if args.config else "baseline"),
    settings=settings,
    source=source,
    device=device,
    metrics_fields=["episode", "reward", "running_mean", "loss", "elapsed_seconds"],
)

CHECKPOINT = run.checkpoint

print(f"Run directory: {run.dir}")
print("Settings: " + ", ".join(
    f"{k}={v}" + ("" if source[k] == "default" else f" ({source[k]})")
    for k, v in sorted(settings.items())
))


policy = Policy(hidden=settings["hidden"]).to(device)


# ------------------------------------------------------------
# RMSProp
#
# This replaces the hand-written rmsprop_cache of the NumPy version.
# ------------------------------------------------------------

optimizer = torch.optim.RMSprop(
    policy.parameters(),
    lr=LEARNING_RATE,
    alpha=DECAY_RATE,
    eps=1e-5,
    momentum=0.0,
    weight_decay=0.0,
)


# ------------------------------------------------------------
# Saving
#
# One writer, used by the periodic save in step 17 and again when the
# loop exits. Without the second call a run shorter than save_every
# episodes would finish having produced no policy at all, and Ctrl+C
# would throw away everything since the last multiple of save_every.
# ------------------------------------------------------------

def save_checkpoint():
    torch.save(
        {
            "episode": episode_number,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "running_reward": running_reward,

            # Carried so a checkpoint can say what produced it without
            # its run directory beside it.
            "settings": settings,
        },
        CHECKPOINT,
    )


# ------------------------------------------------------------
# Discount rewards
#
# This is almost literally the reference NumPy implementation.
# ------------------------------------------------------------

def discount_rewards(rewards):
    """
    rewards looks something like:

        [0, 0, 0, ..., -1, 0, 0, ..., +1, ...]

    Work backwards and assign discounted future reward
    to every action.
    """

    discounted = [0.0] * len(rewards)

    running_reward = 0.0

    for t in reversed(range(len(rewards))):

        # Pong-specific:
        #
        # +1 or -1 means one rally has finished.
        # Don't propagate reward across rally boundaries.
        if rewards[t] != 0:
            running_reward = 0.0

        running_reward = (
            running_reward * GAMMA
            + rewards[t]
        )

        discounted[t] = running_reward

    return torch.tensor(
        discounted,
        dtype=torch.float32,
        device=device,
    )


# ------------------------------------------------------------
# Training state
# ------------------------------------------------------------

previous_frame = None

log_probs = []
rewards = []

episode_number = 0
reward_sum = 0.0
running_reward = None


# ------------------------------------------------------------
# Resume from a checkpoint
#
# Checkpoints are written every save_every episodes, and save_every is a
# multiple of batch_size, so a save always lands just after step 15
# called optimizer.zero_grad(). No partially accumulated gradient is ever
# in flight at save time, which is what makes resuming exact rather than
# approximate: weights, optimizer state, episode count and running
# reward are the whole picture.
#
# Resuming is explicit -- --resume or --resume-from PATH -- and off by
# default. Silently continuing from whatever checkpoint happened to be
# lying around is how you end up reporting a gamma=0.90 result that was
# mostly trained at 0.99.
# ------------------------------------------------------------

resume_from = args.resume_from or (default_checkpoint() if args.resume else None)

if resume_from is not None:

    checkpoint = torch.load(
        resume_from,
        map_location=device,
        weights_only=False,
    )

    # A checkpoint's weight shapes are fixed at training time, so a
    # different hidden size cannot be continued -- only compared.
    saved = checkpoint.get("settings", {})
    saved_hidden = saved.get("hidden", DEFAULTS["hidden"])

    if saved_hidden != settings["hidden"]:
        run.discard()
        raise SystemExit(
            f"{resume_from} was trained with hidden={saved_hidden}, "
            f"but this run wants hidden={settings['hidden']}. "
            f"Pass --hidden {saved_hidden} to continue it."
        )

    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    episode_number = checkpoint["episode"]
    running_reward = checkpoint["running_reward"]

    print(
        f"Resuming from {resume_from} at episode {episode_number}"
        + (
            f" | running reward {running_reward:.3f}"
            if running_reward is not None
            else ""
        )
    )

else:

    print("Starting from scratch.")


# Important:
#
# PyTorch stores gradients in parameter.grad.
#
# We deliberately DON'T call optimizer.zero_grad()
# after every episode.
#
# That means gradients accumulate for BATCH_SIZE=10
# episodes, standing in for the NumPy version's grad_buffer.

optimizer.zero_grad()


# ------------------------------------------------------------
# Main RL loop
# ------------------------------------------------------------

try:

    while True:

        # ----------------------------------------------------
        # 1. Process current screen
        # ----------------------------------------------------

        current_frame = preprocess(observation, device)


        # ----------------------------------------------------
        # 2. Compute difference image
        #
        # Reference NumPy version:
        #
        # x = cur_x - prev_x
        # ----------------------------------------------------

        if previous_frame is None:
            x = torch.zeros(
                D,
                dtype=torch.float32,
                device=device,
            )
        else:
            x = current_frame - previous_frame

        previous_frame = current_frame


        # ----------------------------------------------------
        # 3. Run policy network
        #
        # probability means:
        #
        #       P(action 2 | current state)
        # ----------------------------------------------------

        probability = policy(x)


        # ----------------------------------------------------
        # 4. Construct Bernoulli distribution
        #
        # Example:
        #
        # probability = 0.7
        #
        # sampled_action = 1 with probability 0.7
        # sampled_action = 0 with probability 0.3
        # ----------------------------------------------------

        distribution = torch.distributions.Bernoulli(
            probs=probability
        )


        # ----------------------------------------------------
        # 5. Sample from policy
        # ----------------------------------------------------

        sampled_action = distribution.sample()


        # ----------------------------------------------------
        # 6. Remember log probability of WHAT WE ACTUALLY DID
        #
        # If sampled_action == 1:
        #
        #     log_prob = log(p)
        #
        # If sampled_action == 0:
        #
        #     log_prob = log(1-p)
        #
        # This replaces the NumPy version's:
        #
        #     y - aprob
        # ----------------------------------------------------

        log_prob = distribution.log_prob(sampled_action)

        log_probs.append(log_prob)


        # ----------------------------------------------------
        # 7. Convert Bernoulli result into Atari action
        #
        # This follows the reference implementation:
        #
        # action 2 if sample == 1
        # action 3 if sample == 0
        # ----------------------------------------------------

        if sampled_action.item() == 1:
            action = 2
        else:
            action = 3


        # ----------------------------------------------------
        # 8. Act in the environment
        # ----------------------------------------------------

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        reward_sum += reward
        rewards.append(float(reward))


        # ----------------------------------------------------
        # Print when a Pong point ends
        # ----------------------------------------------------

        if reward != 0:

            marker = " !!!!!!!!" if reward == 1 else ""

            print(
                f"episode {episode_number}: "
                f"point finished, "
                f"reward {reward:+.0f}"
                f"{marker}"
            )


        # ----------------------------------------------------
        # 9. If the whole Pong episode finished...
        # ----------------------------------------------------

        if done:

            episode_number += 1


            # ------------------------------------------------
            # 10. Calculate discounted return R_t
            # ------------------------------------------------

            returns = discount_rewards(rewards)


            # ------------------------------------------------
            # 11. Normalize returns
            #
            # Reference NumPy version:
            #
            # discounted_epr -= mean
            # discounted_epr /= std
            #
            # This behaves like a crude advantage estimate.
            # ------------------------------------------------

            returns = (
                returns - returns.mean()
            ) / (
                returns.std(unbiased=False) + 1e-8
            )


            # ------------------------------------------------
            # 12. Stack log probabilities
            # ------------------------------------------------

            episode_log_probs = torch.stack(log_probs)


            # ------------------------------------------------
            # 13. REINFORCE objective
            #
            # The NumPy version effectively calculates:
            #
            #       return * ∇ log π(a|s)
            #
            # In PyTorch we write the scalar objective:
            #
            #       loss =
            #       - Σ return_t log π(a_t | s_t)
            #
            # and let autograd differentiate it.
            # ------------------------------------------------

            loss = -(
                episode_log_probs * returns
            ).sum()


            # ------------------------------------------------
            # 14. Backpropagation
            #
            # THIS replaces policy_backward().
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # 15. Every 10 episodes update the network
            #
            # Since we did not zero gradients above,
            # .grad contains the sum of gradients from
            # all 10 episodes.
            #
            # This replaces the NumPy version's grad_buffer.
            # ------------------------------------------------

            if episode_number % BATCH_SIZE == 0:

                optimizer.step()

                optimizer.zero_grad()

                print(
                    f"\n*** parameter update "
                    f"after episode {episode_number} ***\n"
                )


            # ------------------------------------------------
            # 16. Running reward
            # ------------------------------------------------

            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = (
                    running_reward * 0.99
                    + reward_sum * 0.01
                )

            print(
                f"episode {episode_number} finished | "
                f"reward: {reward_sum:.1f} | "
                f"running mean: {running_reward:.3f} | "
                f"loss: {loss.item():.3f}"
            )

            run.log(
                episode=episode_number,
                reward=reward_sum,
                running_mean=round(running_reward, 4),
                loss=round(loss.item(), 4),
            )


            # ------------------------------------------------
            # 17. Save occasionally
            # ------------------------------------------------

            if episode_number % SAVE_EVERY == 0:

                save_checkpoint()

                print(
                    f"Saved checkpoint to {CHECKPOINT}"
                )


            # ------------------------------------------------
            # 17b. Stop if --episodes was given
            # ------------------------------------------------

            if MAX_EPISODES is not None and episode_number >= MAX_EPISODES:
                print(f"Reached --episodes {MAX_EPISODES}; stopping.")
                break


            # ------------------------------------------------
            # 18. Reset episode memory
            # ------------------------------------------------

            reward_sum = 0.0

            log_probs = []
            rewards = []

            observation, info = env.reset()

            previous_frame = None


finally:
    env.close()

    # Includes KeyboardInterrupt: stopping a run should not lose it.
    if episode_number > 0:
        save_checkpoint()
        print(f"\nSaved checkpoint at episode {episode_number} to {CHECKPOINT}")
        run.close()
        print(f"Run directory: {run.dir}")
    else:
        # Interrupted before the first episode finished: there is no result,
        # so do not leave a directory claiming there is one.
        run.discard()
        print("\nStopped before the first episode finished; discarded the run.")