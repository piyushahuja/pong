# Pong from pixels, in PyTorch

A policy-gradient agent that learns to play Atari Pong from raw frames, plus a
four-notebook walkthrough of how it works.

The agent is never told what Pong is. It gets no ball position, no velocity, no
paddle coordinates, no notion that some pixels matter and others do not — only
100,800 numbers per frame and a sparse reward of +1 or -1 when a rally ends.
From that it learns to beat the built-in opponent.

Two weight matrices, 1,280,200 parameters, no convolutions:

```
6400 (80x80 difference image) --> 200 (ReLU) --> 1 (sigmoid) --> P(move up)
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/piyushahuja/pong-pytorch.git
cd pong-pytorch
uv sync

uv run play.py --mode human        # watch the tracked policy play
```

A trained policy is committed (see [Checkpoints](#checkpoints)), so you can
watch a competent agent immediately without training one.

### In Google Colab

```python
!git clone https://github.com/piyushahuja/pong-pytorch.git
%cd pong-pytorch
!pip install -q gymnasium ale-py matplotlib
```

Then open `Notebook-4-watch-agent.ipynb`, or inline:

```python
import replay
replay.watch(seed=1)        # HTML5 player, no ffmpeg needed
```

`--mode human` will not work in Colab — a live pygame window needs an OS window
that a notebook cannot host. Use `replay.watch()` instead, which renders to
`rgb_array` and animates the frames inline.

## Layout

| Path | What it is |
|---|---|
| `pong.py` | The training loop. Runs until you stop it. |
| `play.py` | Load a checkpoint and play: live window or headless frame capture. |
| `replay.py` | `watch()` — an inline HTML5 player for a single episode. |
| `pong.ipynb` | Exploratory notebook version of `pong.py`, kept for its narrative. |
| `Notebook-1-setup.ipynb` | uv, ALE, Gymnasium; what `reset()` and `step()` return. |
| `Notebook-2-neural-network.ipynb` | Defining a net with `torch.nn`; tensor shapes and batch dims. |
| `Notebook-3-train-step.ipynb` | probability → Bernoulli sample → log-prob → loss → step. |
| `Notebook-4-watch-agent.ipynb` | What a checkpoint holds; loading it; replaying an episode. |
| `checkpoints/` | Archived policies. One is tracked; the rest stay local. |
| `test_pong.py` | Scratch script: drives the env with random actions. |
| `main.py` | uv's generated entry-point stub; nothing depends on it. |
| `assets/` | Explanatory figures for the notebooks. |

Notebooks 1–3 read without executing anything; the figures are explanatory.

## Checkpoints

`pong.py` writes `pong_policy.pt` into the project root every 100 episodes,
overwriting it each time. That file is **gitignored** — checkpoints stay on the
machine that produced them.

Snapshots worth keeping get copied into `checkpoints/` under a name that says
what they are:

```
checkpoints/pong-ep<episodes>-reward<running reward>.pt
```

```bash
# archive the current state of a live training run
cp pong_policy.pt "checkpoints/pong-ep99400-reward+6.96.pt"
```

`checkpoints/` is gitignored too, with one deliberate exception listed in
`.gitignore`: a single trained policy is tracked so a fresh clone can watch the
agent play without training first.

| File | Episodes | Running reward | Tracked? |
|---|---|---|---|
| `checkpoints/pong-ep99400-reward+6.96.pt` | 99,400 | +6.96 — beats the built-in opponent | yes |
| `pong_policy.pt` | whatever your run is at | — | no, gitignored |

To track a newer policy instead, add the file and update the negation in
`.gitignore` to match its name:

```bash
git add -f "checkpoints/pong-ep120000-reward+9.10.pt"
```

A checkpoint is a dict, not a bare model — `episode`, `model_state_dict`,
`optimizer_state_dict`, `running_reward`. Only `model_state_dict` is needed to
play. Notebook 4 unpacks it.

### Which checkpoint gets loaded

Nothing hardcodes a path. `play.default_checkpoint()` resolves in order:

1. `--checkpoint`, if you passed one
2. `pong_policy.pt` in the project root — a live training run on this machine
3. the newest `.pt` in `checkpoints/` — what a fresh clone has

So the same code does the right thing whether you are mid-training locally or
on a bare clone in Colab.

## Training

```bash
uv run pong.py
```

Prints per-episode reward and a running mean, and saves every 100 episodes. It
runs until interrupted.

This is slow. The tracked policy took roughly 99,000 episodes — days of CPU
time. Running reward climbs from about -21 (losing every point) through 0
(even) to positive territory. The intermediate checkpoint kept locally at
22,400 episodes was still at -9.49, so it loses most points but has clearly
learned to track the ball.

Note that `pong.py` always starts from scratch: it saves checkpoints but does
not load them, so it cannot resume an interrupted run.

The network runs on CPU by default. For a net this small and this sequential,
CPU is usually the right call — per-step overhead dominates, so `mps` does not
obviously help. Change `device` in `pong.py` to try it.

## Watching a trained agent

```bash
uv run play.py --mode human                  # argmax, live window
uv run play.py --mode human --sample         # sample, as it was trained
uv run play.py --mode human --episodes 5 --seed 1
uv run play.py --mode human --sticky 0.0     # no action-repeat noise
uv run play.py --checkpoint path/to/another.pt
```

`--sticky` sets `repeat_action_probability`. The default `0.25` reproduces
training conditions (and the old `Pong-v0` default); `0.0` removes the
action-repeat noise and the agent looks sharper, but it is no longer the
distribution it was trained on.

`--sample` draws from the policy's Bernoulli, which is how it behaved during
training. Without it, actions are taken greedily.

In a notebook, `replay.watch()` returns an HTML5 player built with matplotlib's
`to_jshtml`. Frames are embedded as base64 PNGs with a JS scrubber, so the
output needs no ffmpeg and survives a kernel restart and `nbconvert`. GitHub
strips the script, so it will not render in the repo's notebook preview — run
the cell to see it.

## Implementation notes

Vanilla REINFORCE, with the pieces worth naming:

- **Preprocessing.** A 210x160x3 frame is cropped to rows 35:195, downsampled
  2x, and binarised: the two background colours go to 0, everything else to 1.
  Result is a 6400-element vector. Done in torch, not NumPy.
- **Difference image.** The network is fed `current - previous` frame, not a raw
  frame. A single frame cannot express which way the ball is moving.
- **No biases.** Two bare weight matrices, initialised `randn / sqrt(fan_in)`.
- **Bernoulli log-probs.** The policy emits one probability; the action is a
  Bernoulli draw, and `dist.log_prob(action)` gives autograd the term it needs.
  Action 2 (up) if the sample is 1, action 3 (down) if 0.
- **Reward discounting** resets at rally boundaries — a `+1` or `-1` ends a
  rally, and credit does not propagate across that line. Returns are then
  normalised per episode.
- **Batched gradients.** `optimizer.zero_grad()` is deliberately *not* called
  each episode. Gradients accumulate across 10 episodes before a step, which
  averages over the noise of a single game.
- **Environment.** `ALE/Pong-v5` configured with `frameskip=(2, 5)` and
  `repeat_action_probability=0.25`, reproducing the old `Pong-v0` behaviour.

Hyperparameters: `H=200`, batch of 10 episodes, learning rate `1e-4`,
`gamma=0.99`, RMSProp with `alpha=0.99`.
