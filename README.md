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

**Contents.** [Quickstart](#quickstart) ·
[Layout](#layout) ·
[Checkpoints](#checkpoints) ·
[Training](#training) ·
[Experiments](#experiments) ·
[Watching a trained agent](#watching-a-trained-agent) ·
[For mentees](#for-mentees) ·
[Known gaps](#known-gaps) ·
[Implementation notes](#implementation-notes)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/piyushahuja/pong-pytorch.git
cd pong-pytorch
uv sync --locked

uv run scripts/play.py --mode human        # watch the tracked policy play
```

A trained policy is committed (see [Checkpoints](#checkpoints)), so you can
watch a competent agent immediately without training one.

`uv sync --locked` installs the exact versions in `uv.lock` and fails rather
than silently re-resolving if `pyproject.toml` and the lockfile have drifted.
Use it everywhere except when you are deliberately changing dependencies.

### In Google Colab

Colab is the one case where the kernel exists before the project does, so
`uv sync` cannot be the entry point. Each notebook in `notebooks/` carries
a bootstrap cell as its second cell: open the notebook from GitHub, run that
cell first, and it clones the repo and installs what Colab lacks. It is a no-op
when you run the same notebook locally.

By default the bootstrap installs only `gymnasium` and `ale-py`, since Colab
already ships torch and matplotlib. That takes seconds. Set `EXACT_ENV = True`
in the cell to install the locked environment instead:

```bash
uv export --locked --no-dev --format requirements.txt -o /tmp/pong-req.txt
pip install -r /tmp/pong-req.txt
```

That reproduces the pinned versions exactly, but downloads roughly a gigabyte
and replaces the torch build Colab ships with. Prefer it when a result needs to
be reproducible, not when you just want to watch the agent play.

To drive it by hand instead:

```python
from pong import replay
replay.watch(seed=1)        # HTML5 player, no ffmpeg needed
```

`--mode human` will not work in Colab — a live pygame window needs an OS window
that a notebook cannot host. Use `replay.watch()` instead, which renders to
`rgb_array` and animates the frames inline.

### On someone else's laptop

Same three commands as the local quickstart. Nothing in the repo assumes macOS,
Apple Silicon, or an absolute path, and `uv.lock` carries per-platform
resolution, so Linux and Intel Macs resolve from the same lockfile.

### On a GPU server

```bash
git clone https://github.com/piyushahuja/pong-pytorch.git
cd pong-pytorch
uv sync --locked
uv run scripts/train.py
```

The repo owns Python, torch, Gymnasium, ALE and the code. The host owns the
NVIDIA driver, the GPU and the OS — different layers, and you should not need
to install a CUDA toolkit by hand for the torch wheels pinned here.

**It defaults to CPU, on purpose.** Set `PONG_DEVICE` to change it:

```bash
PONG_DEVICE=cuda uv run scripts/train.py     # explicit
PONG_DEVICE=auto uv run scripts/train.py     # cuda, else mps, else cpu
```

CPU is the default because this network is 1.28M parameters stepped one frame
at a time: per-step overhead dominates, not matrix multiplication, so a GPU
often loses to CPU here. Every checkpoint so far was also trained on CPU, so
staying there keeps a resumed run consistent with what it resumes. If you run
the default and `nvidia-smi` looks idle, that is expected, not a
misconfiguration.

## Versions

| File | Says |
|---|---|
| `.python-version` | `3.12` — the interpreter uv provisions |
| `pyproject.toml` | `requires-python = ">=3.12,<3.13"` |
| `uv.lock` | the exact resolved versions, one per package |

The `requires-python` range is deliberately narrow. When it was `>=3.9`, the
lockfile had to carry parallel resolutions for every supported interpreter —
three numpy versions, two matplotlib, three contourpy — and which one you got
depended on your machine. Pinning to 3.12 collapsed `uv.lock` from 4118 lines
to 1108 and made "the locked environment" mean one thing.

## Layout

```
pong-pytorch/
├── pyproject.toml          dependencies, and the pong package
├── uv.lock                 the exact resolved environment
├── .python-version         3.12
│
├── src/pong/               the library: imported, never copied
│   ├── model.py            Policy — the network, defined once
│   ├── env.py              make_env, preprocess, and the settings a
│   │                       checkpoint is tied to
│   ├── checkpoints.py      finding and loading saved policies
│   ├── evaluate.py         rollout: playing without training
│   ├── experiment.py       configs, run directories, provenance
│   ├── replay.py           watch() — inline animation for notebooks
│   └── utils.py            project root, and device selection
│
├── scripts/                entry points: argument parsing, then the library
│   ├── train.py            the training loop
│   ├── play.py             watch a trained policy
│   └── explore_env.py      scratch: drives the env with random actions
│
├── notebooks/
│   ├── 01_setup.ipynb              uv, ALE, Gymnasium; reset() and step()
│   ├── 02_neural_network.ipynb     torch.nn, tensor shapes, batch dims
│   ├── 03_train_step.ipynb         probability → sample → log-prob → step
│   ├── 04_watch_agent.ipynb        checkpoints, loading, replaying
│   └── pong_exploration.ipynb      the original working notebook
│
├── configs/                one TOML file per experiment
│   ├── baseline.toml       what the tracked policy was trained with
│   └── gamma-090.toml      …and deviations from it
│
├── tests/                  fast contract tests, no training
├── checkpoints/            archived policies; one tracked, rest local
├── outputs/                one directory per run (gitignored)
└── assets/                 figures for the notebooks
```

The split is `src/` versus `scripts/`: anything a notebook or another script
might need lives in the package and is imported; the scripts only parse
arguments and call it. Nothing is ever defined twice.

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
| `pong_policy.pt` | whatever your run is at | — | no, gitignored; a run resumes from it |

To track a newer policy instead, add the file and update the negation in
`.gitignore` to match its name:

```bash
git add -f "checkpoints/pong-ep120000-reward+9.10.pt"
```

A checkpoint is a dict, not a bare model — `episode`, `model_state_dict`,
`optimizer_state_dict`, `running_reward`. Only `model_state_dict` is needed to
play. Notebook 4 unpacks it.

### Which checkpoint gets loaded

Nothing hardcodes a path. `agent.default_checkpoint()` — also reachable as
`play.default_checkpoint()`, which is how the notebooks call it — resolves in
order:

1. `--checkpoint`, if you passed one
2. `pong_policy.pt` in the project root — where runs used to write, and where a
   long-lived training run may still be writing
3. the newest `outputs/<run>/policy.pt` — your most recent experiment
4. the newest `.pt` in `checkpoints/` — what a fresh clone has

So the same code does the right thing whether you are mid-training locally or
on a bare clone in Colab. Paths resolve against the project root rather than
the working directory, so running from a subdirectory finds the same files.

## Training

```bash
uv run scripts/train.py
```

Prints per-episode reward and a running mean, writes a checkpoint every 100
episodes into a fresh directory under `outputs/`, and runs until you stop it.
See [Experiments](#experiments) for configs, provenance and resuming.

This is slow. The tracked policy took roughly 99,000 episodes — days of CPU
time. Running reward climbs from about -21 (losing every point) through 0
(even) to positive territory. The intermediate checkpoint kept locally at
22,400 episodes was still at -9.49, so it loses most points but has clearly
learned to track the ball.

The resume is exact rather than approximate, and that is a property of where
saves land: gradients are stepped every 10 episodes and checkpoints written
every 100, so a save always happens immediately after `optimizer.zero_grad()`.
No partially accumulated gradient is ever in flight when the file is written,
which is why weights, optimizer state, episode count and running reward are the
whole picture.

Training runs on CPU unless you say otherwise — `--device auto`, `--device
cuda`, or `PONG_DEVICE`. See the [GPU server](#on-a-gpu-server) section for why
CPU is the default here.

## Experiments

An experiment is this code plus a configuration, never a copy of the code.
Comparing three discount factors is three invocations, not three files:

```bash
uv run scripts/train.py --config gamma-090
uv run scripts/train.py --config gamma-095
uv run scripts/train.py --config baseline
```

Settings come from three layers, each overriding the one before: the defaults
in `pong.py`, then a TOML file in `configs/`, then command-line flags.

```bash
uv run scripts/train.py --gamma 0.95 --seed 3          # no config file needed
uv run scripts/train.py --config gamma-090 --seed 3    # config, with one override
uv run scripts/train.py --help                         # every setting is a flag
```

A setting a config names but the code does not know is an error, not something
quietly ignored — a typo like `gama = 0.9` fails at startup rather than
producing a run that silently used 0.99.

Configs are TOML rather than YAML so this needs no dependency beyond the
standard library: `tomllib` is built in, and the project pins Python 3.12.

### What a run leaves behind

Every run gets its own directory and writes nothing outside it, so two runs can
never overwrite each other:

```
outputs/2026-09-28T03-19-53Z-gamma-090/
├── config.json      the settings, and which layer each came from
├── metadata.json    commit, dirty flag, seed, device, platform, versions, argv
├── metrics.csv      one row per episode, flushed as it goes
└── policy.pt        the checkpoint, carrying its settings inside it
```

`config.json` records not just the values but where each came from:

```json
"settings": { "gamma": 0.9,     "seed": 7,   "hidden": 200 },
"source":   { "gamma": "config", "seed": "cli", "hidden": "default" }
```

That answers the question you actually have when reading an old run back —
which of these did I set, and which was just the default?

`metadata.json` records `git_commit` **and** `git_dirty`. The dirty flag matters
as much as the commit: a run made with uncommitted edits is not reproducible
from that commit alone, and recording the fact is the difference between a
usable result and a misleading one.

The settings also go inside `policy.pt`, so a checkpoint can still say what
produced it if it gets separated from its directory.

### Seeds reproduce

`--seed` seeds torch and the first `env.reset()`. Only the first — seeding every
reset would make every episode identical, which is a single game on a loop
rather than reproducibility. Two runs at the same seed produce identical
metrics, down to the loss:

```
episode,reward,running_mean,loss
1,-19.0,-19.0,-0.3787
2,-21.0,-19.02,-1.1829
```

### Stopping and resuming

A run always leaves its checkpoint, including when you Ctrl+C it — the save is
in a `finally`, so stopping a run mid-episode costs you that episode, not
everything since the last periodic save. Interrupting before the first episode
finishes discards the directory instead of leaving one that claims a result.

```bash
uv run scripts/train.py --resume                        # continue the newest checkpoint
uv run scripts/train.py --resume-from outputs/<run>/policy.pt
uv run scripts/train.py --episodes 500                  # stop after N episodes
```

Resuming is explicit and off by default. Silently continuing from whatever
checkpoint happened to be lying around is how you end up reporting a
`gamma=0.90` result that was mostly trained at `0.99`. Resuming a checkpoint
whose `hidden` differs from the current run is refused outright, since the
weight shapes cannot match.

## Watching a trained agent

```bash
uv run scripts/play.py --mode human                  # argmax, live window
uv run scripts/play.py --mode human --sample         # sample, as it was trained
uv run scripts/play.py --mode human --episodes 5 --seed 1
uv run scripts/play.py --mode human --sticky 0.0     # no action-repeat noise
uv run scripts/play.py --checkpoint path/to/another.pt
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

## For mentees

Start here if you have been handed this repo to work on. The detail is in
[CONTRIBUTING.md](CONTRIBUTING.md); this is the path through it.

### Day one

You do not have write access to this repo, so work from a fork.

```bash
gh repo fork piyushahuja/pong-pytorch --clone
cd pong-pytorch

git remote -v                    # origin should be your fork
# gh usually adds the parent as `upstream`. If `git remote -v` does not show it:
git remote add upstream https://github.com/piyushahuja/pong-pytorch.git

uv sync --locked                 # installs Python 3.12, the pinned deps, and the pong package
uv run pytest                    # ~1s. If this passes, your environment is correct.
```

Then see the thing work before you change anything:

```bash
uv run scripts/play.py --mode human
```

That loads the trained policy committed in `checkpoints/` and plays a full game in a
window. It should win. You have not trained anything — that is the point of shipping one
checkpoint: you get to see the finished behaviour on day one, before you understand it.

If `--mode human` cannot open a window, use `--mode rgb`, or open
`notebooks/04_watch_agent.ipynb` and call `replay.watch()` for an inline animation.

### Then read, in this order

| Order | File | Why |
|---|---|---|
| 1 | `notebooks/01_setup.ipynb` | what the agent is and is not told |
| 2 | `notebooks/02_neural_network.ipynb` | the network, and why it is only two layers |
| 3 | `notebooks/03_train_step.ipynb` | how one update works |
| 4 | `notebooks/04_watch_agent.ipynb` | what a checkpoint holds, and playing it back |
| 5 | `scripts/train.py` | the real loop. It reads top to bottom, in the same order as the notebooks. |

### The loop you will repeat

```
   upstream/main ── the known-working baseline
        │
        │   git fetch upstream && git rebase upstream/main
        ▼
   your fork
        │
        ├── git switch -c experiment/reward-normalisation
        │        │
        │        ├── change a config, or change src/pong/
        │        ├── uv run scripts/train.py --config ...   (several seeds)
        │        ├── uv run pytest
        │        │
        │        └── gh pr create
        │
        └── review: the diff, and whether the numbers support the claim
```

One branch per question. Two questions is two branches — otherwise the diff cannot tell
you which change moved the numbers.

### The steps, concretely

**1. Branch off an up-to-date main.**

```bash
git fetch upstream
git switch main && git rebase upstream/main
git switch -c experiment/reward-normalisation
```

**2. Make the change a config if it can be one.**

```bash
cp configs/baseline.toml configs/my-experiment.toml    # then edit the one line
# (configs/gamma-090.toml and gamma-095.toml already exist as examples)
```

Editing `src/pong/` is for a change in *method* — a different loss, extra preprocessing,
another layer. If you are editing a script to change a number, that number should have
been a setting. Every setting is already a flag: `uv run scripts/train.py --help`.

**3. Run it, on more than one seed.**

```bash
for seed in 0 1 2 3 4; do
  uv run scripts/train.py --config gamma-090 --seed $seed --episodes 2000
done
```

Reward variance between seeds is large enough that one run of a worse configuration
routinely beats one run of a better one. One seed is not a result.

**Budget the time before you start.** An episode takes a couple of seconds of CPU, so
2,000 episodes is roughly an hour and five seeds of that is most of a day. The tracked
policy took about 99,000 episodes. Decide how many episodes actually settle your question,
start one seed to measure the rate on your machine, and run the rest in parallel or
overnight. `--episodes` stops cleanly, and Ctrl+C still saves the run.

**4. Check nothing else broke.**

```bash
uv run pytest
```

**5. Open the pull request.**

```bash
git push -u origin experiment/reward-normalisation
gh pr create --fill
```

The template asks for your run directories. Quote them — each already records its commit,
seed, settings and whether the tree was dirty, so a reviewer can see what produced a
number without asking you.

### Good first tasks

In rough order of difficulty, none of which require understanding the whole thing:

1. Run `uv run scripts/train.py --episodes 200` and read `metrics.csv` from the run
   directory it prints. Plot reward against episode.
2. Play the tracked policy with `--sample` instead of the default argmax, over the same
   seed, and describe the difference in how it moves — not just the score. (Only one
   checkpoint is tracked; the weaker ones live on the machine that trained them, so ask
   for one if you want a before/after.)
3. Compare `--gamma 0.90` against baseline over five seeds. Report which won, and say
   whether the gap is bigger than the seed-to-seed spread.
4. Write the missing comparison script — read several run directories and report which
   configuration won. It is in [Known gaps](#known-gaps) because it does not exist yet.

### The rules

1. Never run `uv init` in the clone. It would overwrite `pyproject.toml`.
2. `uv sync --locked` to set up, `uv run ...` to run anything.
3. Do not `pip install` into the environment. `uv add X` instead, and commit
   `pyproject.toml` and `uv.lock` **together** — one without the other breaks everyone
   else's `uv sync --locked`, which is what CI catches.
4. Experiment differences go in configs or arguments — `--config gamma-090`, never
   `pong_gamma95.py`.
5. Never hardcode a path. Use `pong.PROJECT_ROOT`.
6. Commit source, configs and small assets. Not `.venv/`, not `outputs/`, not routine
   checkpoints.
7. Report results from a run directory, not terminal scrollback. Every run already
   records its seed, commit, dirty flag and settings.
8. `uv run pytest` before pushing. About a second, and CI runs the same thing.

A pull request comes back if it contains a copied script with numbers changed, a result
from one seed, a notebook with its own copy of `Policy`, a hardcoded path, a
`pyproject.toml` without its `uv.lock`, or a number with no run directory behind it.

### Who owns what

When something breaks, this usually says where to look.

| Owned by the repo | Owned by your machine |
|---|---|
| Python version (`.python-version`) | OS |
| dependencies (`pyproject.toml`, `uv.lock`) | NVIDIA driver |
| research code | GPU model |
| hyperparameters | RAM |
| seeds | the transient `.venv/` |

## Known gaps

Worth knowing before handing this to someone:

- **`notebooks/pong_exploration.ipynb` still defines its own policy with
  `H = 300`**, where everything else uses 200. It is the original working
  notebook, kept for its narrative, and has not been folded onto the package.
  Treat it as a historical document, not a second implementation to trust.
- **Nothing compares runs.** Each run writes its own `metrics.csv`; there is no
  script that reads several and reports which configuration won over how many
  seeds. Until there is, that comparison is done by hand, which is exactly the
  step where a mistake is easiest.
- **`scripts/train.py` runs at module level**, so it cannot be imported and the
  loop itself is not unit-tested. Deliberate: it reads top to bottom in the
  order the notebooks teach. The pieces it calls are all tested.
- **Committed notebook outputs.** `02_neural_network.ipynb` and
  `pong_exploration.ipynb` carry saved outputs, which makes their diffs noisy.
  Either strip them before committing or decide they are documentation.

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
