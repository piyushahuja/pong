# Working on this repo

The point of the setup here is that an experiment can be reviewed. Not "here is
my `notebook_final2.ipynb`", but a branch whose diff says what changed and a run
directory that says what happened.

## Setting up

```bash
git clone https://github.com/piyushahuja/pong.git
cd pong
uv sync --locked
uv run pytest
```

If `uv run pytest` passes you have a working environment. Do not run `uv init`,
do not create a virtualenv by hand, and do not `pip install` into it.

## The loop

```
  main ── the known-working baseline
   │
   ├── branch: experiment/reward-normalisation
   │     │
   │     ├── a config, or a change in src/pong/
   │     ├── uv run scripts/train.py --config ...     (several seeds)
   │     ├── uv run pytest
   │     │
   │     └── pull request, quoting the run directories
   │
   └── review: the diff, and whether the numbers support the claim
```

### 1. Branch

```bash
git switch -c experiment/reward-normalisation
```

One branch per question. If you are asking two questions, that is two branches,
because otherwise the diff cannot tell you which change moved the numbers.

### 2. Make the change a config where it can be

A different hyperparameter is a config, not an edit:

```bash
cp configs/baseline.toml configs/my-experiment.toml   # then edit the one line
uv run scripts/train.py --config my-experiment --seed 0

# gamma-090.toml and gamma-095.toml already exist; use them as examples rather
# than overwriting them.
```

Editing `src/pong/` is for a change in *method*, a different loss, a different
preprocessing step, an extra network layer. If you find yourself editing
`scripts/train.py` to change a number, that number should have been a setting.

### 3. Run it more than once

```bash
for seed in 0 1 2 3 4; do
  uv run scripts/train.py --config gamma-090 --seed $seed
done
```

A single seed tells you very little here. Reward variance between seeds is
large enough that one run of a worse configuration routinely beats one run of a
better one. Five seeds is a reasonable minimum for a claim.

### 4. Check nothing else broke

```bash
uv run pytest
```

CI runs the same tests plus `uv sync --locked`, which fails if you changed
dependencies without committing the lockfile.

### 5. Open a pull request

```bash
git push -u origin experiment/reward-normalisation
gh pr create
```

The template asks for the run directories. Quote them, each one already
records its commit, seed, settings and whether the tree was dirty, so a
reviewer can tell what produced the number without asking you.

## What gets committed

| Commit | Do not commit |
|---|---|
| `src/`, `scripts/`, `tests/` | `.venv/` |
| `configs/*.toml` | `outputs/`, runs are reproducible from config + commit |
| `pyproject.toml` and `uv.lock` together | checkpoints, except a deliberate archive |
| notebooks, with their outputs | datasets, videos, saved animations |

`outputs/` and `*.pt` are gitignored, so this mostly takes care of itself. The
one thing that does not: if you add a dependency with `uv add`, commit both
`pyproject.toml` and `uv.lock` in the same commit. Committing one without the
other breaks `uv sync --locked` for everyone else, which is exactly what CI
catches.

## Notebook outputs

The worksheets save their outputs on purpose, so they can be read before being run. Text
and small figures are worth the diff noise.

One output is not: `replay.watch()` embeds every frame of an episode as base64 PNG. That
single cell once made a notebook 7.5 MB, and because each re-run writes a fresh blob, five
copies ended up in history at roughly 37 MB. It also does not render on GitHub, so saving it
buys nothing.

Clear that cell before committing. `uv run pytest` enforces a size budget per notebook and
per output, so you will find out before CI does.

## Reporting a result

Quote the run directory, not the terminal:

> `outputs/2026-09-28T11-02-14Z-gamma-090/` and four more seeds. Mean running
> reward after 20k episodes: +2.1, against +0.4 for baseline over the
> same seeds. `git_dirty` false on all five.

If `git_dirty` is `true` in a run you want to report, the run is not
reproducible from its commit. Commit the change and run it again.

## Things that will get a PR sent back

- A new script that is a copy of an old script with numbers changed.
- A result from one seed.
- A notebook containing its own copy of `Policy` rather than importing it.
- A hardcoded path, `/Users/you/...`, or a relative path assuming a working
  directory. Use `pong.PROJECT_ROOT`.
- `pyproject.toml` changed without `uv.lock`.
- A reported number with no run directory behind it.
