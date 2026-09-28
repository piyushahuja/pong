## What changed

<!-- One or two sentences. If this is a hyperparameter, say which config. -->

## Is it a config or a method change?

- [ ] A config only (`configs/*.toml`), no change to `src/`
- [ ] A change in method (`src/pong/...`)
- [ ] Both

## Runs

<!-- One line per run directory. Each already records its commit, seed,
     settings and dirty flag, so this is all a reviewer needs. -->

| Run directory | Seed | Episodes | Final running reward |
|---|---|---|---|
| `outputs/…` | | | |

Baseline compared against:

| Run directory | Seed | Episodes | Final running reward |
|---|---|---|---|
| `outputs/…` | | | |

## Checks

- [ ] `uv run pytest` passes locally
- [ ] Run over at least 3 seeds, not 1
- [ ] `git_dirty` is `false` in every run I am quoting
- [ ] `uv.lock` committed if `pyproject.toml` changed
- [ ] No hardcoded or working-directory-relative paths

## What I expected, and what happened

<!-- Including when they disagree. A negative result with clean provenance is
     worth more than a positive one without it. -->
