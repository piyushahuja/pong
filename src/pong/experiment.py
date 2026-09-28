"""
Configs, run directories and provenance.

An experiment is the same code plus a different configuration. Nothing here
knows about Pong; it answers three questions that any run has to answer:

    what settings was this?      config.json in the run directory
    what produced it?            metadata.json -- commit, seed, device, versions
    what happened?               metrics.csv, one row per episode

Configs are TOML rather than YAML so this needs no dependency outside the
standard library: tomllib has been built in since Python 3.11, and the project
pins 3.12.
"""

import csv
import json
import platform
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from pong.utils import CONFIG_DIR, OUTPUT_DIR, PROJECT_ROOT


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

def load_config(path):
    """
    Read a TOML config. A bare name resolves inside configs/, so both

        --config configs/gamma-090.toml
        --config gamma-090

    work.
    """
    path = Path(path)

    if not path.exists() and path.parent == Path("."):
        for candidate in (CONFIG_DIR / path.name,
                          CONFIG_DIR / f"{path.name}.toml"):
            if candidate.is_file():
                path = candidate
                break

    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.toml"))
        raise FileNotFoundError(
            f"No config at {path}. Available in configs/: "
            + (", ".join(available) or "none")
        )

    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_config(defaults, config=None, overrides=None):
    """
    Merge three layers, later winning: defaults, then a config file, then
    command-line overrides. Only keys already in defaults are accepted, so a
    typo in a config file is an error rather than a silently ignored setting.

    Returns (settings, provenance) where provenance records where each value
    came from -- worth keeping, because "which of these did I actually set?"
    is the first question you ask when reading an old run back.
    """
    settings = dict(defaults)
    source = {key: "default" for key in defaults}

    for layer_name, layer in (("config", config or {}),
                              ("cli", overrides or {})):
        for key, value in layer.items():
            if value is None:                 # argparse default for "unset"
                continue
            if key not in defaults:
                raise KeyError(
                    f"Unknown setting {key!r} from {layer_name}. "
                    f"Known settings: {', '.join(sorted(defaults))}"
                )
            settings[key] = value
            source[key] = layer_name

    return settings, source


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

def _git(*args):
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_metadata(device=None, extra=None):
    """
    What a result needs attached to be believable six months on.

    git_dirty matters as much as the commit: a run made with uncommitted
    edits is not reproducible from that commit alone, and recording the fact
    is the difference between a usable result and a misleading one.
    """
    import torch          # imported here only because nothing else in this module needs it

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device) if device is not None else None,
    }

    if device is not None and getattr(device, "type", None) == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(0)

    if extra:
        metadata.update(extra)

    return metadata


# ------------------------------------------------------------
# Run directory
# ------------------------------------------------------------

class Run:
    """
    One experiment's output directory:

        outputs/2026-09-28T10-30-00Z-baseline/
        ├── config.json      the settings, and where each came from
        ├── metadata.json    commit, dirtiness, seed, device, versions
        ├── metrics.csv      one row per episode
        └── policy.pt        the checkpoint, rewritten as training proceeds

    Nothing is written outside this directory, so two runs never collide and
    deleting one deletes all of it.
    """

    def __init__(self, directory, metrics_fields):
        self.dir = Path(directory)
        self.checkpoint = self.dir / "policy.pt"
        self.metrics_path = self.dir / "metrics.csv"
        self._metrics_fields = list(metrics_fields)
        self._metrics_file = None
        self._writer = None
        self._started = time.time()

    def log(self, **row):
        """
        Append one row to metrics.csv, flushed so a killed run keeps it.

        The file is opened on the first call rather than up front, so a run
        that produced no episodes has no metrics.csv -- which is how discard()
        tells an empty run from a real one.
        """
        row.setdefault("elapsed_seconds", round(time.time() - self._started, 1))

        if self._writer is None:
            new = not self.metrics_path.exists()
            self._metrics_file = open(self.metrics_path, "a", newline="")
            self._writer = csv.DictWriter(
                self._metrics_file, fieldnames=self._metrics_fields
            )
            if new:
                self._writer.writeheader()

        self._writer.writerow({k: row.get(k) for k in self._metrics_fields})
        self._metrics_file.flush()

    def close(self):
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._writer = None

    def discard(self):
        """
        Remove the directory, but only if the run produced nothing.

        A run that failed before its first episode has config and metadata and
        nothing else; keeping it would fill outputs/ with directories that
        record no result. Anything with metrics or a checkpoint is kept.
        """
        self.close()

        if self.checkpoint.exists() or self.metrics_path.exists():
            return False

        for child in self.dir.iterdir():
            child.unlink()
        self.dir.rmdir()
        return True


def start_run(name, settings, source=None, device=None, metrics_fields=(),
              root=None, timestamp=None):
    """
    Create the run directory and write config.json and metadata.json before
    training starts -- so an interrupted run still says what it was trying.
    """
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    directory = Path(root or OUTPUT_DIR) / f"{stamp}-{name}"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "config.json").write_text(
        json.dumps({"name": name, "settings": settings, "source": source or {}},
                   indent=2, sort_keys=True) + "\n"
    )

    (directory / "metadata.json").write_text(
        json.dumps(
            collect_metadata(device=device,
                             extra={"seed": settings.get("seed"),
                                    "argv": sys.argv}),
            indent=2, sort_keys=True,
        ) + "\n"
    )

    return Run(directory, metrics_fields)
