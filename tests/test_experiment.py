"""
Checks on the config/provenance layer.

The point of recording provenance is that a result stays interpretable later,
so these assert the things that would make a run *un*interpretable: a setting
that silently didn't apply, a missing commit, a config typo that was ignored.
"""

import csv
import json

import pytest

import experiment


DEFAULTS = {"gamma": 0.99, "hidden": 200, "seed": 0}


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

def test_project_root_is_the_repo():
    assert (experiment.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (experiment.PROJECT_ROOT / "agent.py").is_file()


def test_project_root_survives_chdir(tmp_path, monkeypatch):
    before = experiment.find_project_root()
    monkeypatch.chdir(tmp_path)
    assert experiment.find_project_root() == before


# ------------------------------------------------------------
# Config layering
# ------------------------------------------------------------

def test_cli_beats_config_beats_default():
    settings, source = experiment.resolve_config(
        DEFAULTS, config={"gamma": 0.9, "hidden": 400}, overrides={"gamma": 0.5}
    )
    assert settings == {"gamma": 0.5, "hidden": 400, "seed": 0}
    assert source == {"gamma": "cli", "hidden": "config", "seed": "default"}


def test_unset_cli_flags_do_not_override():
    """argparse gives None for flags nobody passed; those must not win."""
    settings, source = experiment.resolve_config(
        DEFAULTS, config={"gamma": 0.9}, overrides={"gamma": None, "hidden": None}
    )
    assert settings["gamma"] == 0.9
    assert source["gamma"] == "config"


def test_unknown_setting_is_an_error():
    with pytest.raises(KeyError, match="gama"):
        experiment.resolve_config(DEFAULTS, config={"gama": 0.9})


def test_shipped_configs_only_name_known_settings():
    for path in sorted((experiment.PROJECT_ROOT / "configs").glob("*.toml")):
        config = experiment.load_config(path)
        experiment.resolve_config(DEFAULTS | {"batch_size": 10,
                                              "learning_rate": 1e-4,
                                              "decay_rate": 0.99,
                                              "save_every": 100}, config)


def test_config_resolves_by_bare_name():
    assert experiment.load_config("baseline")["gamma"] == 0.99


def test_missing_config_lists_what_exists():
    with pytest.raises(FileNotFoundError, match="baseline"):
        experiment.load_config("does-not-exist")


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

def test_metadata_records_what_a_result_needs():
    meta = experiment.collect_metadata(device="cpu", extra={"seed": 7})

    assert meta["git_commit"] and len(meta["git_commit"]) == 40
    assert meta["git_dirty"] in (True, False)
    assert meta["torch"] and meta["python"].startswith("3.12")
    assert meta["device"] == "cpu"
    assert meta["seed"] == 7
    assert meta["started_at"].endswith("+00:00")


# ------------------------------------------------------------
# Run directory
# ------------------------------------------------------------

def test_start_run_writes_config_and_metadata_before_training(tmp_path):
    settings, source = experiment.resolve_config(DEFAULTS, config={"gamma": 0.9})

    run = experiment.start_run(
        "gamma-090", settings, source, device="cpu",
        metrics_fields=["episode", "reward", "elapsed_seconds"],
        root=tmp_path, timestamp="2026-01-01T00-00-00Z",
    )

    assert run.dir == tmp_path / "2026-01-01T00-00-00Z-gamma-090"

    config = json.loads((run.dir / "config.json").read_text())
    assert config["settings"]["gamma"] == 0.9
    assert config["source"]["gamma"] == "config"
    assert config["name"] == "gamma-090"

    meta = json.loads((run.dir / "metadata.json").read_text())
    assert meta["git_commit"]
    assert meta["argv"]

    run.close()


def test_metrics_are_written_and_flushed(tmp_path):
    run = experiment.start_run(
        "m", {"seed": 0}, {}, device="cpu",
        metrics_fields=["episode", "reward", "elapsed_seconds"],
        root=tmp_path, timestamp="t",
    )

    run.log(episode=1, reward=-21.0)
    run.log(episode=2, reward=-19.0)

    # Read before close(): a killed run must still have its metrics on disk.
    rows = list(csv.DictReader(open(run.metrics_path)))
    assert [r["episode"] for r in rows] == ["1", "2"]
    assert rows[0]["reward"] == "-21.0"
    assert float(rows[0]["elapsed_seconds"]) >= 0

    run.close()


def test_runs_do_not_collide(tmp_path):
    a = experiment.start_run("a", {}, {}, device="cpu", root=tmp_path, timestamp="t1")
    b = experiment.start_run("b", {}, {}, device="cpu", root=tmp_path, timestamp="t2")
    assert a.dir != b.dir
    assert a.checkpoint.parent == a.dir and b.checkpoint.parent == b.dir
    a.close()
    b.close()


def test_discard_removes_a_run_that_produced_nothing(tmp_path):
    run = experiment.start_run("dead", {}, {}, device="cpu",
                               root=tmp_path, timestamp="t")
    assert run.dir.exists()
    assert run.discard() is True
    assert not run.dir.exists()


def test_discard_keeps_a_run_with_results(tmp_path):
    run = experiment.start_run("live", {}, {}, device="cpu",
                               metrics_fields=["episode", "elapsed_seconds"],
                               root=tmp_path, timestamp="t")
    run.log(episode=1)
    assert run.discard() is False
    assert run.dir.exists()
