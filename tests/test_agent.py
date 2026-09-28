"""
Fast checks on the contracts a checkpoint depends on.

Nothing here trains. The point is to catch the breakages that would quietly
invalidate every saved policy: a changed network shape, preprocessing that
stops producing a 6400-vector, or environment settings drifting away from
what the weights were trained against.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import pong as agent
import pong as pong_pkg
from pong import checkpoints as agent_checkpoints


def test_shape_constants():
    assert agent.D == 80 * 80 == 6400
    assert agent.H == 200


def test_preprocess_shape_and_values():
    # Rows are cropped to [35:195] then decimated 2x, so a pixel survives only
    # at an even offset from row 35 and an even column.
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[35 + 20, 50] = 200                 # kept: even row offset, even column
    frame[35 + 21, 52] = 200                 # dropped by the row decimation
    frame[35 + 22, 53] = 200                 # dropped by the column decimation
    frame[35 + 24, 54] = 144                 # kept by the grid, zeroed as background
    frame[35 + 26, 56] = 109                 # the other background colour

    x = agent.preprocess(frame)

    assert x.shape == (agent.D,)
    assert x.dtype == torch.float32
    assert set(x.unique().tolist()) <= {0.0, 1.0}, "output must be binary"
    assert x.sum() == 1.0, "only the one on-grid, non-background pixel survives"

    # And it lands where the 80x80 -> 6400 flattening says it should.
    assert x[(20 // 2) * 80 + (50 // 2)] == 1.0


def test_preprocess_crops_scoreboard():
    frame = np.full((210, 160, 3), 200, dtype=np.uint8)
    frame[:35] = 0                           # rows above the crop
    frame[195:] = 0                          # rows below it
    assert agent.preprocess(frame).sum() == agent.D, "crop window moved"


def test_policy_forward_is_a_probability():
    policy = agent.Policy()
    p = policy(torch.zeros(agent.D))
    assert p.shape == ()
    assert 0.0 <= p.item() <= 1.0


def test_policy_has_no_biases_and_expected_size():
    policy = agent.Policy()
    assert all("bias" not in name for name, _ in policy.named_parameters())
    assert sum(p.numel() for p in policy.parameters()) == 1_280_200


def test_get_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("PONG_DEVICE", raising=False)
    assert agent.get_device().type == "cpu"


def test_get_device_respects_explicit_spec(monkeypatch):
    monkeypatch.setenv("PONG_DEVICE", "cpu")
    assert agent.get_device().type == "cpu"
    assert agent.get_device("cpu").type == "cpu"


def test_get_device_auto_is_available():
    # Whatever it picks must be a device torch can actually place a tensor on.
    device = agent.get_device("auto")
    torch.zeros(1).to(device)


def test_env_observation_contract():
    env = agent.make_env()
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (210, 160, 3)
        assert agent.preprocess(observation).shape == (agent.D,)

        # Actions 2 and 3 are the up/down the policy maps onto.
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[2] == "RIGHT" and meanings[3] == "LEFT"
    finally:
        env.close()


def test_env_settings_match_training():
    env = agent.make_env()
    try:
        assert env.spec.kwargs["frameskip"] == agent.FRAMESKIP
        assert env.spec.kwargs["repeat_action_probability"] == agent.STICKY
    finally:
        env.close()


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    """A file written in pong.py's format must load back through load_policy."""
    trained = agent.Policy()
    path = tmp_path / "policy.pt"
    torch.save(
        {
            "episode": 1234,
            "model_state_dict": trained.state_dict(),
            "optimizer_state_dict": {},
            "running_reward": -3.5,
        },
        path,
    )

    monkeypatch.chdir(tmp_path)
    loaded = agent.load_policy(path)

    for (name, a), (_, b) in zip(trained.state_dict().items(),
                                 loaded.state_dict().items()):
        assert torch.equal(a, b), f"{name} changed across save/load"


def test_default_checkpoint_picks_the_newest(tmp_path, monkeypatch):
    """Paths come from the project root, so point the constants at a fake one."""
    monkeypatch.setattr(agent_checkpoints, "CHECKPOINT", tmp_path / "pong_policy.pt")
    monkeypatch.setattr(agent_checkpoints, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(agent_checkpoints, "OUTPUT_DIR", tmp_path / "outputs")

    with pytest.raises(FileNotFoundError):
        agent.default_checkpoint()

    # Only the archive exists, which is a fresh clone.
    archive = tmp_path / "checkpoints"
    archive.mkdir()
    archived = archive / "pong-ep100-reward-20.00.pt"
    archived.touch()
    os.utime(archived, (1_000_000, 1_000_000))
    assert agent.default_checkpoint() == archived

    # A run directory that is newer wins.
    run = tmp_path / "outputs" / "2026-01-01T00-00-00Z-baseline"
    run.mkdir(parents=True)
    policy = run / "policy.pt"
    policy.touch()
    os.utime(policy, (2_000_000, 2_000_000))
    assert agent.default_checkpoint() == policy

    # A root checkpoint older than the run does NOT win, which is the bug this
    # replaced: a fixed order preferred the root and resumed stale weights.
    legacy = tmp_path / "pong_policy.pt"
    legacy.touch()
    os.utime(legacy, (1_500_000, 1_500_000))
    assert agent.default_checkpoint() == policy

    # ...but it does when it really is the newest.
    os.utime(legacy, (3_000_000, 3_000_000))
    assert agent.default_checkpoint() == legacy


def test_paths_are_root_relative_not_cwd_relative(tmp_path, monkeypatch):
    """Running from a subdirectory must not change which files are found."""
    before = agent_checkpoints.CHECKPOINT_DIR
    monkeypatch.chdir(tmp_path)
    assert agent_checkpoints.CHECKPOINT_DIR == before
    assert agent_checkpoints.CHECKPOINT_DIR.is_absolute()


def test_package_surface_is_the_submodules():
    """`import pong` must expose the same objects the submodules define."""
    from pong import checkpoints, env, evaluate, model

    assert pong_pkg.Policy is model.Policy
    assert pong_pkg.preprocess is env.preprocess
    assert pong_pkg.make_env is env.make_env
    assert pong_pkg.load_policy is checkpoints.load_policy
    assert pong_pkg.rollout is evaluate.rollout
