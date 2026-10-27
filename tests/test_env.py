import numpy as np
import pytest

from src.envs import EnvironmentConfig, make_env


def test_spaces_and_horizon():
    env = make_env(EnvironmentConfig(episode_horizon=3), seed=1)
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == (17,)
        assert env.action_space.shape == (6,)
        for _ in range(3):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert truncated and not terminated
        assert np.isfinite(reward)
    finally:
        env.close()


def test_seeded_actions_reproduce():
    first = make_env(EnvironmentConfig(), seed=4)
    second = make_env(EnvironmentConfig(), seed=4)
    try:
        first.reset(seed=4); second.reset(seed=4)
        assert all(np.array_equal(first.action_space.sample(), second.action_space.sample()) for _ in range(3))
    finally:
        first.close(); second.close()
