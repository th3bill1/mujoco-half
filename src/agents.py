"""Policies used by checkpoint 1."""

import gymnasium as gym


def sample_random_action(env: gym.Env):
    """Return one fresh action-space sample; there is no policy state."""
    return env.action_space.sample()
