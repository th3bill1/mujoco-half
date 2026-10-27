"""Configuration and construction of the Gymnasium HalfCheetah environment."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    env_id: str = "HalfCheetah-v5"
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.1
    reset_noise_scale: float = 0.1
    exclude_current_positions_from_observation: bool = True
    frame_skip: int = 5
    episode_horizon: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.env_id != "HalfCheetah-v5":
            raise ValueError("This checkpoint supports only HalfCheetah-v5")
        if self.frame_skip <= 0 or self.episode_horizon <= 0:
            raise ValueError("frame_skip and episode_horizon must be positive")
        if self.forward_reward_weight < 0 or self.ctrl_cost_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if self.reset_noise_scale < 0:
            raise ValueError("reset_noise_scale must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvironmentConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown environment configuration keys: {sorted(unknown)}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def make_env(
    config: EnvironmentConfig,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Create exactly one explicit time-limit wrapper and seed both spaces."""
    kwargs: dict[str, Any] = {
        "forward_reward_weight": config.forward_reward_weight,
        "ctrl_cost_weight": config.ctrl_cost_weight,
        "reset_noise_scale": config.reset_noise_scale,
        "exclude_current_positions_from_observation": config.exclude_current_positions_from_observation,
        "frame_skip": config.frame_skip,
        # Disable the registry's implicit TimeLimit before adding our configured one.
        "max_episode_steps": None,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(config.env_id, **kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.episode_horizon)
    actual_seed = config.seed if seed is None else seed
    env.reset(seed=actual_seed)
    env.action_space.seed(actual_seed)
    env.observation_space.seed(actual_seed)
    return env
