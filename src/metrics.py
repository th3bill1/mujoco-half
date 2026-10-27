"""Episode metric aggregation and reward checks."""

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int
    return_: float
    total_forward_reward: float
    mean_forward_reward: float
    total_control_cost: float
    mean_control_cost: float
    initial_x: float
    final_x: float
    displacement: float
    mean_forward_velocity: float
    mean_squared_action: float
    episode_length: int
    terminated: bool
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["return"] = result.pop("return_")
        return result


def reward_components(info: dict[str, Any], reward: float) -> tuple[float, float]:
    """Extract forward reward and positive control cost from Gymnasium info."""
    if "reward_forward" not in info or "reward_ctrl" not in info:
        raise KeyError("HalfCheetah info must contain reward_forward and reward_ctrl")
    forward = float(info["reward_forward"])
    raw_ctrl = float(info["reward_ctrl"])
    # Gymnasium reports reward_ctrl as a negative term; tolerate positive penalty APIs.
    cost = -raw_ctrl if raw_ctrl <= 0 else raw_ctrl
    if not np.isclose(reward, forward - cost, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Reward decomposition mismatch: reward={reward}, forward={forward}, cost={cost}")
    return forward, cost


def aggregate_episode(
    episode: int,
    rewards: list[float],
    forward_rewards: list[float],
    control_costs: list[float],
    velocities: list[float],
    squared_actions: list[float],
    initial_x: float,
    final_x: float,
    terminated: bool,
    truncated: bool,
) -> EpisodeMetrics:
    n = len(rewards)
    if not n or not all(len(values) == n for values in (
        forward_rewards, control_costs, velocities, squared_actions
    )):
        raise ValueError("All episode step arrays must be non-empty and have equal length")
    return EpisodeMetrics(
        episode=episode,
        return_=float(np.sum(rewards)),
        total_forward_reward=float(np.sum(forward_rewards)),
        mean_forward_reward=float(np.mean(forward_rewards)),
        total_control_cost=float(np.sum(control_costs)),
        mean_control_cost=float(np.mean(control_costs)),
        initial_x=initial_x,
        final_x=final_x,
        displacement=final_x - initial_x,
        mean_forward_velocity=float(np.mean(velocities)),
        mean_squared_action=float(np.mean(squared_actions)),
        episode_length=n,
        terminated=terminated,
        truncated=truncated,
    )
