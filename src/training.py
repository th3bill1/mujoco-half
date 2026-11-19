"""SAC training, checkpointing, and deterministic evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .envs import EnvironmentConfig, make_env
from .metrics import aggregate_episode, reward_components
from .sac import SACAgent
from .utils import write_json


@dataclass(frozen=True)
class SACConfig:
    total_timesteps: int = 100_000
    learning_starts: int = 10_000
    buffer_size: int = 100_000
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"
    checkpoint_freq: int = 10_000
    seeds: tuple[int, ...] = (42,)
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_timesteps", "learning_starts", "buffer_size", "batch_size",
                     "train_freq", "gradient_steps", "checkpoint_freq"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(self.seeds, int):
            seed_values = (self.seeds,)
        elif isinstance(self.seeds, (list, tuple)):
            seed_values = tuple(self.seeds)
        else:
            raise ValueError("seeds must be a list of integers")
        if self.seed is not None:
            if seed_values != (42,) or len(seed_values) != 1:
                raise ValueError("Use either seed or seeds, not both")
            seed_values = (int(self.seed),)
        if not seed_values:
            raise ValueError("seeds must contain at least one seed")
        if any(not isinstance(seed, int) for seed in seed_values):
            raise ValueError("seeds must contain only integers")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seed_values)
        object.__setattr__(self, "seed", seed_values[0])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SACConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown SAC configuration keys: {sorted(unknown)}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_sac(
    environment: EnvironmentConfig,
    training: SACConfig,
    *,
    run_dir: Path,
    resume: Path | None = None,
) -> Path:
    """Train SAC and save checkpoints plus per-episode structured metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(environment, seed=training.seed)
    metrics_rows: list[dict[str, Any]] = []
    action_size = int(env.action_space.shape[0])
    observation_size = int(env.observation_space.shape[0])
    agent = (
        SACAgent.load(resume, device="cpu")
        if resume
        else SACAgent(
            observation_size,
            action_size,
            learning_rate=training.learning_rate,
            gamma=training.gamma,
            tau=training.tau,
            buffer_size=training.buffer_size,
            batch_size=training.batch_size,
            learning_starts=training.learning_starts,
            train_freq=training.train_freq,
            gradient_steps=training.gradient_steps,
            ent_coef=training.ent_coef,
            seed=training.seed,
            device="cpu",
        )
    )

    observation, _ = env.reset(seed=training.seed)
    initial_x = float(env.unwrapped.data.qpos[0])
    episode = 0
    rewards: list[float] = []
    forwards: list[float] = []
    costs: list[float] = []
    velocities: list[float] = []
    squared_actions: list[float] = []
    start_timestep = agent.num_timesteps

    try:
        while agent.num_timesteps - start_timestep < training.total_timesteps:
            if agent.num_timesteps < training.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.predict(observation, deterministic=False)
            next_observation, reward, terminated, truncated, info = env.step(action)
            forward, cost = reward_components(info, reward)
            done = bool(terminated or truncated)
            agent.add_transition(
                observation, action, float(reward), next_observation, done
            )
            rewards.append(float(reward))
            forwards.append(forward)
            costs.append(cost)
            velocities.append(float(info.get("x_velocity", 0.0)))
            squared_actions.append(float(np.mean(np.asarray(action) ** 2)))
            observation = next_observation

            if agent.should_update():
                for _ in range(agent.gradient_steps):
                    agent.update()

            if agent.num_timesteps % training.checkpoint_freq == 0:
                checkpoint = run_dir / "checkpoints" / f"model_{agent.num_timesteps}"
                agent.save(str(checkpoint.with_suffix(".zip")))

            if done:
                episode += 1
                final_x = float(info.get("x_position", env.unwrapped.data.qpos[0]))
                metrics = aggregate_episode(
                    episode, rewards, forwards, costs, velocities, squared_actions,
                    initial_x, final_x, terminated, truncated,
                ).as_dict()
                metrics["timesteps"] = agent.num_timesteps
                metrics_rows.append(metrics)
                observation, _ = env.reset(
                    seed=training.seed + episode
                )
                env.action_space.seed(training.seed + episode)
                initial_x = float(env.unwrapped.data.qpos[0])
                rewards.clear()
                forwards.clear()
                costs.clear()
                velocities.clear()
                squared_actions.clear()
        agent.save(str(run_dir / "model_final.zip"))
    finally:
        env.close()
    _write_rows(run_dir / "training_metrics.csv", metrics_rows)
    write_json(run_dir / "run.json", {
        "environment": environment.as_dict(),
        "training": training.as_dict(),
        "resumed_from": str(resume) if resume else None,
        "episodes_logged": len(metrics_rows),
    })
    return run_dir / "model_final.zip"


def evaluate_sac(
    environment: EnvironmentConfig,
    model_path: str | Path,
    *,
    episodes: int,
    seed: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Evaluate a saved SAC policy without exploration noise."""
    model = SACAgent.load(model_path, device="cpu")
    env = make_env(environment, seed=seed)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed + episode - 1)
            env.action_space.seed(seed + episode - 1)
            initial_x = float(env.unwrapped.data.qpos[0])
            rewards: list[float] = []
            forwards: list[float] = []
            costs: list[float] = []
            velocities: list[float] = []
            squared_actions: list[float] = []
            terminated = truncated = False
            while not (terminated or truncated):
                action = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                forward, cost = reward_components(info, reward)
                rewards.append(float(reward))
                forwards.append(forward)
                costs.append(cost)
                velocities.append(float(info.get("x_velocity", 0.0)))
                squared_actions.append(float(np.mean(np.asarray(action) ** 2)))
            rows.append(aggregate_episode(
                episode, rewards, forwards, costs, velocities, squared_actions,
                initial_x, float(info["x_position"]), terminated, truncated,
            ).as_dict())
    finally:
        env.close()
    _write_rows(output_path, rows)
    write_json(output_path.with_suffix(".json"), {
        "model": str(model_path),
        "environment": environment.as_dict(),
        "seed": seed,
        "episodes": rows,
        "deterministic": True,
    })
    return rows
