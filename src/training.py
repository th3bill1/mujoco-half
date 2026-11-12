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
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("total_timesteps", "learning_starts", "buffer_size", "batch_size",
                     "train_freq", "gradient_steps", "checkpoint_freq"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

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


def _require_sb3() -> tuple[Any, Any]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the train extra: python -m pip install -e \".[train]\""
        ) from exc
    return SAC, BaseCallback


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
    SAC, BaseCallback = _require_sb3()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(environment, seed=training.seed)
    metrics_rows: list[dict[str, Any]] = []

    class MetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episode = 0
            self.rewards: list[float] = []
            self.forwards: list[float] = []
            self.costs: list[float] = []
            self.velocities: list[float] = []
            self.squared_actions: list[float] = []
            self.initial_x = 0.0

        def _on_training_start(self) -> None:
            self.initial_x = float(self.training_env.envs[0].unwrapped.data.qpos[0])

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]
            reward = float(self.locals["rewards"][0])
            action = np.asarray(self.locals["actions"][0])
            forward, cost = reward_components(info, reward)
            self.rewards.append(reward)
            self.forwards.append(forward)
            self.costs.append(cost)
            self.velocities.append(float(info.get("x_velocity", 0.0)))
            self.squared_actions.append(float(np.mean(action ** 2)))
            if self.locals["dones"][0]:
                final_x = float(info.get("x_position", self.initial_x))
                truncated = bool(info.get("TimeLimit.truncated", False))
                metrics = aggregate_episode(
                    self.episode + 1, self.rewards, self.forwards, self.costs,
                    self.velocities, self.squared_actions, self.initial_x, final_x,
                    not truncated, truncated,
                ).as_dict()
                metrics["timesteps"] = self.num_timesteps
                metrics_rows.append(metrics)
                if self.num_timesteps % training.checkpoint_freq == 0:
                    self.model.save(str(run_dir / "checkpoints" / f"model_{self.num_timesteps}"))
                self.episode += 1
                self.rewards.clear()
                self.forwards.clear()
                self.costs.clear()
                self.velocities.clear()
                self.squared_actions.clear()
                self.initial_x = float(self.training_env.envs[0].unwrapped.data.qpos[0])
            return True

    callback = MetricsCallback()
    model_kwargs = {
        "learning_rate": training.learning_rate,
        "buffer_size": training.buffer_size,
        "learning_starts": training.learning_starts,
        "batch_size": training.batch_size,
        "train_freq": training.train_freq,
        "gradient_steps": training.gradient_steps,
        "gamma": training.gamma,
        "tau": training.tau,
        "ent_coef": training.ent_coef,
        "seed": training.seed,
        "verbose": 1,
        "tensorboard_log": str(run_dir / "tensorboard"),
    }
    model = SAC.load(str(resume), env=env) if resume else SAC("MlpPolicy", env, **model_kwargs)
    try:
        model.learn(total_timesteps=training.total_timesteps, callback=callback, reset_num_timesteps=resume is None)
        model.save(str(run_dir / "model_final"))
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
    SAC, _ = _require_sb3()
    model = SAC.load(str(model_path), device="cpu")
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
                action, _ = model.predict(obs, deterministic=True)
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
