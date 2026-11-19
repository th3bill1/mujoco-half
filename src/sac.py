"""A from-scratch Soft Actor-Critic implementation using PyTorch."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int, action_size: int, seed: int) -> None:
        self.capacity = capacity
        self.observation_size = observation_size
        self.action_size = action_size
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices]),
            torch.as_tensor(self.actions[indices]),
            torch.as_tensor(self.rewards[indices]),
            torch.as_tensor(self.next_observations[indices]),
            torch.as_tensor(self.dones[indices]),
        )


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class Actor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.network = MLP(observation_size, hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
        self.action_size = action_size

    def forward(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.network(observations))
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -20.0, 2.0)
        if deterministic:
            unbounded_action = mean
        else:
            unbounded_action = mean + torch.exp(log_std) * torch.randn_like(mean)
        actions = torch.tanh(unbounded_action)

        if not with_log_prob:
            return actions, torch.zeros(observations.shape[0], 1, device=observations.device)

        normal = torch.distributions.Normal(mean, torch.exp(log_std))
        raw_actions = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
        log_prob = normal.log_prob(raw_actions) - torch.log(
            1.0 - actions.square() + 1e-6
        )
        return actions, log_prob.sum(dim=-1, keepdim=True)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.network(observations))
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -20.0, 2.0)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        raw_actions = normal.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = normal.log_prob(raw_actions) - torch.log(
            1.0 - actions.square() + 1e-6
        )
        return actions, log_prob.sum(dim=-1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.q1 = MLP(observation_size + action_size, 1, hidden_size)
        self.q2 = MLP(observation_size + action_size, 1, hidden_size)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([observations, actions], dim=-1)
        return self.q1(inputs), self.q2(inputs)


class SACAgent:
    """Soft Actor-Critic with twin critics and optional learned temperature."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str | float = "auto",
        seed: int = 42,
        hidden_size: int = 256,
        device: str | None = None,
    ) -> None:
        self.seed = seed
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.actor = Actor(observation_size, action_size, hidden_size).to(self.device)
        self.critic = TwinCritic(observation_size, action_size, hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.target_entropy = -float(action_size)

        if isinstance(ent_coef, str) and ent_coef == "auto":
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.fixed_alpha = float(ent_coef)

        self.replay = ReplayBuffer(buffer_size, observation_size, action_size, seed)
        self.num_timesteps = 0

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, device=self.device)

    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            actions, _ = self.actor(
                observation_tensor,
                deterministic=deterministic,
                with_log_prob=False,
            )
        return actions.squeeze(0).cpu().numpy()

    def add_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(
            np.asarray(observation, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            reward,
            np.asarray(next_observation, dtype=np.float32),
            done,
        )
        self.num_timesteps += 1

    def should_update(self) -> bool:
        return (
            self.num_timesteps >= self.learning_starts
            and self.num_timesteps % self.train_freq == 0
        )

    def update(self) -> dict[str, float]:
        if self.replay.size < self.batch_size:
            return {}
        observations, actions, rewards, next_observations, dones = self.replay.sample(
            self.batch_size
        )
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_observations = next_observations.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_observations)
            target_q1, target_q2 = self.critic_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(observations, actions)
        critic_loss = (current_q1 - target).square().mean() + (
            current_q2 - target
        ).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_prob = self.actor.sample(observations)
        q1, q2 = self.critic(observations, sampled_actions)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for target_parameter, parameter in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_parameter.data.mul_(1.0 - self.tau)
            target_parameter.data.add_(self.tau * parameter.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": "mujoco-half-sac-v1",
            "seed": self.seed,
            "observation_size": self.replay.observation_size,
            "action_size": self.action_size,
            "hidden_size": self.actor.network.network[0].out_features,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_entropy": self.target_entropy,
            "num_timesteps": self.num_timesteps,
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "fixed_alpha": None if self.log_alpha is not None else self.fixed_alpha,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "SACAgent":
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        if payload.get("format") != "mujoco-half-sac-v1":
            raise ValueError("Unsupported SAC model format")
        agent = cls(
            observation_size=payload["observation_size"],
            action_size=payload["action_size"],
            gamma=payload["gamma"],
            tau=payload["tau"],
            batch_size=payload["batch_size"],
            learning_starts=payload["learning_starts"],
            train_freq=payload["train_freq"],
            gradient_steps=payload["gradient_steps"],
            ent_coef="auto" if payload["log_alpha"] is not None else payload["fixed_alpha"],
            seed=payload["seed"],
            hidden_size=payload["hidden_size"],
            device=device,
        )
        agent.actor.load_state_dict(payload["actor_state"])
        agent.critic.load_state_dict(payload["critic_state"])
        agent.num_timesteps = int(payload.get("num_timesteps", 0))
        if payload["log_alpha"] is not None:
            agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        return agent
