"""Train the baseline SAC policy."""

import argparse
from pathlib import Path

from src.envs import EnvironmentConfig
from src.training import SACConfig, train_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/sac_baseline.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/sac-baseline"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    environment = EnvironmentConfig.from_yaml(args.env_config)
    training = SACConfig.from_yaml(args.config)
    for seed in training.seeds:
        seed_training = SACConfig(
            total_timesteps=training.total_timesteps,
            learning_starts=training.learning_starts,
            buffer_size=training.buffer_size,
            batch_size=training.batch_size,
            train_freq=training.train_freq,
            gradient_steps=training.gradient_steps,
            learning_rate=training.learning_rate,
            gamma=training.gamma,
            tau=training.tau,
            ent_coef=training.ent_coef,
            checkpoint_freq=training.checkpoint_freq,
            seed=seed,
        )
        seed_dir = args.run_dir / f"seed-{seed}"
        model_path = train_sac(
            environment,
            seed_training,
            run_dir=seed_dir,
            resume=args.resume,
        )
        print(f"Saved seed {seed} model: {model_path}")


if __name__ == "__main__":
    main()
