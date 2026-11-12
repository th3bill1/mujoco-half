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
    model_path = train_sac(
        EnvironmentConfig.from_yaml(args.env_config),
        SACConfig.from_yaml(args.config),
        run_dir=args.run_dir,
        resume=args.resume,
    )
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
