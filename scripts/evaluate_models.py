"""Evaluate a saved SAC model deterministically."""

import argparse
from pathlib import Path

from src.envs import EnvironmentConfig
from src.training import evaluate_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--env-config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/evaluation.csv"))
    args = parser.parse_args()
    rows = evaluate_sac(
        EnvironmentConfig.from_yaml(args.env_config),
        args.model,
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Evaluated {len(rows)} episodes; results: {args.output}")


if __name__ == "__main__":
    main()
