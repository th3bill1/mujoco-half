"""Evaluate a saved SAC model deterministically."""

import argparse
from pathlib import Path

from src.envs import EnvironmentConfig
from src.training import SACConfig, evaluate_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", type=Path)
    parser.add_argument("--env-config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/sac_baseline.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/sac-baseline"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/evaluation"))
    args = parser.parse_args()
    environment = EnvironmentConfig.from_yaml(args.env_config)
    training = SACConfig.from_yaml(args.config)

    if args.models:
        evaluations = [(training.seed if training.seed is not None else args.seed, model)
                       for model in args.models]
    else:
        evaluations = [
            (seed, args.run_dir / f"seed-{seed}" / "model_final.zip")
            for seed in training.seeds
        ]

    all_rows: dict[str, list[dict[str, object]]] = {}
    for seed, model in evaluations:
        if not model.exists():
            raise FileNotFoundError(f"Saved SAC model not found: {model}")
        output = args.output
        if len(evaluations) > 1:
            output = args.output / f"seed-{seed}-evaluation.csv"
        else:
            if args.output.suffix:
                output = args.output
            else:
                output = args.output.with_suffix(".csv")
        rows = evaluate_sac(
            environment,
            model,
            episodes=args.episodes,
            seed=args.seed,
            output_path=output,
        )
        all_rows[str(seed)] = rows
        print(f"Evaluated seed {seed}: {len(rows)} episodes; results: {output}")


if __name__ == "__main__":
    main()
