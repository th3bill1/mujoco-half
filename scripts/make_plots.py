"""Plot mean episode return with across-run uncertainty."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.training import SACConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_baseline.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/sac-baseline"))
    parser.add_argument("--label", default="SAC")
    parser.add_argument("--output", type=Path, default=Path("artifacts/plots/learning-curves.png"))
    args = parser.parse_args()
    if args.inputs:
        inputs = args.inputs
    else:
        training = SACConfig.from_yaml(args.config)
        inputs = [
            args.run_dir / f"seed-{seed}" / "training_metrics.csv"
            for seed in training.seeds
        ]
    grouped_by_run: list[dict[int, float]] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Training metrics not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[int, float] = {}
        for row in rows:
            grouped[int(row["episode"])] = float(row["return"])
        grouped_by_run.append(grouped)
    episodes = np.array(sorted(set().union(*(run for run in grouped_by_run))))
    values = np.full((len(episodes), len(grouped_by_run)), np.nan)
    for column, run in enumerate(grouped_by_run):
        for row, episode in enumerate(episodes):
            if episode in run:
                values[row, column] = run[episode]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)
    plt.plot(episodes, mean, label=args.label)
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    plt.close()
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
