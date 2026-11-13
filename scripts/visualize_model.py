"""Record a saved SAC policy as an MP4 video."""

import argparse
from pathlib import Path

from src.envs import EnvironmentConfig
from src.record import record_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--env-config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-length", type=int)
    args = parser.parse_args()

    video_paths = record_sac(
        EnvironmentConfig.from_yaml(args.env_config),
        args.model,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        fps=args.fps,
        clip_length=args.clip_length,
    )
    for video_path in video_paths:
        print(f"Saved video: {video_path}")


if __name__ == "__main__":
    main()
