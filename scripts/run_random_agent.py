"""Run a truly random HalfCheetah policy and optionally record MP4 video."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.envs import EnvironmentConfig  # noqa: E402
from src.record import run_random  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/environment.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/results")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-length", type=int)
    args = parser.parse_args()
    config = EnvironmentConfig.from_yaml(args.config)
    seed = config.seed if args.seed is None else args.seed
    rows = run_random(config, seed=seed, episodes=args.episodes, max_steps=args.max_steps,
                      output_dir=args.output_dir, record_video=args.record_video, fps=args.fps,
                      clip_length=args.clip_length)
    for row in rows:
        print(f"Random policy episode {row['episode']}: return={row['return']:.3f}, "
              f"steps={row['episode_length']}, displacement={row['displacement']:.3f}")


if __name__ == "__main__":
    main()
