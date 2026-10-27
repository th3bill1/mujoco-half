"""Inspect HalfCheetah spaces, reward diagnostics and runtime versions."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import gymnasium
import mujoco
import numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.envs import EnvironmentConfig, make_env  # noqa: E402
from src.utils import runtime_info, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/environment.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = EnvironmentConfig.from_yaml(args.config)
    seed = config.seed if args.seed is None else args.seed
    env = make_env(config, seed=seed)
    try:
        obs, info = env.reset(seed=seed)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        report = {
            "environment": config.env_id, "config": config.as_dict(),
            "observation": {"shape": env.observation_space.shape, "dtype": str(env.observation_space.dtype)},
            "action": {"shape": env.action_space.shape, "dtype": str(env.action_space.dtype),
                       "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist()},
            "physical_dt": env.unwrapped.dt, "horizon": config.episode_horizon,
            "termination": {"terminated": terminated, "truncated": truncated},
            "info_keys": sorted(step_info), "reward": reward,
            "runtime": {**runtime_info(), "gymnasium": gymnasium.__version__,
                        "mujoco": mujoco.__version__, "numpy": numpy.__version__,
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "not detected")},
        }
        print(json.dumps(report, indent=2, default=str))
        print("HalfCheetah controls six joints: back/front thigh, shin and foot.")
        print("The default 17-value observation contains 8 qpos and 9 qvel values; global x is excluded.")
        print("Task termination is normally false; the 1000-step TimeLimit causes truncation.")
        write_json(ROOT / "artifacts/results/environment-inspection.json", report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
