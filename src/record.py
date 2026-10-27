"""Random-agent rollout and optional Gymnasium video recording."""

from pathlib import Path
from typing import Any
import csv
import time

from .agents import sample_random_action
from .envs import EnvironmentConfig, make_env
from .metrics import aggregate_episode, reward_components
from .utils import write_json


def _render_centered(env: Any) -> Any:
    """Render with the offscreen camera following the cheetah's root x position."""
    renderer = env.unwrapped.mujoco_renderer
    renderer.render("rgb_array")
    camera = renderer.viewer.cam
    camera.lookat[0] = float(env.unwrapped.data.qpos[0])
    return renderer.render("rgb_array")


def run_random(
    config: EnvironmentConfig,
    *,
    seed: int,
    episodes: int,
    max_steps: int | None,
    output_dir: Path,
    record_video: bool,
    fps: int,
    clip_length: int | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir.parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config, render_mode="rgb_array" if record_video else None, seed=seed)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        if record_video:
            raise RuntimeError("Video recording requires the dev dependency imageio[ffmpeg]") from exc
        imageio = None
    rows: list[dict[str, Any]] = []
    video_records: list[tuple[Path, int, int]] = []
    try:
        for episode in range(1, episodes + 1):
            obs, info = env.reset(seed=seed + episode - 1)
            env.action_space.seed(seed + episode - 1)
            video_path = video_dir / f"random-policy-seed{seed}-episode{episode}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            writer = imageio.get_writer(video_path, fps=fps) if record_video and imageio else None
            frame_count = 0
            if writer is not None:
                writer.append_data(_render_centered(env))
                frame_count += 1
            initial_x = float(env.unwrapped.data.qpos[0])
            rewards: list[float] = []
            forwards: list[float] = []
            costs: list[float] = []
            velocities: list[float] = []
            squared: list[float] = []
            terminated = truncated = False
            limit = max_steps or config.episode_horizon
            for _ in range(limit):
                action = sample_random_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                forward, cost = reward_components(info, reward)
                rewards.append(float(reward))
                forwards.append(forward)
                costs.append(cost)
                velocities.append(float(info.get("x_velocity", 0.0)))
                squared.append(float((action ** 2).mean()))
                if writer is not None:
                    writer.append_data(_render_centered(env))
                    frame_count += 1
                if clip_length is not None and len(rewards) >= clip_length:
                    truncated = True
                if terminated or truncated:
                    break
            if writer is not None:
                writer.close()
                video_records.append((video_path, episode, frame_count))
            final_x = float(info.get("x_position", env.unwrapped.data.qpos[0]))
            metrics = aggregate_episode(
                episode, rewards, forwards, costs, velocities, squared,
                initial_x, final_x, terminated, truncated,
            ).as_dict()
            rows.append(metrics)
    finally:
        env.close()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = output_dir / f"random-policy-seed{seed}-{stamp}"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(stem.with_suffix(".json"), {"config": config.as_dict(), "seed": seed, "episodes": rows})
    if record_video:
        for video, episode, frame_count in video_records:
            write_json(video.with_suffix(".json"), {
                "policy": "Random policy", "seed": seed, "environment": config.env_id,
                "config": config.as_dict(), "episode": episode,
                "frame_count": frame_count, "fps": fps,
                "camera": "centered on the cheetah root x-position",
                "environment_steps": rows[episode - 1]["episode_length"],
                "generation_command": " ".join(__import__("sys").argv),
            })
    return rows
