# MuJoCo HalfCheetah

Checkpoint-1 foundations for Gymnasium `HalfCheetah-v5`. The official research
question is not assigned yet. The green difficulty level leaves room for later
ablations; SAC is planned for checkpoint 2 and PPO is optional.

## Install

```powershell
py -m pip install -e ".[dev]"
```

The package supports Python 3.10 and newer versions supported by the pinned
dependencies. The checked environment uses Python 3.15.

## Inspect and run a random policy

```powershell
python scripts\inspect_env.py --config configs\environment.yaml --seed 42
python scripts\run_random_agent.py --config configs\environment.yaml --seed 42 --episodes 3 --record-video
```

The inspection report is written to `artifacts/results/environment-inspection.json`.
Rollout CSV/JSON files are written to `artifacts/results/`; MP4 and sidecar
metadata are written to `artifacts/videos/`. The policy samples a new action
from the full action space at every step. Video FPS is an output rate and is
not the MuJoCo action rate. During recording, the offscreen camera follows the
cheetah root x-position so backward or forward motion stays visible. Use `--max-steps`, `--fps`, `--output-dir` and
`--clip-length` to shorten or relocate a run. Open the MP4 with any local
video player.

## Test

```powershell
pytest -q
```

## Environment defaults

| Value | Default |
|---|---:|
| Environment | `HalfCheetah-v5` |
| Forward reward weight | 1.0 |
| Control cost weight | 0.1 |
| Reset noise scale | 0.1 |
| Exclude absolute x from observation | true |
| Frame skip | 5 |
| Action horizon | 1000 steps |
| Physical timestep | 0.05 s per action |

The six controls are back/front thigh, shin and foot torques. The default
observation has 8 relative `qpos` values plus 9 `qvel` values (17 values);
actions have six values in `[-1, 1]`. The reward is

`forward_reward_weight * (x_after - x_before) / dt - ctrl_cost_weight * sum(action ** 2)`.

Displacement and forward velocity are physical movement measures. Mean
action-squared is only a control-effort proxy, not measured physical energy.
See the [official Gymnasium documentation](https://gymnasium.farama.org/main/environments/mujoco/half_cheetah/).

## Troubleshooting

Install `.[dev]` for the FFmpeg-backed video encoder. On headless systems,
`rgb_array` rendering should work without a display; if MuJoCo or encoding is
missing, the command reports the dependency error and video tests are skipped
only when the encoder is unavailable.
