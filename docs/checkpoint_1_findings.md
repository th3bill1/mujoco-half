# Checkpoint 1 findings

This file records the local verification run on 27 October 2026. No
trained-agent result or research question is claimed before the course
assignment.

Commands:

```powershell
python scripts\inspect_env.py --config configs\environment.yaml --seed 42
python scripts\run_random_agent.py --config configs\environment.yaml --seed 42 --episodes 3 --record-video
pytest -q
```

The generated inspection JSON, rollout CSV/JSON, MP4, and MP4 sidecar are the
machine-readable evidence for the checkpoint. The random baseline reports
return, forward reward, control cost, displacement, velocity, action-squared
proxy, episode length, and termination/truncation flags. SAC is the planned
checkpoint-2 baseline; PPO remains optional.

## Verified environment

- Python 3.15.1, Windows 11, AMD64, Gymnasium 1.4.0, MuJoCo 3.17.0,
  NumPy 2.5.4.
- Observation shape/dtype: `(17,)`, `float64`; action shape/dtype: `(6,)`,
  `float32`, bounds `[-1, 1]`.
- Physical action timestep: `0.05 s`; configured horizon: 1000 steps.
- One inspection step returned finite values and info keys
  `reward_ctrl`, `reward_forward`, `x_position`, and `x_velocity`.
- The focused test suite passed: `5 passed`.

## Random baseline sample

The short seeded smoke rollout used seed 42, two episodes, and a 20-step
maximum. Both episodes ended before the time limit and had `terminated=False`,
`truncated=False`.

| Episode | Return | Displacement | Mean forward velocity | Mean action-squared |
|---:|---:|---:|---:|---:|
| 1 | -6.9594 | -0.1696 | -0.1696 | 0.2972 |
| 2 | -2.4297 | 0.0701 | 0.0701 | 0.3193 |

The generated videos were
`artifacts/videos/random-policy-seed42-episode1-20261027-173347.mp4` and
`artifacts/videos/random-policy-seed42-episode2-20261027-173448.mp4`.
ImageIO/FFmpeg decoded each as a nonzero H.264 MP4 with 21 frames at 30 FPS
(one initial frame plus one per environment step). The output FPS is distinct
from the 20 Hz environment action rate.

Repeated no-video rollouts with seed 77, two episodes and a 10-step limit
produced identical return, displacement, action-squared proxy, and length
arrays, confirming the seeded environment and action-space streams are
reproducible on this installation.

The installed Gymnasium 1.4.0 `RecordVideo` wrapper requires MoviePy, so the
implementation uses the already-installed ImageIO/FFmpeg writer instead. This
keeps frame memory bounded and preserves a JSON sidecar with seed, config,
episode, frame count, FPS, and command.
