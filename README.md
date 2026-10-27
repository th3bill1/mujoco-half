# MuJoCo HalfCheetah — Reinforcement Learning Project

Semester project for the **Reinforcement Learning** course at Warsaw University of Technology (2026/27).

**Status:** Checkpoint 1

**Authors:** Aleksandra Wieczorek, Wojciech Wójcik

## Project objective

Train and evaluate an agent that learns to move a simulated planar cheetah forward efficiently by applying continuous torques to six joints. The primary environment is [Gymnasium's `HalfCheetah-v5`](https://gymnasium.farama.org/main/environments/mujoco/half_cheetah/), powered by MuJoCo.

### Environment specification

| Property | Default |
| --- | --- |
| Task | Maximize forward motion while limiting control effort |
| Agents | One; no opponents or game modes |
| Observation | 17 continuous values: 8 positions (`qpos`, excluding global x) and 9 velocities (`qvel`) |
| Action | 6 continuous joint controls in `[-1, 1]`, for the back/front thigh, shin and foot |
| Reward | `forward_reward_weight * forward_velocity - ctrl_cost_weight * sum(action ** 2)` |
| Reward weights | Forward: `1.0`; control: `0.1` |
| Initial-state noise | `reset_noise_scale=0.1` |
| Episode limit | 1,000 environment steps (time-limit truncation); no built-in task termination |
| Default time step | `frame_skip=5`, simulation time `0.01 s` per physics frame, or `0.05 s` per action |

The default 17-dimensional observation is **not an image**. The absolute x-position is omitted to encourage position-independent control, but can be included (`exclude_current_positions_from_observation=False`, giving 18 dimensions). The environment returns useful diagnostics in `info`, such as `x_position`, `reward_forward` and `reward_ctrl`.

### Configurable parameters

- `forward_reward_weight`: how strongly forward speed is rewarded.
- `ctrl_cost_weight`: penalty for large actions.
- `reset_noise_scale`: variation in starting state.
- `exclude_current_positions_from_observation`: include/exclude global x-position.
- `frame_skip`: number of simulation frames per agent action.
- `xml_file`: optional custom MuJoCo model; out of scope unless required by the research question.
- Episode length via Gymnasium's `TimeLimit` wrapper; distinguish this from environment termination.

## Planned repository structure

```text
halfcheetah-rl/
├── README.md                       # Project overview and usage
├── plan.md                         # Checklist and milestones
├── pyproject.toml                  # Dependencies, scripts, tool config
├── .gitignore
├── configs/
│   ├── environment.yaml            # Canonical HalfCheetah parameters
│   ├── sac_baseline.yaml           # First trainable agent
│   └── experiments/                # Controlled research variants
├── src/
│   ├── __init__.py
│   ├── envs.py                     # Environment factory and wrappers
│   ├── agents.py                   # Random and trained policy interfaces
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Deterministic evaluation
│   ├── record.py                   # Video recording
│   ├── metrics.py                  # Episode-level metrics
│   └── utils.py                    # Seeds, paths, configuration
├── scripts/
│   ├── inspect_env.py              # Space/reward/termination sanity checks
│   ├── run_random_agent.py         # Checkpoint 1 random-agent rollout
│   ├── train_baseline.py           # Baseline training command
│   ├── evaluate_models.py          # Batch evaluation
│   └── make_plots.py               # Learning curves and comparisons
├── tests/
│   ├── test_env.py
│   ├── test_metrics.py
│   └── test_reproducibility.py
├── presentations/
│   ├── checkpoint_1/
│   ├── checkpoint_2/
│   └── final/
├── artifacts/
│   ├── videos/
│   ├── models/
│   ├── logs/
│   ├── results/
│   └── plots/
└── docs/
    ├── experiment_protocol.md
    └── research_log.md
```

## Install

```powershell
py -m pip install -e ".[dev]"
```

The package supports Python 3.10 and newer versions supported by the pinned
dependencies. The checked environment uses Python 3.15.

## Inspect and run a random policy

```powershell
python scripts\run_random_agent.py --episodes 1 --clip-length 300 --fps 30 --record-video
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

## References

- [Gymnasium: HalfCheetah](https://gymnasium.farama.org/main/environments/mujoco/half_cheetah/)
- [Gymnasium: MuJoCo environments](https://gymnasium.farama.org/main/environments/mujoco/)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [Sutton and Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book.html)

