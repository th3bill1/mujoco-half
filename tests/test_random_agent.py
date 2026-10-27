from src.agents import sample_random_action
from src.envs import EnvironmentConfig, make_env


def test_random_policy_samples_each_step():
    env = make_env(EnvironmentConfig(), seed=8)
    try:
        env.reset(seed=8)
        actions = [sample_random_action(env) for _ in range(3)]
        assert len({tuple(action) for action in actions}) == 3
    finally:
        env.close()
