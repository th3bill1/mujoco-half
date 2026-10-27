import pytest
from src.metrics import aggregate_episode, reward_components


def test_reward_components_and_metrics():
    assert reward_components({"reward_forward": 2.0, "reward_ctrl": -0.5}, 1.5) == (2.0, 0.5)
    metrics = aggregate_episode(1, [1, 2], [1, 2], [0, 0.1], [3, 5], [0.2, 0.4],
                                1, 4, False, True)
    assert metrics.displacement == 3
    assert metrics.episode_length == 2


def test_reward_mismatch_rejected():
    with pytest.raises(ValueError):
        reward_components({"reward_forward": 2, "reward_ctrl": -0.5}, 9)
