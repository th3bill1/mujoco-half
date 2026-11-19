import numpy as np

from src.sac import SACAgent


def test_sac_prediction_save_and_load(tmp_path):
    agent = SACAgent(
        observation_size=17,
        action_size=6,
        learning_starts=8,
        batch_size=8,
        buffer_size=100,
        seed=7,
        device="cpu",
    )
    observation = np.zeros(17, dtype=np.float32)
    action = agent.predict(observation, deterministic=True)
    assert action.shape == (6,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    for _ in range(20):
        agent.add_transition(
            observation,
            np.zeros(6, dtype=np.float32),
            1.0,
            observation,
            False,
        )
    update = agent.update()
    assert {"critic_loss", "actor_loss", "alpha_loss", "alpha"} <= set(update)

    model_path = tmp_path / "model.zip"
    action = agent.predict(observation, deterministic=True)
    agent.save(model_path)
    loaded = SACAgent.load(model_path, device="cpu")
    loaded_action = loaded.predict(observation, deterministic=True)
    assert loaded_action.shape == (6,)
    assert np.allclose(action, loaded_action)
