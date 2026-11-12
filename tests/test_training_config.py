import pytest

from src.training import SACConfig


def test_sac_config_loads_and_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "sac.yaml"
    config_path.write_text("total_timesteps: 20\nseed: 7\n", encoding="utf-8")
    config = SACConfig.from_yaml(config_path)
    assert config.total_timesteps == 20
    assert config.seed == 7

    config_path.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown SAC configuration keys"):
        SACConfig.from_yaml(config_path)
