"""配置校验测试。作者：晨星"""
import pytest

from core.config import ENV_PREFIX, load_config
from core.errors import ConfigError


def test_default_config_valid():
    cfg = load_config()
    cfg.validate()


def test_overrides():
    cfg = load_config({"n_users": 10, "n_items": 5, "n_ratings": 100})
    assert cfg.n_users == 10
    assert cfg.n_items == 5


def test_invalid_n_ratings_raises():
    with pytest.raises(ConfigError):
        load_config({"n_users": 50, "n_items": 10, "n_ratings": 50, "min_interactions": 3})


def test_env_override(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "SEED", "99")
    monkeypatch.setenv(ENV_PREFIX + "K_LIST", "3,7")
    cfg = load_config()
    assert cfg.seed == 99
    assert cfg.k_list == [3, 7]
