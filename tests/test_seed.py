"""确定性入口测试。作者：晨星"""
import numpy as np

from core.seed import set_all


def test_set_all_reproducible():
    set_all(7)
    a = np.random.default_rng(7).standard_normal(5)
    set_all(7)
    b = np.random.default_rng(7).standard_normal(5)
    assert np.array_equal(a, b)


def test_set_all_returns_seed():
    assert set_all(123) == 123
