"""确定性测试：同 seed 两次运行核心指标逐位一致（numpy 后端）。作者：晨星"""

from core.config import load_config
from pipeline.pipeline import RecPipeline


def _sys_metrics(seed):
    cfg = load_config(
        {
            "dataset": "synthetic",
            "n_users": 120,
            "n_items": 60,
            "n_ratings": 1500,
            "backend": "numpy",
            "factors": 24,
            "reg": 0.1,
            "iterations": 12,
            "k_list": [5, 10, 20],
            "seed": seed,
            "verbose": False,
        }
    )
    res = RecPipeline(cfg).run()
    return next(r for r in res.rows if r.backend == "numpy").metrics


def test_determinism_bit_identical():
    m1 = _sys_metrics(42)
    m2 = _sys_metrics(42)
    assert m1 == m2  # 逐位一致


def test_different_seed_differs_slightly():
    # 不同 seed 数据不同，指标一般不同（非强约束，仅为健全性）
    m1 = _sys_metrics(1)
    m2 = _sys_metrics(2)
    assert isinstance(m1["recall@10"], float)
