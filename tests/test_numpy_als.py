"""Numpy ALS 训练 + 胜出随机基线测试（离线兜底路径）。作者：晨星"""
from data.synthetic import SyntheticRatings
from preprocess.split import LeaveOneOutSplitter
from recommenders.als import NumpyALSRecommender


def _split():
    ds = SyntheticRatings(seed=21, n_users=120, n_items=60, n_ratings=1500,
                          min_interactions=3).generate()
    return LeaveOneOutSplitter().split(ds, seed=21)


def test_numpy_als_beats_random():
    # 使用已验证的默认长尾场景：numpy ALS 离线兜底必须胜出强基线 MostPopular
    from core.config import load_config
    from pipeline.pipeline import RecPipeline

    cfg = load_config({
        "dataset": "synthetic", "n_users": 500, "n_items": 200, "n_ratings": 8000,
        "backend": "numpy", "factors": 32, "reg": 0.1, "iterations": 15,
        "k_list": [10], "seed": 42, "verbose": False,
    })
    res = RecPipeline(cfg).run()
    sysr = next(r for r in res.rows if r.backend == "numpy").metrics
    mp = next(r for r in res.rows if r.model == "MostPopular").metrics
    # 核心：ALS 离线兜底胜出强基线（S-DoD 实质）
    assert sysr["recall@10"] > mp["recall@10"]
    assert sysr["ndcg@10"] > mp["ndcg@10"]


def test_numpy_als_recommend_excludes_seen():
    split = _split()
    als = NumpyALSRecommender(seed=21).fit(split.train)
    seen = split.train.user_items().get(0, [])
    rec = als.recommend(0, 10, exclude=seen)
    assert len(rec) == 10
    assert set(rec).isdisjoint(set(seen))
