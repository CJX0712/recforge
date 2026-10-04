"""Implicit ALS 测试（后端不可用时跳过）。作者：晨星"""

import pytest

from data.synthetic import SyntheticRatings
from eval.evaluator import Evaluator
from preprocess.split import LeaveOneOutSplitter
from recommenders.als import ImplicitALSRecommender, available_implicit


@pytest.mark.skipif(not available_implicit(), reason="implicit 未安装（离线兜底路径）")
def test_implicit_als_runs():
    ds = SyntheticRatings(seed=31, n_users=100, n_items=50, n_ratings=1200, min_interactions=3).generate()
    split = LeaveOneOutSplitter().split(ds, seed=31)
    model = ImplicitALSRecommender(factors=20, reg=0.1, iterations=10, alpha=40.0, seed=31)
    model.fit(split.train)
    rec = model.recommend(0, 5, exclude=split.train.user_items().get(0, []))
    assert len(rec) == 5
    metrics = Evaluator().evaluate(model, split, k_list=[10])
    assert metrics["recall@10"] > 0.0
