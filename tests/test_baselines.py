"""基线推荐有效性测试。作者：晨星"""
from data.synthetic import SyntheticRatings
from preprocess.split import LeaveOneOutSplitter
from recommenders.baselines import MostPopularRecommender, RandomRecommender


def _split():
    ds = SyntheticRatings(seed=8, n_users=50, n_items=25, n_ratings=400,
                          min_interactions=3).generate()
    return LeaveOneOutSplitter().split(ds, seed=8)


def test_mostpopular_recommend_valid():
    split = _split()
    mp = MostPopularRecommender(seed=8).fit(split.train)
    rec = mp.recommend(0, 5, exclude=split.train.user_items().get(0, []))
    assert len(rec) == 5
    assert len(set(rec)) == 5  # 无重复
    # 不推荐已交互
    seen = set(split.train.user_items().get(0, []))
    assert seen.isdisjoint(rec)


def test_random_deterministic():
    split = _split()
    r1 = RandomRecommender(seed=11).fit(split.train)
    r2 = RandomRecommender(seed=11).fit(split.train)
    rec1 = r1.recommend(0, 5)
    rec2 = r2.recommend(0, 5)
    assert rec1 == rec2  # 同 seed 一致
