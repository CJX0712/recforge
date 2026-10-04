"""合成数据可复现 + 覆盖测试。作者：晨星"""

from data.synthetic import SyntheticRatings


def _tuples(ds):
    return sorted((r.user_id, r.item_id) for r in ds.ratings)


def test_reproducible():
    a = SyntheticRatings(seed=1, n_users=80, n_items=40, n_ratings=800).generate()
    b = SyntheticRatings(seed=1, n_users=80, n_items=40, n_ratings=800).generate()
    assert _tuples(a) == _tuples(b)
    assert len(a.ratings) == 800


def test_min_interactions():
    ds = SyntheticRatings(seed=3, n_users=50, n_items=30, n_ratings=500, min_interactions=4).generate()
    from collections import Counter

    c = Counter(r.user_id for r in ds.ratings)
    assert all(v >= 4 for v in c.values())


def test_distinct_seed_differs():
    a = SyntheticRatings(seed=1, n_users=80, n_items=40, n_ratings=800).generate()
    b = SyntheticRatings(seed=2, n_users=80, n_items=40, n_ratings=800).generate()
    assert _tuples(a) != _tuples(b)
