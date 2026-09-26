"""切分无泄漏测试。作者：晨星"""
from data.synthetic import SyntheticRatings
from preprocess.split import LeaveOneOutSplitter


def test_no_leakage():
    ds = SyntheticRatings(seed=5, n_users=60, n_items=30, n_ratings=600,
                          min_interactions=3).generate()
    split = LeaveOneOutSplitter().split(ds, seed=5)
    train_set = {(r.user_id, r.item_id) for r in split.train.ratings}
    # 任一 held-out 都不应出现在 train
    for u, items in split.test_heldout.items():
        for i in items:
            assert (u, i) not in train_set
    # 总量守恒
    assert len(split.train.ratings) + len(split.test.ratings) == len(ds.ratings)
    # 每 test 用户 train 非空
    train_by_user = {}
    for r in split.train.ratings:
        train_by_user.setdefault(r.user_id, 0)
        train_by_user[r.user_id] += 1
    for u in split.test_heldout:
        assert train_by_user.get(u, 0) >= 1
