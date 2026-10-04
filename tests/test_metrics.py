"""Top-K 指标正确性测试（手算对照）。作者：晨星"""

import numpy as np

from eval.metrics import hr_at_k, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k({1, 2}, [1, 3, 2], 2) == 0.5
    assert recall_at_k({1, 2}, [1, 2, 3], 2) == 1.0
    assert recall_at_k({1}, [3, 4], 2) == 0.0
    assert recall_at_k(set(), [1, 2], 2) == 0.0


def test_ndcg():
    # item 1 在位置 2（1-indexed），DCG=1/log2(3)=0.6309，IDCG=1
    val = ndcg_at_k({1}, [3, 1, 2], 3)
    assert abs(val - 1.0 / np.log2(3)) < 1e-9
    assert ndcg_at_k({1}, [1, 2, 3], 3) == 1.0
    assert ndcg_at_k({1}, [2, 3], 3) == 0.0


def test_hr():
    assert hr_at_k({1}, [3, 1], 2) == 1.0
    assert hr_at_k({1}, [2, 3], 2) == 0.0
