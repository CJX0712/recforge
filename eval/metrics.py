"""
eval/metrics.py · Top-K 排序指标（手写，避免 sklearn 别名递归坑）
作者：晨星
约定：分数越大越推荐；recall/ndcg/hr 越大越好。
relevant：用户真正感兴趣的 item 集合（本系统为 held-out item）。
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_set(relevant: Iterable[int]) -> set[int]:
    return set(relevant)


def recall_at_k(relevant: Iterable[int], recommended: list[int], k: int) -> float:
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    rec_k = set(recommended[:k])
    return len(rel & rec_k) / len(rel)


def hr_at_k(relevant: Iterable[int], recommended: list[int], k: int) -> float:
    """HitRate@k：held-out 是否出现在 top-k（单相关项时与 recall 等价）。"""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    rec_k = set(recommended[:k])
    return 1.0 if (rel & rec_k) else 0.0


def ndcg_at_k(relevant: Iterable[int], recommended: list[int], k: int) -> float:
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in rel:
            dcg += 1.0 / np.log2(idx + 1)
    idcg = 0.0
    for idx in range(1, min(len(rel), k) + 1):
        idcg += 1.0 / np.log2(idx + 1)
    return dcg / idcg if idcg > 0 else 0.0


__all__ = ["hr_at_k", "ndcg_at_k", "recall_at_k"]
