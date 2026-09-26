"""
recommenders/baselines.py · 强基线（打假用）
作者：晨星
- MostPopularRecommender：全局热门召回，top-K 推荐的标准强基线。
- RandomRecommender：随机召回，作为下界 sanity check。
（ItemAverage 在隐式反馈下等价于 MostPopular，故不单列，见 model_card。）
"""
from __future__ import annotations

import numpy as np

from core.errors import ModelError
from core.types import InteractionDataset


class MostPopularRecommender:
    """全局热门基线：所有用户推荐相同的高频物品（排除已交互）。"""

    name = "MostPopular"
    backend = "baseline"

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._order: list[int] = []
        self._train_user_items: dict[int, list[int]] = {}

    def fit(self, train: InteractionDataset) -> MostPopularRecommender:
        counts = np.zeros(train.n_items, dtype=np.int64)
        for r in train.ratings:
            counts[r.item_id] += 1
        self._order = list(np.argsort(-counts).tolist())
        self._train_user_items = train.user_items()
        if counts.sum() == 0:
            raise ModelError("MostPopular: 训练集无交互")
        return self

    def recommend(self, user_id: int, k: int, exclude: list[int] | None = None) -> list[int]:
        ex = set(exclude if exclude is not None else self._train_user_items.get(user_id, []))
        out: list[int] = []
        for i in self._order:
            if i not in ex:
                out.append(int(i))
            if len(out) >= k:
                break
        return out

    def predict(self, user_id: int, item_id: int) -> float:
        # 全局热门度（归一化计数），无用户个性化
        return float(item_id in self._order[: max(1, len(self._order) // 10)])


class RandomRecommender:
    """随机基线：下界 sanity check。"""

    name = "Random"
    backend = "baseline"

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.n_items = 0
        self._train_user_items: dict[int, list[int]] = {}

    def fit(self, train: InteractionDataset) -> RandomRecommender:
        self.n_items = train.n_items
        self._train_user_items = train.user_items()
        return self

    def recommend(self, user_id: int, k: int, exclude: list[int] | None = None) -> list[int]:
        ex = set(exclude if exclude is not None else self._train_user_items.get(user_id, []))
        pool = [i for i in range(self.n_items) if i not in ex]
        if len(pool) <= k:
            return pool
        idx = self._rng.choice(len(pool), size=k, replace=False)
        return [int(pool[i]) for i in idx]

    def predict(self, user_id: int, item_id: int) -> float:
        return 0.0


__all__ = ["MostPopularRecommender", "RandomRecommender"]
