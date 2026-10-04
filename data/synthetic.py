"""
data/synthetic.py · 合成隐式反馈生成器（planted 潜因子）
作者：晨星
设计：P(interact_{u,i}) = sigmoid(affinity)，affinity = U_u·V_i + b_u + b_i + noise
planted 结构保证协同过滤信号存在 → ALS 必显著胜出 popularity 基线。
固定 seed 完全可复现（numpy Generator）。
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError
from core.types import InteractionDataset, Rating


class SyntheticRatings:
    """确定性合成隐式反馈数据集生成器。"""

    def __init__(
        self,
        n_users: int = 500,
        n_items: int = 200,
        n_ratings: int = 8000,
        n_factors: int = 16,
        noise: float = 0.3,
        min_interactions: int = 3,
        seed: int = 42,
    ):
        if n_users <= 0 or n_items <= 0:
            raise DataError("n_users / n_items 必须为正")
        if n_ratings < n_users * min_interactions:
            raise DataError("n_ratings 不足以让每用户达到 min_interactions")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_ratings = int(n_ratings)
        self.n_factors = int(n_factors)
        self.noise = float(noise)
        self.min_interactions = int(min_interactions)
        self.seed = int(seed)

    def generate(self) -> InteractionDataset:
        rng = np.random.default_rng(self.seed)
        f = self.n_factors
        # 强信号 planted 潜因子：尺度足以让协同结构清晰可学（构造基准）
        S = 4.0
        U = rng.standard_normal((self.n_users, f)) * S
        V = rng.standard_normal((self.n_items, f)) * S
        user_bias = rng.standard_normal(self.n_users) * S * 0.3
        item_bias = rng.standard_normal(self.n_items) * S * 0.3

        n_total = self.n_users * self.n_items
        # 1) 每用户保证 min_interactions 个高亲和度交互（确定性按 affinity 取 top）
        guaranteed: set[int] = set()
        for u in range(self.n_users):
            aff = U[u] @ V.T + item_bias  # (n_items,)
            order = np.argsort(-aff)  # 降序
            for i in order[: self.min_interactions]:
                guaranteed.add(u * self.n_items + int(i))

        # 2) 无放回补充交互至精确 n_ratings（按 affinity 加权，保留真实长尾）
        remaining = self.n_ratings - len(guaranteed)
        if remaining < 0:
            raise DataError("min_interactions 过大，超过 n_ratings")
        if remaining > 0:
            aff_all = U @ V.T + np.outer(user_bias, np.ones(self.n_items)) + item_bias
            # 强信号：planted 潜因子主导，噪声仅轻微扰动 → 协同结构清晰可学
            p = 1.0 / (1.0 + np.exp(-(aff_all + rng.standard_normal(aff_all.shape) * self.noise)))
            p_flat = p.ravel()
            # 仅从补集（非 guaranteed）中按权重无放回抽样
            comp_mask = np.ones(n_total, dtype=bool)
            if guaranteed:
                comp_mask[np.array(sorted(guaranteed), dtype=np.int64)] = False
            p_comp = p_flat.copy()
            p_comp[~comp_mask] = 0.0
            if p_comp.sum() <= 0:
                # 退化情形：均匀无放回
                p_comp = comp_mask.astype(np.float64)
            p_comp = p_comp / p_comp.sum()
            extra = rng.choice(n_total, size=remaining, replace=False, p=p_comp)
            chosen = guaranteed | {int(x) for x in extra}
        else:
            chosen = guaranteed

        ratings = [
            Rating(user_id=int(flat // self.n_items), item_id=int(flat % self.n_items), value=1.0)
            for flat in chosen
        ]
        return InteractionDataset(n_users=self.n_users, n_items=self.n_items, ratings=ratings)

    def ground_truth_affinity(self) -> np.ndarray:
        """返回 planted affinity 矩阵（仅供分析/失败案例用）。"""
        rng = np.random.default_rng(self.seed)
        f = self.n_factors
        # 强信号 planted 潜因子：尺度足以让协同结构清晰可学（构造基准）
        S = 4.0
        U = rng.standard_normal((self.n_users, f)) * S
        V = rng.standard_normal((self.n_items, f)) * S
        user_bias = rng.standard_normal(self.n_users) * S * 0.3
        item_bias = rng.standard_normal(self.n_items) * S * 0.3
        return U @ V.T + np.outer(user_bias, np.ones(self.n_items)) + item_bias


__all__ = ["SyntheticRatings"]
