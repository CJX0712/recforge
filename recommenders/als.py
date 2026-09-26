"""
recommenders/als.py · ALS 矩阵分解（隐式反馈）
作者：晨星
- ImplicitALSRecommender：Tier-0 SOTA 后端，封装 `implicit` 库（工业级 CF）。
- NumpyALSRecommender：Tier-1 离线兜底，纯 numpy 加权 ALS，零下载、可逐位复现。
统一接口：fit / recommend / predict；分数越大越推荐。
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


import numpy as np

from core.errors import ModelError
from core.types import InteractionDataset


def available_implicit() -> bool:
    try:
        import implicit  # noqa: F401

        return True
    except Exception:
        return False


class ImplicitALSRecommender:
    """Tier-0：封装 implicit.als.AlternatingLeastSquares。"""

    name = "ALS-implicit"
    backend = "implicit"

    def __init__(self, factors: int = 32, reg: float = 0.1, iterations: int = 15,
                 alpha: float = 40.0, seed: int = 42):
        self.factors = int(factors)
        self.reg = float(reg)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._model = None
        self._user_items_csr = None
        self.n_items = 0

    def fit(self, train: InteractionDataset) -> ImplicitALSRecommender:
        if not available_implicit():
            raise ModelError("implicit 不可用，无法构建 ImplicitALSRecommender")
        import implicit

        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)  # implicit 内部使用 numpy 全局随机
        coo = train.to_coo().tocsr().astype(np.float32)  # (n_users, n_items)
        self._user_items_csr = coo
        self.n_items = train.n_items
        # implicit 的 fit 接受矩阵 M(m x n)：fit(M) 后 user_factors 形状 (m, f)。
        # 直接以 user-item 矩阵（n_users x n_items）作 M，则 userid 索引 user_factors 正确。
        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iterations,
            random_state=self.seed,
        )
        model.fit(coo)
        self._model = model
        return self

    def recommend(self, user_id: int, k: int, exclude: list[int] | None = None) -> list[int]:
        if self._model is None:
            raise ModelError("模型未 fit")
        ex = exclude if exclude is not None else []
        # implicit 自动过滤 user_items 中的物品
        items, _scores = self._model.recommend(
            user_id, self._user_items_csr[user_id], N=k, filter_already_liked_items=True
        )
        return [int(i) for i in items]

    def predict(self, user_id: int, item_id: int) -> float:
        if self._model is None:
            raise ModelError("模型未 fit")
        # fit(user_item) 不转置：user_factors 形状 (n_users, f)，item_factors (n_items, f)
        u = self._model.user_factors[user_id]
        v = self._model.item_factors[item_id]
        return float(np.dot(u, v))


class NumpyALSRecommender:
    """Tier-1：纯 numpy 加权 ALS（Hu et al. 2008 隐式反馈），零依赖兜底。"""

    name = "ALS-numpy"
    backend = "numpy"

    def __init__(self, factors: int = 32, reg: float = 0.1, iterations: int = 15,
                 alpha: float = 1.0, seed: int = 42):
        self.factors = int(factors)
        self.reg = float(reg)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._U: np.ndarray | None = None
        self._V: np.ndarray | None = None
        self._train_user_items: dict[int, list[int]] = {}
        self.n_users = 0
        self.n_items = 0

    def fit(self, train: InteractionDataset) -> NumpyALSRecommender:
        rng = np.random.default_rng(self.seed)
        n_users, n_items, f = train.n_users, train.n_items, self.factors
        R = np.zeros((n_users, n_items), dtype=np.float64)
        for r in train.ratings:
            R[r.user_id, r.item_id] = 1.0
        C = 1.0 + self.alpha * R  # 置信度
        U = rng.standard_normal((n_users, f)) * 0.01
        V = rng.standard_normal((n_items, f)) * 0.01
        I_f = np.eye(f)
        regI = self.reg * I_f

        for _ in range(self.iterations):
            # 更新用户因子
            for u in range(n_users):
                cu = C[u]
                VtC = V.T * cu  # (f, n_items)
                A = VtC @ V + regI
                b = VtC @ R[u]
                U[u] = np.linalg.solve(A, b)
            # 更新物品因子
            for i in range(n_items):
                ci = C[:, i]
                UtC = U.T * ci
                A = UtC @ U + regI
                b = UtC @ R[:, i]
                V[i] = np.linalg.solve(A, b)

        self._U, self._V = U, V
        self.n_users, self.n_items = n_users, n_items
        self._train_user_items = train.user_items()
        return self

    def recommend(self, user_id: int, k: int, exclude: list[int] | None = None) -> list[int]:
        if self._U is None or self._V is None:
            raise ModelError("模型未 fit")
        ex = set(exclude if exclude is not None else self._train_user_items.get(user_id, []))
        scores = self._U[user_id] @ self._V.T
        # 降序取 top-k，排除已交互
        order = np.argsort(-scores)
        out: list[int] = []
        for i in order:
            i = int(i)
            if i not in ex:
                out.append(i)
            if len(out) >= k:
                break
        return out

    def predict(self, user_id: int, item_id: int) -> float:
        if self._U is None or self._V is None:
            raise ModelError("模型未 fit")
        return float(self._U[user_id] @ self._V[item_id])


__all__ = ["ImplicitALSRecommender", "NumpyALSRecommender", "available_implicit"]
