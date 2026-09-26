"""
eval/evaluator.py · 评测器（leave-one-out 协议）
作者：晨星
对每个 test 用户：exclude 其 train 物品 → 取 max(k) 推荐 → 在各级 k 上算指标 → 跨用户平均。
"""
from __future__ import annotations

import numpy as np

from core.errors import EvalError
from core.types import SplitResult

from .metrics import hr_at_k, ndcg_at_k, recall_at_k


class Evaluator:
    def evaluate(
        self, recommender, split: SplitResult, k_list: list[int]
    ) -> dict[str, float]:
        if not k_list:
            raise EvalError("k_list 为空")
        max_k = max(k_list)
        train_user_items = split.train.user_items()
        heldout = split.test_heldout

        if not heldout:
            raise EvalError("test_heldout 为空，无法评测")

        acc: dict[str, list[float]] = {f"recall@{k}": [] for k in k_list}
        for k in k_list:
            acc[f"ndcg@{k}"] = []
            acc[f"hr@{k}"] = []

        per_user_hit = []
        for u, rel_items in heldout.items():
            exclude = train_user_items.get(u, [])
            try:
                rec = recommender.recommend(u, max_k, exclude=exclude)
            except Exception as e:
                raise EvalError(f"recommend 失败 user={u}: {e}")
            for k in k_list:
                acc[f"recall@{k}"].append(recall_at_k(rel_items, rec, k))
                acc[f"ndcg@{k}"].append(ndcg_at_k(rel_items, rec, k))
                acc[f"hr@{k}"].append(hr_at_k(rel_items, rec, k))
            per_user_hit.append(1.0 if (set(rel_items) & set(rec[:max_k])) else 0.0)

        out: dict[str, float] = {}
        for key, vals in acc.items():
            out[key] = float(np.mean(vals)) if vals else 0.0
        # 附带整体 hit rate（= recall@max_k 均值）
        out["users"] = float(len(heldout))
        out["overall_hit_rate"] = float(np.mean(per_user_hit))
        return out


__all__ = ["Evaluator"]
