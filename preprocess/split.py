"""
preprocess/split.py · Leave-One-Out 切分（无泄漏）
作者：晨星
规则：每个用户按交互顺序保留前 N-1 条入 train，最后 1 条入 test。
保证 train 仅用于 fit；test 仅用于评测，绝不与 train 交换信息。
确定性：依赖 ratings 的插入顺序（由生成器固定），无需额外随机源。
"""
from __future__ import annotations

from collections import OrderedDict

from core.errors import DataError
from core.types import InteractionDataset, Rating, SplitResult


class LeaveOneOutSplitter:
    """每个用户留一法切分。"""

    name = "LeaveOneOut"

    def split(self, dataset: InteractionDataset, seed: int = 42) -> SplitResult:
        # 保留插入顺序
        by_user: OrderedDict[int, list[Rating]] = OrderedDict()
        for r in dataset.ratings:
            by_user.setdefault(r.user_id, []).append(r)

        train_ratings: list[Rating] = []
        test_ratings: list[Rating] = []
        heldout: dict[int, list[int]] = {}

        for u, rs in by_user.items():
            if len(rs) < 2:
                # 不足 2 条无法留一，整体放入 train（避免 test 泄漏/空 train）
                train_ratings.extend(rs)
                continue
            test_ratings.append(rs[-1])
            heldout[u] = [rs[-1].item_id]
            train_ratings.extend(rs[:-1])

        if not train_ratings:
            raise DataError("切分后 train 为空，请检查 min_interactions")
        if not test_ratings:
            raise DataError("切分后 test 为空")

        train = InteractionDataset(
            n_users=dataset.n_users, n_items=dataset.n_items, ratings=train_ratings
        )
        test = InteractionDataset(
            n_users=dataset.n_users, n_items=dataset.n_items, ratings=test_ratings
        )
        return SplitResult(train=train, test=test, test_heldout=heldout)


__all__ = ["LeaveOneOutSplitter"]
