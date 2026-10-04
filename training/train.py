"""
training/train.py · 训练封装（计时 + 异常隔离）
作者：晨星
"""

from __future__ import annotations

import time

from core.types import InteractionDataset


def train_model(recommender, train: InteractionDataset):
    """拟合模型并返回 (fitted, train_sec)。失败时标记 available=False。"""
    t0 = time.perf_counter()
    try:
        fitted = recommender.fit(train)
        sec = time.perf_counter() - t0
        return fitted, sec, None
    except Exception as e:  # 捕获后端不可用/训练异常，隔离不影响整体
        sec = time.perf_counter() - t0
        return recommender, sec, str(e)


__all__ = ["train_model"]
