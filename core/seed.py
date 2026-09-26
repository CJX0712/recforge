"""
core/seed.py · 全局确定性入口
唯一 seed 入口：numpy / random 一次设齐；torch 可选。
作者：晨星
"""
from __future__ import annotations

import os
import random


def set_all(seed: int = 42) -> int:
    """设置全部随机源，保证可复现。返回生效的 seed。"""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch 可选
        pass
    return seed


def set_numpy(seed: int = 42):
    """仅设置 numpy，返回 np.random.Generator（推荐用于新代码）。"""
    import numpy as np

    return np.random.default_rng(int(seed))
