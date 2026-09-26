"""
data/movielens.py · 可选 MovieLens-100k 载入（需网络/本地文件）
作者：晨星
默认 demo 用合成数据，保证离线可复现；MovieLens 为可选真实基准。
"""
from __future__ import annotations

import os
import urllib.request
import zipfile

from core.errors import DataError
from core.types import InteractionDataset, Rating

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_DIR = "ml-100k"


def available_movielens() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


class MovieLensSource:
    """MovieLens-100k 源（隐式反馈：评分 >= 4 视为交互）。"""

    def __init__(self, path: str | None = None, seed: int = 42):
        self.path = path or ML100K_DIR
        self.seed = int(seed)

    def _ensure(self) -> str:
        u_data = os.path.join(self.path, "u.data")
        if not os.path.exists(u_data):
            zip_path = os.path.join(self.path + ".zip")
            os.makedirs(self.path, exist_ok=True)
            try:
                urllib.request.urlretrieve(ML100K_URL, zip_path)
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(os.path.dirname(self.path))
            except Exception as e:
                raise DataError(f"MovieLens 下载/解压失败（需网络）: {e}")
        return u_data

    def load(self, path: str | None = None) -> InteractionDataset:
        import pandas as pd

        u_data = self._ensure()
        df = pd.read_csv(
            u_data, sep="\t", header=None,
            names=["user", "item", "rating", "ts"],
        )
        # 重映射为连续 id
        u_map = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
        i_map = {i: j for j, i in enumerate(sorted(df["item"].unique()))}
        ratings = [
            Rating(
                user_id=u_map[int(r.user)],
                item_id=i_map[int(r.item)],
                value=1.0 if r.rating >= 4 else 0.0,
            )
            for r in df.itertuples()
        ]
        # 仅保留正反馈
        ratings = [r for r in ratings if r.value > 0]
        return InteractionDataset(
            n_users=len(u_map), n_items=len(i_map), ratings=ratings
        )

    def generate(self) -> InteractionDataset:
        return self.load(self.path)


__all__ = ["MovieLensSource", "available_movielens"]
