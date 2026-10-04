"""
core/types.py · 跨模块统一数据类型
作者：晨星
约定：分数越大越推荐；指标越小越好者单独标注（如 sMAPE）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Rating:
    """单条隐式/显式反馈。value 默认 1.0 表示隐式交互。"""

    user_id: int
    item_id: int
    value: float = 1.0


@dataclass
class InteractionDataset:
    """用户-物品交互集合（稀疏）。"""

    n_users: int
    n_items: int
    ratings: list[Rating] = field(default_factory=list)

    def to_coo(self):
        import numpy as np
        from scipy.sparse import coo_matrix

        if not self.ratings:
            return coo_matrix((self.n_users, self.n_items))
        rows = np.fromiter((r.user_id for r in self.ratings), dtype=np.int64, count=len(self.ratings))
        cols = np.fromiter((r.item_id for r in self.ratings), dtype=np.int64, count=len(self.ratings))
        data = np.fromiter((r.value for r in self.ratings), dtype=np.float64, count=len(self.ratings))
        return coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def user_items(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for r in self.ratings:
            out.setdefault(r.user_id, []).append(r.item_id)
        return out


@dataclass
class SplitResult:
    """切分结果：train 仅用于 fit，test 仅用于评测，禁止泄漏。"""

    train: InteractionDataset
    test: InteractionDataset
    test_heldout: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    value: float


@dataclass
class BenchmarkRow:
    model: str
    backend: str
    metrics: dict[str, float] = field(default_factory=dict)
    train_sec: float = 0.0
    available: bool = True
    note: str = ""


@dataclass
class BenchmarkResult:
    seed: int = 42
    dataset: str = "synthetic"
    n_users: int = 0
    n_items: int = 0
    n_train: int = 0
    n_test: int = 0
    k_list: list[int] = field(default_factory=list)
    rows: list[BenchmarkRow] = field(default_factory=list)
    ablation: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        rows = [BenchmarkRow(**r) for r in d.get("rows", [])]
        kw = {k: v for k, v in d.items() if k != "rows"}
        return cls(rows=rows, **kw)


__all__ = [
    "BenchmarkResult",
    "BenchmarkRow",
    "InteractionDataset",
    "MetricResult",
    "Rating",
    "SplitResult",
]
