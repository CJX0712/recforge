"""
core/interfaces.py · 模块间统一 Protocol 接口
作者：晨星
单向无环：cli → pipeline → {data, hpo, training, domain, eval} → core
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import InteractionDataset, SplitResult


@runtime_checkable
class DataSource(Protocol):
    def generate(self) -> InteractionDataset: ...
    def load(self, path: str) -> InteractionDataset: ...


@runtime_checkable
class Splitter(Protocol):
    def split(self, dataset: InteractionDataset, seed: int) -> SplitResult: ...


@runtime_checkable
class Recommender(Protocol):
    name: str
    backend: str

    def fit(self, train: InteractionDataset) -> Recommender: ...
    def recommend(self, user_id: int, k: int, exclude: list[int] | None = None) -> list[int]: ...
    def predict(self, user_id: int, item_id: int) -> float: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self, recommender: Recommender, split: SplitResult, k_list: list[int]
    ) -> dict[str, float]: ...


__all__ = ["DataSource", "Evaluator", "Recommender", "Splitter"]
