"""
RecForge · recommenders 包（领域模块）
作者：晨星
包含：强基线（MostPopular / Random）、SOTA 后端（implicit ALS）、离线兜底（numpy ALS）、工厂。
"""

from .als import (
    ImplicitALSRecommender,
    NumpyALSRecommender,
    available_implicit,
)
from .baselines import MostPopularRecommender, RandomRecommender
from .factory import build_baselines, build_recommender, list_models

__all__ = [
    "ImplicitALSRecommender",
    "MostPopularRecommender",
    "NumpyALSRecommender",
    "RandomRecommender",
    "available_implicit",
    "build_baselines",
    "build_recommender",
    "list_models",
]
