"""
recommenders/factory.py · 模型工厂与可用性探测
作者：晨星
backend='auto'：优先 implicit，缺失则降级 numpy（离线兜底）。
"""
from __future__ import annotations

from core.config import Config
from core.errors import ModelError

from .als import ImplicitALSRecommender, NumpyALSRecommender, available_implicit
from .baselines import MostPopularRecommender, RandomRecommender


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "implicit" if available_implicit() else "numpy"
    if backend == "implicit" and not available_implicit():
        raise ModelError("backend=implicit 但 implicit 不可用，请改用 numpy 或 auto")
    return backend


def build_recommender(name: str, cfg: Config) -> object:
    backend = resolve_backend(cfg.backend)
    if name in ("implicit_als", "als"):
        if backend == "implicit":
            return ImplicitALSRecommender(
                factors=cfg.factors, reg=cfg.reg, iterations=cfg.iterations,
                alpha=40.0, seed=cfg.seed,
            )
        return NumpyALSRecommender(
            factors=cfg.factors, reg=cfg.reg, iterations=cfg.iterations,
            alpha=cfg.alpha, seed=cfg.seed,
        )
    if name == "numpy_als":
        return NumpyALSRecommender(
            factors=cfg.factors, reg=cfg.reg, iterations=cfg.iterations,
            alpha=cfg.alpha, seed=cfg.seed,
        )
    raise ModelError(f"未知模型: {name}")


def build_baselines(cfg: Config) -> list[object]:
    return [
        MostPopularRecommender(seed=cfg.seed),
        RandomRecommender(seed=cfg.seed),
    ]


def list_models() -> list[str]:
    out = ["implicit_als", "numpy_als", "MostPopular", "Random"]
    return out


__all__ = ["build_baselines", "build_recommender", "list_models", "resolve_backend"]
