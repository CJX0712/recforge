"""
core/config.py · 配置（ENV_RECFORGE_* 覆盖 + schema 校验）
作者：晨星
优先级：显式 overrides > 环境变量 > 默认值。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import ConfigError

ENV_PREFIX = "RECFORGE_"


@dataclass
class Config:
    # 确定性
    seed: int = 42
    # 数据（合成）
    dataset: str = "synthetic"  # synthetic | movielens
    n_users: int = 500
    n_items: int = 200
    n_ratings: int = 8000
    n_factors_data: int = 16  # 合成数据planted潜因子数
    affinity_noise: float = 0.05
    # 模型
    backend: str = "auto"  # auto | implicit | numpy
    factors: int = 32
    reg: float = 0.1
    iterations: int = 15
    alpha: float = 1.0  # 置信度权重（numpy ALS）
    # 评测
    k_list: list[int] = field(default_factory=lambda: [5, 10, 20])
    # 切分
    min_interactions: int = 3  # 每用户最少交互，保证 train 非空
    # 可选 MovieLens 路径
    movielens_path: str | None = None
    # 输出
    out: str = "benchmark.json"
    verbose: bool = True

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("seed 必须为非负整数")
        if self.n_users <= 0 or self.n_items <= 0:
            raise ConfigError("n_users / n_items 必须为正")
        if self.n_ratings <= self.n_users * self.min_interactions:
            raise ConfigError("n_ratings 不足以让每用户达到 min_interactions")
        if self.factors <= 0:
            raise ConfigError("factors 必须为正")
        if self.reg < 0:
            raise ConfigError("reg 必须 >= 0")
        if self.iterations <= 0:
            raise ConfigError("iterations 必须为正")
        if not self.k_list:
            raise ConfigError("k_list 不能为空")
        if self.backend not in ("auto", "implicit", "numpy"):
            raise ConfigError(f"未知 backend: {self.backend}")


_ENV_MAP = {
    "seed": int,
    "n_users": int,
    "n_items": int,
    "n_ratings": int,
    "n_factors_data": int,
    "affinity_noise": float,
    "factors": int,
    "reg": float,
    "iterations": int,
    "alpha": float,
    "min_interactions": int,
    "verbose": lambda s: s.lower() in ("1", "true", "yes"),
}


def load_config(overrides: dict | None = None) -> Config:
    """从默认值 + 环境变量 + 显式 overrides 合成配置并校验。"""
    cfg = Config()
    # 环境变量
    for key, caster in _ENV_MAP.items():
        env_val = os.environ.get(ENV_PREFIX + key.upper())
        if env_val is not None:
            try:
                setattr(cfg, key, caster(env_val))
            except (ValueError, TypeError) as e:
                raise ConfigError(f"环境变量 {ENV_PREFIX}{key.upper()} 解析失败: {e}")
    env_k = os.environ.get(ENV_PREFIX + "K_LIST")
    if env_k is not None:
        cfg.k_list = [int(x) for x in env_k.split(",") if x.strip()]
    env_ds = os.environ.get(ENV_PREFIX + "DATASET")
    if env_ds is not None:
        cfg.dataset = env_ds
    env_backend = os.environ.get(ENV_PREFIX + "BACKEND")
    if env_backend is not None:
        cfg.backend = env_backend
    env_out = os.environ.get(ENV_PREFIX + "OUT")
    if env_out is not None:
        cfg.out = env_out
    env_ml = os.environ.get(ENV_PREFIX + "MOVIELENS_PATH")
    if env_ml is not None:
        cfg.movielens_path = env_ml
    # 显式覆盖
    if overrides:
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ConfigError(f"未知配置项: {k}")
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


__all__ = ["ENV_PREFIX", "Config", "load_config"]
