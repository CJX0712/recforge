"""
examples/run_demo.py · 端到端演示（落盘 benchmark.json）
作者：晨星
默认：合成数据 + auto 后端（implicit 可用则用 SOTA，否则降级 numpy），固定 seed 可复现。
运行：python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import load_config
from pipeline.pipeline import RecPipeline

DEFAULTS = {
    "dataset": "synthetic",
    "n_users": 500,
    "n_items": 200,
    "n_ratings": 8000,
    "backend": "auto",  # implicit 优先，缺失降级 numpy
    "factors": 32,
    "reg": 0.1,
    "iterations": 15,
    "alpha": 1.0,
    "k_list": [5, 10, 20],
    "seed": 42,
    "out": os.path.join(ROOT, "benchmark.json"),
}


def main() -> int:
    cfg = load_config(DEFAULTS)
    pipe = RecPipeline(cfg)
    result = pipe.benchmark()
    print(pipe.report(result))
    print(f"\nbenchmark.json 已写入：{result and cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
