"""
cli.py · RecForge 命令行入口
作者：晨星
用法：
  python cli.py                         # 默认合成数据 + auto 后端，生成 benchmark.json
  python cli.py --dataset synthetic --backend numpy --k 5,10,20
  python cli.py --dataset movielens      # 可选真实基准（需网络）
  python cli.py --out results.json --quiet
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import load_config
from pipeline.pipeline import RecPipeline


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="recforge",
        description="RecForge · 世界顶级推荐系统（ALS 协同过滤）",
    )
    p.add_argument("--dataset", default=None, help="synthetic | movielens")
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--n-items", type=int, default=None)
    p.add_argument("--n-ratings", type=int, default=None)
    p.add_argument("--backend", default=None, help="auto | implicit | numpy")
    p.add_argument("--factors", type=int, default=None)
    p.add_argument("--reg", type=float, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--k", default=None, help="逗号分隔，如 5,10,20")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="benchmark.json 输出路径")
    p.add_argument("--quiet", action="store_true", help="不打印报告")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = {}
    if args.dataset is not None:
        overrides["dataset"] = args.dataset
    if args.n_users is not None:
        overrides["n_users"] = args.n_users
    if args.n_items is not None:
        overrides["n_items"] = args.n_items
    if args.n_ratings is not None:
        overrides["n_ratings"] = args.n_ratings
    if args.backend is not None:
        overrides["backend"] = args.backend
    if args.factors is not None:
        overrides["factors"] = args.factors
    if args.reg is not None:
        overrides["reg"] = args.reg
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.k is not None:
        overrides["k_list"] = [int(x) for x in args.k.split(",") if x.strip()]
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.out is not None:
        overrides["out"] = args.out
    if args.quiet:
        overrides["verbose"] = False

    cfg = load_config(overrides)
    pipe = RecPipeline(cfg)
    result = pipe.benchmark()
    if not args.quiet:
        print(pipe.report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
