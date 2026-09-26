"""
hpo/tuner.py · Optuna 超参调优（可选，缺失则降级跳过）
作者：晨星
目标：在 train 折上 HPO 后仅在 holdout 评测，杜绝泄漏。
"""
from __future__ import annotations

from core.types import InteractionDataset


def available_optuna() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except Exception:
        return False


def tune_als(train: InteractionDataset, seed: int = 42, n_trials: int = 20) -> dict | None:
    """返回最优超参 {factors, reg, iterations}。Optuna 不可用时返回 None。"""
    if not available_optuna():
        return None
    import optuna

    from eval.evaluator import Evaluator
    from preprocess.split import LeaveOneOutSplitter
    from recommenders.als import NumpyALSRecommender

    # 二次切分：在 train 内再留一，HPO 只用其 holdout
    inner = LeaveOneOutSplitter().split(train, seed=seed + 1)
    evaluator = Evaluator()

    def objective(trial):
        factors = trial.suggest_int("factors", 8, 48, step=8)
        reg = trial.suggest_float("reg", 1e-3, 1.0, log=True)
        iterations = trial.suggest_int("iterations", 8, 20)
        model = NumpyALSRecommender(
            factors=factors, reg=reg, iterations=iterations, alpha=1.0, seed=seed
        )
        model.fit(inner.train)
        metrics = evaluator.evaluate(model, inner, k_list=[10])
        return -metrics.get("recall@10", 0.0)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return {
        "factors": int(best["factors"]),
        "reg": float(best["reg"]),
        "iterations": int(best["iterations"]),
    }


__all__ = ["available_optuna", "tune_als"]
