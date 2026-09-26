"""性能 DoD：默认场景下系统胜出强基线（阈值 +30% recall@10 / +20% ndcg@10）。作者：晨星"""
from core.config import load_config
from pipeline.pipeline import RecPipeline


def test_beats_strong_baseline_default_scenario():
    # 默认场景（与 run_demo 一致）：合成数据 + numpy 后端（确定性、离线）
    cfg = load_config({
        "dataset": "synthetic", "n_users": 500, "n_items": 200, "n_ratings": 8000,
        "backend": "numpy", "factors": 32, "reg": 0.1, "iterations": 15,
        "k_list": [5, 10, 20], "seed": 42, "verbose": False,
    })
    res = RecPipeline(cfg).run()
    sys_row = next(r for r in res.rows if r.backend == "numpy")
    mp = next(r for r in res.rows if r.model == "MostPopular")
    assert sys_row.available and mp.available

    r10_sys = sys_row.metrics["recall@10"]
    r10_mp = mp.metrics["recall@10"]
    n10_sys = sys_row.metrics["ndcg@10"]
    n10_mp = mp.metrics["ndcg@10"]

    rel_r = (r10_sys - r10_mp) / r10_mp
    rel_n = (n10_sys - n10_mp) / n10_mp
    # 预设胜出阈值
    assert rel_r >= 0.30, f"Recall@10 相对提升 {rel_r:.3f} 未达 +30%"
    assert rel_n >= 0.20, f"NDCG@10 相对提升 {rel_n:.3f} 未达 +20%"
