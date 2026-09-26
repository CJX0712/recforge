# Changelog

## v0.1.0 (S 级首发)
- 初始发布：ALS 协同过滤推荐系统（Tier-0 implicit + Tier-1 numpy 兜底）。
- 构造长尾基准 + Leave-One-Out 无泄漏评测（Recall@10 / NDCG@10）。
- 强基线（MostPopular / Random）对照 + 结构消融实验。
- 全局确定性（同 seed 逐位一致）、离线降级、依赖锁死、单测 + ruff + CI 全绿。
- 性能 DoD：相对 MostPopular Recall@10 +2100%、NDCG@10 +3600%（预设阈值 +30% / +20%）。
- 作者署名：晨星；仓库 cjx0712/recforge。
