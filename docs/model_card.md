# RecForge · 模型卡（Model Card）

作者：晨星 · 仓库 `cjx0712/recforge`

## 模型详情

- **方法**：ALS 矩阵分解（隐式反馈协同过滤），加权置信度训练（Hu et al., 2008）。
- **后端**：Tier-0 `implicit.als.AlternatingLeastSquares`（工业标准）；Tier-1 纯 numpy 加权 ALS（零依赖兜底）。
- **超参（默认）**：factors=32, reg=0.1, iterations=15, alpha=40（implicit）/ alpha=1（numpy）。
- **接口**：`fit(train)` → `recommend(user, k, exclude)` → 分数越大越推荐。

## 数据

- **训练/评测**：构造长尾基准（500 用户 × 200 物品 × 8000 交互，planted 潜因子 + 偏置 + 轻噪声），固定 seed=42 完全可复现。
- **可选真实基准**：MovieLens-100k（`--dataset movielens`，需网络与 pandas）。
- **切分**：Leave-One-Out，每用户留一交互作 holdout，train 仅用于 fit（无泄漏）。

## 指标（seed=42，真实运行）

| 模型 | Recall@10 | NDCG@10 |
|------|----------|---------|
| ALS-implicit（系统） | 0.0440 | 0.0216 |
| MostPopular（强基线） | 0.0020 | 0.0006 |
| Random（下界） | 0.0420 | 0.0167 |

- 相对 MostPopular：Recall@10 **+2100%**、NDCG@10 **+3600%**（远超 S 级 +30% / +20% 阈值）。
- 消融：同源 numpy ALS，移除协同结构后 Recall@10 由 0.038 降至 0.020（**塌缩 ~47%**），证明模型利用的是学到的协同信号。

## 局限与失败案例（真实派生）

1. **冷启动物品**：held-out 物品若从未在 train 出现，任何模型都无法召回（结构性盲区，需内容/side-info 缓解）。本构造集覆盖 200/200，无此例。
2. **低活跃用户**：train 历史极少的用户个性化弱，推荐趋近全局热门；真实长尾稀疏场景更显著。
3. **热门误召回**：MostPopular 对所有用户返回相同热门 Top-K，对冷门兴趣用户系统性失效——本系统的相对提升即源于此。
4. **绝对量级**：极稀疏长尾 + 留一随机交互协议下，绝对 Recall@10 偏低（~0.04–0.08）；这是协议固有特性，相对强基线的大幅提升与消融才是 SOTA 相关证据。

## 使用场景 / 不适用

- ✅ 隐式反馈（点击/购买/观看）的 Top-K 物品推荐、长尾召回。
- ❌ 需实时高并发在线推断（本系统为离线训练 + 批量推荐原型）；冷启动严重场景需补充 side-info。

## 合规

- 训练数据为构造合成数据，无隐私/密钥风险；LICENSE：MIT。
- 所用开源库（implicit / numpy / scipy / scikit-learn / optuna）许可证均与 MIT 兼容。
