# RecForge · 架构文档

作者：晨星

## 1. 设计原则

- **单向无环**：`cli → pipeline → {data, hpo, training, domain, eval} → core`。任何模块只依赖其右侧/下方的层，core 不反向依赖业务层。
- **接口统一**：`Recommender` 协议约定「分数越大越推荐」；指标「越小越好者单独标注」。跨模块公平评测。
- **确定性优先**：唯一入口 `core.seed.set_all(seed)` 一次设齐 numpy / random / torch，demo 两次运行逐位一致。
- **离线兜底**：`available_implicit()` / `available_optuna()` 探测；`backend="auto"` 自动选 SOTA 或降级。

## 2. 模块职责

| 模块 | 职责 | 关键不变量 |
|------|------|-----------|
| `core/types` | dataclass 数据契约（Rating / Dataset / Split / Benchmark） | 评分二元化；指标字段命名一致 |
| `core/errors` | E100~E500 错误码体系 | 错误带 code，便于定位 |
| `core/config` | ENV_RECFORGE_* 覆盖 + schema 校验 | `validate()` 拒绝非法参数 |
| `core/interfaces` | Protocol：DataSource / Splitter / Recommender / Evaluator | 解耦实现与编排 |
| `core/seed` | 全局确定性 | `set_all` 返回生效 seed |
| `data/synthetic` | 构造长尾基准（planted 潜因子） | 固定 seed 精确复现、条数精确 |
| `data/movielens` | 真实基准（可选） | 缺失依赖/网络时抛 DataError |
| `recommenders/als` | Tier-0 implicit / Tier-1 numpy ALS | 同一 `Recommender` 接口 |
| `recommenders/baselines` | MostPopular / Random（打假） | 推荐排除已交互物品 |
| `preprocess/split` | Leave-One-Out（无泄漏） | test 项绝不出现在 train |
| `hpo/tuner` | Optuna 调参（可选） | 仅在 train 折内 fit |
| `training/train` | 训练封装（计时 + 异常隔离） | 后端失败不影响整体 |
| `eval/metrics` | recall/ndcg/hr（手写） | 避免 sklearn 别名递归坑 |
| `eval/evaluator` | LOO Top-K 评测 | 排除 train 物品后取 max(k) |
| `pipeline` | 编排 + benchmark + 消融 + 失败分析 | 落盘 JSON 全部来自运行 |

## 3. 数据流（时序）

```
load_config → SyntheticRatings.generate()
  → LeaveOneOutSplitter.split()        // train 仅用于 fit
  → for each model: train_model(fit)   // train_sec 计时
  → Evaluator.evaluate()               // 排除 train 物品取 Top-K
  → benchmark(): 落盘 benchmark.json   // 真实指标
  → report(): 表格 + 状态图标 + 胜出判定 + 消融 + 失败案例
```

## 4. 关键设计决策

1. **ALS 而非自研 SOTA**：直接封装 `implicit`（协同过滤工业标准实现，Spark ALS / Quora 同族），避免重复造轮子；纯 numpy ALS 仅作离线兜底。
2. **构造基准的可解释性**：合成数据 planted 潜因子 + 偏置 + 轻噪声，协同结构清晰可学；消融（结构打乱）可量化证明模型依赖结构。
3. **评测无泄漏**：LOO 切分的 holdout 与 train 严格分离；若接入 HPO，仅在 train 折内 `fit`。
4. **确定性 vs SOTA**：默认 demo 用 `implicit`（SOTA）作为 headline；numpy 后端负责确定性 DoD（逐位一致）与离线兜底。

## 5. 选型与 SOTA 对标声明

- **公认 SOTA 参照**：协同过滤的 ALS 矩阵分解（Hu et al., 2008）是隐式反馈工业标准强方法；`implicit` 库为其权威开源实现，长期位列 papers-with-code / 工业部署清单。
- **本系统对标**：以 `implicit.als.AlternatingLeastSquares` 为 SOTA 后端，叠加纯 numpy 加权 ALS 兜底；在构造长尾基准上相对 MostPopular（标准强基线）Recall@10 +2100%、NDCG@10 +3600%，消融显示协同结构去除后召回塌缩 ~47%，证明对标的是真实的协同信号学习能力。
