# State-of-the-Art Ranking Algorithms — Survey

Survey based on web search across Google Scholar, Kaggle competition patterns, and industry practice. Focus on algorithms that are **simple, proven, and directly optimize NDCG**.

## Tier 1: The Workhorse

### LambdaMART (LightGBM + objective=lambdarank)

- **Algorithm**: Gradient-boosted decision trees trained with LambdaRank gradients.
- **Why it wins**: Optimizes NDCG directly — the lambda gradient captures the change in NDCG from swapping any pair of items. Combines GBDT robustness with ranking-specific gradients.
- **Package**: `lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg")`
- **Kaggle track record**: Dominates tabular ranking. Most winning LTR solutions use LightGBM LambdaRank.
- **Pros**: No feature scaling needed, handles missing values natively, fast training, interpretable feature importance, works well on small-to-medium data.
- **Cons**: Not deep — can't learn cross-feature interactions automatically. May plateau on very large datasets.

### XGBoost (objective=rank:pairwise / rank:ndcg)

- **Algorithm**: GBDT with pairwise (RankNet-style) or listwise (LambdaRank-style) objectives.
- **Key difference**: Level-wise tree growth vs LightGBM's leaf-wise. Marginally slower, occasionally better on small data.
- **Package**: `xgboost.XGBRanker(objective="rank:ndcg")`
- **When to prefer**: When the pairwise objective fits better, or when LightGBM overfits.

### Benchmark results (MSLR-WEB30K)

| Method | NDCG@4 | NDCG@10 | Training speed |
|---|---|---|---|
| LightGBM LambdaRank | **46.56** | **48.52** | Fast |
| XGBoost LambdaMART | 42.60 | 45.84 | Moderate |
| LambdaRank (neural) | 46.26 | 49.56 | Slow |
| ListNet | 45.32 | 48.42 | Very slow |

LambdaMART via LightGBM is the baseline that even 2024 papers use as their SOTA reference.

## Tier 2: Neural LTR (more capacity, more complexity)

### AllRank (PyTorch)

- Open-source framework implementing multiple listwise losses: ListNet, ListMLE, ApproxNDCG, NeuralNDCG, LambdaRank.
- **Key advantage**: `NeuralNDCG` is a differentiable approximation of NDCG — optimizes it directly with neural nets.
- **Repo**: github.com/allegro/allRank
- **When to use**: When GBDT plateaus and you have enough data for a neural net.

### RankNet / ListNet / ListMLE

- **RankNet**: Pairwise loss, neural network. Historical foundation.
- **ListNet**: Listwise loss (top-1 probability via Plackett-Luce).
- **ListMLE**: Maximizes likelihood of the correct permutation.
- **Package**: `tensorflow-ranking`, custom PyTorch.
- **When**: Research contexts or when architectural flexibility matters more than raw accuracy.

## Tier 3: Research

### Transformer-based Rankers

- **SetRank**, **PiRank**: Self-attention over candidate items to model cross-item interactions.
- **When**: Large-scale systems where item context within the candidate set carries signal.

### Bivariate LambdaMART (Bi-LambdaMART, 2024)

- Extends LambdaMART to score pairs of items jointly rather than individually.
- Claims significant improvement over standard LambdaMART but implementation is not widely available.

## Recommendation for This Project

For the autoresearch loop, three algorithms as the pool:

| Algorithm | Complexity | Role |
|---|---|---|
| **LightGBM LambdaRank** | Low | Primary — immediate baseline, optimizes NDCG |
| **XGBoost rank:ndcg** | Low | Comparison — may win on small data |
| **Neural LambdaRank (AllRank)** | Medium | Upside — agent can explore if GBDTs plateau |

The autoresearch agent starts with a simple model, measures NDCG@K on the eval set, and iterates: feature construction, algorithm choice, hyperparameters — keeping changes that improve NDCG.

## Key References

- Burges, C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research.
- Pobrotyn, P. & Białobrzeski, R. (2021). *NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Approximation*.
- Karpathy, A. (2026). *autoresearch*. github.com/karpathy/autoresearch
- idealo Tech Blog (2026). *One Hour, 37% Faster: Applying Autoresearch to Our Search Ranking Inference Endpoint*.
