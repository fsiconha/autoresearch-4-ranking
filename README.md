# autoresearch-4-ranking

Autonomous AI research agent that iteratively improves a **learning-to-rank (LTR) model** by editing `train.py`, running experiments, measuring NDCG, and keeping only changes that move the metric.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and the [idealo Tech Blog post](https://medium.com/idealo-tech-blog/one-hour-37-faster-applying-autoresearch-to-our-search-ranking-inference-endpoint-34cffc08e373) that applied the same loop to their search ranking inference endpoint (37% faster in one hour).

## How it works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. Edit      │────>│ 2. Train &   │────>│ 3. Measure   │
│  train.py    │     │  Evaluate    │     │  NDCG@5/@10  │
└──────────────┘     └──────────────┘     └──────────────┘
       ^                                        │
       │          ┌──────────────┐              │
       └──────────│ 4. Keep if   │<─────────────┘
                  │   improved   │
                  └──────────────┘
                         │ discard otherwise
                         ▼
                    git reset --hard
```

The agent reads `program.md` for instructions, studies `results.tsv` to avoid repeating failures, forms a hypothesis, edits `train.py`, commits, runs `make run`, checks the NDCG output, and keeps or reverts based on the result. Then it loops again. Forever.

## Project structure

```
├── src/
│   ├── generate_data.py   # Synthetic data generation (1000 users, 200 items, 90 days)
│   ├── data.py            # Data loading utilities
│   └── splitting.py       # Temporal Global Split + ideal ranking construction
├── tests/                 # Unit tests (63 tests, 3 modules)
├── data/                  # CSV files (generated, not committed)
├── prepare.py             # Fixed feature engineering & NDCG evaluation (DO NOT EDIT)
├── train.py               # Model — THIS IS WHAT THE AGENT EDITS
├── program.md             # Agent instructions & experiment protocol
├── sota-ranking.md        # SOTA ranking algorithm survey
├── results.tsv            # Experiment log (agent appends, never commits)
├── Makefile               # make test | make data | make run
├── pyproject.toml         # Dependencies (lightgbm, numpy, pandas)
└── uv.lock                # Locked dependency versions
```

## Quick start

```bash
# Install dependencies
uv sync

# Generate synthetic data
make data

# Run the full training pipeline
make run

# Run unit tests
make test
```

## The three files

Following Karpathy's pattern, the project revolves around three files:

| File | Role | Who edits it |
|------|------|-------------|
| `prepare.py` | Data loading, feature engineering, NDCG computation | **Human** — fixed ground truth |
| `train.py` | Model choice, hyperparameters, feature construction | **Agent** — this is the experiment surface |
| `program.md` | Instructions, constraints, loop protocol | **Human** — defines the rules of the game |

## Dataset

Synthetic user-item interaction data designed to simulate a real e-commerce ranking problem:

- **1,000 users** interacting with **200 items** across 4 types (x, y, z, w)
- **~82k interactions** over a 90-day window (view/click/purchase)
- **Temporal Global Split** (Gusak et al., RecSys 2025): hard 80/20 temporal cutoff
- **Ideal rankings** built from the evaluation window using max-interaction aggregation
- **Graded relevance**: view=1, click=2, purchase=5

## Evaluation

NDCG@5 and NDCG@10 averaged across all users. Higher is better (max 1.0).

```
$ make run
---
ndcg_5: 0.378435
ndcg_10: 0.466632
```

## Results (autoresearch/jun13)

| Experiment | NDCG@5 | NDCG@10 | Outcome |
|-----------|--------|---------|---------|
| Baseline (exponential gain) | 0.3588 | 0.4533 | — |
| **Linear label_gain** | **0.3784** | **0.4666** | **+5.5% NDCG@5** |

Key finding: switching from exponential `label_gain` (LightGBM default, `2^rel-1`) to linear `[0,1,2,0,0,5]` was the single effective improvement. The 21 aggregate features from `prepare.py` create a feature ceiling — the model converges at tree #1.

See [PR #5](https://github.com/fsiconha/autoresearch-4-ranking/pull/5) for the full 12-experiment report.

## Algorithms

The agent can choose from (see `sota-ranking.md` for benchmarks):

| Algorithm | Package | When to use |
|-----------|---------|------------|
| LightGBM LambdaRank | `lightgbm` | Primary — optimizes NDCG directly |
| XGBoost rank:ndcg | `xgboost` | May win on small data |

## The experiment loop

From `program.md`:

```bash
# 1. Review past results
cat results.tsv

# 2. Form hypothesis, edit train.py
# ...

# 3. Commit the experiment
git add train.py && git commit -m "exp: description"

# 4. Run
make run

# 5. Read NDCG output, log to results.tsv

# 6. Keep if improved, revert otherwise
git reset --hard HEAD~1  # on failure
```

## References

- [Karpathy, A. (2026). *autoresearch*.](https://github.com/karpathy/autoresearch)
- [idealo Tech Blog (2026). *One Hour, 37% Faster: Applying Autoresearch to Our Search Ranking Inference Endpoint*.](https://medium.com/idealo-tech-blog/one-hour-37-faster-applying-autoresearch-to-our-search-ranking-inference-endpoint-34cffc08e373)
- [Burges, C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research.](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [Gusak et al. (2025). *Temporal Global Split for RecSys evaluation*. RecSys 2025.](https://dl.acm.org/doi/abs/10.1145/3696410.3714975)
