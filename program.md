# autoresearch — Learning to Rank

This is an experiment to have an LLM autonomously improve a ranking model. The agent modifies `train.py`, runs the full train → evaluate cycle, and keeps changes that improve NDCG.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun13`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current feat branch.
3. **Read the in-scope files**:
   - `README.md` — repository context
   - `sota-ranking.md` — algorithm survey and benchmarks
   - `prepare.py` — **DO NOT MODIFY.** Fixed data loading, feature engineering, and NDCG evaluation.
   - `train.py` — **THE FILE YOU EDIT.** Model choice, hyperparameters, and feature construction.
4. **Verify data exists**: Check that `data/train_interactions.csv`, `data/ideal_rankings.csv`, and `data/items.csv` exist.
5. **Install dependencies**: `pip install lightgbm pandas numpy` (only what's needed).
6. **Confirm setup and begin experimentation**.

## Experimentation

Each experiment runs the full training pipeline:

```bash
python train.py
```

**What you CAN do:**
- Modify `train.py` — model algorithm, hyperparameters, feature construction, training strategy. Everything is fair game.
- Switch between LightGBM LambdaRank, XGBoost, or other ranking algorithms.
- Add new features derived from the training window.
- Tune hyperparameters.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the evaluation metric and feature engineering utilities.
- Use data from `eval_interactions.csv` or leak information from the eval window into training.
- Install packages beyond `lightgbm`, `xgboost`, `pandas`, `numpy`, `scikit-learn`.

**The goal is simple: maximize NDCG@5 and NDCG@10.**

## Output format

After training, `prepare.py` prints results like this:

```
---
ndcg_5: 0.234567
ndcg_10: 0.345678
```

These are the NDCG scores averaged across all users in the validation set. Higher is better (max 1.0).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 4 columns:

```
commit	ndcg_5	ndcg_10	description
```

Example:
```
commit	ndcg_5	ndcg_10	description
a1b2c3d	0.234567	0.345678	baseline LightGBM LambdaRank default params
b2c3d4e	0.251234	0.362145	increase num_leaves to 127
c3d4e5f	0.229876	0.341234	discard — add feature_interaction=True (overfit)
```

## The experiment loop

Each experiment runs on a dedicated branch (e.g. `autoresearch/jun13` or `autoresearch/jun13-v2`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit.
2. Read the current `results.tsv` to understand what's been tried so far.
3. Form a hypothesis based on: past results, `sota-ranking.md` algorithm guidance, and feature engineering intuition.
4. Edit `train.py` with your change.
5. `git add train.py && git commit -m "<brief description of experiment>"`
6. Run: `python train.py`
7. Read the NDCG output.
8. Record results in `results.tsv` (do NOT commit results.tsv).
9. If ANY ndcg_k improved (higher): **keep** the commit.
10. If ALL ndcg_k equal or worse: **revert** (`git reset --hard HEAD~1`).
11. Log the result and go to step 1.

## Feature engineering guidance

Features are built from `train_interactions.csv` (the training window only). The fixed utilities in `prepare.py` provide:

- `build_user_features(train_df)` — per-user stats (activity, engagement, click/purchase rates)
- `build_item_features(train_df)` — per-item stats (popularity, user count, rates)
- `build_user_item_features(train_df)` — joint features (interaction count, recency)
- `build_training_dataset(train_df, items_df, ideal_df)` — full feature matrix with labels

You can add new features in `train.py` by computing them from the training data and joining into the dataset. **Never use eval window data for feature computation.**

## Algorithm options (from sota-ranking.md)

1. **LightGBM LambdaRank** (current baseline) — `objective="lambdarank"`. Fast, optimizes NDCG directly.
2. **XGBoost** — `objective="rank:ndcg"`. Sometimes better on small data.
3. **Hyperparameter tuning** within LightGBM — num_leaves, learning_rate, min_data_in_leaf, lambda_l2, feature_fraction, bagging.

## Strategy tips

- Review `results.tsv` before each experiment. Don't repeat failed ideas.
- If a change improves NDCG@5 but hurts NDCG@10 (or vice versa), use your judgment — prefer improvements at the k that matters more for our use case.
- Small, incremental changes are safer than radical rewrites.
- If you're stuck, review `sota-ranking.md` for algorithm ideas or try feature engineering.
- Deleting unnecessary code that doesn't hurt NDCG is a win (simplicity matters).

## Timeout and crashes

- Each experiment should complete in under 2 minutes. If it hangs, kill it and treat as a crash.
- If a run crashes (error or OOM): fix trivial issues and re-run. If the idea is fundamentally broken, log it as `crash` in results.tsv, revert, and move on.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask if you should continue. The human might be gone and expects you to work autonomously until manually stopped. You are an autonomous researcher. If you run out of ideas, think harder — re-read `sota-ranking.md`, study the feature distributions, try combining near-misses, try a different algorithm entirely. The loop runs until interrupted.
