"""
Ranking model training and evaluation. AGENT EDITS THIS FILE.

Objective: maximize NDCG@5 and NDCG@10 on the eval set.
The agent can modify model choice, hyperparameters, and feature construction.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from prepare import (
    build_training_dataset,
    evaluate_ndcg,
    get_feature_columns,
    load_ideal,
    load_items,
    load_train,
    print_results,
    NDCG_K_VALUES,
    RANDOM_SEED,
)


def _prepare_ltr_dataset(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    """Sort by user and return (X, y, groups) for LightGBM LambdaRank."""
    df = df.sort_values("user_id")
    X = df[feature_cols].values
    y = df["relevance"].values
    groups = df.groupby("user_id", sort=False).size().values.tolist()
    return X, y, groups


def train_model(X, y, groups, X_val, y_val, groups_val, params=None):
    """Train a LightGBM LambdaRank model. AGENT CAN MODIFY PARAMS."""
    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": NDCG_K_VALUES,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.08,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
        "lambda_l2": 0.1,
        "label_gain": [0, 1, 2, 0, 0, 5],
        "verbosity": -1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    train_ds = lgb.Dataset(X, label=y, group=groups)
    valid_ds = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_ds)

    model = lgb.train(
        default_params,
        train_ds,
        num_boost_round=800,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model


def main():
    train_df = load_train()
    items_df = load_items()
    ideal_df = load_ideal()

    df = build_training_dataset(train_df, items_df, ideal_df)
    feature_cols = get_feature_columns(df)

    # Temporal split within training set: last 20% of users as validation
    users = sorted(df["user_id"].unique())
    split_idx = int(len(users) * 0.8)
    train_users = set(users[:split_idx])
    val_users = set(users[split_idx:])

    train_mask = df["user_id"].isin(train_users)
    val_mask = df["user_id"].isin(val_users)

    X, y, groups = _prepare_ltr_dataset(df[train_mask], feature_cols)
    X_val, y_val, groups_val = _prepare_ltr_dataset(df[val_mask], feature_cols)

    model = train_model(X, y, groups, X_val, y_val, groups_val)

    # Predict and evaluate
    df_val = df[val_mask].copy()
    df_val["prediction"] = model.predict(df_val[feature_cols].values)
    ndcg_scores = evaluate_ndcg(df_val, score_col="prediction")

    print_results(ndcg_scores)


if __name__ == "__main__":
    main()
