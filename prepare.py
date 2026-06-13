"""
Fixed data loading, feature engineering, and evaluation harness for autoresearch.

DO NOT MODIFY THIS FILE. The agent edits train.py only.
This file is the ground truth: data loading, metric computation, and evaluation.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = "data"
NDCG_K_VALUES = [5, 10]
RANDOM_SEED = 42

INTERACTION_WEIGHT = {"view": 1, "click": 2, "purchase": 5}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/train_interactions.csv", parse_dates=["timestamp"])


def load_ideal() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/ideal_rankings.csv")


def load_items() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/items.csv")


# ---------------------------------------------------------------------------
# Feature engineering (fixed utilities — agent calls these in train.py)
# ---------------------------------------------------------------------------

def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user statistics from the training window."""
    df = train_df.copy()
    df["weight"] = df["interaction_type"].map(INTERACTION_WEIGHT)

    user = df.groupby("user_id").agg(
        total_interactions=("timestamp", "count"),
        unique_items=("item_id", "nunique"),
        active_days=("timestamp", lambda x: x.dt.date.nunique()),
        total_weight=("weight", "sum"),
        avg_weight=("weight", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    ).reset_index()

    user["days_active"] = user["active_days"]
    user["interactions_per_day"] = user["total_interactions"] / user["active_days"].clip(lower=1)
    user["repeat_ratio"] = 1 - (user["unique_items"] / user["total_interactions"].clip(lower=1))

    for itype in ["view", "click", "purchase"]:
        counts = (
            df[df["interaction_type"] == itype]
            .groupby("user_id").size()
            .reindex(user["user_id"], fill_value=0).values
        )
        user[f"n_{itype}"] = counts

    user["click_rate"] = user["n_click"] / user["total_interactions"].clip(lower=1)
    user["purchase_rate"] = user["n_purchase"] / user["total_interactions"].clip(lower=1)

    return user


def build_item_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-item statistics from the training window."""
    df = train_df.copy()
    df["weight"] = df["interaction_type"].map(INTERACTION_WEIGHT)

    item = df.groupby("item_id").agg(
        total_interactions=("timestamp", "count"),
        unique_users=("user_id", "nunique"),
        total_weight=("weight", "sum"),
    ).reset_index()

    item["popularity"] = item["total_interactions"]
    item["avg_weight"] = item["total_weight"] / item["total_interactions"].clip(lower=1)

    for itype in ["view", "click", "purchase"]:
        counts = (
            df[df["interaction_type"] == itype]
            .groupby("item_id").size()
            .reindex(item["item_id"], fill_value=0).values
        )
        item[f"n_{itype}"] = counts

    item["click_rate"] = item["n_click"] / item["total_interactions"].clip(lower=1)
    item["purchase_rate"] = item["n_purchase"] / item["total_interactions"].clip(lower=1)

    return item


def build_user_item_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user-item interaction features from the training window."""
    df = train_df.copy()
    df["weight"] = df["interaction_type"].map(INTERACTION_WEIGHT)

    ui = df.groupby(["user_id", "item_id"]).agg(
        interaction_count=("timestamp", "count"),
        max_weight=("weight", "max"),
        avg_weight=("weight", "mean"),
        first_interaction=("timestamp", "min"),
        last_interaction=("timestamp", "max"),
    ).reset_index()

    ui["interaction_span_days"] = (
        (ui["last_interaction"] - ui["first_interaction"]).dt.total_seconds() / 86400
    )
    ui["recency_days"] = (
        (df["timestamp"].max() - ui["last_interaction"]).dt.total_seconds() / 86400
    )

    return ui


def build_training_dataset(
    train_df: pd.DataFrame, items_df: pd.DataFrame, ideal_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the full feature matrix with relevance labels.

    For each (user, item) pair in the training window, compute features and
    attach the relevance label from the ideal rankings (0 if not present).
    """
    user_feat = build_user_features(train_df)
    item_feat = build_item_features(train_df)
    ui_feat = build_user_item_features(train_df)

    pairs = ui_feat[["user_id", "item_id"]].copy()

    pairs = pairs.merge(items_df, on="item_id", how="left")
    pairs = pairs.merge(user_feat, on="user_id", how="left")
    pairs = pairs.merge(item_feat, on="item_id", how="left")
    pairs = pairs.merge(ui_feat, on=["user_id", "item_id"], how="left")

    labels = ideal_df.set_index(["user_id", "item_id"])["relevance"]
    pairs["relevance"] = pairs.set_index(["user_id", "item_id"]).index.map(labels).fillna(0).values

    return pairs


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature column names."""
    exclude = {"user_id", "item_id", "item_type", "relevance",
               "first_seen", "last_seen", "first_interaction", "last_interaction"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("int64", "float64")]


# ---------------------------------------------------------------------------
# NDCG computation (fixed evaluation metric)
# ---------------------------------------------------------------------------

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = np.asarray(relevance, dtype=float)[:k]
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    dcg = dcg_at_k(y_true_sorted, k)
    idcg = dcg_at_k(np.sort(y_true)[::-1], k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ndcg(
    df: pd.DataFrame, score_col: str = "prediction", k_values: list[int] | None = None
) -> dict[int, float]:
    """Compute NDCG@k averaged across all users.

    df must have columns: user_id, relevance, and the score column.
    """
    if k_values is None:
        k_values = NDCG_K_VALUES
    results = {}
    for k in k_values:
        ndcg_scores = df.groupby("user_id").apply(
            lambda g: ndcg_at_k(g["relevance"].values, g[score_col].values, k),
            include_groups=False,
        )
        results[k] = float(ndcg_scores.mean())
    return results


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_results(ndcg_scores: dict[int, float]) -> None:
    print("---")
    for k, score in sorted(ndcg_scores.items()):
        print(f"ndcg_{k}: {score:.6f}")
