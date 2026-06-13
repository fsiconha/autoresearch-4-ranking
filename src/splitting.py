"""Temporal Global Split (GTS) and ideal ranking generation.

Implements a hard temporal cutoff to avoid look-ahead bias, following
Gusak et al., RecSys 2025. The ideal ranking is built exclusively from
the evaluation window using max-interaction aggregation.
"""

from datetime import timedelta

import pandas as pd

SCORE_MAP = {"view": 1, "click": 2, "purchase": 5}
TRAIN_RATIO = 0.80


def compute_cutoff(interactions: pd.DataFrame) -> pd.Timestamp:
    start = interactions["timestamp"].min()
    end = interactions["timestamp"].max()
    return start + (end - start) * TRAIN_RATIO


def temporal_split(
    interactions: pd.DataFrame, cutoff: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train (before cutoff) and eval (after cutoff)."""
    if cutoff is None:
        cutoff = compute_cutoff(interactions)
    train = interactions[interactions["timestamp"] < cutoff].copy()
    eval_df = interactions[interactions["timestamp"] >= cutoff].copy()
    return train, eval_df


def build_ideal_rankings(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-user ideal ranking from evaluation window interactions.

    Aggregates interactions for each (user, item) by taking the strongest
    signal (purchase > click > view), then sorts by score descending.
    """
    scored = eval_df.copy()
    scored["relevance"] = scored["interaction_type"].map(SCORE_MAP)

    relevance = (
        scored.groupby(["user_id", "item_id"])["relevance"]
        .max()
        .reset_index()
        .sort_values(["user_id", "relevance"], ascending=[True, False])
    )

    return relevance.reset_index(drop=True)


def generate_splits() -> None:
    """Run the full GTS pipeline and save train, eval, and ideal rankings."""
    interactions = pd.read_csv(
        "data/interactions.csv", parse_dates=["timestamp"]
    )

    cutoff = compute_cutoff(interactions)
    train, eval_df = temporal_split(interactions, cutoff)
    ideal = build_ideal_rankings(eval_df)

    train.to_csv("data/train_interactions.csv", index=False)
    eval_df.to_csv("data/eval_interactions.csv", index=False)
    ideal.to_csv("data/ideal_rankings.csv", index=False)

    n_users_with_interactions = ideal["user_id"].nunique()
    n_items_in_ideal = ideal["item_id"].nunique()

    print(f"Cutoff date:        {cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training window:    {len(train):>8,} interactions")
    print(f"Evaluation window:  {len(eval_df):>8,} interactions")
    print(f"Ideal ranking rows: {len(ideal):>8,} (user, item) pairs")
    print(f"Users with eval interactions: {n_users_with_interactions}")
    print(f"Items in ideal rankings:      {n_items_in_ideal}")
    print(f"\nRelevance distribution:")
    print(ideal["relevance"].value_counts().sort_index())


if __name__ == "__main__":
    generate_splits()
