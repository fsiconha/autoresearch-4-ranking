"""Generate synthetic user-item interaction data for ranking problem."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 1000
N_ITEMS = 200
ITEM_TYPES = ["x", "y", "z", "w"]
INTERACTION_WEIGHTS = {"view": 0.70, "click": 0.20, "purchase": 0.10}
INTERACTION_TYPES = list(INTERACTION_WEIGHTS.keys())
INTERACTION_PROBABILITIES = list(INTERACTION_WEIGHTS.values())
DAYS_WINDOW = 90
BASE_DATE = datetime(2026, 1, 1)
MIN_ITEMS_PER_USER = 5
MAX_ITEMS_PER_USER = 50
MAX_INTERACTIONS_PER_ITEM = 5


def build_item_catalog(n_items: int) -> dict[str, str]:
    """Return a mapping of item_id -> item_type."""
    return {
        f"item_{i:04d}": np.random.choice(ITEM_TYPES)
        for i in range(n_items)
    }


def random_timestamp() -> str:
    days = np.random.uniform(0, DAYS_WINDOW)
    seconds = np.random.randint(0, 86400)
    ts = BASE_DATE + timedelta(days=days, seconds=seconds)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def generate_user_interactions(user_id: str, item_catalog: dict[str, str]) -> list[dict]:
    n_items = np.random.randint(MIN_ITEMS_PER_USER, MAX_ITEMS_PER_USER + 1)
    chosen_items = np.random.choice(list(item_catalog.keys()), size=n_items, replace=False)

    records = []
    for item_id in chosen_items:
        n_interactions = np.random.randint(1, MAX_INTERACTIONS_PER_ITEM + 1)
        for _ in range(n_interactions):
            records.append({
                "user_id": user_id,
                "item_id": item_id,
                "item_type": item_catalog[item_id],
                "interaction_type": np.random.choice(
                    INTERACTION_TYPES, p=INTERACTION_PROBABILITIES
                ),
                "timestamp": random_timestamp(),
            })
    return records


def generate_all_interactions(item_catalog: dict[str, str]) -> pd.DataFrame:
    all_records = []
    for user_idx in range(N_USERS):
        user_id = f"user_{user_idx:04d}"
        all_records.extend(generate_user_interactions(user_id, item_catalog))

    df = pd.DataFrame(all_records)
    return df.sort_values("timestamp").reset_index(drop=True)


def save_data(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
    items_df.to_csv("data/items.csv", index=False)
    interactions_df.to_csv("data/interactions.csv", index=False)


def print_summary(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
    print(f"Items: {len(items_df)}")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Users: {interactions_df['user_id'].nunique()}")
    print(f"Date range: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
    print(f"\nItem type distribution:\n{items_df['item_type'].value_counts()}")
    print(f"\nInteraction type distribution:\n{interactions_df['interaction_type'].value_counts()}")
    print(f"\nInteractions per user stats:\n{interactions_df.groupby('user_id').size().describe()}")
    print(f"\nSample:\n{interactions_df.head(10)}")


if __name__ == "__main__":
    item_catalog = build_item_catalog(N_ITEMS)
    items_df = pd.DataFrame(
        [{"item_id": k, "item_type": v} for k, v in item_catalog.items()]
    )
    interactions_df = generate_all_interactions(item_catalog)
    save_data(items_df, interactions_df)
    print_summary(items_df, interactions_df)
