"""Data loading and preprocessing for the ranking problem."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_interactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "interactions.csv", parse_dates=["timestamp"])
    return df


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "items.csv")
