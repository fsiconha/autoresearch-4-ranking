"""Data loading and preprocessing for the ranking problem."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_interactions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "interactions.csv", parse_dates=["timestamp"])


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "items.csv")


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train_interactions.csv", parse_dates=["timestamp"])


def load_eval() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "eval_interactions.csv", parse_dates=["timestamp"])


def load_ideal_rankings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ideal_rankings.csv")
