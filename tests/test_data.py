"""Unit tests for src.data — data loading functions."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    load_eval,
    load_ideal_rankings,
    load_interactions,
    load_items,
    load_train,
)


@pytest.fixture
def data_dir():
    return Path("data")


class TestLoadItems:
    def test_returns_dataframe(self):
        df = load_items()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_items()
        assert list(df.columns) == ["item_id", "item_type"]

    def test_not_empty(self):
        df = load_items()
        assert len(df) > 0

    def test_item_ids_are_unique(self):
        df = load_items()
        assert df["item_id"].is_unique


class TestLoadInteractions:
    def test_returns_dataframe(self):
        df = load_interactions()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_interactions()
        assert set(df.columns) >= {"user_id", "item_id", "item_type", "interaction_type", "timestamp"}

    def test_timestamp_is_datetime(self):
        df = load_interactions()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_not_empty(self):
        df = load_interactions()
        assert len(df) > 0

    def test_valid_interaction_types(self):
        df = load_interactions()
        assert set(df["interaction_type"].unique()) <= {"view", "click", "purchase"}


class TestLoadTrain:
    def test_returns_dataframe(self):
        df = load_train()
        assert isinstance(df, pd.DataFrame)

    def test_timestamp_is_datetime(self):
        df = load_train()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_subset_of_full_interactions(self):
        train = load_train()
        full = load_interactions()
        assert len(train) < len(full)


class TestLoadEval:
    def test_returns_dataframe(self):
        df = load_eval()
        assert isinstance(df, pd.DataFrame)

    def test_timestamp_is_datetime(self):
        df = load_eval()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_eval_after_train_no_overlap(self):
        train = load_train()
        eval_df = load_eval()
        train_max = train["timestamp"].max()
        eval_min = eval_df["timestamp"].min()
        assert eval_min >= train_max


class TestLoadIdealRankings:
    def test_returns_dataframe(self):
        df = load_ideal_rankings()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_ideal_rankings()
        assert set(df.columns) == {"user_id", "item_id", "relevance"}

    def test_relevance_values_are_valid(self):
        df = load_ideal_rankings()
        assert set(df["relevance"].unique()).issubset({1, 2, 5})

    def test_sorted_by_user_and_relevance_descending(self):
        df = load_ideal_rankings()
        for user_id, group in df.groupby("user_id"):
            assert group["relevance"].is_monotonic_decreasing
