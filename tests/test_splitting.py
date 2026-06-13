"""Unit tests for src.splitting — Temporal Global Split and ideal rankings."""

from datetime import datetime

import pandas as pd
import pytest

from src.splitting import (
    SCORE_MAP,
    TRAIN_RATIO,
    build_ideal_rankings,
    compute_cutoff,
    temporal_split,
)


def make_interactions(timestamps: list[str]) -> pd.DataFrame:
    """Helper to create minimal interactions DataFrame with given timestamps."""
    return pd.DataFrame({
        "user_id": ["user_0001"] * len(timestamps),
        "item_id": [f"item_{i:04d}" for i in range(len(timestamps))],
        "item_type": ["x"] * len(timestamps),
        "interaction_type": ["view"] * len(timestamps),
        "timestamp": pd.to_datetime(timestamps),
    })


class TestScoreMap:
    def test_view_is_1(self):
        assert SCORE_MAP["view"] == 1

    def test_click_is_2(self):
        assert SCORE_MAP["click"] == 2

    def test_purchase_is_5(self):
        assert SCORE_MAP["purchase"] == 5

    def test_ordinal_relationship(self):
        assert SCORE_MAP["view"] < SCORE_MAP["click"] < SCORE_MAP["purchase"]


class TestTrainRatio:
    def test_is_80_percent(self):
        assert TRAIN_RATIO == 0.80


class TestComputeCutoff:
    def test_returns_timestamp(self):
        df = make_interactions(["2026-01-01", "2026-01-31"])
        cutoff = compute_cutoff(df)
        assert isinstance(cutoff, pd.Timestamp)

    def test_splits_at_80_percent(self):
        df = make_interactions(["2026-01-01 00:00:00", "2026-01-11 00:00:00"])
        cutoff = compute_cutoff(df)
        expected = pd.Timestamp("2026-01-01") + (pd.Timestamp("2026-01-11") - pd.Timestamp("2026-01-01")) * 0.80
        assert cutoff == expected

    def test_uniform_timestamps(self):
        df = make_interactions(["2026-01-01", "2026-04-01"])
        cutoff = compute_cutoff(df)
        expected = pd.Timestamp("2026-01-01") + pd.Timedelta(days=90 * 0.80)
        assert cutoff == expected


class TestTemporalSplit:
    def test_returns_two_dataframes(self):
        df = make_interactions(["2026-01-01", "2026-01-15"])
        cutoff = pd.Timestamp("2026-01-10")
        train, eval_df = temporal_split(df, cutoff)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(eval_df, pd.DataFrame)

    def test_train_before_cutoff(self):
        df = make_interactions(["2026-01-01", "2026-01-15", "2026-01-20"])
        cutoff = pd.Timestamp("2026-01-10")
        train, _ = temporal_split(df, cutoff)
        assert (train["timestamp"] < cutoff).all()

    def test_eval_after_or_equal_cutoff(self):
        df = make_interactions(["2026-01-01", "2026-01-15", "2026-01-20"])
        cutoff = pd.Timestamp("2026-01-10")
        _, eval_df = temporal_split(df, cutoff)
        assert (eval_df["timestamp"] >= cutoff).all()

    def test_no_data_lost(self):
        df = make_interactions(["2026-01-01", "2026-01-05", "2026-01-15", "2026-01-20"])
        cutoff = pd.Timestamp("2026-01-10")
        train, eval_df = temporal_split(df, cutoff)
        assert len(train) + len(eval_df) == len(df)

    def test_none_cutoff_computes_automatically(self):
        df = make_interactions(["2026-01-01", "2026-01-31"])
        train, eval_df = temporal_split(df, None)
        assert len(train) + len(eval_df) == len(df)

    def test_all_before_cutoff(self):
        df = make_interactions(["2026-01-01", "2026-01-05"])
        cutoff = pd.Timestamp("2026-02-01")
        train, eval_df = temporal_split(df, cutoff)
        assert len(train) == len(df)
        assert len(eval_df) == 0

    def test_all_after_cutoff(self):
        df = make_interactions(["2026-02-01", "2026-02-05"])
        cutoff = pd.Timestamp("2026-01-01")
        train, eval_df = temporal_split(df, cutoff)
        assert len(train) == 0
        assert len(eval_df) == len(df)


class TestBuildIdealRankings:
    def test_returns_dataframe_with_expected_columns(self):
        df = pd.DataFrame({
            "user_id": ["user_0001", "user_0001"],
            "item_id": ["item_0001", "item_0002"],
            "interaction_type": ["view", "purchase"],
            "timestamp": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        })
        result = build_ideal_rankings(df)
        assert list(result.columns) == ["user_id", "item_id", "relevance"]

    def test_max_relevance_per_user_item(self):
        df = pd.DataFrame({
            "user_id": ["user_0001", "user_0001"],
            "item_id": ["item_0001", "item_0001"],
            "interaction_type": ["view", "purchase"],
            "timestamp": pd.to_datetime(["2026-02-01", "2026-02-02"]),
        })
        result = build_ideal_rankings(df)
        assert len(result) == 1
        assert result.iloc[0]["relevance"] == 5

    def test_sorted_by_user_asc_relevance_desc(self):
        df = pd.DataFrame({
            "user_id": ["user_0002", "user_0001", "user_0001"],
            "item_id": ["item_0001", "item_0001", "item_0002"],
            "interaction_type": ["view", "purchase", "click"],
            "timestamp": pd.to_datetime(["2026-02-01", "2026-02-01", "2026-02-02"]),
        })
        result = build_ideal_rankings(df)
        users = result["user_id"].tolist()
        assert users == sorted(users)
        for _, group in result.groupby("user_id"):
            assert group["relevance"].is_monotonic_decreasing

    def test_empty_eval_returns_empty(self):
        df = pd.DataFrame({
            "user_id": [],
            "item_id": [],
            "interaction_type": [],
            "timestamp": pd.to_datetime([]),
        })
        result = build_ideal_rankings(df)
        assert len(result) == 0

    def test_single_interaction(self):
        df = pd.DataFrame({
            "user_id": ["user_0001"],
            "item_id": ["item_0001"],
            "interaction_type": ["click"],
            "timestamp": pd.to_datetime(["2026-02-01"]),
        })
        result = build_ideal_rankings(df)
        assert len(result) == 1
        assert result.iloc[0]["relevance"] == 2

    def test_multiple_users_independent_ranking(self):
        df = pd.DataFrame({
            "user_id": ["user_0001", "user_0001", "user_0002"],
            "item_id": ["item_0001", "item_0002", "item_0001"],
            "interaction_type": ["view", "purchase", "click"],
            "timestamp": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-01"]),
        })
        result = build_ideal_rankings(df)
        assert len(result) == 3
        user1 = result[result["user_id"] == "user_0001"]
        assert user1.iloc[0]["relevance"] >= user1.iloc[1]["relevance"]
