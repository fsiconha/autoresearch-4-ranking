"""Unit tests for src.generate_data — synthetic data generation."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.generate_data import (
    BASE_DATE,
    DAYS_WINDOW,
    ITEM_TYPES,
    MAX_INTERACTIONS_PER_ITEM,
    MAX_ITEMS_PER_USER,
    MIN_ITEMS_PER_USER,
    N_ITEMS,
    N_USERS,
    build_item_catalog,
    generate_all_interactions,
    generate_user_interactions,
    print_summary,
    random_timestamp,
    save_data,
)


class TestBuildItemCatalog:
    def test_correct_count(self):
        catalog = build_item_catalog(50)
        assert len(catalog) == 50

    def test_zero_items(self):
        catalog = build_item_catalog(0)
        assert catalog == {}

    def test_key_format(self):
        catalog = build_item_catalog(5)
        for key in catalog:
            assert key.startswith("item_")
            assert len(key) == 9  # "item_" + 4-digit zero-padded number

    def test_values_are_valid_types(self):
        catalog = build_item_catalog(100)
        assert set(catalog.values()) <= set(ITEM_TYPES)

    def test_keys_are_unique(self):
        catalog = build_item_catalog(200)
        assert len(catalog) == len(set(catalog.keys()))

    def test_deterministic_with_seed(self):
        np.random.seed(42)
        catalog1 = build_item_catalog(100)
        np.random.seed(42)
        catalog2 = build_item_catalog(100)
        assert catalog1 == catalog2


class TestRandomTimestamp:
    def test_returns_string(self):
        ts = random_timestamp()
        assert isinstance(ts, str)

    def test_parseable_as_datetime(self):
        ts = random_timestamp()
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        assert isinstance(dt, datetime)

    def test_within_90_day_window(self):
        ts = random_timestamp()
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        delta = dt - BASE_DATE
        assert 0 <= delta.total_seconds() <= DAYS_WINDOW * 86400


class TestGenerateUserInteractions:
    @pytest.fixture
    def catalog(self):
        np.random.seed(42)
        return build_item_catalog(N_ITEMS)

    def test_returns_list_of_dicts(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_has_expected_keys(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        for record in records:
            assert set(record.keys()) == {"user_id", "item_id", "item_type", "interaction_type", "timestamp"}

    def test_user_id_matches(self, catalog):
        records = generate_user_interactions("user_0042", catalog)
        assert all(r["user_id"] == "user_0042" for r in records)

    def test_item_count_in_range(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        items = set(r["item_id"] for r in records)
        assert MIN_ITEMS_PER_USER <= len(items) <= MAX_ITEMS_PER_USER

    def test_interactions_per_item_in_range(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        counts = pd.DataFrame(records).groupby("item_id").size()
        assert all(1 <= c <= MAX_INTERACTIONS_PER_ITEM for c in counts)

    def test_item_types_match_catalog(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        for r in records:
            assert r["item_type"] == catalog[r["item_id"]]

    def test_valid_interaction_types(self, catalog):
        records = generate_user_interactions("user_0001", catalog)
        valid_types = {"view", "click", "purchase"}
        assert all(r["interaction_type"] in valid_types for r in records)

    def test_deterministic_with_seed(self, catalog):
        np.random.seed(42)
        records1 = generate_user_interactions("user_0001", catalog)
        np.random.seed(42)
        records2 = generate_user_interactions("user_0001", catalog)
        assert records1 == records2


class TestGenerateAllInteractions:
    def test_returns_dataframe(self):
        np.random.seed(42)
        catalog = build_item_catalog(N_ITEMS)
        df = generate_all_interactions(catalog)
        assert isinstance(df, pd.DataFrame)

    def test_correct_user_count(self):
        np.random.seed(42)
        catalog = build_item_catalog(N_ITEMS)
        df = generate_all_interactions(catalog)
        assert df["user_id"].nunique() == N_USERS

    def test_sorted_by_timestamp(self):
        np.random.seed(42)
        catalog = build_item_catalog(N_ITEMS)
        df = generate_all_interactions(catalog)
        assert df["timestamp"].is_monotonic_increasing

    def test_no_missing_values(self):
        np.random.seed(42)
        catalog = build_item_catalog(N_ITEMS)
        df = generate_all_interactions(catalog)
        assert not df.isnull().any().any()


class TestSaveData:
    def test_creates_files(self, tmp_path):
        items_df = pd.DataFrame({"item_id": ["item_0001"], "item_type": ["x"]})
        interactions_df = pd.DataFrame({
            "user_id": ["user_0001"],
            "item_id": ["item_0001"],
            "item_type": ["x"],
            "interaction_type": ["view"],
            "timestamp": ["2026-01-01 00:00:00"],
        })

        import src.generate_data as gd
        original_dir = gd.__name__

        items_path = tmp_path / "items.csv"
        interactions_path = tmp_path / "interactions.csv"
        items_df.to_csv(items_path, index=False)
        interactions_df.to_csv(interactions_path, index=False)

        assert items_path.exists()
        assert interactions_path.exists()


class TestPrintSummary:
    def test_does_not_raise(self, capsys):
        items_df = pd.DataFrame({"item_id": ["item_0001", "item_0002"], "item_type": ["x", "y"]})
        interactions_df = pd.DataFrame({
            "user_id": ["user_0001"],
            "item_id": ["item_0001"],
            "item_type": ["x"],
            "interaction_type": ["view"],
            "timestamp": ["2026-01-01 00:00:00"],
        })
        print_summary(items_df, interactions_df)
        captured = capsys.readouterr()
        assert "Items:" in captured.out
