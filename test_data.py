import pandas as pd

from src.data_cleaning import REQUIRED_COLUMNS, clean_returns, validate_columns
from src.utils import CATEGORIES


def test_columns_present(small_data):
    returns = small_data[3]
    assert validate_columns(returns) == []
    for c in ("return_id", "reverse_shipping_cost", "estimated_resale_value", "recovered_value", "actual_disposition"):
        assert c in returns.columns


def test_no_impossible_values(small_data):
    r = small_data[3]
    assert (r["original_value"] > 0).all()
    assert (r["purchase_price"] > 0).all()
    assert (r[["reverse_shipping_cost", "warehouse_processing_cost", "handling_cost"]] >= 0).all().all()
    assert r["condition_score"].between(0, 100).all()


def test_dates_and_categories(small_data):
    r = small_data[3]
    assert (pd.to_datetime(r["return_date"]) >= pd.to_datetime(r["delivery_date"])).all()
    assert (pd.to_datetime(r["delivery_date"]) >= pd.to_datetime(r["order_date"])).all()
    assert set(r["category"]).issubset(set(CATEGORIES))
    assert r["return_id"].is_unique


def test_cleaning_removes_bad_rows(small_data):
    r = small_data[3].copy().head(50)
    r.loc[r.index[0], "original_value"] = -5
    dup = pd.concat([r, r.head(3)])
    cleaned, rep = clean_returns(dup)
    assert rep["duplicates_removed"] == 3 and rep["invalid_rows_removed"] == 1
    assert (cleaned["original_value"] > 0).all()
