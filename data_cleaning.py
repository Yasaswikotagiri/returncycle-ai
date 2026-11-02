"""Input validation / cleaning for both single records and whole dataframes."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils import CATEGORY_PROFILES, CATEGORIES, clip, default_logistics

REQUIRED_COLUMNS = [
    "return_id", "order_id", "customer_id", "product_id", "category", "original_value",
    "return_reason", "return_date", "order_date", "cosmetic_damage", "functional_damage",
    "component_health", "condition_score", "condition_grade", "best_pathway", "best_net_recovery",
]


def _f(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def normalise_record(rec: dict) -> dict:
    """Validate one product record and fill missing fields with safe defaults.

    Raises ValueError with a friendly message for unusable input.
    """
    cat = rec.get("category")
    if cat not in CATEGORY_PROFILES:
        raise ValueError(f"Unknown product category: {cat!r}. Choose one of: {', '.join(CATEGORIES)}.")
    p = CATEGORY_PROFILES[cat]
    orig = _f(rec.get("original_value"))
    if orig is None or orig <= 0:
        raise ValueError("Original value must be a positive number.")

    r = dict(rec)
    r["category"] = cat
    r["original_value"] = orig
    r["product_age_days"] = clip(_f(rec.get("product_age_days"), 0.0), 0, 20000)
    r["product_weight"] = max(0.01, _f(rec.get("product_weight"), p["weight"]))
    r["expected_lifetime"] = max(30.0, _f(rec.get("expected_lifetime"), p["lifetime"]))
    r["cosmetic_damage"] = clip(_f(rec.get("cosmetic_damage"), 0.0), 0, 100)
    r["functional_damage"] = clip(_f(rec.get("functional_damage"), 0.0), 0, 100)
    r["missing_accessories"] = int(clip(round(_f(rec.get("missing_accessories"), 0.0)), 0, p["max_acc"]))
    r["packaging_condition"] = clip(_f(rec.get("packaging_condition"), 100.0), 0, 100)
    r["component_health"] = clip(_f(rec.get("component_health"), 100.0), 0, 100)
    r["water_damage"] = 1 if _f(rec.get("water_damage"), 0.0) > 0.5 else 0
    r["usage_level"] = clip(_f(rec.get("usage_level"), 0.0), 0, 100)
    r["screen_condition"] = clip(_f(rec.get("screen_condition"), 100.0), 0, 100) if p["has_screen"] else None
    r["battery_health"] = clip(_f(rec.get("battery_health"), 100.0), 0, 100) if p["has_battery"] else None

    dl = default_logistics(cat, r["product_weight"], orig)
    for k, default in dl.items():
        v = _f(rec.get(k))
        r[k] = max(0.0, v) if v is not None else default
    r["estimated_processing_days"] = clip(r["estimated_processing_days"], 0, 60)

    ro = _f(rec.get("repair_cost_override"))
    r["repair_cost_override"] = ro if (ro is not None and ro > 0) else None
    return r


def validate_columns(df: pd.DataFrame, required=None) -> list[str]:
    required = required or REQUIRED_COLUMNS
    return [c for c in required if c not in df.columns]


def clean_returns(df: pd.DataFrame):
    """Clean the raw return table. Returns (clean_df, report_dict)."""
    report = {"rows_in": int(len(df))}
    df = df.copy()
    if "return_id" in df.columns:
        n0 = len(df)
        df = df.drop_duplicates(subset="return_id")
        report["duplicates_removed"] = int(n0 - len(df))
    for c in ("order_date", "delivery_date", "return_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = [c for c in df.columns if c not in (
        "return_id", "order_id", "customer_id", "product_id", "category", "seller_type", "return_reason",
        "product_name", "order_date", "delivery_date", "return_date", "condition_grade", "best_pathway",
        "alt_pathway", "actual_disposition", "recovery_pathway")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    mask = df["category"].isin(CATEGORIES) & (df["original_value"] > 0)
    if "return_date" in df.columns:
        mask &= df["return_date"].notna()
    df = df[mask]
    report["invalid_rows_removed"] = int(n0 - len(df))
    for c in ("cosmetic_damage", "functional_damage", "component_health", "usage_level", "packaging_condition"):
        if c in df.columns:
            df[c] = df[c].clip(0, 100).fillna(df[c].median() if df[c].notna().any() else 0)
    for c in ("screen_condition", "battery_health"):  # NaN is meaningful (not applicable)
        if c in df.columns:
            df[c] = df[c].clip(0, 100)
    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report
