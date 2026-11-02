"""Feature engineering for the ML models. Inspection-time information only (no pathway outcomes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import CATEGORIES

BASE_FEATURES = ["original_value", "product_age_days", "product_weight", "cosmetic_damage", "functional_damage",
                 "missing_accessories", "packaging_condition", "screen_condition", "battery_health",
                 "component_health", "water_damage", "usage_level", "estimated_processing_days"]
ECON_FEATURES = ["condition_score", "estimated_repair_cost", "estimated_resale_value"]
PATHWAY_TARGET = "best_pathway"
VALUE_TARGET = "recovered_value"
GRADE_TARGET = "condition_grade"


def build_features(df: pd.DataFrame, include_econ: bool = True) -> pd.DataFrame:
    """Return a numeric feature matrix with a stable column order."""
    X = pd.DataFrame(index=df.index)
    for c in BASE_FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    X["screen_condition"] = X["screen_condition"].fillna(-1)   # -1 = not applicable
    X["battery_health"] = X["battery_health"].fillna(-1)
    X["processing_cost"] = (pd.to_numeric(df.get("reverse_shipping_cost"), errors="coerce").fillna(0)
                            + pd.to_numeric(df.get("warehouse_processing_cost"), errors="coerce").fillna(0)
                            + pd.to_numeric(df.get("handling_cost"), errors="coerce").fillna(0))
    if include_econ:
        for c in ECON_FEATURES:
            X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        X["repair_to_value"] = X["estimated_repair_cost"] / X["original_value"].clip(lower=1)
    for cat in CATEGORIES:
        X[f"cat_{cat}"] = (df["category"] == cat).astype(int)
    return X.fillna(X.median(numeric_only=True)).fillna(0)
