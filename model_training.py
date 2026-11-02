"""Train, evaluate and save the three ReturnCycle AI models.

    python src/model_training.py

1. Condition grade classifier   (inspection features  -> grade A-E)
2. Recovery pathway classifier  (inspection + early economics -> best next life)
3. Recovery value regressor     (same features -> recovered value in INR)

All metrics are computed on a held-out test split and are NOT fabricated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,  # noqa: E402
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from src.data_cleaning import clean_returns  # noqa: E402
from src.feature_engineering import GRADE_TARGET, PATHWAY_TARGET, VALUE_TARGET, build_features  # noqa: E402
from src.utils import CATEGORIES, DATA_PROCESSED, DATA_SAMPLE, MODELS_DIR  # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAVE_XGB = False


def _clf(n_classes: int):
    if HAVE_XGB:
        return XGBClassifier(n_estimators=160, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                             eval_metric="mlogloss" if n_classes > 2 else "logloss", n_jobs=2, random_state=42,
                             verbosity=0)
    return RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=3, n_jobs=2, random_state=42)


def _reg():
    if HAVE_XGB:
        return XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                            n_jobs=2, random_state=42, verbosity=0)
    return RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=3, n_jobs=2, random_state=42)


def load_training_data() -> pd.DataFrame:
    for path in (DATA_PROCESSED / "returns_enriched.csv.gz", DATA_SAMPLE / "sample_returns.csv"):
        if path.exists():
            return pd.read_csv(path, parse_dates=["order_date", "return_date"])
    raise FileNotFoundError("No dataset found. Run: python src/data_generator.py --mode demo")


def _split(X, y, seed=42, stratify=True):
    strat = y if stratify and pd.Series(y).value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)


def _importance(model, cols):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    s = pd.Series(imp, index=cols).sort_values(ascending=False).head(15)
    return {k: float(v) for k, v in s.items()}


def _classification(name, target, df, X, verbose):
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))
    Xtr, Xte, ytr, yte = _split(X, y)
    # Re-encode so the training labels are contiguous (needed by XGBoost)
    present = np.unique(ytr)
    remap = {old: i for i, old in enumerate(present)}
    model = _clf(len(present))
    model.fit(Xtr, np.array([remap[v] for v in ytr]))
    inv = {i: old for old, i in remap.items()}
    pred = np.array([inv[int(v)] for v in model.predict(Xte)])
    classes = list(le.classes_)
    majority = float(pd.Series(yte).value_counts(normalize=True).iloc[0])
    cm = confusion_matrix(yte, pred, labels=range(len(classes)))
    m = {"model": type(model).__name__, "target": target, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
         "accuracy": float(accuracy_score(yte, pred)),
         "precision_weighted": float(precision_score(yte, pred, average="weighted", zero_division=0)),
         "recall_weighted": float(recall_score(yte, pred, average="weighted", zero_division=0)),
         "f1_weighted": float(f1_score(yte, pred, average="weighted", zero_division=0)),
         "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
         "majority_class_baseline_accuracy": majority, "classes": classes, "confusion_matrix": cm.tolist(),
         "feature_importance": _importance(model, X.columns)}
    if verbose:
        print(f"  {name}: acc={m['accuracy']:.3f} (baseline {majority:.3f}) f1={m['f1_weighted']:.3f}")
    bundle = {"model": model, "label_encoder": le, "features": list(X.columns), "remap_inv": inv}
    return bundle, m


def train_all(df: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("1/6 Loading data …")
    df = load_training_data() if df is None else df
    if verbose:
        print("2/6 Cleaning …")
    df, report = clean_returns(df)
    if len(df) < 200:
        raise ValueError("Need at least 200 returns to train. Generate the demo dataset first.")
    if verbose:
        print(f"3/6 Engineering features … ({len(df):,} rows, cleaning report: {report})")
    X_cond = build_features(df, include_econ=False)
    X_full = build_features(df, include_econ=True)

    if verbose:
        print("4/6 Training + 5/6 evaluating models …")
    cond_bundle, cond_m = _classification("condition grade", GRADE_TARGET, df, X_cond, verbose)
    rec_bundle, rec_m = _classification("recovery pathway", PATHWAY_TARGET, df, X_full, verbose)

    y = df[VALUE_TARGET].astype(float).values
    Xtr, Xte, ytr, yte = _split(X_full, y, stratify=False)
    reg = _reg().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    val_m = {"model": type(reg).__name__, "target": VALUE_TARGET, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
             "rmse": float(np.sqrt(mean_squared_error(yte, pred))), "mae": float(mean_absolute_error(yte, pred)),
             "r2": float(r2_score(yte, pred)), "baseline_rmse_predict_mean": float(np.sqrt(np.mean((yte - ytr.mean()) ** 2))),
             "feature_importance": _importance(reg, X_full.columns)}
    if verbose:
        print(f"  recovery value: RMSE=₹{val_m['rmse']:,.0f} R²={val_m['r2']:.3f}")
    val_bundle = {"model": reg, "features": list(X_full.columns)}

    if verbose:
        print("6/6 Saving models …")
    joblib.dump(cond_bundle, MODELS_DIR / "condition_model.pkl")
    joblib.dump(rec_bundle, MODELS_DIR / "recovery_model.pkl")
    joblib.dump(val_bundle, MODELS_DIR / "value_model.pkl")
    prep = {"categories": CATEGORIES, "condition_features": list(X_cond.columns),
            "full_features": list(X_full.columns), "cleaning_report": report}
    joblib.dump(prep, MODELS_DIR / "preprocessing.pkl")
    metrics = {"condition": cond_m, "recovery": rec_m, "value": val_m, "n_rows": int(len(df)),
               "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
               "algorithm": "XGBoost" if HAVE_XGB else "RandomForest (XGBoost not installed)",
               "note": "Trained on synthetic data; metrics describe how well the model reproduces the synthetic engine."}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if verbose:
        print(f"Done in {metrics['seconds']}s. Metrics saved to models/metrics.json")
    return metrics


if __name__ == "__main__":
    train_all()
