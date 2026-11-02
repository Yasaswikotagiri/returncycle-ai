"""Synthetic data generator for ReturnCycle AI.

    python src/data_generator.py --mode demo     # ~25k orders, ~9k returns
    python src/data_generator.py --mode full     # ~100k orders, ~37k returns

Everything here is SYNTHETIC. No Amazon / Flipkart / private company data is used.
The demo set is "return-enriched": return rates are higher than real-world averages so
that a 25k-order sample yields enough returns (8-12k) to analyse.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.next_life_engine import analyze_dataframe  # noqa: E402
from src.utils import (CATEGORIES, CATEGORY_PROFILES, DATA_PROCESSED, DATA_RAW, DATA_SAMPLE,  # noqa: E402
                       DATASET_LABEL, DISPOSITION_FOR, LABEL_TO_KEY, PATHWAY_LABELS, SELLER_TYPES, get_config)

BRANDS = ["Aurex", "Nimbus", "Vertia", "Kestrel", "Lumo", "Orbis", "Zenta", "Halcyon", "Brio", "Tandem"]
APPAREL = {"Clothing", "Shoes"}
BULKY = {"Furniture", "Home Appliances", "Kitchen Appliances"}
REASON_W = {  # order follows RETURN_REASONS_LIST; skewed toward damage so the demo shows every pathway
    "apparel": [0.12, 0.10, 0.12, 0.05, 0.15, 0.30, 0.00, 0.04, 0.09, 0.03],
    "bulky":   [0.30, 0.18, 0.10, 0.05, 0.08, 0.04, 0.06, 0.03, 0.12, 0.04],
    "tech":    [0.24, 0.30, 0.07, 0.04, 0.08, 0.00, 0.06, 0.03, 0.15, 0.03],
}
# (cosmetic mean/sd, functional mean/sd, packaging mean/sd, P(missing accessory), P(water), usage mean)
REASON_PROFILE = {
    "Damaged on arrival":      (45, 22, 25, 25, 35, 20, 0.05, 0.03, 3),
    "Defective / not working": (12, 10, 62, 22, 80, 15, 0.08, 0.04, 25),
    "Not as described":        (8, 8, 5, 6, 85, 12, 0.06, 0.00, 8),
    "Wrong item received":     (3, 4, 2, 3, 90, 8, 0.04, 0.00, 3),
    "Changed mind":            (6, 8, 3, 4, 85, 12, 0.12, 0.00, 10),
    "Size / fit issue":        (8, 8, 3, 4, 80, 15, 0.02, 0.00, 12),
    "Missing accessories":     (6, 6, 3, 4, 85, 10, 1.00, 0.00, 8),
    "Better price elsewhere":  (4, 5, 2, 3, 90, 8, 0.05, 0.00, 4),
    "Accidental damage":       (55, 22, 45, 25, 70, 20, 0.25, 0.20, 45),
    "Late delivery":           (3, 4, 2, 3, 92, 6, 0.03, 0.00, 3),
}
CRACK_P = {"Damaged on arrival": 0.22, "Accidental damage": 0.45}
# Legacy "damaged = recycle" style policy used to simulate CURRENT practice (not the optimiser).
LEGACY = {"A": "resale", "B": "resale", "C": "material", "D": "material", "E": "disposal"}


def _grp(cat: str) -> str:
    return "apparel" if cat in APPAREL else "bulky" if cat in BULKY else "tech"


def generate_raw(n_orders: int, n_customers: int, n_products: int, seed: int = 42):
    cfg = get_config()["data"]
    rng = np.random.default_rng(seed)

    # ---- customers ------------------------------------------------------------
    customers = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n_customers + 1)],
        "return_propensity": np.clip(rng.gamma(2.0, 0.5, n_customers), 0.2, 3.0).round(3)})

    # ---- products -------------------------------------------------------------
    counts = {c: max(3, int(round(n_products * CATEGORY_PROFILES[c]["mix"]))) for c in CATEGORIES}
    rows = []
    pid = 1
    for cat, n in counts.items():
        p = CATEGORY_PROFILES[cat]
        prices = np.clip(rng.lognormal(np.log(p["price_median"]), p["price_sigma"], n),
                         0.3 * p["price_median"], 6 * p["price_median"])
        step = 100 if p["price_median"] > 5000 else 50 if p["price_median"] > 1500 else 10
        prices = (np.round(prices / step) * step).clip(min=step)
        for i in range(n):
            rows.append({"product_id": f"P{pid:05d}", "category": cat,
                         "product_name": f"{BRANDS[rng.integers(len(BRANDS))]} {cat[:-1] if cat.endswith('s') else cat} "
                                         f"{rng.choice(list('ABCDEFGHJK'))}{rng.integers(10, 99)}",
                         "original_value": float(prices[i]),
                         "product_weight": round(float(p["weight"] * rng.lognormal(0, 0.25)), 3),
                         "expected_lifetime": int(p["lifetime"] * rng.uniform(0.8, 1.2)),
                         "base_age_days": int(rng.integers(*p["age_range"])),
                         "popularity": float(rng.lognormal(0, 0.8))})
            pid += 1
    products = pd.DataFrame(rows)
    mix = products["category"].map(lambda c: CATEGORY_PROFILES[c]["mix"])
    w = products["popularity"] * mix / products.groupby("category")["popularity"].transform("sum")
    w = (w / w.sum()).values

    # ---- orders ---------------------------------------------------------------
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    span = (end - start).days
    pi = rng.choice(len(products), n_orders, p=w)
    ci = rng.integers(0, n_customers, n_orders)
    order_date = start + pd.to_timedelta(rng.integers(0, span + 1, n_orders), unit="D")
    orders = pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(1, n_orders + 1)],
        "customer_id": customers["customer_id"].values[ci], "product_id": products["product_id"].values[pi],
        "order_date": order_date, "delivery_date": order_date + pd.to_timedelta(rng.integers(2, 9, n_orders), unit="D"),
        "category": products["category"].values[pi],
        "seller_type": rng.choice(SELLER_TYPES, n_orders, p=[0.55, 0.30, 0.15])})
    orders["purchase_price"] = (products["original_value"].values[pi] * rng.uniform(0.75, 1.0, n_orders)).round(0)
    base_rate = orders["category"].map(lambda c: CATEGORY_PROFILES[c]["ret_rate"]).values
    seller_f = orders["seller_type"].map({"Marketplace Seller": 1.10, "First-party": 0.95, "Brand Store": 0.85}).values
    p_ret = np.clip(base_rate * customers["return_propensity"].values[ci] * seller_f, 0, 0.9)
    orders["is_returned"] = rng.random(n_orders) < p_ret

    # ---- returns + condition ---------------------------------------------------
    ret = orders[orders["is_returned"]].copy()
    n = len(ret)
    prod = products.set_index("product_id")
    ret = ret.join(prod[["product_name", "original_value", "product_weight", "expected_lifetime", "base_age_days"]],
                   on="product_id")
    ret["days_since_delivery"] = np.clip(rng.gamma(2.0, 4.0, n), 1, 30).round().astype(int)
    ret.loc[ret["category"].isin(APPAREL), "days_since_delivery"] = ret["days_since_delivery"].clip(upper=15)
    ret["return_date"] = ret["delivery_date"] + pd.to_timedelta(ret["days_since_delivery"], unit="D")
    ret["product_age_days"] = ret["base_age_days"] + ret["days_since_delivery"]
    reasons = np.empty(n, dtype=object)
    for g in ("apparel", "bulky", "tech"):
        m = ret["category"].map(_grp).values == g
        wt = np.array(REASON_W[g], dtype=float)
        reasons[m] = rng.choice(RETURN_REASONS_LIST, m.sum(), p=wt / wt.sum())
    ret["return_reason"] = reasons

    cos = np.zeros(n); fun = np.zeros(n); pack = np.zeros(n); usage = np.zeros(n)
    miss = np.zeros(n, dtype=int); water = np.zeros(n, dtype=int)
    crack = np.zeros(n, dtype=bool)
    for reason, (cm, cs, fm, fs, pm, ps, mp, wp, um) in REASON_PROFILE.items():
        m = reasons == reason
        k = int(m.sum())
        if not k:
            continue
        cos[m] = np.clip(rng.normal(cm, cs, k), 0, 100)
        fun[m] = np.clip(rng.normal(fm, fs, k), 0, 100)
        pack[m] = np.clip(rng.normal(pm, ps, k), 0, 100)
        usage[m] = np.clip(rng.normal(um, 10, k), 0, 100)
        miss[m] = (rng.random(k) < mp) * rng.choice([1, 1, 1, 2, 3], k)
        water[m] = (rng.random(k) < wp).astype(int)
        crack[m] = rng.random(k) < CRACK_P.get(reason, 0.02)
    cat_arr = ret["category"].values
    max_acc = np.array([CATEGORY_PROFILES[c]["max_acc"] for c in cat_arr])
    ret["cosmetic_damage"] = cos.round(1)
    ret["functional_damage"] = fun.round(1)
    ret["packaging_condition"] = pack.round(1)
    ret["usage_level"] = usage.round(1)
    ret["missing_accessories"] = np.minimum(miss, max_acc)
    ret["water_damage"] = water
    ret["component_health"] = np.clip(100 - 0.55 * fun - 25 * water + rng.normal(0, 6, n), 0, 100).round(1)
    has_screen = np.array([CATEGORY_PROFILES[c]["has_screen"] for c in cat_arr])
    has_batt = np.array([CATEGORY_PROFILES[c]["has_battery"] for c in cat_arr])
    scr = np.clip(100 - 0.35 * cos - 0.35 * fun + rng.normal(0, 4, n), 0, 100)
    scr = np.where(crack, rng.uniform(5, 40, n), scr)
    batt = np.clip(rng.normal(96 - 0.2 * usage - 0.01 * ret["product_age_days"].values - 20 * water, 6), 20, 100)
    ret["screen_condition"] = np.where(has_screen, scr.round(1), np.nan)
    ret["battery_health"] = np.where(has_batt, batt.round(1), np.nan)

    prof = lambda c, k: CATEGORY_PROFILES[c][k]  # noqa: E731
    wgt = ret["product_weight"].values
    orig = ret["original_value"].values
    ret["reverse_shipping_cost"] = [round(prof(c, "ship_base") + prof(c, "ship_kg") * w_ * rng.uniform(0.8, 1.4), 2)
                                    for c, w_ in zip(cat_arr, wgt)]
    ret["warehouse_processing_cost"] = [round(prof(c, "wh_base") * rng.uniform(0.8, 1.3) + 0.002 * o, 2)
                                        for c, o in zip(cat_arr, orig)]
    ret["handling_cost"] = [round(0.35 * prof(c, "wh_base") * rng.uniform(0.8, 1.2), 2) for c in cat_arr]
    ret["estimated_processing_days"] = np.clip(np.round(rng.normal([prof(c, "queue_days") for c in cat_arr], 1.5)), 1, 20).astype(int)

    ret = ret.sort_values("return_date").reset_index(drop=True)
    ret["return_id"] = [f"RET-{i:06d}" for i in range(1, len(ret) + 1)]
    ret["customer_return_history"] = ret.groupby("customer_id").cumcount()
    tot_orders = orders.groupby("customer_id").size()
    tot_rets = ret.groupby("customer_id").size()
    ret["return_frequency"] = (ret["customer_id"].map(tot_rets) / ret["customer_id"].map(tot_orders)).round(3)
    ret = ret.drop(columns=["base_age_days"])
    return customers, products.drop(columns=["popularity", "base_age_days"]), orders, ret


RETURN_REASONS_LIST = ["Damaged on arrival", "Defective / not working", "Not as described", "Wrong item received",
                       "Changed mind", "Size / fit issue", "Missing accessories", "Better price elsewhere",
                       "Accidental damage", "Late delivery"]


def add_outcomes(ret: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Run every return through the engine and simulate what CURRENT practice would have done."""
    rng = np.random.default_rng(seed + 1)
    eng = analyze_dataframe(ret)
    df = pd.concat([ret, eng], axis=1)
    best_key = df["best_pathway"].map(LABEL_TO_KEY)
    legacy_key = df["condition_grade"].map(LEGACY)
    use_best = rng.random(len(df)) < 0.15  # a minority of teams already choose well
    actual_key = np.where(use_best, best_key, legacy_key)
    df["recovery_pathway"] = [PATHWAY_LABELS[k] for k in actual_key]
    df["actual_disposition"] = [DISPOSITION_FOR[k] for k in actual_key]
    nets = np.array([df.loc[i, f"net_{k}"] for i, k in enumerate(actual_key)])
    df["recovered_value"] = (nets * rng.lognormal(0, 0.10, len(df))).round(2)
    return df


def build(mode: str = "demo", seed: int | None = None, orders_override: int | None = None, write: bool = True):
    cfg = get_config()
    seed = cfg["seed"] if seed is None else seed
    size = dict(cfg["data"][mode])
    if orders_override:
        scale = orders_override / size["orders"]
        size = {"orders": orders_override, "customers": max(200, int(size["customers"] * scale)),
                "products": max(60, int(size["products"] * scale))}
    t0 = time.time()
    customers, products, orders, ret = generate_raw(size["orders"], size["customers"], size["products"], seed)
    returns = add_outcomes(ret, seed)
    returns["order_date"] = pd.to_datetime(returns["order_date"])
    if write:
        for d in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLE):
            d.mkdir(parents=True, exist_ok=True)
        customers.to_csv(DATA_RAW / "customers.csv.gz", index=False)
        products.to_csv(DATA_RAW / "products.csv.gz", index=False)
        orders.drop(columns=["is_returned"]).to_csv(DATA_RAW / "orders.csv.gz", index=False)
        ret.to_csv(DATA_RAW / "returns.csv.gz", index=False)
        returns.to_csv(DATA_PROCESSED / "returns_enriched.csv.gz", index=False)
        (orders.groupby([orders["order_date"].dt.normalize().rename("order_date"), "category", "seller_type"])
         .size().rename("orders").reset_index().to_csv(DATA_PROCESSED / "orders_summary.csv.gz", index=False))
        returns.sample(min(1500, len(returns)), random_state=seed).sort_values("return_date") \
            .to_csv(DATA_SAMPLE / "sample_returns.csv", index=False)
        meta = {"label": DATASET_LABEL, "mode": mode, "seed": seed, "orders": int(len(orders)),
                "customers": int(len(customers)), "products": int(len(products)), "returns": int(len(returns)),
                "note": "Return-enriched synthetic sample; return rates are higher than real-world averages.",
                "generated_seconds": round(time.time() - t0, 1)}
        (DATA_PROCESSED / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return customers, products, orders, returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic ReturnCycle AI dataset.")
    ap.add_argument("--mode", choices=["demo", "full"], default="demo")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    c, p, o, r = build(args.mode, args.seed)
    print(DATASET_LABEL)
    print(f"mode={args.mode}: {len(o):,} orders · {len(c):,} customers · {len(p):,} products · {len(r):,} returns "
          f"({len(r) / len(o):.1%} of orders)")
    print(f"Files written to {DATA_RAW.parent}")
