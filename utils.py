"""Shared constants, paths, formatting helpers and per-category economics."""
from __future__ import annotations

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"

DATASET_LABEL = "Synthetic demonstration dataset created for ReturnCycle AI."
TAGLINE_1 = "A returned product is not necessarily a lost product."
TAGLINE_2 = "ReturnCycle AI evaluates what the product can become next."

# ----------------------------------------------------------------------------
# Config (defaults are built in, so a missing config.yaml never breaks the app)
# ----------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "demo": {"orders": 25000, "customers": 6000, "products": 800},
        "full": {"orders": 100000, "customers": 20000, "products": 2000},
        "start_date": "2025-11-01",
        "end_date": "2026-09-30",
    },
    "grading": {"A": 85, "B": 70, "C": 50, "D": 30},
    "economics": {"holding_cost_per_day": 0.0012, "outbound_share_of_inbound": 0.5},
    "optimizer_weights": {"economic": 0.50, "suitability": 0.12, "usability": 0.06,
                          "market": 0.10, "cost": 0.06, "time": 0.08, "risk": 0.08},
}
_CONFIG_CACHE: dict | None = None


def _merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in (new or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg = dict(_DEFAULT_CONFIG)
        try:
            import yaml
            with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
                cfg = _merge(_DEFAULT_CONFIG, yaml.safe_load(fh) or {})
        except Exception:
            pass
        _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


# ----------------------------------------------------------------------------
# Pathways & grades
# ----------------------------------------------------------------------------
PATHWAYS = ["resale", "refurbish", "component", "transformation", "material", "disposal"]
PATHWAY_LABELS = {
    "resale": "Resale", "refurbish": "Refurbish", "component": "Component Recovery",
    "transformation": "Transformation", "material": "Material Recovery", "disposal": "Disposal",
}
PATHWAY_HEADLINE = {
    "resale": "Resell as-is", "refurbish": "Refurbish & Resell", "component": "Recover Components",
    "transformation": "Transform / Upcycle", "material": "Material Recovery", "disposal": "Dispose",
}
LABEL_TO_KEY = {v: k for k, v in PATHWAY_LABELS.items()}
PATHWAY_COLORS = {
    "Resale": "#4C78A8", "Refurbish": "#3E8E6B", "Component Recovery": "#C08A3E",
    "Transformation": "#7C6BA6", "Material Recovery": "#6B8794", "Disposal": "#B4544B",
}
PATHWAY_DESC = {
    "resale": "Sell with minimal intervention (open-box / as-is grading).",
    "refurbish": "Repair, clean, test and resell as certified refurbished.",
    "component": "Harvest working components and sell them as spares.",
    "transformation": "Convert reusable parts/material into another product (estimated / simulated).",
    "material": "Recover metals, plastics, textiles or wood by weight.",
    "disposal": "Compliant disposal; only when other options lose more value.",
}
GRADES = ["A", "B", "C", "D", "E"]
GRADE_LABELS = {
    "A": "Resale-ready", "B": "Minor refurbishment", "C": "Repair required",
    "D": "Component recovery candidate", "E": "Material recovery / disposal candidate",
}
DISPOSITION_FOR = {
    "resale": "Restocked / open-box sale", "refurbish": "Sent to refurbisher",
    "component": "Parts harvesting", "transformation": "Upcycling partner",
    "material": "Sent to recycler", "disposal": "Scrapped / disposed",
}

RETURN_REASONS = [
    "Damaged on arrival", "Defective / not working", "Not as described", "Wrong item received",
    "Changed mind", "Size / fit issue", "Missing accessories", "Better price elsewhere",
    "Accidental damage", "Late delivery",
]
SELLER_TYPES = ["Marketplace Seller", "First-party", "Brand Store"]


# ----------------------------------------------------------------------------
# Category economics (synthetic assumptions, deliberately different per category)
# components: (name, share_of_original_value, health_source)
# transformations: (name, value_share, cost_frac, fixed_cost, min_reusable, days, demand)
# ----------------------------------------------------------------------------
def _P(**kw):
    base = dict(dep_floor=0.12, yield_=0.75, comp_demand=0.75, cosm_frac=0.06, acc_frac=0.02,
                screen_frac=0.0, battery_frac=0.0, repair_frac=0.4, has_screen=False,
                has_battery=False, max_acc=3, refurb_factor=0.82, refurb_overhead=0.025,
                material_rate=300.0, material_recovery=0.85, disp_base=60.0, disp_kg=25.0,
                queue_days=3, ret_rate=0.3)
    base.update(kw)
    base["yield"] = base.pop("yield_")
    return base


CATEGORY_PROFILES = {
    "Smartphones": _P(
        mix=0.14, ret_rate=0.30, price_median=22000, price_sigma=0.60, weight=0.2, lifetime=1100,
        age_range=(30, 700), dep_rate=0.30, has_screen=True, has_battery=True, repair_frac=0.45,
        screen_frac=0.22, battery_frac=0.08, acc_frac=0.02, refurb_factor=0.82, material_rate=1800,
        ship_base=90, ship_kg=60, wh_base=90, proc_base=150, queue_days=3, comp_demand=0.85,
        components=[("Display module", 0.22, "screen"), ("Battery", 0.07, "battery"),
                    ("Camera module", 0.10, "comp"), ("Mainboard / SoC", 0.20, "comp"),
                    ("Casing", 0.03, "casing")],
        transformations=[("Certified spare-parts kit", 0.14, 0.02, 150, 0.40, 4, 0.80),
                         ("Secondary-use device (kiosk / dashcam)", 0.10, 0.03, 250, 0.55, 5, 0.50)]),
    "Laptops": _P(
        mix=0.09, ret_rate=0.28, price_median=52000, price_sigma=0.50, weight=2.0, lifetime=1500,
        age_range=(60, 900), dep_rate=0.28, has_screen=True, has_battery=True, repair_frac=0.50,
        screen_frac=0.18, battery_frac=0.09, acc_frac=0.03, refurb_factor=0.80, refurb_overhead=0.03,
        material_rate=350, ship_base=180, ship_kg=45, wh_base=160, proc_base=400, queue_days=4,
        comp_demand=0.85,
        components=[("Display panel", 0.16, "screen"), ("SSD", 0.08, "comp"), ("RAM", 0.06, "comp"),
                    ("Keyboard", 0.04, "comp"), ("Battery", 0.06, "battery"),
                    ("Mainboard", 0.15, "comp"), ("Casing", 0.03, "casing")],
        transformations=[("Home-server / NAS kit (SSD + RAM + board)", 0.16, 0.03, 300, 0.50, 6, 0.60),
                         ("Spare-parts bundle for repair shops", 0.14, 0.02, 250, 0.40, 5, 0.75),
                         ("External SSD enclosure kit", 0.05, 0.01, 100, 0.40, 3, 0.70)]),
    "Tablets": _P(
        mix=0.06, ret_rate=0.30, price_median=28000, price_sigma=0.55, weight=0.5, lifetime=1200,
        age_range=(40, 800), dep_rate=0.30, has_screen=True, has_battery=True, repair_frac=0.45,
        screen_frac=0.22, battery_frac=0.08, acc_frac=0.02, refurb_factor=0.82, material_rate=900,
        ship_base=100, ship_kg=55, wh_base=110, proc_base=200, queue_days=3, comp_demand=0.80,
        components=[("Display module", 0.24, "screen"), ("Battery", 0.07, "battery"),
                    ("Mainboard", 0.20, "comp"), ("Camera module", 0.05, "comp"),
                    ("Casing", 0.03, "casing")],
        transformations=[("Certified spare-parts kit", 0.14, 0.02, 150, 0.40, 4, 0.75),
                         ("Digital photo-frame / smart-panel kit", 0.09, 0.03, 200, 0.55, 5, 0.50)]),
    "Headphones": _P(
        mix=0.11, ret_rate=0.38, price_median=4500, price_sigma=0.70, weight=0.3, lifetime=900,
        age_range=(20, 600), dep_rate=0.35, has_battery=True, repair_frac=0.35, battery_frac=0.10,
        acc_frac=0.04, refurb_factor=0.78, refurb_overhead=0.04, material_rate=700,
        ship_base=60, ship_kg=50, wh_base=50, proc_base=80, queue_days=3, comp_demand=0.65,
        components=[("Driver units", 0.12, "comp"), ("Battery", 0.06, "battery"),
                    ("Ear cushions / headband", 0.04, "casing"), ("PCB", 0.08, "comp")],
        transformations=[("Wired conversion / replacement-part kit", 0.12, 0.03, 60, 0.50, 3, 0.55)]),
    "Smartwatches": _P(
        mix=0.07, ret_rate=0.36, price_median=9000, price_sigma=0.60, weight=0.06, lifetime=1000,
        age_range=(30, 600), dep_rate=0.33, has_screen=True, has_battery=True, repair_frac=0.40,
        screen_frac=0.20, battery_frac=0.09, acc_frac=0.03, refurb_factor=0.80, material_rate=1500,
        ship_base=60, ship_kg=60, wh_base=55, proc_base=100, queue_days=3, comp_demand=0.75,
        components=[("Display", 0.20, "screen"), ("Battery", 0.08, "battery"),
                    ("Sensor module", 0.12, "comp"), ("Strap / casing", 0.05, "casing")],
        transformations=[("Sensor-module spare kit", 0.10, 0.03, 80, 0.50, 4, 0.55)]),
    "Cameras": _P(
        mix=0.03, ret_rate=0.30, price_median=38000, price_sigma=0.70, weight=0.8, lifetime=2000,
        age_range=(60, 1000), dep_rate=0.20, has_screen=True, has_battery=True, repair_frac=0.45,
        screen_frac=0.07, battery_frac=0.04, acc_frac=0.03, refurb_factor=0.84, refurb_overhead=0.03,
        material_rate=500, ship_base=140, ship_kg=50, wh_base=140, proc_base=300, queue_days=4,
        comp_demand=0.85,
        components=[("Lens assembly", 0.25, "comp"), ("Image sensor", 0.20, "comp"),
                    ("LCD screen", 0.06, "screen"), ("Battery", 0.04, "battery"), ("Body", 0.05, "casing")],
        transformations=[("Lens / sensor spare-parts kit", 0.20, 0.02, 300, 0.45, 5, 0.65)]),
    "Home Appliances": _P(
        mix=0.08, ret_rate=0.32, price_median=18000, price_sigma=0.70, weight=12, lifetime=3000,
        age_range=(30, 900), dep_rate=0.12, repair_frac=0.40, acc_frac=0.015, refurb_factor=0.80,
        refurb_overhead=0.03, material_rate=90, ship_base=450, ship_kg=18, wh_base=380, proc_base=500,
        queue_days=6, comp_demand=0.60, disp_base=200, disp_kg=14, max_acc=2,
        components=[("Motor", 0.20, "comp"), ("Control board", 0.10, "comp"),
                    ("Compressor / heating element", 0.15, "comp"), ("Panels", 0.06, "casing")],
        transformations=[("Repair-shop spare-parts bundle", 0.16, 0.03, 300, 0.40, 6, 0.55)]),
    "Furniture": _P(
        mix=0.08, ret_rate=0.34, price_median=14000, price_sigma=0.80, weight=25, lifetime=3650,
        age_range=(30, 900), dep_rate=0.15, repair_frac=0.35, cosm_frac=0.10, acc_frac=0.015,
        refurb_factor=0.78, refurb_overhead=0.04, material_rate=22, ship_base=800, ship_kg=14,
        wh_base=650, proc_base=700, queue_days=6, comp_demand=0.45, disp_base=300, disp_kg=10,
        max_acc=2, dep_floor=0.18,
        components=[("Frame / structure", 0.25, "frame"), ("Panels / boards", 0.15, "casing"),
                    ("Hardware & fittings", 0.06, "comp"), ("Upholstery / surface", 0.08, "casing")],
        transformations=[("Small storage unit", 0.38, 0.08, 500, 0.50, 7, 0.65),
                         ("Desk accessory set", 0.20, 0.05, 250, 0.40, 5, 0.60),
                         ("Replacement panel / component kit", 0.12, 0.04, 200, 0.45, 4, 0.50),
                         ("Decorative product", 0.07, 0.04, 200, 0.40, 5, 0.45)]),
    "Clothing": _P(
        mix=0.13, ret_rate=0.52, price_median=1800, price_sigma=0.60, weight=0.4, lifetime=700,
        age_range=(10, 400), dep_rate=0.35, dep_floor=0.10, repair_frac=0.25, cosm_frac=0.14,
        acc_frac=0.0, refurb_factor=0.75, refurb_overhead=0.05, material_rate=40, ship_base=60,
        ship_kg=40, wh_base=35, proc_base=40, queue_days=2, comp_demand=0.30, disp_base=20,
        disp_kg=8, max_acc=1,
        components=[("Fabric panels", 0.30, "casing"), ("Zips / buttons / trims", 0.04, "comp"),
                    ("Lining", 0.05, "casing")],
        transformations=[("Altered / upcycled garment", 0.50, 0.08, 40, 0.45, 4, 0.60),
                         ("Fabric accessories (bags, scrunchies)", 0.24, 0.05, 25, 0.35, 3, 0.55)]),
    "Shoes": _P(
        mix=0.08, ret_rate=0.48, price_median=3200, price_sigma=0.60, weight=0.8, lifetime=800,
        age_range=(10, 400), dep_rate=0.35, dep_floor=0.10, repair_frac=0.30, cosm_frac=0.12,
        acc_frac=0.01, refurb_factor=0.76, refurb_overhead=0.05, material_rate=35, ship_base=80,
        ship_kg=35, wh_base=45, proc_base=60, queue_days=2, comp_demand=0.35, disp_base=25,
        disp_kg=8, max_acc=1,
        components=[("Sole unit", 0.22, "comp"), ("Upper", 0.22, "casing"), ("Insole / lining", 0.05, "casing")],
        transformations=[("Sole-swap refurbished pair", 0.30, 0.10, 60, 0.60, 4, 0.50),
                         ("Upcycled accessories", 0.10, 0.05, 30, 0.45, 3, 0.45)]),
    "Kitchen Appliances": _P(
        mix=0.08, ret_rate=0.34, price_median=5500, price_sigma=0.70, weight=3.5, lifetime=2500,
        age_range=(30, 800), dep_rate=0.18, repair_frac=0.38, acc_frac=0.03, refurb_factor=0.79,
        refurb_overhead=0.04, material_rate=120, ship_base=220, ship_kg=28, wh_base=190, proc_base=250,
        queue_days=4, comp_demand=0.60, disp_base=90, disp_kg=14, max_acc=3,
        components=[("Motor", 0.22, "comp"), ("Heating element", 0.12, "comp"),
                    ("Control panel", 0.08, "comp"), ("Jar / vessel", 0.10, "casing")],
        transformations=[("Spare-parts bundle (jar + motor)", 0.16, 0.04, 120, 0.45, 4, 0.55)]),
    "Gaming Accessories": _P(
        mix=0.05, ret_rate=0.38, price_median=3500, price_sigma=0.80, weight=0.5, lifetime=1000,
        age_range=(20, 700), dep_rate=0.32, has_battery=True, repair_frac=0.38, battery_frac=0.08,
        acc_frac=0.04, refurb_factor=0.80, refurb_overhead=0.04, material_rate=600, ship_base=70,
        ship_kg=50, wh_base=55, proc_base=80, queue_days=3, comp_demand=0.70,
        components=[("Controller PCB", 0.15, "comp"), ("Analog sticks / buttons", 0.10, "comp"),
                    ("Battery", 0.05, "battery"), ("Shell", 0.06, "casing")],
        transformations=[("Repair-kit (sticks + buttons)", 0.14, 0.03, 60, 0.45, 3, 0.60)]),
}

CATEGORIES = list(CATEGORY_PROFILES.keys())


def default_logistics(category: str, weight: float, original_value: float) -> dict:
    """Deterministic logistics estimate used when a record has no logistics data."""
    p = CATEGORY_PROFILES[category]
    return {
        "reverse_shipping_cost": round(p["ship_base"] + p["ship_kg"] * weight, 2),
        "warehouse_processing_cost": round(p["wh_base"] + 0.002 * original_value, 2),
        "handling_cost": round(0.35 * p["wh_base"], 2),
        "estimated_processing_days": p["queue_days"],
    }


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
def inr(x, decimals: int = 0) -> str:
    """Indian-style grouping: ₹1,23,456."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(x) or math.isinf(x):
        return "–"
    sign = "-" if round(x, decimals) < 0 else ""
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = head + "," + tail
    return f"{sign}₹{whole}" + (f".{frac}" if frac else "")


def inr_short(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(x):
        return "–"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:.1f}K"
    return f"{sign}₹{a:.0f}"


def pct(x, d: int = 1) -> str:
    try:
        return f"{float(x):.{d}f}%"
    except (TypeError, ValueError):
        return "–"


def clip(x, lo, hi):
    return max(lo, min(hi, x))
