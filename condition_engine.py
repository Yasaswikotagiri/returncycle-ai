"""Condition intelligence: score (0-100), grade (A-E) and an explanation of the main factors."""
from __future__ import annotations

from src.data_cleaning import normalise_record
from src.utils import CATEGORY_PROFILES, GRADE_LABELS, GRADES, clip, get_config

# Maximum points per factor (sum = 100 when a product has both a screen and a battery).
BASE_WEIGHTS = {"functional": 20, "components": 12, "cosmetic": 12, "screen": 20, "battery": 10,
                "accessories": 8, "water": 8, "usage": 4, "packaging": 3, "age": 3}
FACTOR_LABELS = {
    "functional": "Functional condition", "components": "Component health", "cosmetic": "Cosmetic condition",
    "screen": "Screen condition", "battery": "Battery health", "accessories": "Accessories complete",
    "water": "No water exposure", "usage": "Low usage wear", "packaging": "Packaging condition",
    "age": "Product age",
}


def assess_condition(rec: dict) -> dict:
    r = normalise_record(rec)
    p = CATEGORY_PROFILES[r["category"]]
    q = {
        "functional": 1 - r["functional_damage"] / 100,
        "components": r["component_health"] / 100,
        "cosmetic": 1 - r["cosmetic_damage"] / 100,
        "screen": r["screen_condition"] / 100 if r["screen_condition"] is not None else None,
        "battery": r["battery_health"] / 100 if r["battery_health"] is not None else None,
        "accessories": 1 - r["missing_accessories"] / max(1, p["max_acc"]),
        "water": 0.0 if r["water_damage"] else 1.0,
        "usage": 1 - r["usage_level"] / 100,
        "packaging": r["packaging_condition"] / 100,
        "age": 1 - min(1.0, r["product_age_days"] / r["expected_lifetime"]),
    }
    active = {k: w for k, w in BASE_WEIGHTS.items() if q[k] is not None}
    scale = 100.0 / sum(active.values())  # N/A factors (no screen / battery) are redistributed

    factors = []
    for k, w in active.items():
        mx = w * scale
        earned = mx * q[k]
        factors.append({"key": k, "label": FACTOR_LABELS[k], "earned": round(earned, 1),
                        "max": round(mx, 1), "delta": round(earned - mx, 1)})

    penalties = []
    screen_def = 1 - q["screen"] if q["screen"] is not None else 0.0
    if screen_def >= 0.5:
        penalties.append(("Critical defect: screen unusable", -8.0))
    if r["functional_damage"] >= 50:
        penalties.append(("Critical defect: major functional failure", -8.0))
    score = clip(sum(f["earned"] for f in factors) + sum(v for _, v in penalties), 0, 100)
    score = round(score, 1)

    cfg = get_config()["grading"]
    grade = "E"
    for g in ("A", "B", "C", "D"):
        if score >= cfg[g]:
            grade = g
            break
    critical = screen_def >= 0.5 or r["functional_damage"] >= 50 or r["water_damage"]
    if critical and grade in ("A", "B"):
        grade = "C"
    if r["functional_damage"] >= 75 and r["component_health"] < 40 and GRADES.index(grade) < 3:
        grade = "D"

    issues = []
    if screen_def >= 0.5:
        issues.append(f"Screen damaged / unusable (condition {r['screen_condition']:.0f}/100)")
    elif screen_def >= 0.2:
        issues.append(f"Screen scratched / degraded (condition {r['screen_condition']:.0f}/100)")
    if r["battery_health"] is not None and r["battery_health"] < 70:
        issues.append(f"Weak battery ({r['battery_health']:.0f}% health)")
    if r["functional_damage"] >= 20:
        issues.append(f"Functional damage {r['functional_damage']:.0f}%")
    if r["cosmetic_damage"] >= 25:
        issues.append(f"Visible cosmetic damage ({r['cosmetic_damage']:.0f}%)")
    if r["missing_accessories"] > 0:
        issues.append(f"{r['missing_accessories']} accessory item(s) missing")
    if r["water_damage"]:
        issues.append("Water / liquid damage detected")
    if r["packaging_condition"] < 50:
        issues.append("Packaging badly damaged")
    if r["component_health"] < 60:
        issues.append(f"Low component health ({r['component_health']:.0f}/100)")

    ordered = sorted(factors, key=lambda f: f["delta"])
    negatives = [f for f in ordered if f["delta"] <= -1.5]
    neg_keys = {f["key"] for f in negatives}
    positives = [f for f in sorted(factors, key=lambda f: -f["earned"])
                 if f["earned"] >= 0.85 * f["max"] and f["max"] >= 6 and f["key"] not in neg_keys]
    lines = [f"Condition Score: {score:.0f}/100 — Grade {grade} ({GRADE_LABELS[grade]})", "Main factors:"]
    lines += [f"- {f['label']}: {f['delta']:+.0f}" for f in negatives[:5]]
    lines += [f"- {n}: {v:+.0f}" for n, v in penalties]
    lines += [f"- {f['label']}: +{f['earned']:.0f} of {f['max']:.0f}" for f in positives[:3]]
    return {"score": score, "grade": grade, "grade_label": GRADE_LABELS[grade], "factors": factors,
            "penalties": penalties, "issues": issues, "critical": bool(critical), "explanation": "\n".join(lines),
            "screen_deficit": screen_def}
