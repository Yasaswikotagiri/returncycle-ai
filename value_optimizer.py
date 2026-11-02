"""Pathway scoring + recommendation.

Score = economic value + condition suitability + component usability + market potential
        - processing cost - time penalty - operational risk      (normalised to 0-100)
"""
from __future__ import annotations

import math

from src.utils import clip, get_config

WEIGHT_LABELS = {"economic": "Economic value", "suitability": "Condition suitability",
                 "usability": "Component usability", "market": "Market potential",
                 "cost": "Processing cost (penalty)", "time": "Time (penalty)", "risk": "Operational risk (penalty)"}


def score_pathways(pathways: dict, original_value: float, weights: dict | None = None) -> dict:
    w = {**get_config()["optimizer_weights"], **(weights or {})}
    # Normalise economics over viable pathways (+ disposal) so a hopeless route cannot distort the scale.
    nets = [p["net"] for p in pathways.values() if p["viable"]]
    lo_net, hi_net = min(nets), max(nets)
    hi_s = w["economic"] + w["suitability"] + w["usability"] + w["market"]
    lo_s = -(w["cost"] + w["time"] + w["risk"])
    span = max(hi_s - lo_s, 1e-9)
    for p in pathways.values():
        econ = clip((p["net"] - lo_net) / (hi_net - lo_net), 0, 1) if hi_net > lo_net else 0.5
        if not p["viable"]:
            econ = 0.0
        cost_n = clip(p["total_cost"] / max(original_value, 1.0) * 2, 0, 1)
        time_n = clip(p["days"] / 30.0, 0, 1)
        parts = {"economic": w["economic"] * econ, "suitability": w["suitability"] * p["suitability"],
                 "usability": w["usability"] * p["usability"], "market": w["market"] * p["market"],
                 "cost": -w["cost"] * cost_n, "time": -w["time"] * time_n, "risk": -w["risk"] * p["risk"]}
        p["score_parts"] = {k: v / span * 100 for k, v in parts.items()}
        p["score"] = round(100 * (sum(parts.values()) - lo_s) / span, 1)
    return pathways


def recommend(pathways: dict) -> dict:
    """Pick the best eligible pathway. Disposal is only eligible when nothing else is viable
    or every viable alternative loses more value than disposal itself."""
    non_disp = [p for k, p in pathways.items() if k != "disposal" and p["viable"]]
    disp = pathways["disposal"]
    eligible = list(non_disp)
    if not non_disp or max(p["net"] for p in non_disp) < disp["net"]:
        eligible.append(disp)
    eligible.sort(key=lambda p: -p["score"])
    ranking = [p["key"] for p in eligible] + [k for k, _ in sorted(pathways.items(), key=lambda kv: -kv[1]["score"])
                                              if k not in {p["key"] for p in eligible}]
    best = eligible[0]
    alt = eligible[1] if len(eligible) > 1 else None
    margin = best["score"] - alt["score"] if alt else 25.0
    conf = 0.50 + 0.38 * math.tanh(margin / 12.0) - 0.20 * best["risk"]
    conf = clip(conf, 0.30, 0.97)
    return {"recommended": best["key"], "alternative": alt["key"] if alt else None,
            "confidence": round(conf, 3), "ranking": ranking, "eligible": [p["key"] for p in eligible],
            "margin": round(margin, 1)}
