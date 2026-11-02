"""Recovery economics: market value, repair estimate, components, transformations and
the net-recovery calculation for each of the six pathways.

    gross_recovery_value - repair - refurbishment - processing - transformation
    - logistics - warehouse - disposal - expected_loss  =  estimated_net_recovery
"""
from __future__ import annotations

from src.utils import CATEGORY_PROFILES, PATHWAY_LABELS, clip, get_config

COST_KEYS = ["repair", "refurbishment", "processing", "transformation", "logistics",
             "warehouse", "disposal", "expected_loss"]
COST_LABELS = {
    "repair": "Repair cost", "refurbishment": "Refurbishment / testing", "processing": "Component / material processing",
    "transformation": "Transformation cost", "logistics": "Logistics (reverse + handling + outbound)",
    "warehouse": "Warehouse & holding", "disposal": "Disposal cost", "expected_loss": "Expected loss (risk)",
}
OVERRIDE_KEYS = ["repair_cost", "resale_value", "refurbished_value", "refurbishment_cost",
                 "logistics_cost", "component_value", "transformation_value", "processing_days"]


def market_factor(r: dict, p: dict) -> float:
    years = r["product_age_days"] / 365.0
    return max(p["dep_floor"], 1 - p["dep_rate"] * years) * (1 - 0.2 * r["usage_level"] / 100)


def repair_estimate(r: dict, p: dict) -> dict:
    o = r["original_value"]
    items = {}
    if r["screen_condition"] is not None and p["screen_frac"] > 0:
        items["Screen replacement"] = o * p["screen_frac"] * (100 - r["screen_condition"]) / 100
    if r["battery_health"] is not None and p["battery_frac"] > 0:
        items["Battery replacement"] = o * p["battery_frac"] * (100 - r["battery_health"]) / 100
    items["Functional repair"] = o * p["repair_frac"] * 0.7 * (r["functional_damage"] / 100) ** 1.2
    items["Cosmetic restoration"] = o * p["cosm_frac"] * r["cosmetic_damage"] / 100
    items["Replace missing accessories"] = o * p["acc_frac"] * r["missing_accessories"]
    items["Water-damage remediation"] = o * 0.10 * r["water_damage"]
    return {k: round(v, 2) for k, v in items.items() if v >= 0.5}


def _component_health(r: dict, name: str, src: str) -> float:
    if src == "screen" and r["screen_condition"] is not None:
        h = r["screen_condition"]
    elif src == "battery" and r["battery_health"] is not None:
        h = r["battery_health"]
    elif src == "casing":
        h = 100 - r["cosmetic_damage"]
    elif src == "frame":
        h = 100 - 0.8 * r["functional_damage"]
    else:
        jitter = (sum(map(ord, name)) % 15) - 7  # deterministic per-component variation
        h = r["component_health"] + jitter
    return clip(h, 0, 100)


def component_breakdown(r: dict, p: dict, mf: float) -> list[dict]:
    out = []
    for name, share, src in p["components"]:
        health = _component_health(r, name, src)
        nominal = r["original_value"] * share * mf
        if health >= 70:
            status, value = "Reusable", nominal * p["yield"] * health / 100
        elif health >= 45:
            status, value = "Refurbish before reuse", nominal * p["yield"] * health / 100 * 0.7
        else:
            status, value = "Material recovery only", 0.0
        out.append({"component": name, "health": round(health), "status": status,
                    "nominal_value": round(nominal, 2), "value": round(value, 2), "usable": value > 0,
                    "route": {"Reusable": "Component recovery", "Refurbish before reuse": "Spare-part reuse after refurbishment",
                              "Material recovery only": "Material recovery"}[status]})
    return out


def reusable_fraction(r: dict) -> float:
    return clip(0.45 * r["component_health"] / 100 + 0.35 * (1 - r["functional_damage"] / 100)
                + 0.20 * (1 - r["cosmetic_damage"] / 100) - 0.30 * r["water_damage"], 0, 1)


def transformation_options(r: dict, p: dict) -> list[dict]:
    """Rule-based, simulated opportunities. Always needs human validation."""
    reusable = reusable_fraction(r)
    o = r["original_value"]
    ideas = []
    for name, share, cost_frac, fixed, min_re, days, demand in p["transformations"]:
        value = o * share * reusable * demand
        cost = o * cost_frac + fixed
        ideas.append({"idea": name, "value": round(value, 2), "cost": round(cost, 2), "net": round(value - cost, 2),
                      "days": days, "min_reusable": min_re, "reusable": round(reusable, 2), "demand": demand,
                      "viable": reusable >= min_re and value > cost})
    return sorted(ideas, key=lambda i: (not i["viable"], -i["net"]))


def _make(key, gross, costs, days, p_success, suit, usab, market, viable):
    full = {k: 0.0 for k in COST_KEYS}
    full.update({k: float(v) for k, v in costs.items()})
    total = sum(full.values())
    return {"key": key, "name": PATHWAY_LABELS[key], "gross": float(gross), "costs": full,
            "total_cost": total, "net": float(gross) - total, "days": float(days), "p_success": p_success,
            "risk": 1 - p_success, "suitability": clip(suit, 0, 1), "usability": clip(usab, 0, 1),
            "market": clip(market, 0, 1), "viable": bool(viable)}


def pathway_economics(r: dict, cond: dict, overrides: dict | None = None) -> dict:
    eco = get_config()["economics"]
    p = CATEGORY_PROFILES[r["category"]]
    ov = {}
    for k, v in (overrides or {}).items():
        if k in OVERRIDE_KEYS and v is not None:
            try:
                ov[k] = max(0.0, float(v))
            except (TypeError, ValueError):
                pass
    orig = r["original_value"]
    mf = market_factor(r, p)
    mv = orig * mf
    score = cond["score"]
    comp = r["component_health"]
    water = r["water_damage"]
    crit = max(cond["screen_deficit"], 0.9 * r["functional_damage"] / 100, float(water))

    # ---- baseline (model-estimated) values --------------------------------
    repair_items = repair_estimate(r, p)
    repair_base = r["repair_cost_override"] if r["repair_cost_override"] else sum(repair_items.values())
    as_is = max(0.03, (score / 100) ** 2.3 * (1 - 0.55 * crit))
    restoration = 0.6 + 0.4 * comp / 100
    components = component_breakdown(r, p, mf)
    ideas = transformation_options(r, p)
    best_idea = next((i for i in ideas if i["viable"]), None)
    base = {
        "repair_cost": repair_base,
        "resale_value": mv * as_is,
        "refurbished_value": mv * p["refurb_factor"] * restoration * (1 - 0.10 * water),
        "refurbishment_cost": orig * p["refurb_overhead"] + 0.4 * p["wh_base"],
        "logistics_cost": r["reverse_shipping_cost"] + r["handling_cost"],
        "component_value": sum(c["value"] for c in components),
        "transformation_value": best_idea["value"] if best_idea else 0.0,
        "processing_days": float(r["estimated_processing_days"]),
    }
    used = {**base, **ov}

    outbound = eco["outbound_share_of_inbound"] * r["reverse_shipping_cost"]
    hold = eco["holding_cost_per_day"] * mv
    q = used["processing_days"]
    wh = lambda days: r["warehouse_processing_cost"] + hold * days  # noqa: E731
    demand = 0.5 + 0.5 * p["comp_demand"]
    common_logi = used["logistics_cost"]
    paths = {}

    # 1. Resale
    g = used["resale_value"]
    d = q + 2
    ps = clip(0.72 + 0.26 * score / 100, 0.5, 0.97) * (1 - 0.1 * crit)
    paths["resale"] = _make("resale", g, {"logistics": common_logi + outbound, "warehouse": wh(d),
                                          "expected_loss": g * (1 - ps)}, d, ps, score / 100,
                            comp / 100, demand * (score / 100) ** 0.5, g > 0)

    # 2. Refurbish
    rc, rg = used["repair_cost"], used["refurbished_value"]
    d = q + 3 + 6 * clip(rc / orig * 3, 0, 1)
    ps = clip(0.50 + 0.45 * comp / 100 - 0.20 * water - (0.08 if rc > 0.5 * rg else 0), 0.25, 0.96)
    suit = clip(1 - rc / rg, 0, 1) * clip(comp / 100 * 1.1, 0, 1) if rg > 0 else 0
    paths["refurbish"] = _make("refurbish", rg, {"repair": rc, "refurbishment": used["refurbishment_cost"],
                                                 "logistics": common_logi + outbound, "warehouse": wh(d),
                                                 "expected_loss": rg * (1 - ps)}, d, ps, suit, comp / 100,
                               demand * 0.95, rg > 0 and rc <= 0.9 * rg)

    # 3. Component recovery
    cg = used["component_value"]
    n_use = sum(1 for c in components if c["usable"])
    if cg > 0:
        n_use = max(n_use, 1)
    proc = (p["proc_base"] * (1 + 0.2 * n_use) + 0.08 * cg) if cg > 0 else 0.0
    d = q + 3 + min(5, 0.8 * n_use)
    ps = clip(0.88 - 0.20 * water - (0.10 if n_use <= 1 else 0), 0.4, 0.92)
    nominal_total = sum(c["nominal_value"] for c in components) * p["yield"]
    usable_share = clip(sum(c["value"] for c in components) / nominal_total, 0, 1) if nominal_total > 0 else 0
    paths["component"] = _make("component", cg, {"processing": proc, "logistics": common_logi, "warehouse": wh(d),
                                                 "expected_loss": cg * (1 - ps)}, d, ps, usable_share,
                               usable_share, p["comp_demand"], cg > 0)

    # 4. Transformation (best viable idea; simulated)
    tg = used["transformation_value"]
    idea = best_idea or (ideas[0] if ideas else None)
    d = q + (idea["days"] if idea else 0) + 2
    ps = 0.72
    reusable = reusable_fraction(r)
    paths["transformation"] = _make("transformation", tg, {"transformation": idea["cost"] if idea else 0.0,
                                                           "logistics": common_logi, "warehouse": wh(d),
                                                           "expected_loss": tg * (1 - ps)}, d, ps, reusable,
                                    reusable, idea["demand"] if idea else 0.4, tg > 0 and idea is not None)

    # 5. Material recovery
    mg = p["material_rate"] * r["product_weight"] * p["material_recovery"]
    d = q + 2 + min(3, 0.1 * r["product_weight"])
    ps = 0.95
    mproc = max(15.0, 0.15 * mg) + 8 * r["product_weight"]
    paths["material"] = _make("material", mg, {"processing": mproc, "logistics": common_logi, "warehouse": wh(d),
                                               "expected_loss": mg * (1 - ps)}, d, ps, 1 - score / 100, 0.2, 0.5, mg > mproc)

    # 6. Disposal
    d = q + 1
    paths["disposal"] = _make("disposal", 0.0, {"disposal": p["disp_base"] + p["disp_kg"] * r["product_weight"],
                                                "logistics": common_logi, "warehouse": wh(d)}, d, 1.0,
                              0.15 if score < 30 else 0.05, 0.0, 0.0, True)

    for pw in paths.values():
        pw["value_preserved"] = pw["net"] / orig * 100.0
    return {"pathways": paths, "market_value": mv, "market_factor": mf, "repair_items": repair_items,
            "components": components, "transformations": ideas, "baseline_values": base, "values_used": used,
            "reusable_fraction": reusable}
