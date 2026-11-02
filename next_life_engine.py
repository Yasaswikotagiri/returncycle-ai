"""Orchestrates the full pipeline for one product:
condition -> economics for every pathway -> scoring -> recommendation -> reasoning."""
from __future__ import annotations

import pandas as pd

from src.condition_engine import assess_condition
from src.data_cleaning import normalise_record
from src.explainability import build_reasoning, why_not
from src.recovery_engine import COST_KEYS, OVERRIDE_KEYS, pathway_economics
from src.utils import CATEGORY_PROFILES, PATHWAY_LABELS, PATHWAYS, clip
from src.value_optimizer import recommend, score_pathways


def scale_damage(r: dict, severity: float) -> dict:
    """Counterfactual: make the damage worse (>1) or milder (<1)."""
    r = dict(r)
    for k in ("cosmetic_damage", "functional_damage"):
        r[k] = clip(r[k] * severity, 0, 100)
    for k in ("screen_condition", "battery_health", "component_health", "packaging_condition"):
        if r[k] is not None:
            r[k] = clip(100 - (100 - r[k]) * severity, 0, 100)
    return r


def analyze_product(record: dict, overrides: dict | None = None, weights: dict | None = None,
                    detailed: bool = True) -> dict:
    r = normalise_record(record)
    ov = dict(overrides or {})
    severity = float(ov.pop("damage_severity", 1.0) or 1.0)
    if severity != 1.0:
        r = scale_damage(r, severity)
    cond = assess_condition(r)
    eco = pathway_economics(r, cond, {k: v for k, v in ov.items() if k in OVERRIDE_KEYS})
    P = score_pathways(eco["pathways"], r["original_value"], weights)
    rec = recommend(P)
    a = {"record": r, "condition": cond, "pathways": P, **{k: v for k, v in eco.items() if k != "pathways"},
         **rec, "overrides": {**{k: v for k, v in ov.items() if k in OVERRIDE_KEYS},
                              **({"damage_severity": severity} if severity != 1.0 else {})}}
    if detailed:
        a["reasoning"] = build_reasoning(a)
        a["why_not"] = why_not(a)
        a["journey"] = journey_steps(a)
    return a


def journey_steps(a: dict) -> list[dict]:
    from src.utils import inr
    best = a["pathways"][a["recommended"]]
    c = a["condition"]
    return [
        {"step": "RETURNED", "detail": f"{a['record']['category']} · original {inr(a['record']['original_value'])}"},
        {"step": "INSPECTED", "detail": f"{len(c['issues'])} issue(s) detected"},
        {"step": "CONDITION GRADED", "detail": f"Grade {c['grade']} · score {c['score']:.0f}/100"},
        {"step": "VALUE ESTIMATED", "detail": f"Market value {inr(a['market_value'])}"},
        {"step": "PATHWAYS GENERATED", "detail": "6 candidate next lives"},
        {"step": "BEST NEXT LIFE", "detail": f"{PATHWAY_LABELS[a['recommended']]} · net {inr(best['net'])}"},
    ]


def pathway_table(a: dict) -> pd.DataFrame:
    rows = []
    for k in PATHWAYS:
        p = a["pathways"][k]
        rows.append({"Pathway": PATHWAY_LABELS[k], "Gross Value": round(p["gross"]), "Costs": round(p["total_cost"]),
                     "Net Recovery": round(p["net"]), "Value Preserved %": round(p["value_preserved"], 1),
                     "Time (days)": round(p["days"], 1), "Score": p["score"],
                     "Status": ("RECOMMENDED" if k == a["recommended"] else
                                "Alternative" if k == a.get("alternative") else
                                "Not viable" if not p["viable"] else
                                "Not eligible" if k not in a["eligible"] else "Viable")})
    return pd.DataFrame(rows)


def cost_table(a: dict, key: str) -> pd.DataFrame:
    from src.recovery_engine import COST_LABELS
    p = a["pathways"][key]
    rows = [{"Item": "Gross recovery value", "Amount": round(p["gross"], 2)}]
    rows += [{"Item": "− " + COST_LABELS[c], "Amount": -round(p["costs"][c], 2)} for c in COST_KEYS if p["costs"][c] > 0.005]
    rows.append({"Item": "= Estimated net recovery", "Amount": round(p["net"], 2)})
    return pd.DataFrame(rows)


def flatten(a: dict) -> dict:
    """One analysis -> one flat row of dataset columns."""
    r, P, p = a["record"], a["pathways"], CATEGORY_PROFILES[a["record"]["category"]]
    best = P[a["recommended"]]
    orig, mv = r["original_value"], a["market_value"]
    c = best["costs"]
    total_loss = orig - best["net"]
    dep = orig - mv
    logi = c["logistics"]
    repair = c["repair"] + c["refurbishment"]
    proc = c["warehouse"] + c["processing"] + c["transformation"] + c["disposal"]
    residual = total_loss - dep - logi - repair - proc
    acc = min(orig * p["acc_frac"] * r["missing_accessories"], max(residual, 0.0))
    row = {
        "condition_score": a["condition"]["score"], "condition_grade": a["condition"]["grade"],
        "market_value": round(mv, 2), "estimated_repair_cost": round(a["values_used"]["repair_cost"], 2),
        "estimated_resale_value": round(P["resale"]["gross"], 2),
        "estimated_refurbished_value": round(P["refurbish"]["gross"], 2),
        "component_recovery_value": round(P["component"]["gross"], 2),
        "transformation_value": round(P["transformation"]["gross"], 2),
        "material_recovery_value": round(P["material"]["gross"], 2),
        "disposal_cost": round(P["disposal"]["costs"]["disposal"], 2),
        "best_pathway": PATHWAY_LABELS[a["recommended"]],
        "alt_pathway": PATHWAY_LABELS[a["alternative"]] if a.get("alternative") else "",
        "best_gross": round(best["gross"], 2), "best_net_recovery": round(best["net"], 2),
        "best_value_preserved": round(best["value_preserved"], 2), "best_confidence": a["confidence"],
        "best_days": round(best["days"], 1),
        "loss_depreciation": round(dep, 2), "loss_damage": round(residual - acc, 2), "loss_accessories": round(acc, 2),
        "loss_logistics": round(logi, 2), "loss_repair": round(repair, 2), "loss_processing": round(proc, 2),
    }
    for k in PATHWAYS:
        row[f"net_{k}"] = round(P[k]["net"], 2)
    return row


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.to_dict("records"):
        rows.append(flatten(analyze_product(rec, detailed=False)))
    return pd.DataFrame(rows, index=df.index)


def transformation_tree(a: dict) -> str:
    """Text tree for the 'Could this become something else?' section."""
    from src.utils import inr
    r = a["record"]
    lines = [f"{r.get('product_name') or r['category']}  (returned · Grade {a['condition']['grade']})", "│"]
    lines.append("├── Usable components / material")
    comps = a["components"]
    for i, c in enumerate(comps):
        branch = "└──" if i == len(comps) - 1 else "├──"
        lines.append(f"│   {branch} {c['component']} [{c['health']}%] → {c['route']}"
                     + (f"  ~{inr(c['value'])}" if c["value"] else ""))
    lines.append("│")
    lines.append("└── Potential products (estimated / simulated — needs human validation)")
    ideas = a["transformations"]
    for i, t in enumerate(ideas):
        branch = "└──" if i == len(ideas) - 1 else "├──"
        flag = "viable" if t["viable"] else "not viable"
        lines.append(f"    {branch} {t['idea']}  value ~{inr(t['value'])}, cost ~{inr(t['cost'])}  ({flag})")
    return "\n".join(lines)
