"""Human-readable reasoning for every recommendation + optional SHAP (never crashes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PATHWAY_HEADLINE, PATHWAY_LABELS, inr


def build_reasoning(a: dict) -> str:
    k = a["recommended"]
    P = a["pathways"]
    best = P[k]
    r = a["record"]
    cond = a["condition"]
    used = a["values_used"]
    comp = r["component_health"]
    parts = []
    if k == "resale":
        parts.append(f"The product is in Grade {cond['grade']} condition (score {cond['score']:.0f}/100) with limited damage, "
                     f"so selling it directly loses less value than repairing or dismantling it.")
    elif k == "refurbish":
        ratio = used["repair_cost"] / max(used["refurbished_value"], 1)
        strength = "strong" if comp >= 75 else "adequate"
        parts.append(f"The product has {strength} functional health (component health {comp:.0f}/100) and the repair cost "
                     f"({inr(used['repair_cost'])}) is only {ratio:.0%} of the expected refurbished resale value "
                     f"({inr(used['refurbished_value'])}).")
    elif k == "component":
        n = sum(1 for c in a["components"] if c["usable"])
        parts.append(f"{n} component(s) are still usable and worth about {inr(used['component_value'])} in total, while "
                     f"repairing the whole product would cost {inr(used['repair_cost'])} against a refurbished value of "
                     f"{inr(used['refurbished_value'])}.")
    elif k == "transformation":
        idea = next((i for i in a["transformations"] if i["viable"]), None)
        nm = idea["idea"] if idea else "a new product"
        parts.append(f"Reusable material/parts ({a['reusable_fraction']:.0%} of the product) can be converted into: {nm}. "
                     f"This is a simulated opportunity and needs human validation.")
    elif k == "material":
        parts.append("Little functional or resale value remains, so recovering material by weight loses the least value.")
    else:
        parts.append("Every recovery route would cost more than it returns, so compliant disposal is the least-loss option.")
    alt = a.get("alternative")
    if alt:
        diff = best["net"] - P[alt]["net"]
        word = "more" if diff >= 0 else "less"
        parts.append(f"{PATHWAY_HEADLINE[k]} is estimated to preserve {best['value_preserved']:.1f}% of the original value "
                     f"(net {inr(best['net'])}); that is {inr(abs(diff))} {word} than {PATHWAY_LABELS[alt]} "
                     f"({inr(P[alt]['net'])}), and its overall score is higher after time, cost and risk penalties.")
    else:
        parts.append(f"It is estimated to preserve {best['value_preserved']:.1f}% of the original value.")
    return " ".join(parts)


def why_not(a: dict) -> list[tuple[str, str]]:
    P, k, used = a["pathways"], a["recommended"], a["values_used"]
    out = []
    for key, p in P.items():
        if key == k:
            continue
        if key == "refurbish" and not p["viable"]:
            why = f"Repair cost ({inr(used['repair_cost'])}) is too close to the refurbished value ({inr(used['refurbished_value'])})."
        elif key == "component" and not p["viable"]:
            why = "No component is healthy enough to resell."
        elif key == "transformation" and not p["viable"]:
            why = "Not enough reusable material for any rule-based transformation idea."
        elif key == "disposal":
            why = "Only chosen when every other pathway loses more value."
        else:
            gap = P[k]["net"] - p["net"]
            why = f"Net {inr(p['net'])} — {inr(abs(gap))} {'below' if gap >= 0 else 'above'} the recommendation; score {p['score']:.0f}."
        out.append((PATHWAY_LABELS[key], why))
    return out


# ----------------------------------------------------------------------------
# SHAP helpers (optional dependency; every function degrades gracefully)
# ----------------------------------------------------------------------------
def _to_3d(sv):
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    return sv[:, :, None] if sv.ndim == 2 else sv


def shap_global(model, X: pd.DataFrame, max_rows: int = 300):
    """Mean |SHAP| per feature, or (None, reason)."""
    try:
        import shap
        Xs = X.sample(min(max_rows, len(X)), random_state=0)
        sv = _to_3d(shap.TreeExplainer(model).shap_values(Xs))
        imp = np.abs(sv).mean(axis=(0, 2))
        return pd.Series(imp, index=X.columns).sort_values(ascending=False), None
    except Exception as exc:  # noqa: BLE001
        return None, f"SHAP unavailable ({type(exc).__name__}); showing built-in feature importance instead."


def shap_local(model, x_row: pd.DataFrame, class_idx: int | None = None):
    try:
        import shap
        sv = _to_3d(shap.TreeExplainer(model).shap_values(x_row))
        k = min(class_idx or 0, sv.shape[2] - 1)
        return pd.Series(sv[0, :, k], index=x_row.columns).sort_values(key=np.abs, ascending=False), None
    except Exception as exc:  # noqa: BLE001
        return None, f"SHAP unavailable ({type(exc).__name__})."
