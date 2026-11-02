import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.next_life_engine import analyze_product
from src.simulator import SCENARIOS, compare, run_all_scenarios, scenario_overrides
from src.utils import PATHWAYS, PATHWAY_LABELS, inr

df, source, meta = ui.get_data()
ui.page_header("Counterfactual Simulator", "Change an assumption and watch every pathway — and the recommendation — recalculate.")
rec, label = ui.product_picker(df, "sim")
ss = st.session_state
try:
    base = analyze_product(rec)
except ValueError as e:
    st.error(f"Could not analyze this product: {e}")
    st.stop()
bv, orig = base["baseline_values"], base["record"]["original_value"]
sig = hashlib.md5(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:8]
FIELDS = [("repair_cost", "Repair cost (₹)"), ("resale_value", "Resale value as-is (₹)"), ("refurbished_value", "Refurbished resale value (₹)"),
          ("refurbishment_cost", "Refurbishment / testing cost (₹)"), ("logistics_cost", "Logistics cost (₹)"),
          ("component_value", "Component value (₹)"), ("transformation_value", "Transformation value (₹)")]
keys = [f"sim_{f}_{sig}" for f, _ in FIELDS] + [f"sim_days_{sig}", f"sim_sev_{sig}"]


def _reset():
    for k in keys:
        ss.pop(k, None)


tab1, tab2 = st.tabs(["Live simulator", "What-if scenarios"])
with tab1:
    left, right = st.columns([2, 3])
    ov = {}
    with left:
        st.markdown("**Assumptions**")
        for f, lbl in FIELDS:
            b = int(round(bv[f]))
            mx = int(max(2 * bv[f], 0.2 * orig, 100))
            v = st.slider(lbl, 0, mx, min(b, mx), step=max(1, mx // 200), key=f"sim_{f}_{sig}")
            if v != b:
                ov[f] = v
        d0 = int(round(bv["processing_days"]))
        d = st.slider("Warehouse processing time (days)", 0, 30, min(d0, 30), key=f"sim_days_{sig}")
        if d != d0:
            ov["processing_days"] = d
        sev = st.slider("Damage severity (×)", 0.5, 1.5, 1.0, 0.05, key=f"sim_sev_{sig}")
        if sev != 1.0:
            ov["damage_severity"] = sev
        st.button("Reset", on_click=_reset)
    res = compare(rec, ov)
    cur, new = res["current"], res["scenario"]
    nb = new["pathways"][new["recommended"]]
    with right:
        if res["changed"]:
            st.success(f"Recommendation changed from **{res['from_label'].upper()}** → **{res['to_label'].upper()}**")
        else:
            st.info(f"Recommendation unchanged: **{res['to_label']}**" + ("" if ov else " (move a slider to explore)"))
        ui.kpi_cards([("Best pathway", res["to_label"], "", "good"), ("Net recovery", inr(nb["net"]), f"{res['delta_best_net']:+,.0f} vs baseline", ""),
                      ("Value preserved", f"{nb['value_preserved']:.1f}%", "", ""),
                      ("Vs current strategy", inr(res["delta_keep_strategy"]), f"if still doing {res['from_label']}", "warn")])
        names = [PATHWAY_LABELS[k] for k in PATHWAYS]
        fig = go.Figure()
        fig.add_bar(name="Current", x=names, y=[cur["pathways"][k]["net"] for k in PATHWAYS], marker_color="#9AA5B1")
        fig.add_bar(name="Scenario", x=names, y=[new["pathways"][k]["net"] for k in PATHWAYS], marker_color="#2E7D5B")
        fig.update_layout(barmode="group", title="Net recovery: current vs scenario")
        ui.plot(fig)
        ui.table(res["table"])
        c1, c2 = st.columns(2)
        if c1.button("Run Simulation", type="primary"):
            hist = ss.setdefault("sim_hist", [])
            hist.append({"Product": label, "Overrides": json.dumps(ov), "Recommended": res["to_label"], "Net recovery": round(nb["net"]),
                         "Changed": res["changed"]})
        hist = ss.get("sim_hist", [])
        if hist:
            c2.download_button("Export Results (CSV)", pd.DataFrame(hist).to_csv(index=False), "simulation_runs.csv", "text/csv")
    if ss.get("sim_hist"):
        st.markdown("**Simulation log**")
        ui.table(pd.DataFrame(ss["sim_hist"]))
    ui.export_buttons(new, "sim", label)
with tab2:
    st.caption("Preset what-if scenarios applied to the baseline assumptions of this product.")
    sw = run_all_scenarios(rec)
    ui.table(sw)
    pick = st.radio("Inspect scenario", list(SCENARIOS), format_func=lambda k: f"{k} · {SCENARIOS[k][0]}", horizontal=True)
    r2 = compare(rec, scenario_overrides(pick, base["baseline_values"]))
    if r2["changed"]:
        st.success(f"Recommendation changes: {r2['from_label']} → {r2['to_label']}")
    else:
        st.info(f"Recommendation stays {r2['to_label']}; best net changes by {inr(r2['delta_best_net'])}.")
    t = r2["table"]
    fig = go.Figure()
    fig.add_bar(name="Current recovery", x=t["Pathway"], y=t["Current net"], marker_color="#9AA5B1")
    fig.add_bar(name="Scenario recovery", x=t["Pathway"], y=t["Scenario net"], marker_color="#2E7D5B")
    fig.update_layout(barmode="group", title=f"Scenario {pick}: {SCENARIOS[pick][0]}")
    ui.plot(fig)
ui.disclaimer()
