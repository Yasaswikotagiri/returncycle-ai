import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.next_life_engine import analyze_product, pathway_table
from src.utils import PATHWAYS, PATHWAY_LABELS, get_config, inr
from src.value_optimizer import WEIGHT_LABELS

df, source, meta = ui.get_data()
ui.page_header("Recovery Optimizer", "Compare every pathway on economics, suitability, time and risk.")
rec, label = ui.product_picker(df, "opt")
ss = st.session_state
defaults = get_config()["optimizer_weights"]
wkeys = [f"w_{k}" for k in defaults]


def _reset():
    for k in wkeys:
        ss.pop(k, None)
    ss.pop("opt_weights", None)


with st.expander("Scoring weights (optional)"):
    with st.form("weights"):
        cols = st.columns(4)
        ws = {k: cols[i % 4].slider(WEIGHT_LABELS[k], 0.0, 1.0, float(v), 0.01, key=f"w_{k}") for i, (k, v) in enumerate(defaults.items())}
        go_ = st.form_submit_button("Compare Pathways", type="primary")
    st.button("Reset weights", on_click=_reset)
    if go_:
        ss["opt_weights"] = ws
weights = ss.get("opt_weights")
try:
    a = analyze_product(rec, weights=weights)
except ValueError as e:
    st.error(f"Could not analyze this product: {e}")
    st.stop()
if weights:
    st.info("Using custom scoring weights. Press **Reset weights** to return to defaults.")

k, alt = a["recommended"], a.get("alternative")
p = a["pathways"][k]
ui.kpi_cards([("Recommended", PATHWAY_LABELS[k], "", "good"), ("Alternative", PATHWAY_LABELS[alt] if alt else "–", "", ""),
              ("Confidence", f"{a['confidence']:.0%}", "heuristic", ""), ("Expected net recovery", inr(p["net"]), f"{p['value_preserved']:.1f}% preserved", "good"),
              ("Processing time", f"{p['days']:.0f} days", "incl. queue", "")])
st.write("")
st.subheader("Pathway comparison")
ui.pathway_table_styled(a)
st.download_button("Export comparison (CSV)", pathway_table(a).to_csv(index=False), "pathway_comparison.csv", "text/csv")
c1, c2 = st.columns(2)
with c1:
    names = [PATHWAY_LABELS[x] for x in PATHWAYS]
    fig = go.Figure()
    fig.add_bar(name="Gross value", x=names, y=[a["pathways"][x]["gross"] for x in PATHWAYS], marker_color="#9AA5B1")
    fig.add_bar(name="Total costs", x=names, y=[a["pathways"][x]["total_cost"] for x in PATHWAYS], marker_color="#B4443C")
    fig.add_bar(name="Net recovery", x=names, y=[a["pathways"][x]["net"] for x in PATHWAYS], marker_color="#2E7D5B")
    fig.update_layout(barmode="group", title="Gross vs cost vs net")
    ui.plot(fig)
with c2:
    fig = go.Figure()
    colors = ["#2E7D5B", "#4C78A8", "#C08A3E", "#6B8794", "#B4443C", "#7C6BA6", "#9AA5B1"]
    for i, part in enumerate(WEIGHT_LABELS):
        fig.add_bar(name=WEIGHT_LABELS[part], x=names, y=[a["pathways"][x]["score_parts"][part] for x in PATHWAYS], marker_color=colors[i])
    fig.update_layout(barmode="relative", title="Score contributions (points)")
    ui.plot(fig)
st.subheader("Reason")
ui.recommendation_card(a)
ui.export_buttons(a, "opt", label)
ui.disclaimer()
