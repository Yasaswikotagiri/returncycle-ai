import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import ui
from src.demo_scenarios import DEMO_PRODUCTS
from src.next_life_engine import analyze_product
from src.utils import (DATASET_LABEL, PATHWAY_COLORS, TAGLINE_1, TAGLINE_2, inr, inr_short, pct)

df, source, meta = ui.get_data()
ui.page_header("ReturnCycle AI", "Turning returned products into their highest-value next life.")
ui.callout(TAGLINE_1, TAGLINE_2)
ui.flow_chips()
st.caption(f"{DATASET_LABEL} Loaded: {source}.")

k = an.kpis(df)
ui.kpi_cards([
    ("Total returns", f"{k['returns']:,}", "return records", ""),
    ("Original product value", inr_short(k["original_value"]), "list value of returned items", ""),
    ("Recoverable value", inr_short(k["recoverable_value"]), "gross, best pathway per item", "good"),
    ("Value currently lost", inr_short(k["value_lost_current"]), "under simulated current practice", "bad"),
    ("Recovery rate", pct(k["recovery_rate_current"]), f"current · optimized {pct(k['recovery_rate_optimized'])}", "warn"),
])
st.write("")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Recovery pathway distribution")
    view = st.radio("View", ["Recommended (optimized)", "Current practice (simulated)"], horizontal=True, label_visibility="collapsed")
    dist = an.pathway_distribution(df, "best_pathway" if view.startswith("Rec") else "recovery_pathway")
    fig = px.pie(dist, names="Pathway", values="Returns", hole=0.55, color="Pathway", color_discrete_map=PATHWAY_COLORS)
    fig.update_traces(textinfo="percent", sort=False)
    ui.plot(fig)
with c2:
    st.subheader("Value recovery funnel")
    f = an.funnel(df)
    fig = go.Figure(go.Funnel(y=f["Stage"], x=f["Value"], textinfo="text", text=[inr_short(v) for v in f["Value"]],
                              marker=dict(color=["#9AA5B1", "#7B8794", "#3E8E6B", "#2E7D5B"])))
    ui.plot(fig)

c3, c4 = st.columns([3, 2])
with c3:
    st.subheader("Top value-loss drivers")
    ld = an.loss_drivers(df)
    fig = go.Figure(go.Bar(y=ld["Driver"], x=ld["Value lost"], orientation="h", marker_color="#B4443C",
                           text=[inr_short(v) for v in ld["Value lost"]], textposition="outside", cliponaxis=False))
    fig.update_layout(xaxis_title="₹ (optimized pathway)", showlegend=False)
    ui.plot(fig)
with c4:
    st.subheader("Try a built-in example")
    st.caption("Open one of five demo products in the Next-Life view.")
    ex = st.selectbox("Example", list(DEMO_PRODUCTS), label_visibility="collapsed")
    if st.button("Analyze example", type="primary"):
        st.session_state["active_src"] = "Demo scenarios"
        st.session_state["active_demo"] = ex
        st.switch_page("pages/next_life.py")
    st.metric("Extra value if optimized", inr_short(k["uplift"]), f"{pct(k['recovery_rate_optimized'] - k['recovery_rate_current'])} pts recovery rate")

st.subheader("Highest recovery opportunities")
st.caption("Largest gap between the optimized recommendation and the simulated current handling. Select a row to open its analysis.")
cat = st.selectbox("Category", ["All"] + sorted(df["category"].unique()))
sub = df if cat == "All" else df[df["category"] == cat]
top = an.top_opportunities(sub, 15).reset_index(drop=True)
show = top.assign(Product=top["product_name"], Condition=top["condition_grade"] + " (" + top["condition_score"].round(0).astype(int).astype(str) + ")",
                  **{"Recommended path": top["best_pathway"], "Net recovery": top["best_net_recovery"].map(inr),
                     "Uplift vs current": top["uplift"].map(inr), "Return ID": top["return_id"], "Category": top["category"]})
show = show[["Return ID", "Product", "Category", "Condition", "Recommended path", "Net recovery", "Uplift vs current"]]
ev = ui.table(show, key="opp_table", selectable=True)
sel = []
try:
    sel = ev.selection.rows
except Exception:
    pass
if sel:
    row = top.iloc[sel[0]]
    rec = ui.row_to_record(row)
    a = analyze_product(rec)
    st.markdown(f"#### Analysis: {row['product_name']} · {row['return_id']}")
    l, r = st.columns([3, 2])
    with l:
        ui.recommendation_card(a)
    with r:
        ui.pathway_net_chart(a, height=280)
    if st.button("Open full analysis"):
        st.session_state["active_src"] = "Returned products (dataset)"
        st.session_state["active_return"] = row["return_id"]
        st.switch_page("pages/next_life.py")
ui.disclaimer()
