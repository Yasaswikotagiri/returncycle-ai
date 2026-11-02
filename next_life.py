import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.next_life_engine import analyze_product, cost_table, transformation_tree
from src.utils import PATHWAYS, PATHWAY_DESC, PATHWAY_HEADLINE, PATHWAY_LABELS, TAGLINE_1, TAGLINE_2, inr

df, source, meta = ui.get_data()
ui.page_header("Next-Life Pathways", "From returned product to best next life — every option, with the calculation behind it.")
ui.callout(TAGLINE_1, TAGLINE_2)
rec, label = ui.product_picker(df, "nl")
try:
    a = analyze_product(rec)
except ValueError as e:
    st.error(f"Could not analyze this product: {e}")
    st.stop()
ui.journey_html(a)
ss = st.session_state
if ss.get("nl_for") != label:
    ss["nl_for"], ss["nl_sel"] = label, a["recommended"]

st.subheader("Candidate next lives")
cols = st.columns(3)
for i, k in enumerate(PATHWAYS):
    p = a["pathways"][k]
    with cols[i % 3].container(border=True):
        tag = "RECOMMENDED" if k == a["recommended"] else ("Alternative" if k == a.get("alternative") else
                                                           "Not viable" if not p["viable"] else "Viable")
        tone = "g" if k == a["recommended"] else "n"
        st.markdown(f"**{PATHWAY_HEADLINE[k].upper()}** &nbsp; {ui.pill(tag, tone)}", unsafe_allow_html=True)
        st.markdown(f"### {inr(p['net'])}")
        st.caption(f"net recovery · {p['value_preserved']:.1f}% preserved · {p['days']:.0f} days · {p['p_success']:.0%} success likelihood")
        if st.button("Show calculation", key=f"card_{k}"):
            ss["nl_sel"] = k
sel = ss.get("nl_sel", a["recommended"])
st.subheader(f"Calculation — {PATHWAY_HEADLINE[sel]}")
st.caption(PATHWAY_DESC[sel])
c1, c2 = st.columns([3, 2])
with c1:
    ui.waterfall_chart(a, sel)
with c2:
    t = cost_table(a, sel)
    t["Amount"] = t["Amount"].map(inr)
    ui.table(t)
    p = a["pathways"][sel]
    st.caption(f"Score {p['score']:.0f}/100 · success likelihood {p['p_success']:.0%} · time {p['days']:.1f} days incl. warehouse queue.")
st.subheader("Why this recommendation?")
ui.recommendation_card(a)
with st.expander("Why not the other pathways?"):
    for n, w in a["why_not"]:
        st.markdown(f"- **{n}** — {w}")

st.subheader("Could this become something else?")
st.caption("Estimated / simulated recovery opportunities — rule-based, not validated for manufacturing feasibility.")
st.code(transformation_tree(a), language="text")
t1, t2 = st.columns(2)
with t1:
    ui.components_table(a)
with t2:
    ti = pd.DataFrame(a["transformations"])
    fig = go.Figure(go.Bar(y=ti["idea"], x=ti["net"], orientation="h", marker_color=["#7C6BA6" if v else "#C9CED6" for v in ti["viable"]],
                           text=[inr(v) for v in ti["net"]], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Transformation ideas — estimated net value", showlegend=False)
    ui.plot(fig)
ui.export_buttons(a, "nl", label)
ui.disclaimer()
