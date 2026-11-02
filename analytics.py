import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import ui
from src.utils import PATHWAY_COLORS, inr, inr_short, pct

df, source, meta = ui.get_data()
ui.page_header("Recovery & Circularity Analytics", "Where value is recovered, where it leaks, and how circular the outcomes are.")
t1, t2, t3, t4, t5 = st.tabs(["Returns", "Recovery", "Loss", "Circularity", "Product explorer"])
with t1:
    c1, c2 = st.columns(2)
    cat = df.groupby("category").size().reset_index(name="Returns").sort_values("Returns")
    fig = px.bar(cat, x="Returns", y="category", orientation="h", title="Returns by category", labels={"category": ""})
    fig.update_traces(marker_color="#4C78A8")
    with c1:
        ui.plot(fig)
    with c2:
        r = df["return_reason"].value_counts().reset_index()
        r.columns = ["Reason", "Returns"]
        fig = px.bar(r.sort_values("Returns"), x="Returns", y="Reason", orientation="h", title="Returns by reason")
        fig.update_traces(marker_color="#52606D")
        ui.plot(fig)
    ui.plot(px.line(an.monthly_trend(df), x="Month", y="Returns", markers=True, title="Return trend"))
with t2:
    rc = an.recovery_by_category(df)
    fig = go.Figure()
    fig.add_bar(name="Current practice", x=rc["category"], y=rc["Current rate %"], marker_color="#9AA5B1")
    fig.add_bar(name="Optimized", x=rc["category"], y=rc["Optimized rate %"], marker_color="#2E7D5B")
    fig.update_layout(barmode="group", title="Recovery rate by category (% of original value)", yaxis_title="%")
    ui.plot(fig)
    npw = an.net_by_pathway(df)
    fig = px.bar(npw, x="Pathway", y="Net", color="Pathway", color_discrete_map=PATHWAY_COLORS, title="Net recovered value by recommended pathway (₹)")
    fig.update_layout(showlegend=False)
    ui.plot(fig)
with t3:
    c1, c2 = st.columns(2)
    for col, by, title in ((c1, "return_reason", "Loss by return reason"), (c2, "category", "Loss by category")):
        lb = an.loss_by(df, by).sort_values("Loss (current practice)")
        fig = go.Figure()
        fig.add_bar(name="Current practice", y=lb[by], x=lb["Loss (current practice)"], orientation="h", marker_color="#B4443C")
        fig.add_bar(name="Optimized", y=lb[by], x=lb["Loss (optimized)"], orientation="h", marker_color="#C9CED6")
        fig.update_layout(barmode="group", title=title)
        with col:
            ui.plot(fig)
    c3, c4 = st.columns(2)
    lg = df.groupby("category")["loss_logistics"].sum().sort_values().reset_index()
    fig = px.bar(lg, x="loss_logistics", y="category", orientation="h", title="Reverse-logistics loss by category (₹)", labels={"loss_logistics": "₹", "category": ""})
    fig.update_traces(marker_color="#C08A3E")
    with c3:
        ui.plot(fig)
    cg = df.groupby("condition_grade")[["original_value", "recovered_value"]].sum().reset_index()
    cg["Loss"] = cg["original_value"] - cg["recovered_value"]
    fig = px.bar(cg, x="condition_grade", y="Loss", title="Condition-related loss by grade (current practice, ₹)", labels={"condition_grade": "Grade"})
    fig.update_traces(marker_color="#B4443C")
    with c4:
        ui.plot(fig)
with t4:
    ci = an.circularity(df)
    sh = an.circular_share(df)
    ui.kpi_cards([("Kept in use (current)", pct(sh["current"]), "not disposed", ""), ("Kept in use (optimized)", pct(sh["optimized"]), "not disposed", "good"),
                  ("Refurbished (optimized)", f"{int(ci.loc[ci.Outcome == 'Refurbished', 'Optimized'].iloc[0]):,}", "items", "")])
    fig = go.Figure()
    fig.add_bar(name="Current practice", x=ci["Outcome"], y=ci["Current practice"], marker_color="#9AA5B1")
    fig.add_bar(name="Optimized", x=ci["Outcome"], y=ci["Optimized"], marker_color="#2E7D5B")
    fig.update_layout(barmode="group", title="Circularity outcomes (items)")
    ui.plot(fig)
    ui.table(ci.round(1))
with t5:
    f, _ = ui.filter_bar(df, "ex", extended=True)
    st.caption(f"{len(f):,} returns match")
    cols = ["return_id", "product_id", "product_name", "category", "return_reason", "condition_grade", "condition_score",
            "original_value", "best_pathway", "best_net_recovery", "best_value_preserved", "recovered_value"]
    ui.table(f[cols].head(1000))
    st.download_button("Export results (CSV)", f[cols].to_csv(index=False), "explorer_results.csv", "text/csv")
ui.disclaimer()
