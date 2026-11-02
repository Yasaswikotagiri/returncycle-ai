import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src import ui
from src.utils import inr_short, pct

df, source, meta = ui.get_data()
ui.page_header("Return Intelligence", "Return volume, reasons, rates and customer / product patterns.")
f, ctx = ui.filter_bar(df, "ri")
orders = ui.load_orders_summary()

top_reason = f["return_reason"].value_counts().index[0]
rep = (f.groupby("customer_id").size() > 1).mean() * 100
rate = an.return_rate_by(f, orders, "category", ctx["date_range"], ctx["categories"], ctx["sellers"])
overall = (rate["Returns"].sum() / rate["Orders"].sum() * 100) if rate is not None and rate["Orders"].sum() else None
ui.kpi_cards([("Return volume", f"{len(f):,}", "after filters", ""),
              ("Return rate", pct(overall) if overall is not None else "n/a", "of orders (enriched sample)", ""),
              ("Value returned", inr_short(f["original_value"].sum()), "original value", ""),
              ("Top reason", top_reason, "most frequent", ""),
              ("Repeat returners", pct(rep), "customers with 2+ returns", "warn")])
st.write("")
c1, c2 = st.columns(2)
with c1:
    tr = an.monthly_trend(f)
    fig = px.line(tr, x="Month", y="Returns", markers=True, title="Returns by month")
    fig.update_traces(line_color="#2E6F5E")
    ui.plot(fig)
with c2:
    rs = f["return_reason"].value_counts().reset_index()
    rs.columns = ["Reason", "Returns"]
    fig = px.bar(rs.sort_values("Returns"), x="Returns", y="Reason", orientation="h", title="Return reasons")
    fig.update_traces(marker_color="#52606D")
    ui.plot(fig)
c3, c4 = st.columns(2)
with c3:
    if rate is not None:
        fig = px.bar(rate.sort_values("Return rate %"), x="Return rate %", y="Category", orientation="h", title="Return rate by category (% of orders)")
        fig.update_traces(marker_color="#C08A3E")
        ui.plot(fig)
    else:
        st.info("Order totals are unavailable, so return rates cannot be computed.")
with c4:
    cc = f.groupby("category").agg(Returns=("return_id", "count"), Grade=("condition_score", "mean")).reset_index()
    fig = px.bar(cc.sort_values("Returns"), x="Returns", y="category", orientation="h", title="Returns by category", labels={"category": ""})
    fig.update_traces(marker_color="#4C78A8")
    ui.plot(fig)
st.subheader("Customer return patterns")
dist, topc = an.customer_patterns(f)
c5, c6 = st.columns([2, 3])
with c5:
    fig = px.bar(dist, x="Returns per customer", y="Customers", title="Customers by number of returns")
    fig.update_traces(marker_color="#7C6BA6")
    ui.plot(fig)
with c6:
    st.caption("Top repeat returners")
    topc = topc.assign(Value=topc["Value"].round(0), Freq=topc["Freq"].round(2)).rename(columns={"customer_id": "Customer", "Freq": "Return frequency"})
    ui.table(topc)
st.subheader("Product-level return patterns")
ui.table(an.product_patterns(f).rename(columns={"product_id": "Product ID", "category": "Category", "Top_reason": "Top reason",
                                                 "Avg_condition": "Avg condition", "Net_recovery": "Net recovery (₹)"}))
st.download_button("Export filtered returns (CSV)", f.to_csv(index=False), "returns_filtered.csv", "text/csv")
ui.disclaimer()
