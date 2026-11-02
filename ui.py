"""Shared Streamlit UI helpers (styling, cards, charts, cached loaders, product picker, filters)."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as an
from src.data_cleaning import validate_columns
from src.demo_scenarios import DEMO_PRODUCTS
from src.next_life_engine import analyze_product, cost_table, pathway_table
from src.report import build_csv, build_json, build_markdown, build_pdf
from src.utils import (DATA_PROCESSED, DATA_SAMPLE, DATASET_LABEL, GRADE_LABELS, GRADES, MODELS_DIR,
                       PATHWAY_COLORS, PATHWAY_HEADLINE, PATHWAY_LABELS, PATHWAYS, inr, inr_short, pct)

INK, MUTED, GOOD, WARN, BAD, LINE, SOFT, SLATE = "#1F2933", "#6B7280", "#2E7D5B", "#B7791F", "#B4443C", "#E5E7EB", "#F4F5F7", "#52606D"

CSS = """
<style>
html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Inter, Roboto, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }
h1 { font-weight: 650; letter-spacing: -0.02em; font-size: 1.9rem !important; margin-bottom: 0.1rem; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.rc-sub { color: #6B7280; font-size: 1.0rem; margin: 0 0 1rem 0; }
.rc-kpi { border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px 16px; background: #fff; height: 100%; }
.rc-kpi .l { font-size: 0.74rem; color: #6B7280; text-transform: uppercase; letter-spacing: .05em; }
.rc-kpi .v { font-size: 1.65rem; font-weight: 650; color: #1F2933; line-height: 1.25; margin-top: 2px; }
.rc-kpi .s { font-size: 0.78rem; color: #6B7280; margin-top: 2px; }
.rc-kpi.good .v { color: #2E7D5B; } .rc-kpi.bad .v { color: #B4443C; } .rc-kpi.warn .v { color: #B7791F; }
.rc-callout { border-left: 3px solid #2E6F5E; background: #F4F5F7; padding: 10px 16px; border-radius: 0 6px 6px 0; margin: 0 0 14px 0; }
.rc-callout b { font-size: 1.02rem; } .rc-callout span { color: #52606D; }
.rc-flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 16px 0; }
.rc-chip { border: 1px solid #D5D9DE; border-radius: 4px; padding: 3px 10px; font-size: 0.74rem; font-weight: 600; letter-spacing: .06em; color: #323F4B; background: #fff; }
.rc-arrow { color: #9AA5B1; font-size: 0.8rem; }
.rc-card { border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px 18px; background: #fff; }
.rc-card.rec { border: 1.5px solid #2E7D5B; background: #F6FBF8; }
.rc-tag { font-size: 0.7rem; font-weight: 700; letter-spacing: .1em; color: #2E7D5B; }
.rc-title { font-size: 1.35rem; font-weight: 650; margin: 2px 0 10px 0; color: #1F2933; }
.rc-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px; }
.rc-grid div span { display: block; font-size: 0.7rem; color: #6B7280; text-transform: uppercase; letter-spacing: .04em; }
.rc-grid div b { font-size: 1.05rem; color: #1F2933; }
.rc-why { color: #3E4C59; font-size: 0.92rem; line-height: 1.5; }
.rc-journey { display: flex; gap: 0; margin: 6px 0 14px 0; flex-wrap: wrap; }
.rc-step { flex: 1 1 140px; border: 1px solid #E5E7EB; padding: 9px 12px; background: #fff; margin-right: -1px; }
.rc-step:first-child { border-radius: 8px 0 0 8px; } .rc-step:last-child { border-radius: 0 8px 8px 0; background: #F6FBF8; border-color: #2E7D5B; }
.rc-step .n { font-size: 0.68rem; letter-spacing: .08em; font-weight: 700; color: #6B7280; }
.rc-step:last-child .n { color: #2E7D5B; }
.rc-step .d { font-size: 0.82rem; color: #1F2933; margin-top: 2px; }
.rc-pill { display: inline-block; padding: 2px 9px; border-radius: 10px; font-size: 0.74rem; font-weight: 600; }
.rc-pill.g { background: #E3F1EA; color: #1F6B49; } .rc-pill.w { background: #FBF0D9; color: #8A5A0B; } .rc-pill.b { background: #F7E1DF; color: #8E2F28; } .rc-pill.n { background: #EEF0F3; color: #52606D; }
.rc-note { font-size: 0.78rem; color: #6B7280; }
[data-testid="stSidebar"] { background: #F7F8FA; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    inject_css()
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="rc-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def callout(line1: str, line2: str):
    st.markdown(f'<div class="rc-callout"><b>{html.escape(line1)}</b><br><span>{html.escape(line2)}</span></div>',
                unsafe_allow_html=True)


def flow_chips():
    steps = ["RETURN", "RECOVER", "REUSE", "REFURBISH", "TRANSFORM", "RECYCLE"]
    inner = '<span class="rc-arrow">→</span>'.join(f'<span class="rc-chip">{s}</span>' for s in steps)
    st.markdown(f'<div class="rc-flow">{inner}</div>', unsafe_allow_html=True)


def kpi_cards(items):
    """items: (label, value, sub, tone) — tone in {'', 'good', 'bad', 'warn'}"""
    cols = st.columns(len(items))
    for col, (label, value, sub, tone) in zip(cols, items):
        col.markdown(f'<div class="rc-kpi {tone}"><div class="l">{html.escape(label)}</div>'
                     f'<div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(str(sub))}</div></div>',
                     unsafe_allow_html=True)


def pill(text: str, tone: str = "n") -> str:
    return f'<span class="rc-pill {tone}">{html.escape(text)}</span>'


def disclaimer():
    st.caption("Synthetic data · simulated values · decision-support estimates only — not a financial guarantee and "
               "not a replacement for physical inspection.")


# ----------------------------------------------------------------------------
# Streamlit-version-safe wrappers
# ----------------------------------------------------------------------------
def plot(fig, key=None):
    fig = style_fig(fig)
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def table(df, key=None, selectable=False, column_config=None, height=None):
    kw = dict(hide_index=True)
    if column_config is not None:
        kw["column_config"] = column_config
    if height is not None:
        kw["height"] = height
    if key is not None:
        kw["key"] = key
    if selectable:
        kw.update(on_select="rerun", selection_mode="single-row")
    try:
        return st.dataframe(df, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig, height=None):
    fig.update_layout(template="simple_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="-apple-system, Segoe UI, Inter, sans-serif", color=INK, size=12),
                      margin=dict(l=10, r=10, t=36, b=10), legend=dict(orientation="h", y=-0.15, x=0),
                      title=dict(font=dict(size=14), x=0))
    if height:
        fig.update_layout(height=height)
    return fig


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset …")
def load_returns():
    """Returns (df | None, source_description, meta_dict)."""
    meta = {}
    try:
        mp = DATA_PROCESSED / "metadata.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        meta = {}
    df, source = None, ""
    try:
        for path, name in ((DATA_PROCESSED / "returns_enriched.csv.gz", "processed dataset"),
                           (DATA_SAMPLE / "sample_returns.csv", "bundled sample dataset")):
            if path.exists():
                df = pd.read_csv(path, parse_dates=["order_date", "return_date"])
                source = name
                break
        if df is None:  # last resort: build a small dataset in memory
            from src.data_generator import build
            _, _, _, df = build("demo", orders_override=3000, write=False)
            source = "in-memory demo dataset (run `python src/data_generator.py --mode demo` for the full demo)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not load data: {type(exc).__name__}: {exc}", meta
    missing = validate_columns(df)
    if missing:
        return None, f"Dataset is missing required columns: {', '.join(missing)}. Regenerate it with the data generator.", meta
    return df, source, meta


@st.cache_data(show_spinner=False)
def load_orders_summary():
    p = DATA_PROCESSED / "orders_summary.csv.gz"
    try:
        return pd.read_csv(p, parse_dates=["order_date"]) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    out = {"models": {}, "metrics": None, "error": None}
    try:
        for n in ("condition", "recovery", "value"):
            p = MODELS_DIR / f"{n}_model.pkl"
            if not p.exists():
                out["error"] = f"Model file '{p.name}' not found."
                return out
            out["models"][n] = joblib.load(p)
        mp = MODELS_DIR / "metrics.json"
        out["metrics"] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
    except Exception as exc:  # noqa: BLE001
        out["models"] = {}
        out["error"] = f"Saved models could not be loaded ({type(exc).__name__}); they may be corrupted or built with a different library version."
    return out


def get_data():
    df, source, meta = load_returns()
    if df is None:
        st.error(source)
        st.info("Create the dataset with `python src/data_generator.py --mode demo`, or use the **About** page → Data tools.")
        st.stop()
    return df, source, meta


def ml_predict(a: dict):
    """ML cross-check for one analysed product; returns None if models are unavailable."""
    bundle = load_models()
    if bundle["error"] or not bundle["models"]:
        return None
    try:
        from src.feature_engineering import build_features
        row = {**a["record"], "condition_score": a["condition"]["score"],
               "estimated_repair_cost": a["values_used"]["repair_cost"],
               "estimated_resale_value": a["pathways"]["resale"]["gross"]}
        d = pd.DataFrame([row])

        def cls(name, include_econ):
            b = bundle["models"][name]
            X = build_features(d, include_econ).reindex(columns=b["features"], fill_value=0)
            proba = b["model"].predict_proba(X)[0]
            classes = b["label_encoder"].classes_
            inv = b.get("remap_inv") or {i: i for i in range(len(proba))}
            probs = {str(classes[inv[i]]): float(p) for i, p in enumerate(proba)}
            best = max(probs, key=probs.get)
            return best, probs[best], probs

        g, gp, _ = cls("condition", False)
        pw, pp, pall = cls("recovery", True)
        vb = bundle["models"]["value"]
        Xv = build_features(d, True).reindex(columns=vb["features"], fill_value=0)
        return {"grade": g, "grade_p": gp, "pathway": pw, "pathway_p": pp, "pathway_probs": pall,
                "value": float(vb["model"].predict(Xv)[0])}
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Product selection (shared across pages through session_state)
# ----------------------------------------------------------------------------
def row_to_record(row) -> dict:
    d = dict(row)
    for k, v in list(d.items()):
        if isinstance(v, float) and np.isnan(v):
            d[k] = None
    return d


def _ret_label(r) -> str:
    return f"{r['return_id']} · {r['category']} · {inr(r['original_value'])} · Grade {r['condition_grade']}"


def product_picker(df, key: str, allow_manual: bool = True):
    ss = st.session_state
    opts = ["Demo scenarios", "Returned products (dataset)"]
    if allow_manual and ss.get("manual_record"):
        opts.append("Manual product")
    cur = ss.get("active_src", opts[0])
    if cur not in opts:
        cur = opts[0]
    src = st.radio("Product source", opts, index=opts.index(cur), horizontal=True, key=f"{key}_src")
    ss["active_src"] = src
    if src == "Demo scenarios":
        names = list(DEMO_PRODUCTS)
        cur_d = ss.get("active_demo", names[0])
        sel = st.selectbox("Built-in example", names, index=names.index(cur_d) if cur_d in names else 0, key=f"{key}_demo")
        ss["active_demo"] = sel
        return dict(DEMO_PRODUCTS[sel]), sel
    if src == "Manual product":
        return dict(ss["manual_record"]), ss["manual_record"].get("product_name") or "Manual product"
    c1, c2 = st.columns([1, 2])
    q = c1.text_input("Search Product ID / Category", key=f"{key}_q", placeholder="e.g. laptops, P00123, RET-000045")
    sub = an.apply_filters(df, search=q).sort_values("original_value", ascending=False)
    if sub.empty:
        st.warning("No returned products match this search.")
        st.stop()
    sub = sub.head(300)
    ids = sub["return_id"].tolist()
    labels = {r["return_id"]: _ret_label(r) for _, r in sub.iterrows()}
    cur_r = ss.get("active_return")
    if cur_r in set(df["return_id"]) and cur_r not in labels:
        row = df[df["return_id"] == cur_r].iloc[0]
        ids.insert(0, cur_r)
        labels[cur_r] = _ret_label(row)
    sel = c2.selectbox(f"Returned product ({len(sub):,} shown)", ids, index=ids.index(cur_r) if cur_r in ids else 0,
                       format_func=lambda i: labels[i], key=f"{key}_ret")
    ss["active_return"] = sel
    row = df[df["return_id"] == sel].iloc[0]
    rec = row_to_record(row)
    return rec, f"{rec.get('product_name') or rec['category']} ({sel})"


# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
def filter_bar(df, key: str, extended: bool = False):
    keys = [f"{key}_date", f"{key}_cat", f"{key}_sell", f"{key}_grade", f"{key}_reason", f"{key}_path",
            f"{key}_val", f"{key}_search"]

    def _reset():
        for k in keys:
            st.session_state.pop(k, None)

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        dmin, dmax = df["order_date"].min().date(), df["order_date"].max().date()
        dr = c1.date_input("Order date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"{key}_date")
        cats = c2.multiselect("Category", sorted(df["category"].unique()), key=f"{key}_cat")
        sellers = c3.multiselect("Seller type", sorted(df["seller_type"].unique()), key=f"{key}_sell")
        c4, c5, c6 = st.columns(3)
        grades = c4.multiselect("Condition grade", GRADES, key=f"{key}_grade",
                                format_func=lambda g: f"{g} · {GRADE_LABELS[g]}")
        reasons = c5.multiselect("Return reason", sorted(df["return_reason"].unique()), key=f"{key}_reason")
        paths = vr = search = None
        if extended:
            paths = c6.multiselect("Recommended pathway", [PATHWAY_LABELS[p] for p in PATHWAYS], key=f"{key}_path")
            lo, hi = float(df["original_value"].min()), float(df["original_value"].max())
            vr = st.slider("Original value range (₹)", lo, hi, (lo, hi), key=f"{key}_val")
            search = st.text_input("Search Product ID / Category / Name", key=f"{key}_search")
        else:
            c6.button("Reset filters", on_click=_reset, key=f"{key}_reset")
        if extended:
            st.button("Reset filters", on_click=_reset, key=f"{key}_reset")
    date_range = tuple(dr) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None
    if date_range is None:
        st.info("Pick an end date to apply the date filter.")
    out = an.apply_filters(df, date_range, cats, sellers, grades, reasons, paths, vr, search)
    if out.empty:
        st.warning("No returns match the current filters. Try widening them or press **Reset filters**.")
        st.stop()
    return out, {"date_range": date_range, "categories": cats, "sellers": sellers}


# ----------------------------------------------------------------------------
# Result renderers
# ----------------------------------------------------------------------------
def recommendation_card(a: dict, label: str | None = None):
    k = a["recommended"]
    p = a["pathways"][k]
    st.markdown(
        f'<div class="rc-card rec"><div class="rc-tag">RECOMMENDED</div>'
        f'<div class="rc-title">{html.escape(PATHWAY_HEADLINE[k])}</div>'
        f'<div class="rc-grid">'
        f'<div><span>Expected net recovery</span><b>{inr(p["net"])}</b></div>'
        f'<div><span>Value preserved</span><b>{p["value_preserved"]:.1f}%</b></div>'
        f'<div><span>Processing time</span><b>{p["days"]:.0f} days</b></div>'
        f'<div><span>Confidence</span><b>{a["confidence"]:.0%}</b></div></div>'
        f'<div class="rc-why"><b>Why?</b> {html.escape(a.get("reasoning", ""))}</div></div>', unsafe_allow_html=True)
    alt = a.get("alternative")
    if alt:
        ap = a["pathways"][alt]
        st.caption(f"Alternative: **{PATHWAY_LABELS[alt]}** — net {inr(ap['net'])}, {ap['days']:.0f} days, score {ap['score']:.0f}. "
                   "Confidence is a heuristic based on score separation and execution risk.")


def journey_html(a: dict):
    steps = "".join(f'<div class="rc-step"><div class="n">{html.escape(s["step"])}</div>'
                    f'<div class="d">{html.escape(s["detail"])}</div></div>' for s in a["journey"])
    st.markdown(f'<div class="rc-journey">{steps}</div>', unsafe_allow_html=True)


def pathway_net_chart(a: dict, title="Net recovery by pathway", height=300):
    names = [PATHWAY_LABELS[k] for k in PATHWAYS]
    nets = [a["pathways"][k]["net"] for k in PATHWAYS]
    colors = [GOOD if k == a["recommended"] else (BAD if a["pathways"][k]["net"] < 0 else "#9AA5B1") for k in PATHWAYS]
    fig = go.Figure(go.Bar(x=names, y=nets, marker_color=colors, text=[inr_short(n) for n in nets], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(title=title, yaxis_title="₹", showlegend=False)
    fig.update_yaxes(zeroline=True, zerolinecolor="#9AA5B1")
    style_fig(fig, height)
    plot(fig)


def waterfall_chart(a: dict, key: str, height=330):
    ct = cost_table(a, key)
    labels = ct["Item"].tolist()
    vals = ct["Amount"].tolist()
    measure = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(x=labels, y=vals[:-1] + [vals[-1]], measure=measure,
                                 increasing=dict(marker_color=GOOD), decreasing=dict(marker_color=BAD),
                                 totals=dict(marker_color=SLATE), connector=dict(line=dict(color="#D5D9DE")),
                                 text=[inr_short(v) for v in vals], textposition="outside"))
    fig.update_layout(title=f"{PATHWAY_LABELS[key]} — from gross value to net recovery", showlegend=False)
    fig.update_xaxes(tickangle=-25)
    style_fig(fig, height)
    plot(fig)


def condition_panel(a: dict):
    c = a["condition"]
    tone = {"A": "g", "B": "g", "C": "w", "D": "w", "E": "b"}[c["grade"]]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="rc-kpi"><div class="l">Condition score</div><div class="v">{c["score"]:.0f}<span style="font-size:1rem;color:#6B7280"> / 100</span></div>'
                    f'<div class="s">{pill("Grade " + c["grade"], tone)} {html.escape(c["grade_label"])}</div></div>', unsafe_allow_html=True)
        st.markdown("**Detected issues**")
        if c["issues"]:
            for i in c["issues"]:
                st.markdown(f"- {i}")
        else:
            st.markdown("- No significant issues detected")
    with right:
        rows = [(f["label"], f["delta"]) for f in c["factors"]] + [(n, v) for n, v in c["penalties"]]
        rows = sorted(rows, key=lambda x: x[1])
        fig = go.Figure(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                               marker_color=[BAD if r[1] <= -1.5 else "#C9CED6" for r in rows],
                               text=[f"{r[1]:+.1f}" for r in rows], textposition="outside", cliponaxis=False))
        fig.update_layout(title="Points lost per factor (0 = full marks)", showlegend=False,
                          xaxis=dict(range=[min(min(r[1] for r in rows) * 1.25, -3), 2]))
        fig.update_yaxes(autorange="reversed")
        style_fig(fig, 300)
        plot(fig)
    with st.expander("Condition score explanation"):
        st.code(c["explanation"], language="text")
        ft = pd.DataFrame([{"Factor": f["label"], "Points earned": f["earned"], "Max points": f["max"]} for f in c["factors"]])
        table(ft)


def components_table(a: dict):
    df = pd.DataFrame(a["components"])
    if df.empty:
        st.info("No component model for this category.")
        return
    out = pd.DataFrame({"Component": df["component"], "Health": df["health"].astype(str) + "%", "Status": df["status"],
                        "Route": df["route"], "Est. value": df["value"].map(inr)})
    table(out)


def pathway_table_styled(a: dict):
    t = pathway_table(a)
    rec = PATHWAY_LABELS[a["recommended"]]

    def hl(row):
        return ["background-color: #E3F1EA; font-weight: 600" if row["Pathway"] == rec else "" for _ in row]

    styled = t.style.apply(hl, axis=1).format({"Gross Value": inr, "Costs": inr, "Net Recovery": inr,
                                               "Value Preserved %": "{:.1f}%", "Time (days)": "{:.1f}", "Score": "{:.0f}"})
    table(styled)


def export_buttons(a: dict, key: str, label: str | None = None):
    st.markdown("**Download report**")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Report (Markdown)", build_markdown(a, label), "returncycle_report.md", "text/markdown", key=f"{key}_md")
    c2.download_button("Report (JSON)", build_json(a, label), "returncycle_report.json", "application/json", key=f"{key}_js")
    c3.download_button("Pathways (CSV)", build_csv(a), "returncycle_pathways.csv", "text/csv", key=f"{key}_csv")
    pdf = build_pdf(a, label)
    if pdf:
        c4.download_button("Report (PDF)", pdf, "returncycle_report.pdf", "application/pdf", key=f"{key}_pdf")
    else:
        c4.button("PDF unavailable (install fpdf2)", disabled=True, key=f"{key}_pdf_x")


def ml_crosscheck(a: dict):
    m = ml_predict(a)
    with st.expander("Machine-learning cross-check (XGBoost models trained on the synthetic data)"):
        if m is None:
            st.info("ML models are not available. Train them with `python src/model_training.py` (or use Model Insights → Train).")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("ML condition grade", m["grade"], f"engine: {a['condition']['grade']}", delta_color="off")
        c2.metric("ML recommended pathway", m["pathway"], f"{m['pathway_p']:.0%} probability", delta_color="off")
        c3.metric("ML value estimate (current practice)", inr(m["value"]))
        agree = m["pathway"] == PATHWAY_LABELS[a["recommended"]]
        st.caption(("The ML model agrees with the economic optimizer. " if agree else
                    "The ML model disagrees with the optimizer — treat the deterministic calculation as authoritative. ")
                   + "ML mimics the engine for fast batch scoring; the value model estimates recovery under the simulated "
                     "current (legacy) handling policy, not the optimized one.")
