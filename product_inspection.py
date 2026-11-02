import streamlit as st

from src import ui
from src.data_cleaning import normalise_record
from src.next_life_engine import analyze_product, pathway_table
from src.recovery_engine import COST_LABELS
from src.utils import CATEGORIES, CATEGORY_PROFILES, inr, TAGLINE_2
import pandas as pd

df, source, meta = ui.get_data()
ui.page_header("Product Inspection", "Grade a returned product's condition and discover what value is still inside it.")
ss = st.session_state
mode = st.radio("Input", ["Select a returned product", "Enter product details manually"], horizontal=True)

if mode.startswith("Select"):
    rec, label = ui.product_picker(df, "insp", allow_manual=False)
    if st.button("Analyze Product", type="primary"):
        ss["insp"] = {"record": rec, "label": label}
else:
    cat = st.selectbox("Category", CATEGORIES, key="m_cat")
    p = CATEGORY_PROFILES[cat]
    keys = ["m_name", "m_orig", "m_age", "m_cos", "m_fun", "m_scr", "m_bat", "m_comp", "m_acc", "m_water", "m_use", "m_pack", "m_rep"]

    def _reset():
        for k in keys:
            ss.pop(k, None)

    with st.form("manual"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Product name", "My returned product", key="m_name")
        orig = c2.number_input("Original value (₹)", 100.0, 5_000_000.0, 50000.0, 500.0, key="m_orig")
        age = c3.number_input("Product age (days)", 0, 5000, 200, key="m_age")
        c4, c5, c6 = st.columns(3)
        cos = c4.slider("Cosmetic damage %", 0, 100, 20, key="m_cos")
        fun = c5.slider("Functional damage %", 0, 100, 10, key="m_fun")
        comp = c6.slider("Component health", 0, 100, 85, key="m_comp")
        c7, c8, c9 = st.columns(3)
        scr = c7.slider("Screen condition", 0, 100, 90, key="m_scr") if p["has_screen"] else None
        bat = c8.slider("Battery health", 0, 100, 90, key="m_bat") if p["has_battery"] else None
        acc = c9.number_input("Missing accessories", 0, p["max_acc"], 0, key="m_acc")
        c10, c11, c12 = st.columns(3)
        use = c10.slider("Usage level %", 0, 100, 20, key="m_use")
        pack = c11.slider("Packaging condition", 0, 100, 80, key="m_pack")
        water = c12.checkbox("Water damage", key="m_water")
        rep = st.number_input("Known repair estimate (₹) — 0 = estimate automatically", 0.0, 5_000_000.0, 0.0, 100.0, key="m_rep")
        go = st.form_submit_button("Analyze Product", type="primary")
    st.button("Reset form", on_click=_reset)
    if go:
        raw = dict(product_name=name, category=cat, original_value=orig, product_age_days=age, cosmetic_damage=cos,
                   functional_damage=fun, component_health=comp, screen_condition=scr, battery_health=bat,
                   missing_accessories=acc, usage_level=use, packaging_condition=pack, water_damage=int(water),
                   repair_cost_override=rep or None)
        try:
            normalise_record(raw)
            ss["manual_record"] = raw
            ss["active_src"] = "Manual product"
            ss["insp"] = {"record": raw, "label": name}
        except ValueError as e:
            st.error(str(e))

res = ss.get("insp")
if not res:
    st.info("Choose a product and press **Analyze Product** to see its condition, recoverable value and best next life.")
    st.stop()
try:
    a = analyze_product(res["record"])
except ValueError as e:
    st.error(f"Could not analyze this product: {e}")
    st.stop()

st.divider()
st.subheader(f"Inspection result — {res['label']}")
ui.condition_panel(a)
st.subheader("Recommended action")
ui.recommendation_card(a)
c1, c2 = st.columns(2)
with c1:
    st.subheader("Recoverable components")
    ui.components_table(a)
with c2:
    st.subheader("Estimated value")
    v = a["values_used"]
    ui.kpi_cards([("Market value (working)", inr(a["market_value"]), "after depreciation & usage", ""),
                  ("Repair estimate", inr(v["repair_cost"]), "to restore", "warn")])
    if a["repair_items"]:
        ui.table(pd.DataFrame({"Repair item": list(a["repair_items"]), "Cost": [inr(x) for x in a["repair_items"].values()]}))
st.subheader("Possible next lives")
ui.pathway_table_styled(a)
ui.ml_crosscheck(a)
ui.export_buttons(a, "insp", res["label"])
ui.disclaimer()
