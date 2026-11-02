import streamlit as st

from src import ui
from src.utils import DATASET_LABEL, TAGLINE_1, TAGLINE_2

df, source, meta = ui.get_data()
ui.page_header("Methodology & Responsible AI", "How ReturnCycle AI reaches a recommendation, and what it does not claim.")
ui.callout(TAGLINE_1, TAGLINE_2)
st.markdown("""
**Pipeline** — Synthetic data → cleaning → feature engineering → condition intelligence → recovery value estimation →
next-life candidate generation → economic optimization → counterfactual simulation → decision dashboard.

**Net recovery per pathway** = gross recovery value − repair − refurbishment − processing − transformation − logistics − warehouse − disposal − expected loss.
**Value preserved %** = net recovery ÷ original product value × 100. Reverse logistics already incurred is included for every pathway so options are comparable.

**Pathway score** = economic value + condition suitability + component usability + market potential − processing cost − time − operational risk, normalised to 0–100.
Disposal is only recommended when every alternative loses more value than disposal itself.

**Techniques** — deterministic rules (condition, economics), a scoring optimizer, rule-based transformation catalogue, and XGBoost models for fast cross-checks.
""")
st.subheader("Responsible AI")
st.markdown(f"""
- {DATASET_LABEL} No Amazon, Flipkart or other proprietary data is used.
- All financial values are simulated assumptions, not guarantees. Actual economics depend on market conditions.
- Recommendations are decision-support estimates and do not replace physical inspection.
- Transformation suggestions are *estimated / simulated opportunities* and require human validation.
- Model outputs are probabilistic; "confidence" is a heuristic from score separation and execution risk.
- The demo set is return-enriched: return rates are higher than real-world averages. The "current practice" baseline is a simulated legacy policy.
""")
st.subheader("Data tools")
st.caption(f"Loaded: {source}. " + (f"{meta.get('returns', '?'):,} returns." if meta else ""))
c1, c2 = st.columns(2)
for col, mode in ((c1, "demo"), (c2, "full")):
    if col.button(f"Generate {mode} dataset"):
        with st.spinner(f"Generating {mode} dataset (this can take a minute) …"):
            try:
                from src.data_generator import build
                build(mode)
                st.cache_data.clear()
                st.success("Dataset generated. Reload the page; retrain models from Model Insights.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Generation failed: {e}")
