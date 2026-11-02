"""ReturnCycle AI — entry point.  Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="ReturnCycle AI", page_icon="♻", layout="wide", initial_sidebar_state="expanded")

pages = {
    "Decision intelligence": [
        st.Page("pages/dashboard.py", title="Dashboard", default=True),
        st.Page("pages/product_inspection.py", title="Product Inspection"),
        st.Page("pages/next_life.py", title="Next-Life Pathways"),
        st.Page("pages/recovery_optimizer.py", title="Recovery Optimizer"),
        st.Page("pages/simulator.py", title="Counterfactual Simulator"),
    ],
    "Analytics": [
        st.Page("pages/return_intelligence.py", title="Return Intelligence"),
        st.Page("pages/analytics.py", title="Recovery & Circularity"),
        st.Page("pages/model_insights.py", title="Model Insights"),
    ],
    "About": [st.Page("pages/about.py", title="Methodology & Responsible AI")],
}
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("**ReturnCycle AI**")
    st.caption("Returned product next-life & value recovery intelligence")
    st.caption("A returned product is not necessarily a lost product.")
    st.caption("Synthetic demonstration dataset — simulated values.")
nav.run()
