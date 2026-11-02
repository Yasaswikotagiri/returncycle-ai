import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.explainability import shap_global, shap_local
from src.feature_engineering import build_features
from src.utils import inr

df, source, meta = ui.get_data()
ui.page_header("Model Insights", "What each technique does, how well the models perform, and what drives them.")
st.markdown("**Which technique is used where**")
ui.table(pd.DataFrame([
    ("Condition score & grade", "Deterministic weighted rules", "Transparent, auditable"),
    ("Recovery economics (net per pathway)", "Deterministic arithmetic", "Money should be computed, not guessed"),
    ("Pathway ranking & recommendation", "Normalised scoring optimizer", "Weights editable on Optimizer page"),
    ("Transformation ideas", "Rule-based catalogue", "Needs human validation"),
    ("Fast grade / pathway / value prediction", "XGBoost models (ML)", "Cross-check & batch scoring"),
], columns=["Component", "Technique", "Why"]))
b = ui.load_models()
if b["error"] or not b["metrics"]:
    st.warning((b["error"] or "No metrics file found.") + " Train the models with `python src/model_training.py`, or press the button below.")
    if st.button("Train models now", type="primary"):
        with st.spinner("Training three models …"):
            try:
                from src.model_training import train_all
                train_all(verbose=False)
                ui.load_models.clear()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Training failed: {e}")
    st.stop()
M = b["metrics"]
st.caption(f"Algorithm: {M['algorithm']} · trained {M['trained_at']} on {M['n_rows']:,} synthetic returns · 80/20 train/test split. {M['note']}")
t1, t2, t3 = st.tabs(["Condition grade", "Recovery pathway", "Recovery value"])
for tab, name in ((t1, "condition"), (t2, "recovery")):
    with tab:
        m = M[name]
        st.write(f"**Target:** `{m['target']}` · **Model:** {m['model']} · test rows {m['n_test']:,}")
        ui.kpi_cards([("Accuracy", f"{m['accuracy']:.3f}", f"baseline {m['majority_class_baseline_accuracy']:.3f}", "good"),
                      ("Precision (weighted)", f"{m['precision_weighted']:.3f}", "", ""), ("Recall (weighted)", f"{m['recall_weighted']:.3f}", "", ""),
                      ("F1 (weighted)", f"{m['f1_weighted']:.3f}", f"macro {m['f1_macro']:.3f}", "")])
        c1, c2 = st.columns(2)
        with c1:
            fi = pd.Series(m["feature_importance"]).sort_values()
            fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#52606D"))
            fig.update_layout(title="Top feature importances", showlegend=False)
            ui.plot(fig)
        with c2:
            fig = go.Figure(go.Heatmap(z=m["confusion_matrix"], x=m["classes"], y=m["classes"], colorscale="Greens", showscale=False,
                                       text=m["confusion_matrix"], texttemplate="%{text}"))
            fig.update_layout(title="Confusion matrix (rows = actual)", xaxis_title="Predicted")
            fig.update_yaxes(autorange="reversed")
            ui.plot(fig)
with t3:
    m = M["value"]
    ui.kpi_cards([("RMSE", inr(m["rmse"]), f"mean-predictor {inr(m['baseline_rmse_predict_mean'])}", ""), ("MAE", inr(m["mae"]), "", ""),
                  ("R²", f"{m['r2']:.3f}", "test set", "good")])
    fi = pd.Series(m["feature_importance"]).sort_values()
    fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#52606D"))
    fig.update_layout(title="Top feature importances", showlegend=False)
    ui.plot(fig)
st.subheader("SHAP explanation")
st.caption("Falls back to built-in feature importance automatically if SHAP is unavailable.")
try:
    model = b["models"]["recovery"]
    X = build_features(df.sample(min(len(df), 2000), random_state=1), True).reindex(columns=model["features"], fill_value=0)
    sg, err = shap_global(model["model"], X)
    if sg is None:
        st.info(err)
    else:
        top = sg.head(12).sort_values()
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#2E6F5E"))
        fig.update_layout(title="Mean |SHAP| — recovery pathway model", showlegend=False)
        ui.plot(fig)
        st.markdown("Top factors influencing the recovery recommendation: " + ", ".join(f"{i + 1}. {n}" for i, n in enumerate(sg.index[:6])))
        pick = st.selectbox("Explain one product", df["return_id"].head(200).tolist())
        row = df[df["return_id"] == pick]
        x1 = build_features(row, True).reindex(columns=model["features"], fill_value=0)
        pred = int(model["model"].predict(x1)[0])
        sl, err2 = shap_local(model["model"], x1, pred)
        if sl is None:
            st.info(err2)
        else:
            t = sl.head(8).iloc[::-1]
            fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color=["#2E7D5B" if v > 0 else "#B4443C" for v in t.values]))
            fig.update_layout(title="Local SHAP for predicted class", showlegend=False)
            ui.plot(fig)
except Exception as e:  # noqa: BLE001
    st.info(f"SHAP could not run ({type(e).__name__}); feature importance above is shown instead.")
if st.button("Retrain models"):
    with st.spinner("Training …"):
        try:
            from src.model_training import train_all
            train_all(verbose=False)
            ui.load_models.clear()
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
ui.disclaimer()
