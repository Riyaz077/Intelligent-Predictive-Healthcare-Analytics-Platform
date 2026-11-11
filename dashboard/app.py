from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import split_features_target
from src.evaluation import evaluate_models, permutation_importance_frame

st.set_page_config(page_title="Healthcare Intelligence", page_icon="🩺", layout="wide")
st.title("🩺 AI-Powered Healthcare Intelligence")
st.caption("Educational machine-learning prototype • Not for diagnosis or treatment decisions")

with st.sidebar:
    st.header("Project navigation")
    page = st.radio("Go to", ["Overview", "Dataset & EDA", "Model Training", "Prediction"])
    st.divider()
    st.warning("Do not upload identifiable patient information. This tool is not clinically validated.")

st.session_state.setdefault("df", None)
st.session_state.setdefault("metrics", None)
st.session_state.setdefault("fitted", None)
st.session_state.setdefault("feature_columns", None)

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction tasks", "3 planned")
    c2.metric("Candidate algorithms", "5+")
    c3.metric("Workflow", "End-to-end")
    st.markdown("""
    This prototype supports independent classification experiments for heart disease,
    diabetes and breast cancer datasets. Upload a prepared CSV, inspect it, select a
    target, train candidate models and explore a demonstration prediction.

    **Workflow:** Data → EDA → Preprocessing → Model training → Evaluation → Prediction.
    """)
    st.info("Start by uploading a CSV in the Dataset & EDA page.")

elif page == "Dataset & EDA":
    st.subheader("Upload and inspect a dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        try:
            st.session_state.df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(st.session_state.df):,} rows and {len(st.session_state.df.columns)} columns.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    df = st.session_state.df
    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(30), use_container_width=True)
        st.subheader("Missing values")
        missing = df.isna().sum().rename("Missing values").reset_index(names="Feature")
        st.dataframe(missing, use_container_width=True)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            st.subheader("Correlation heatmap")
            corr = numeric.corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", title="Numeric feature correlations"), use_container_width=True)
            selected = st.selectbox("Numeric feature distribution", numeric.columns)
            st.plotly_chart(px.histogram(df, x=selected, color_discrete_sequence=None), use_container_width=True)
        st.subheader("Target distribution")
        target = st.selectbox("Choose target column", df.columns, key="eda_target")
        st.plotly_chart(px.histogram(df, x=target, title=f"Distribution of {target}"), use_container_width=True)

elif page == "Model Training":
    st.subheader("Train and compare classification models")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first in Dataset & EDA.")
    else:
        target = st.selectbox("Target column", df.columns, key="train_target")
        if st.button("Train candidate models", type="primary"):
            try:
                X, y = split_features_target(df, target)
                with st.spinner("Training and evaluating models..."):
                    metrics, fitted = evaluate_models(X, y)
                st.session_state.metrics = metrics
                st.session_state.fitted = fitted
                st.session_state.feature_columns = X.columns.tolist()
                st.session_state.target = target
                st.success("Training completed. Results are based on a held-out test split.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        if st.session_state.metrics is not None:
            st.dataframe(st.session_state.metrics.style.format(precision=3), use_container_width=True)
            st.plotly_chart(px.bar(st.session_state.metrics, x="Model", y="F1_weighted",
                                   title="Weighted F1 comparison", range_y=[0, 1]), use_container_width=True)
            model_name = st.selectbox("Inspect model", list(st.session_state.fitted))
            item = st.session_state.fitted[model_name]
            cm = item["confusion_matrix"]
            st.plotly_chart(px.imshow(cm, text_auto=True, title=f"{model_name}: confusion matrix",
                                      labels=dict(x="Predicted", y="Actual")), use_container_width=True)
            try:
                importance = permutation_importance_frame(item["pipeline"], item["X_test"], item["y_test"])
                st.plotly_chart(px.bar(importance.head(15), x="Importance", y="Feature",
                                       orientation="h", title="Permutation feature importance"), use_container_width=True)
            except Exception as exc:
                st.caption(f"Feature importance unavailable: {exc}")
            st.download_button("Download metrics CSV", st.session_state.metrics.to_csv(index=False),
                               file_name="model_metrics.csv", mime="text/csv")

elif page == "Prediction":
    st.subheader("Demonstration prediction")
    df = st.session_state.df
    fitted = st.session_state.fitted
    if df is None or not fitted:
        st.warning("Upload data and train at least one model first.")
    else:
        model_name = st.selectbox("Choose trained model", list(fitted))
        pipeline = fitted[model_name]["pipeline"]
        features = st.session_state.feature_columns
        st.caption("Enter a sample feature vector. Values are for software demonstration only.")
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            series = df[feature]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    median = float(series.median()) if series.notna().any() else 0.0
                    values[feature] = st.number_input(feature, value=median, key=f"pred_{feature}")
                else:
                    choices = series.dropna().astype(str).unique().tolist()
                    values[feature] = st.selectbox(feature, choices or [""], key=f"pred_{feature}")
        if st.button("Generate demonstration prediction", type="primary"):
            try:
                sample = pd.DataFrame([values], columns=features)
                prediction = pipeline.predict(sample)[0]
                st.metric("Predicted dataset class", str(prediction))
                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(sample)[0]
                    classes = pipeline.named_steps["classifier"].classes_
                    prob_df = pd.DataFrame({"Class": classes.astype(str), "Probability": probabilities})
                    st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", range_y=[0, 1],
                                           title="Model class probabilities"), use_container_width=True)
                st.warning("This is a model output, not a medical diagnosis or an estimate of clinical risk.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
