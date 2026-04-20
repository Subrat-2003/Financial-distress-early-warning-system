import streamlit as st
import pandas as pd
import numpy as np
import os, warnings, gc
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLD_PATH    = "./data/gold_dataset.parquet"
SENT_PATH    = "./data/mda_sentiment_signals.parquet"
LSTM_PATH    = "./data/final_lstm_dataset.parquet"
XGB_PATH     = "./models/xgb_model.json"
SCALER_PATH  = "./models/scaler.pkl"
HISTORY_PATH = "./models/training_history_v2.csv"

# ── Feature columns — exact training order ────────────────────────────────────
FEATURE_COLS = [
    "current_ratio", "quick_ratio", "cash_ratio",
    "roa", "profit_margin", "operating_margin", "roe",
    "debt_to_assets", "debt_to_equity", "asset_turnover",
    "interest_coverage", "retained_earnings_ratio",
    "revenue_growth_rate", "sentiment_signal",
    "persistent_distress_flag", "Assets", "Revenues"
]

st.set_page_config(page_title="SEC Risk Dashboard", layout="wide", page_icon="📉")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.caption("v2.1 | Gold Dataset | ~9k companies")

page = st.sidebar.radio("Page", [
    "Overview", "Predictions", "SHAP Explainer",
    "LSTM Analysis", "Sentiment Signals", "Raw Data"
])

# ── Startup health check ──────────────────────────────────────────────────────
def validate_assets():
    missing = []
    for label, path in [
        ("Gold dataset",  GOLD_PATH),
        ("XGBoost model", XGB_PATH),
        ("Scaler",        SCALER_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"❌ {label} not found at: `{path}`")
    if missing:
        for msg in missing:
            st.error(msg)
        st.stop()

validate_assets()

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_gold():
    df = pd.read_parquet(GOLD_PATH)
    gc.collect()
    return df

@st.cache_data(ttl=3600)
def load_sentiment():
    if os.path.exists(SENT_PATH):
        df = pd.read_parquet(SENT_PATH)
        gc.collect()
        return df
    return None

@st.cache_data(ttl=3600)
def load_lstm_data():
    if os.path.exists(LSTM_PATH):
        df = pd.read_parquet(LSTM_PATH)
        gc.collect()
        return df, "final_lstm_dataset.parquet"
    return None, None

@st.cache_data(ttl=3600)
def load_training_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return None

@st.cache_resource
def load_xgb_model():
    model = XGBClassifier()
    model.load_model(XGB_PATH)
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_label(df):
    if "crash_label" in df.columns:
        return df
    if "target_crash" in df.columns:
        df["crash_label"] = df["target_crash"]
        return df
    df["crash_label"] = 0
    return df

def get_model_input(df):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    return df[feat].fillna(0), feat

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    primary_df = load_gold()
    sent_df    = load_sentiment()

primary_df   = ensure_label(primary_df)
feature_cols = [c for c in FEATURE_COLS if c in primary_df.columns]

st.sidebar.metric("Companies loaded", f"{len(primary_df):,}")

try:
    model = load_xgb_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.warning(f"XGB model load failed: {e}")

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("📉 SEC Financial Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total companies",  f"{len(primary_df):,}")
    c2.metric("Companies",        f"{primary_df['cik'].nunique():,}" if "cik" in primary_df.columns else "N/A")
    c3.metric("Crash rate",       f"{primary_df['crash_label'].mean():.1%}")
    c4.metric("Features loaded",  len(feature_cols))

    st.subheader("Crash label distribution")
    vc = primary_df["crash_label"].value_counts().reset_index()
    vc.columns = ["label","count"]
    fig = px.bar(vc, x="label", y="count", color="label",
                 color_discrete_map={0:"#3B8BD4", 1:"#E24B4A"},
                 labels={"label":"0 = Safe  |  1 = Crash"})
    st.plotly_chart(fig, use_container_width=True)

    if feature_cols:
        st.subheader("Feature distributions by crash label")
        sel  = st.selectbox("Feature", feature_cols)
        fig2 = px.violin(primary_df, y=sel, color="crash_label",
                         box=True, points="outliers",
                         color_discrete_map={0:"#3B8BD4", 1:"#E24B4A"})
        st.plotly_chart(fig2, use_container_width=True)

    year_col = next((c for c in ["year","fiscal_year","fy"] if c in primary_df.columns), None)
    if year_col:
        st.subheader("Crash rate by year")
        trend = primary_df.groupby(year_col)["crash_label"].mean().reset_index()
        fig3  = px.line(trend, x=year_col, y="crash_label", markers=True,
                        labels={"crash_label":"Crash rate"})
        st.plotly_chart(fig3, use_container_width=True)

# ── Predictions ───────────────────────────────────────────────────────────────
elif page == "Predictions":
    st.title("🔮 Crash Predictions")

    if not model_loaded:
        st.error("XGB model not loaded. Check: " + XGB_PATH)
        st.stop()
    if not feature_cols:
        st.error("No matching feature columns found.")
        st.stop()

    thresh = st.sidebar.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)
    company_list     = ["All Companies"] + sorted(primary_df["name"].dropna().unique().tolist())
    selected_company = st.sidebar.selectbox("🏢 Select company", company_list)

    X, feat = get_model_input(primary_df)
    try:
        scaler   = load_scaler()
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X.values

    import xgboost as xgb
    dmatrix = xgb.DMatrix(X_scaled)
    probs = model.get_booster().predict(dmatrix)
    pred_df = primary_df.copy()
    pred_df["crash_prob"] = probs
    pred_df["predicted"]  = (probs >= thresh).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk companies", f"{(probs >= thresh).sum():,}")
    c2.metric("Threshold",           f"{thresh:.0%}")
    c3.metric("Avg risk score",      f"{probs.mean():.3f}")

    fig = px.histogram(pred_df, x="crash_prob", color="crash_label",
                       nbins=50, barmode="overlay",
                       color_discrete_map={0:"#3B8BD4", 1:"#E24B4A"},
                       labels={"crash_prob":"Predicted crash probability"})
    fig.add_vline(x=thresh, line_dash="dash", line_color="orange",
                  annotation_text=f"Threshold={thresh}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 30 highest-risk companies")
    display_cols = ["name","cik","crash_prob","predicted","crash_label"] + feat
    display_cols = [c for c in display_cols if c in pred_df.columns]
    result_df    = pred_df.sort_values("crash_prob", ascending=False)
    if selected_company != "All Companies":
        result_df = result_df[result_df["name"] == selected_company]
    st.dataframe(result_df[display_cols].head(30), use_container_width=True)

# ── SHAP Explainer ────────────────────────────────────────────────────────────
elif page == "SHAP Explainer":
    st.title("🔍 SHAP Explainability")

    if not model_loaded or not feature_cols:
        st.error("Need XGB model + features.")
        st.stop()

    X, feat = get_model_input(primary_df)
    try:
        scaler   = load_scaler()
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feat)
    except Exception:
        X_scaled = X.reset_index(drop=True)

    sample_size = st.sidebar.slider("Sample size for SHAP", 100, min(1000, len(X_scaled)), 300, 100)
    X_sample    = X_scaled.sample(sample_size, random_state=42)

    with st.spinner("Computing SHAP values…"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    st.subheader("Global feature importance")
    fig1, _ = plt.subplots()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig1); plt.clf()

    st.subheader("SHAP beeswarm")
    fig2, _ = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig2); plt.clf()

    st.subheader("Single company waterfall")
    sample_indices = X_sample.index.tolist()
    if "name" in primary_df.columns:
        name_options = primary_df.loc[sample_indices, "name"].fillna("Unknown").tolist()
        idx = st.selectbox("Select company", range(len(name_options)),
                           format_func=lambda i: f"{name_options[i]} (row {i})")
    else:
        idx = st.number_input("Row index", 0, len(X_sample)-1, 0)

    fig3, _ = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[idx],
            feature_names=feat
        ), show=False
    )
    st.pyplot(fig3); plt.clf()

# ── LSTM Analysis ─────────────────────────────────────────────────────────────
elif page == "LSTM Analysis":
    st.title("🧠 LSTM Analysis")

    lstm_df, fname = load_lstm_data()
    history_df     = load_training_history()

    if history_df is not None:
        st.subheader("Training history")
        fig = go.Figure()
        for col in history_df.columns:
            fig.add_trace(go.Scatter(y=history_df[col], name=col, mode="lines"))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    if lstm_df is not None:
        st.subheader(f"LSTM dataset: `{fname}`")
        st.write(f"Shape: {lstm_df.shape}")
        num_cols = lstm_df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel  = st.selectbox("Column to plot", num_cols[:20])
            fig2 = px.histogram(lstm_df, x=sel, nbins=50)
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(lstm_df.head(100), use_container_width=True)
    else:
        st.info("No LSTM dataset found.")

# ── Sentiment Signals ─────────────────────────────────────────────────────────
elif page == "Sentiment Signals":
    st.title("📰 MDA Sentiment Signals")

    if sent_df is None:
        st.info("mda_sentiment_signals.parquet not found.")
        st.stop()

    st.write(f"Shape: {sent_df.shape}")
    st.dataframe(sent_df.head(50), use_container_width=True)

    num_cols = sent_df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        sel = st.selectbox("Visualize", num_cols[:15])
        fig = px.histogram(sent_df, x=sel, nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    if "crash_label" in sent_df.columns and num_cols:
        st.subheader("Sentiment vs crash label")
        fig2 = px.box(sent_df, x="crash_label", y=num_cols[0], color="crash_label",
                      color_discrete_map={0:"#3B8BD4", 1:"#E24B4A"})
        st.plotly_chart(fig2, use_container_width=True)

# ── Raw Data ──────────────────────────────────────────────────────────────────
elif page == "Raw Data":
    st.title("📂 Raw Data Explorer")

    tab1, tab2, tab3, tab4 = st.tabs(["Gold Dataset","Sentiment","LSTM","File Status"])

    with tab1:
        st.write(f"Shape: {primary_df.shape}")
        st.dataframe(primary_df.head(200), use_container_width=True)

    with tab2:
        if sent_df is not None:
            st.dataframe(sent_df.head(200), use_container_width=True)
        else:
            st.info("Sentiment file not found.")

    with tab3:
        lstm_df, fname = load_lstm_data()
        if lstm_df is not None:
            st.write(fname)
            st.dataframe(lstm_df.head(200), use_container_width=True)
        else:
            st.info("LSTM data not found.")

    with tab4:
        st.write("**File Status**")
        st.write(f"- gold_dataset.parquet: {'✅' if os.path.exists(GOLD_PATH) else '❌'}")
        st.write(f"- mda_sentiment_signals.parquet: {'✅' if os.path.exists(SENT_PATH) else '❌'}")
        st.write(f"- final_lstm_dataset.parquet: {'✅' if os.path.exists(LSTM_PATH) else '❌'}")
        st.write(f"- xgb_model.json: {'✅' if model_loaded else '❌'}")
        st.write(f"- scaler.pkl: {'✅' if os.path.exists(SCALER_PATH) else '❌'}")
        st.write(f"- training_history_v2.csv: {'✅' if os.path.exists(HISTORY_PATH) else '❌'}")