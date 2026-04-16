import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Trader ML Dashboard", layout="wide")

OUTPUT_DIR = Path("outputs")


@st.cache_data
def load_data():
    files = {
        "daily": OUTPUT_DIR / "daily_trader_features.csv",
        "preds": OUTPUT_DIR / "predictions_test_split.csv",
        "latest": OUTPUT_DIR / "latest_next_day_predictions.csv",
        "archetypes": OUTPUT_DIR / "account_archetypes.csv",
        "profiles": OUTPUT_DIR / "cluster_profiles.csv",
        "metrics": OUTPUT_DIR / "metrics_summary.json",
    }

    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        return None, missing

    daily = pd.read_csv(files["daily"], parse_dates=["date"])
    preds = pd.read_csv(files["preds"], parse_dates=["date"])
    latest = pd.read_csv(files["latest"], parse_dates=["date"])
    archetypes = pd.read_csv(files["archetypes"])
    profiles = pd.read_csv(files["profiles"])

    with open(files["metrics"], "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "daily": daily,
        "preds": preds,
        "latest": latest,
        "archetypes": archetypes,
        "profiles": profiles,
        "metrics": metrics,
    }, None


def show_missing_message(missing_files):
    st.error("Required output files are missing. Run pipeline first:")
    st.code("python trader_ml_pipeline.py")
    st.write("Missing files:")
    for path in missing_files:
        st.write(f"- {path}")


def model_tab(data):
    st.subheader("Next-Day Profitability Bucket Model")

    metrics = data["metrics"]["model_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}")
    c3.metric("Train/Test Rows", f"{metrics['train_rows']}/{metrics['test_rows']}")

    cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["confusion_matrix_labels"], columns=metrics["confusion_matrix_labels"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    preds = data["preds"].copy()
    st.markdown("### Prediction Distribution")
    dist = preds["predicted_bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
    fig_dist = px.bar(dist, x="bucket", y="count", color="bucket", title="Predicted Bucket Counts")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Test Predictions")
    st.dataframe(preds.sort_values("date", ascending=False).head(200), use_container_width=True)


def clustering_tab(data):
    st.subheader("Trader Behavioral Archetypes")

    archetypes = data["archetypes"].copy()
    profiles = data["profiles"].copy()

    st.markdown("### Cluster Profiles")
    st.dataframe(profiles, use_container_width=True)

    if "archetype" in archetypes.columns:
        count_df = archetypes["archetype"].value_counts().rename_axis("archetype").reset_index(name="trader_count")
        fig_count = px.pie(count_df, names="archetype", values="trader_count", title="Archetype Share")
        st.plotly_chart(fig_count, use_container_width=True)

    fig_scatter = px.scatter(
        archetypes,
        x="avg_daily_trades",
        y="avg_daily_pnl",
        size="avg_daily_volume",
        color="archetype",
        hover_data=["Account", "avg_win_rate", "avg_leverage", "active_days"],
        title="Trader Archetypes: Activity vs PnL",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Trader-Level Cluster Table")
    st.dataframe(archetypes.sort_values(["cluster", "avg_daily_pnl"], ascending=[True, False]), use_container_width=True)


def exploration_tab(data):
    st.subheader("Feature Exploration")

    daily = data["daily"].copy()
    latest = data["latest"].copy()

    accounts = sorted(daily["Account"].unique().tolist())
    selected_accounts = st.multiselect("Filter Accounts", options=accounts, default=accounts[: min(5, len(accounts))])

    if selected_accounts:
        daily = daily[daily["Account"].isin(selected_accounts)]
        latest = latest[latest["Account"].isin(selected_accounts)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", len(daily))
    c2.metric("Unique Accounts", daily["Account"].nunique())
    c3.metric("Date Range", f"{daily['date'].min().date()} to {daily['date'].max().date()}")

    pnl_ts = (
        daily.groupby("date", as_index=False)["realized_pnl"]
        .mean()
        .rename(columns={"realized_pnl": "avg_realized_pnl"})
    )
    fig_ts = px.line(pnl_ts, x="date", y="avg_realized_pnl", title="Average Daily Realized PnL")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("### Latest Next-Day Predictions by Account")
    st.dataframe(latest.sort_values("date", ascending=False), use_container_width=True)

    prob_cols = [c for c in latest.columns if c.startswith("prob_")]
    if prob_cols:
        melted = latest[["Account"] + prob_cols].melt(id_vars="Account", var_name="bucket", value_name="probability")
        fig_prob = px.bar(melted, x="Account", y="probability", color="bucket", barmode="stack", title="Latest Prediction Probabilities")
        st.plotly_chart(fig_prob, use_container_width=True)


def process_tab():
    st.subheader("Step-by-Step ML Process")
    steps = [
        "1. Load and clean raw trader records from final_dataset.csv.",
        "2. Convert key numeric fields and parse trading date.",
        "3. Aggregate trade-level data into account-day behavioral features.",
        "4. Create target: next-day realized PnL bucket (Loss/Neutral/Profit).",
        "5. Split by time (80/20 chronological split) to avoid leakage.",
        "6. Train classifier pipeline with preprocessing + RandomForest.",
        "7. Evaluate model (accuracy, balanced accuracy, confusion matrix).",
        "8. Build account-level features and perform KMeans clustering.",
        "9. Auto-select cluster count using silhouette score and assign archetype names.",
        "10. Save artifacts and CSV outputs for operational use.",
    ]
    for step in steps:
        st.write(step)


st.title("Trader Profitability + Behavioral Archetype Dashboard")

loaded, missing = load_data()
if missing:
    show_missing_message(missing)
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Model", "Clustering", "Explore", "ML Process"])

with tab1:
    model_tab(loaded)
with tab2:
    clustering_tab(loaded)
with tab3:
    exploration_tab(loaded)
with tab4:
    process_tab()
