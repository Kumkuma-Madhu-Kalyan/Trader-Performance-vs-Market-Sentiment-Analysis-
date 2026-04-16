import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_FILE = "final_dataset.csv"
OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)


def safe_read_data(path: str) -> pd.DataFrame:
    # Rename duplicate columns safely by appending a numeric suffix.
    raw = pd.read_csv(path)
    cols = pd.Series(raw.columns)
    for col_name in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == col_name].index.tolist()
        for i, idx in enumerate(dup_idx):
            cols[idx] = col_name if i == 0 else f"{col_name}_{i}"
    raw.columns = cols
    return raw


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    numeric_cols = [
        "Size USD",
        "Closed PnL",
        "Fee",
        "leverage_proxy",
        "value",
        "Execution Price",
        "Size Tokens",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    data["win_num"] = data.get("win", False).astype(str).str.lower().map({"true": 1, "false": 0})
    data["win_num"] = data["win_num"].fillna(0)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    elif "Timestamp IST" in data.columns:
        data["date"] = pd.to_datetime(data["Timestamp IST"], errors="coerce").dt.date
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    else:
        raise ValueError("No date column found. Expected 'date' or 'Timestamp IST'.")

    data["Side"] = data.get("Side", "").astype(str).str.upper()
    data["classification"] = data.get("classification", "Unknown").fillna("Unknown").astype(str)
    data["leverage_group"] = data.get("leverage_group", "Unknown").fillna("Unknown").astype(str)
    data["freq_group"] = data.get("freq_group", "Unknown").fillna("Unknown").astype(str)
    data["consistency_group"] = data.get("consistency_group", "Unknown").fillna("Unknown").astype(str)

    data = data.dropna(subset=["Account", "date"])
    return data


def build_daily_account_features(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["Account", "date"], as_index=False)

    daily = grouped.agg(
        trades_count=("Trade ID", "count"),
        unique_coins=("Coin", pd.Series.nunique),
        total_volume_usd=("Size USD", lambda s: np.nansum(np.abs(s))),
        avg_trade_size_usd=("Size USD", lambda s: np.nanmean(np.abs(s))),
        realized_pnl=("Closed PnL", "sum"),
        mean_trade_pnl=("Closed PnL", "mean"),
        total_fees=("Fee", "sum"),
        mean_leverage_proxy=("leverage_proxy", "mean"),
        max_leverage_proxy=("leverage_proxy", "max"),
        win_rate=("win_num", "mean"),
        fear_greed_value=("value", "mean"),
    )

    buy_counts = (
        data.assign(is_buy=(data["Side"] == "BUY").astype(int))
        .groupby(["Account", "date"], as_index=False)["is_buy"]
        .sum()
        .rename(columns={"is_buy": "buy_count"})
    )

    daily = daily.merge(buy_counts, on=["Account", "date"], how="left")
    daily["buy_count"] = daily["buy_count"].fillna(0)
    daily["sell_count"] = daily["trades_count"] - daily["buy_count"]
    daily["buy_ratio"] = np.where(daily["trades_count"] > 0, daily["buy_count"] / daily["trades_count"], 0)

    cat_daily = (
        data.groupby(["Account", "date"], as_index=False)
        .agg(
            market_regime=("classification", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
            leverage_bucket=("leverage_group", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
            frequency_bucket=("freq_group", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
            consistency_bucket=("consistency_group", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        )
    )

    daily = daily.merge(cat_daily, on=["Account", "date"], how="left")
    daily = daily.sort_values(["Account", "date"]).reset_index(drop=True)

    daily["next_day_realized_pnl"] = daily.groupby("Account")["realized_pnl"].shift(-1)
    daily["next_day_trades"] = daily.groupby("Account")["trades_count"].shift(-1)

    return daily


def add_profitability_bucket(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    valid = df["next_day_realized_pnl"].dropna()

    if valid.empty:
        raise ValueError("No next-day target available after feature engineering.")

    q_low, q_high = valid.quantile([0.33, 0.67]).tolist()

    def bucketize(x: float) -> str:
        if x <= q_low:
            return "Loss"
        if x <= q_high:
            return "Neutral"
        return "Profit"

    df["profitability_bucket"] = df["next_day_realized_pnl"].apply(
        lambda x: bucketize(x) if pd.notnull(x) else np.nan
    )
    df["bucket_low_threshold"] = q_low
    df["bucket_high_threshold"] = q_high

    return df


def train_predictive_model(model_df: pd.DataFrame) -> dict:
    use_df = model_df.dropna(subset=["profitability_bucket"]).copy()

    feature_cols_num = [
        "trades_count",
        "unique_coins",
        "total_volume_usd",
        "avg_trade_size_usd",
        "realized_pnl",
        "mean_trade_pnl",
        "total_fees",
        "mean_leverage_proxy",
        "max_leverage_proxy",
        "win_rate",
        "fear_greed_value",
        "buy_ratio",
    ]
    feature_cols_cat = [
        "market_regime",
        "leverage_bucket",
        "frequency_bucket",
        "consistency_bucket",
    ]

    for col in feature_cols_num:
        if col not in use_df.columns:
            use_df[col] = np.nan
    for col in feature_cols_cat:
        if col not in use_df.columns:
            use_df[col] = "Unknown"

    use_df[feature_cols_num] = use_df[feature_cols_num].replace([np.inf, -np.inf], np.nan)
    use_df[feature_cols_num] = use_df[feature_cols_num].clip(lower=-1e12, upper=1e12)

    feature_cols = feature_cols_num + feature_cols_cat

    split_date = use_df["date"].quantile(0.80)
    train_df = use_df[use_df["date"] <= split_date].copy()
    test_df = use_df[use_df["date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        unique_dates = sorted(use_df["date"].dropna().unique())
        cutoff_idx = int(max(1, len(unique_dates) * 0.8))
        split_date = unique_dates[cutoff_idx - 1]
        train_df = use_df[use_df["date"] <= split_date].copy()
        test_df = use_df[use_df["date"] > split_date].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["profitability_bucket"]
    X_test = test_df[feature_cols]
    y_test = test_df["profitability_bucket"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    classes = model.named_steps["classifier"].classes_

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix_labels": classes.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_date": str(pd.Timestamp(split_date).date()),
    }

    preds = test_df[["Account", "date", "profitability_bucket", "next_day_realized_pnl"]].copy()
    preds["predicted_bucket"] = y_pred
    for idx, cls in enumerate(classes):
        preds[f"prob_{cls}"] = y_prob[:, idx]

    latest_rows = use_df.sort_values("date").groupby("Account", as_index=False).tail(1)
    latest_X = latest_rows[feature_cols]
    latest_pred = model.predict(latest_X)
    latest_prob = model.predict_proba(latest_X)

    latest_predictions = latest_rows[["Account", "date", "realized_pnl", "trades_count"]].copy()
    latest_predictions["predicted_next_day_bucket"] = latest_pred
    for idx, cls in enumerate(classes):
        latest_predictions[f"prob_{cls}"] = latest_prob[:, idx]

    joblib.dump(model, ARTIFACT_DIR / "profitability_model.joblib")

    return {
        "metrics": metrics,
        "predictions": preds,
        "latest_predictions": latest_predictions,
        "feature_data": use_df,
        "feature_cols": feature_cols,
    }


def choose_k_with_silhouette(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 6) -> tuple[int, dict]:
    best_k = k_min
    best_score = -1
    scores = {}

    upper = min(k_max, max(k_min, X_scaled.shape[0] - 1))
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) <= 1:
            continue
        score = silhouette_score(X_scaled, labels)
        scores[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores


def assign_archetype_names(cluster_profile: pd.DataFrame) -> dict:
    names = {}
    for _, row in cluster_profile.iterrows():
        cid = int(row["cluster"])
        if row["avg_daily_trades"] >= cluster_profile["avg_daily_trades"].quantile(0.75) and row[
            "avg_leverage"
        ] >= cluster_profile["avg_leverage"].quantile(0.75):
            names[cid] = "Aggressive High-Volume"
        elif row["avg_daily_pnl"] > 0 and row["avg_win_rate"] >= cluster_profile["avg_win_rate"].quantile(0.65):
            names[cid] = "Consistent Winners"
        elif row["avg_daily_trades"] <= cluster_profile["avg_daily_trades"].quantile(0.25):
            names[cid] = "Low-Activity Traders"
        elif row["pnl_volatility"] >= cluster_profile["pnl_volatility"].quantile(0.75):
            names[cid] = "Volatile Opportunists"
        else:
            names[cid] = "Balanced Swing Traders"
    return names


def run_clustering(feature_df: pd.DataFrame) -> dict:
    account_df = (
        feature_df.groupby("Account", as_index=False)
        .agg(
            active_days=("date", "nunique"),
            avg_daily_trades=("trades_count", "mean"),
            avg_daily_volume=("total_volume_usd", "mean"),
            avg_daily_pnl=("realized_pnl", "mean"),
            pnl_volatility=("realized_pnl", "std"),
            avg_win_rate=("win_rate", "mean"),
            avg_leverage=("mean_leverage_proxy", "mean"),
            avg_coin_diversity=("unique_coins", "mean"),
            avg_buy_ratio=("buy_ratio", "mean"),
            avg_fear_greed=("fear_greed_value", "mean"),
        )
        .fillna(0)
    )

    cluster_features = [
        "active_days",
        "avg_daily_trades",
        "avg_daily_volume",
        "avg_daily_pnl",
        "pnl_volatility",
        "avg_win_rate",
        "avg_leverage",
        "avg_coin_diversity",
        "avg_buy_ratio",
        "avg_fear_greed",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(account_df[cluster_features])

    best_k, k_scores = choose_k_with_silhouette(X_scaled)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    account_df["cluster"] = kmeans.fit_predict(X_scaled)

    profile = (
        account_df.groupby("cluster", as_index=False)[
            [
                "active_days",
                "avg_daily_trades",
                "avg_daily_volume",
                "avg_daily_pnl",
                "pnl_volatility",
                "avg_win_rate",
                "avg_leverage",
                "avg_coin_diversity",
            ]
        ]
        .mean()
        .sort_values("cluster")
    )

    archetype_map = assign_archetype_names(profile)
    account_df["archetype"] = account_df["cluster"].map(archetype_map)
    profile["archetype"] = profile["cluster"].map(archetype_map)

    joblib.dump(
        {
            "model": kmeans,
            "scaler": scaler,
            "features": cluster_features,
            "best_k": best_k,
            "silhouette_scores": k_scores,
        },
        ARTIFACT_DIR / "trader_clustering.joblib",
    )

    return {
        "account_clusters": account_df,
        "cluster_profile": profile,
        "best_k": int(best_k),
        "silhouette_scores": k_scores,
    }


def save_outputs(model_result: dict, cluster_result: dict, full_daily_df: pd.DataFrame) -> None:
    model_result["feature_data"].to_csv(OUTPUT_DIR / "daily_trader_features.csv", index=False)
    model_result["predictions"].to_csv(OUTPUT_DIR / "predictions_test_split.csv", index=False)
    model_result["latest_predictions"].to_csv(OUTPUT_DIR / "latest_next_day_predictions.csv", index=False)

    cluster_result["account_clusters"].to_csv(OUTPUT_DIR / "account_archetypes.csv", index=False)
    cluster_result["cluster_profile"].to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)

    full_daily_df.to_csv(OUTPUT_DIR / "daily_features_with_target.csv", index=False)

    summary = {
        "model_metrics": model_result["metrics"],
        "clustering": {
            "best_k": cluster_result["best_k"],
            "silhouette_scores": cluster_result["silhouette_scores"],
        },
    }

    with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    print("[1/7] Creating folders...")
    ensure_dirs()

    print("[2/7] Reading data...")
    raw = safe_read_data(DATA_FILE)

    print("[3/7] Preprocessing raw trade records...")
    data = preprocess_raw(raw)

    print("[4/7] Building daily account features...")
    daily = build_daily_account_features(data)

    print("[5/7] Building profitability buckets and training model...")
    model_df = add_profitability_bucket(daily)
    model_result = train_predictive_model(model_df)

    print("[6/7] Running trader archetype clustering...")
    cluster_result = run_clustering(model_result["feature_data"])

    print("[7/7] Saving artifacts and reports...")
    save_outputs(model_result, cluster_result, model_df)

    print("\nPipeline complete.")
    print(f"Model accuracy: {model_result['metrics']['accuracy']:.4f}")
    print(f"Model balanced accuracy: {model_result['metrics']['balanced_accuracy']:.4f}")
    print(f"Optimal clusters (k): {cluster_result['best_k']}")
    print("Outputs saved under ./outputs and ./artifacts")


if __name__ == "__main__":
    main()
