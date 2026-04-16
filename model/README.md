# Trader ML Project: Simple End-to-End Guide

This project uses your trader data to do 3 things:

1. Predict each trader's next-day profitability bucket (Loss, Neutral, Profit)
2. Group traders into behavior archetypes (clustering)
3. Show results in a lightweight Streamlit dashboard

The goal is to keep the process simple, practical, and easy to run.

## 1) What Is In This Project

- `final_dataset.csv`: Main input data (trade-level records)
- `trader_ml_pipeline.py`: Training + feature engineering + clustering + saving outputs
- `dashboard_app.py`: Streamlit app to explore model and clustering results
- `requirements.txt`: Python packages needed
- `outputs/`: CSV and JSON reports produced by the pipeline
- `artifacts/`: Saved ML models (`.joblib` files)

## 2) Step-by-Step ML Process (Simple Words)

### Step 1: Read and clean data
The pipeline loads `final_dataset.csv`, fixes duplicate-like column names, and converts important fields to proper numeric/date types.

### Step 2: Build daily trader features
Raw trades are converted into one row per account per day, with features like:

- number of trades
- total volume
- average trade size
- realized PnL
- total fees
- win rate
- leverage behavior
- buy ratio
- market regime category

### Step 3: Create the prediction target
For each account-day row, the script looks at next day's realized PnL and converts it into:

- Loss
- Neutral
- Profit

This becomes the target label for classification.

### Step 4: Train/test split by time
The data is split chronologically (about 80% train, 20% test). This helps avoid time leakage.

### Step 5: Train predictive model
A scikit-learn pipeline is used:

- numeric preprocessing: median imputation + standard scaling
- categorical preprocessing: most-frequent imputation + one-hot encoding
- model: RandomForestClassifier

### Step 6: Evaluate model
The script computes:

- accuracy
- balanced accuracy
- confusion matrix
- classification report

### Step 7: Predict latest next-day bucket per account
The latest available day for each account is used to produce a next-day bucket prediction with class probabilities.

### Step 8: Build account-level behavior profile
Daily rows are aggregated per account to create behavior vectors:

- activity level
- average volume
- pnl level and volatility
- average leverage
- average win rate
- diversification

### Step 9: Cluster traders into archetypes
KMeans is trained on standardized account-level features.
The script chooses cluster count `k` using silhouette score.
Then it assigns readable labels like:

- Aggressive High-Volume
- Consistent Winners
- Low-Activity Traders
- Volatile Opportunists
- Balanced Swing Traders

### Step 10: Save outputs and artifacts
All key files are saved to `outputs/` and models are saved to `artifacts/`.

## 3) Function-by-Function Explanation

## In `trader_ml_pipeline.py`

### `ensure_dirs()`
Creates `outputs/` and `artifacts/` folders if they do not exist.

### `safe_read_data(path)`
Loads CSV and safely renames duplicate-like column names so downstream code does not break.

### `preprocess_raw(df)`
Cleans raw data:

- converts important numeric columns
- handles inf/-inf values
- parses date
- normalizes category columns
- creates numeric `win_num`
- drops rows without account/date

### `build_daily_account_features(data)`
Builds one row per `Account` per `date` using grouped stats and category modes.
Also creates shifted next-day fields (`next_day_realized_pnl`, `next_day_trades`).

### `add_profitability_bucket(daily)`
Turns next-day PnL into 3 target buckets using quantiles (33%, 67%).
Adds thresholds for reproducibility.

### `train_predictive_model(model_df)`
Trains the next-day bucket classifier and returns:

- metrics
- test predictions
- latest account predictions
- feature dataset used

It also saves `artifacts/profitability_model.joblib`.

### `choose_k_with_silhouette(X_scaled, k_min=2, k_max=6)`
Tries multiple `k` values and picks the best one using silhouette score.

### `assign_archetype_names(cluster_profile)`
Adds human-readable names to numeric clusters based on profile statistics.

### `run_clustering(feature_df)`
Builds account-level clustering dataset, scales features, runs KMeans, creates profiles, and saves `artifacts/trader_clustering.joblib`.

### `save_outputs(model_result, cluster_result, full_daily_df)`
Writes all output CSV/JSON files to `outputs/`.

### `main()`
Runs all steps in order from start to finish.

## In `dashboard_app.py`

### `load_data()`
Reads pipeline outputs from `outputs/` and loads them into memory.

### `show_missing_message(missing_files)`
If output files are missing, shows a clear instruction to run the pipeline first.

### `model_tab(data)`
Displays model metrics, confusion matrix, and test prediction tables/charts.

### `clustering_tab(data)`
Displays archetype distribution, cluster profiles, and trader scatter plots.

### `exploration_tab(data)`
Lets you filter by account and inspect trends and latest prediction probabilities.

### `process_tab()`
Shows a short educational summary of the full ML workflow.

## 4) How To Run (Windows)

Open terminal in project folder and run:

### Install packages
```powershell
pip install -r requirements.txt
```

### Run ML pipeline
```powershell
python trader_ml_pipeline.py
```

### Start dashboard
```powershell
streamlit run dashboard_app.py
```

After starting Streamlit, open the local URL shown in terminal (usually `http://localhost:8501`).

## 5) Expected Output Files

After running pipeline, you should see:

- `outputs/daily_trader_features.csv`
- `outputs/daily_features_with_target.csv`
- `outputs/predictions_test_split.csv`
- `outputs/latest_next_day_predictions.csv`
- `outputs/account_archetypes.csv`
- `outputs/cluster_profiles.csv`
- `outputs/metrics_summary.json`

And model files:

- `artifacts/profitability_model.joblib`
- `artifacts/trader_clustering.joblib`

## 6) How To Read The Results Quickly

- `metrics_summary.json`: Best place for quick model quality check
- `latest_next_day_predictions.csv`: Next-day bucket prediction per account
- `account_archetypes.csv`: Archetype assigned to each account
- `cluster_profiles.csv`: Mean behavior values for each cluster

## 7) Troubleshooting

### If dashboard says files are missing
Run pipeline first:

```powershell
python trader_ml_pipeline.py
```

### If package import errors occur
Reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### If model metrics look weak
That is normal for first baseline models. Improve by:

- adding richer time-series features (rolling stats, lag features)
- tuning model hyperparameters
- trying boosted tree models (XGBoost/LightGBM)

## 8) One-Command Flow (After Setup)

1. `python trader_ml_pipeline.py`
2. `streamlit run dashboard_app.py`

That is all you need for regular use.
