# Trader Performance vs Market Sentiment Analysis

This notebook studies how trader behavior changes across Fear / Greed market sentiment conditions by combining:

- `fear_greed_index - fear_greed_index.csv`
- `historical_data.csv`

It prepares the data, merges both sources by date, and produces the sentiment-vs-trading charts and tables used in the analysis.

## How To Run

### In VS Code

1. Open `Trader Performance vs Market Sentiment Analysis.ipynb`.
2. Select the project Python kernel if prompted.
3. Run the notebook cells from top to bottom.

### In Jupyter

```powershell
jupyter notebook "Trader Performance vs Market Sentiment Analysis.ipynb"
```

Then run all cells in order.

## Output Charts And Tables

The notebook renders the following outputs inline:

### Charts

- Average PnL by sentiment
- Win rate across sentiments
- Drawdown / worst-loss comparison by sentiment
- Trade frequency by sentiment
- Average position size by sentiment
- Buy vs sell activity by sentiment
- Leverage-based trader segmentation
- Frequency-based trader segmentation
- Consistency-based trader segmentation

### Tables

- Sentiment dataset column overview
- Trade dataset column overview
- Missing values and duplicate checks
- Date range checks for both data sources
- Merged dataset preview
- Daily PnL per trader
- Win rate by account
- Average trade size by account
- Trades per day
- Long vs short counts by sentiment
- Leverage proxy table
- Trader consistency table
- Sentiment-level performance summary table

## What The Notebook Shows

- Traders tend to perform better in Fear and Extreme Greed than in Neutral or Greed alone.
- Trade activity increases during Fear and Greed, which suggests traders are highly reactive to market sentiment.
- Position sizing is larger during Fear and Greed, while Extreme Greed shows smaller average trade size.
- The notebook also segments traders into leverage, frequency, and consistency groups for a more practical view of behavior.

## Notes

- The charts and tables are generated inside the notebook; there are no separate image exports by default.
- If you change the input CSV files, rerun the notebook from the top so the merged dataset and plots stay consistent.

