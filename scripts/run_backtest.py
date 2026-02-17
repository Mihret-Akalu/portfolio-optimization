import pandas as pd
from src.backtest import backtest

# --------------------------------------------------
# Load returns
# --------------------------------------------------
returns = pd.read_csv(
    "data/processed/daily_returns.csv",
    index_col="Date",
    parse_dates=True
)

# --------------------------------------------------
# Load portfolio weights from file (production-safe)
# --------------------------------------------------
weights_df = pd.read_csv(
    "data/processed/portfolio_weights.csv"
)

weights_strategy = dict(
    zip(weights_df.Asset, weights_df.Weight)
)

# --------------------------------------------------
# Benchmark weights
# --------------------------------------------------
weights_benchmark = {
    "TSLA": 0.0,
    "SPY": 0.6,
    "BND": 0.4
}

# --------------------------------------------------
# Run backtest using src module
# --------------------------------------------------
results = backtest(
    returns=returns,
    weights=weights_strategy,
    benchmark_weights=weights_benchmark,
    transaction_cost=0.001,
    save_results=True
)

# --------------------------------------------------
# Save combined cumulative file for dashboard
# --------------------------------------------------
cumulative_df = pd.DataFrame({
    "Date": results["strategy_cumulative"].index,
    "Strategy": results["strategy_cumulative"].values,
    "Benchmark": results["benchmark_cumulative"].values
})

cumulative_df.to_csv(
    "data/processed/backtest_cumulative.csv",
    index=False
)

print("Backtest complete.")
print("Files saved:")
print("data/processed/backtest_strategy.csv")
print("data/processed/backtest_benchmark.csv")
print("data/processed/backtest_cumulative.csv")
print("reports/figures/backtest_cumulative.png")
