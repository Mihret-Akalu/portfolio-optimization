import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Load returns
# --------------------------------------------------
returns = pd.read_csv(
    "data/processed/daily_returns.csv",
    index_col="Date",
    parse_dates=True
)

# --------------------------------------------------
# Portfolio and Benchmark weights
# --------------------------------------------------
weights_strategy = {
    "TSLA": 0.4909838082747182,
    "SPY": 0.2083481312104325,
    "BND": 0.3006680605148493
}

weights_benchmark = {
    "TSLA": 0.0,
    "SPY": 0.6,
    "BND": 0.4
}

# --------------------------------------------------
# Simulation function
# --------------------------------------------------
def simulate_portfolio(returns_df, weights):

    values = pd.Series(index=returns_df.index, dtype=float)

    values.iloc[0] = 1.0

    weight_array = np.array(
        [weights[asset] for asset in returns_df.columns]
    )

    for i in range(1, len(returns_df)):

        daily_ret = np.dot(
            weight_array,
            returns_df.iloc[i].values
        )

        values.iloc[i] = values.iloc[i - 1] * (1 + daily_ret)

    return values


# --------------------------------------------------
# Run backtests
# --------------------------------------------------
strategy_vals = simulate_portfolio(
    returns,
    weights_strategy
)

benchmark_vals = simulate_portfolio(
    returns,
    weights_benchmark
)

# --------------------------------------------------
# Save cumulative values for dashboard
# --------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

cumulative_df = pd.DataFrame({
    "Date": strategy_vals.index,
    "Strategy": strategy_vals.values,
    "Benchmark": benchmark_vals.values
})

cumulative_df.to_csv(
    "data/processed/backtest_cumulative.csv",
    index=False
)

print("Cumulative backtest data saved.")

# --------------------------------------------------
# Compute performance metrics
# --------------------------------------------------
def compute_metrics(values):

    returns = values.pct_change().dropna()

    total_return = values.iloc[-1] - 1

    ann_return = returns.mean() * 252

    ann_vol = returns.std() * np.sqrt(252)

    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    drawdown = (
        values / values.cummax() - 1
    ).min()

    return {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown
    }


metrics_strategy = compute_metrics(strategy_vals)

metrics_benchmark = compute_metrics(benchmark_vals)

metrics_df = pd.DataFrame(
    [metrics_strategy, metrics_benchmark],
    index=["Strategy", "Benchmark"]
)

metrics_df.to_csv(
    "data/processed/backtest_metrics.csv"
)

print("Performance metrics saved.")

# --------------------------------------------------
# Plot cumulative returns
# --------------------------------------------------
os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(strategy_vals, label="Strategy")

plt.plot(benchmark_vals, label="Benchmark")

plt.xlabel("Date")

plt.ylabel("Portfolio Value")

plt.title("Backtest: Strategy vs Benchmark")

plt.legend()

plt.grid(True)

plt.savefig(
    "reports/figures/backtest_cumulative.png",
    dpi=300,
    bbox_inches="tight"
)

print("Plot saved.")

plt.show()
