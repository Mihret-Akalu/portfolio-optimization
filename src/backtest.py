import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def backtest(
    returns: pd.DataFrame,
    weights: dict,
    benchmark_weights: dict = None,
    transaction_cost: float = 0.001,
    save_results: bool = True
):

    # Ensure correct order
    weights_array = np.array([weights[col] for col in returns.columns])

    # Compute strategy returns
    strategy_returns = returns.dot(weights_array)

    # Transaction cost modeling
    turnover = np.sum(np.abs(weights_array))
    strategy_returns = strategy_returns - turnover * transaction_cost / 252

    # Cumulative performance
    strategy_cumulative = (1 + strategy_returns).cumprod()

    results = {
        "strategy_returns": strategy_returns,
        "strategy_cumulative": strategy_cumulative
    }

    # Benchmark comparison
    if benchmark_weights is not None:

        benchmark_array = np.array(
            [benchmark_weights[col] for col in returns.columns]
        )

        benchmark_returns = returns.dot(benchmark_array)

        benchmark_cumulative = (1 + benchmark_returns).cumprod()

        results["benchmark_returns"] = benchmark_returns
        results["benchmark_cumulative"] = benchmark_cumulative

    # -----------------------------------
    # Save results and automated reporting
    # -----------------------------------
    if save_results:

        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)

        # Save cumulative returns
        strategy_cumulative.to_csv(
            "data/processed/backtest_strategy.csv"
        )

        if benchmark_weights is not None:

            benchmark_cumulative.to_csv(
                "data/processed/backtest_benchmark.csv"
            )

        # -----------------------------------
        # Compute performance metrics
        # -----------------------------------

        def compute_metrics(values):

            returns = values.pct_change().dropna()

            total_return = values.iloc[-1] - 1

            ann_return = returns.mean() * 252

            ann_vol = returns.std() * np.sqrt(252)

            sharpe = (
                ann_return / ann_vol if ann_vol != 0 else np.nan
            )

            max_dd = (
                values / values.cummax() - 1
            ).min()

            return {
                "Total Return": total_return,
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd
            }

        metrics_strategy = compute_metrics(
            strategy_cumulative
        )

        metrics_df = pd.DataFrame(
            [metrics_strategy],
            index=["Strategy"]
        )

        if benchmark_weights is not None:

            metrics_benchmark = compute_metrics(
                benchmark_cumulative
            )

            metrics_df.loc["Benchmark"] = metrics_benchmark

        # Save metrics automatically
        metrics_df.to_csv(
            "data/processed/backtest_metrics.csv"
        )

        # -----------------------------------
        # Plot cumulative performance
        # -----------------------------------

        plt.figure(figsize=(12, 6))

        plt.plot(
            strategy_cumulative,
            label="Strategy"
        )

        if benchmark_weights is not None:

            plt.plot(
                benchmark_cumulative,
                label="Benchmark"
            )

        plt.title("Portfolio Backtest")

        plt.xlabel("Date")

        plt.ylabel("Portfolio Value")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "reports/figures/backtest_cumulative.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    return results
