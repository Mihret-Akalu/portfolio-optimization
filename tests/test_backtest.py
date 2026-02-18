import numpy as np
import pandas as pd
import os

from src.backtest import backtest


def test_backtest_with_saving():

    returns = pd.DataFrame(
        np.random.randn(100, 3) * 0.01,
        columns=["A", "B", "C"]
    )

    weights = {"A": 0.3, "B": 0.4, "C": 0.3}

    result = backtest(
        returns,
        weights,
        benchmark_weights=weights,
        save_results=False  # fast test
    )

    assert "strategy_cumulative" in result


def test_backtest_full_pipeline():

    returns = pd.DataFrame(
        np.random.normal(0.001, 0.01, (252, 3)),
        columns=["A", "B", "C"]
    )

    weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    result = backtest(
        returns,
        weights,
        benchmark_weights=weights,
        save_results=True
    )

    # Verify files created
    assert os.path.exists("data/processed/backtest_strategy.csv")
    assert os.path.exists("data/processed/backtest_metrics.csv")
    assert os.path.exists("reports/figures/backtest_cumulative.png")

    # Verify metrics exist
    assert "strategy_cumulative" in result
