import pandas as pd
import numpy as np
from src.risk_metrics import compute_risk_metrics

def test_compute_risk_metrics_returns_dataframe():

    returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        columns=["A", "B", "C"]
    )

    metrics = compute_risk_metrics(returns)

    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 3  # 3 assets
    assert "Sharpe Ratio" in metrics.columns


def test_sharpe_ratio_calculation():

    returns = pd.DataFrame(
        np.random.normal(0.001, 0.01, 252),
        columns=["TEST"]
    )

    metrics = compute_risk_metrics(returns)

    assert "TEST" in metrics.index
