import pandas as pd
import numpy as np


def compute_risk_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compute financial risk metrics.
    """

    mean_returns = returns.mean() * 252

    volatility = returns.std() * np.sqrt(252)

    sharpe_ratio = (mean_returns - risk_free_rate) / volatility

    var_95 = returns.quantile(0.05)

    metrics = pd.DataFrame({

        "Annualized Return": mean_returns,

        "Annualized Volatility": volatility,

        "Sharpe Ratio": sharpe_ratio,

        "VaR 95%": var_95
    })

    return metrics
