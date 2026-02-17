import pandas as pd
import numpy as np


def compute_risk_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:

    mean_returns = returns.mean() * 252

    volatility = returns.std() * np.sqrt(252)

    sharpe_ratio = (mean_returns - risk_free_rate) / volatility

    # Downside deviation
    downside = returns.copy()
    downside[downside > 0] = 0

    downside_vol = downside.std() * np.sqrt(252)

    sortino_ratio = (mean_returns - risk_free_rate) / downside_vol

    # VaR
    var_95 = returns.quantile(0.05)

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()

    metrics = pd.DataFrame({

        "Annual Return": mean_returns,

        "Volatility": volatility,

        "Sharpe Ratio": sharpe_ratio,

        "Sortino Ratio": sortino_ratio,

        "VaR 95%": var_95,

        "Max Drawdown": max_drawdown
    })

    return metrics
