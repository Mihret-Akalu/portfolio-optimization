import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier

from pypfopt.risk_models import sample_cov


def optimize_portfolio(returns):

    mu = returns.mean()

    S = sample_cov(returns)

    ef = EfficientFrontier(mu, S)

    weights = ef.max_sharpe()

    return weights
