import numpy as np


def backtest(returns, weights):

    portfolio = returns.dot(weights)

    cumulative = (1 + portfolio).cumprod()

    return cumulative
