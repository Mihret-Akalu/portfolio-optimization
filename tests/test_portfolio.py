import numpy as np
import pandas as pd

from src.portfolio import optimize_portfolio


def test_weights_sum_to_one():

    # Force positive mean returns
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.01, (200, 3)),
        columns=["A", "B", "C"]
    )

    weights = optimize_portfolio(returns)

    total_weight = sum(weights.values())

    assert round(total_weight, 5) == 1.0
