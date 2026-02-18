import pandas as pd
import numpy as np

from src.preprocessing import clean_price_data, compute_returns, split_time_series


def test_clean_price_data():

    df = pd.DataFrame({
        "TSLA": [100, 101, 102]
    }, index=["2024-01-01", "2024-01-02", "2024-01-03"])

    cleaned = clean_price_data(df)

    assert isinstance(cleaned.index, pd.DatetimeIndex)


def test_compute_returns():

    df = pd.DataFrame({
        "TSLA": [100, 101, 102]
    })

    returns = compute_returns(df)

    assert len(returns) == 2


def test_split_time_series():

    dates = pd.date_range("2024-01-01", periods=500)

    series = pd.Series(np.random.randn(500), index=dates)

    train, test = split_time_series(series)

    assert len(train) > 0
    assert len(test) > 0
