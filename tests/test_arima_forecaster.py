import pandas as pd
import numpy as np

from src.train_arima import train_arima


def test_arima_forecasting_pipeline():

    series = pd.Series(np.random.randn(300))

    model = train_arima(series)

    forecast = model.forecast(steps=5)

    assert len(forecast) == 5
