import numpy as np
import pandas as pd
import os

from src.arima_forecaster import (
    train_arima,
    forecast_arima_with_intervals
)


def test_arima_forecast_with_intervals():

    series = pd.Series(np.random.randn(300))

    model = train_arima(series)

    forecast_df = forecast_arima_with_intervals(model, steps=5)

    assert "forecast" in forecast_df.columns
    assert len(forecast_df) == 5
