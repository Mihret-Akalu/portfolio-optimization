import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_arima(series, order=(5,1,0)):
    """
    Train ARIMA model
    """
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    return fitted_model


def forecast_arima_with_intervals(model, steps=252, alpha=0.05):
    """
    Forecast future values with confidence intervals

    steps = forecast horizon (252 trading days = 1 year)
    alpha = significance level (0.05 = 95% confidence interval)
    """

    forecast_result = model.get_forecast(steps=steps)

    forecast_df = pd.DataFrame({
        "forecast": forecast_result.predicted_mean,
        "lower_bound": forecast_result.conf_int(alpha=alpha).iloc[:, 0],
        "upper_bound": forecast_result.conf_int(alpha=alpha).iloc[:, 1]
    })

    return forecast_df
