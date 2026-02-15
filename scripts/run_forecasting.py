import pandas as pd
from src.data_loader import load_data, get_tsla_close
from src.arima_forecaster import train_arima, forecast_arima_with_intervals

from sklearn.preprocessing import MinMaxScaler
from src.lstm_forecaster import bootstrap_lstm_forecast


print("Loading data...")
df = load_data()
tsla = get_tsla_close(df)

# =========================
# ARIMA Forecast
# =========================

print("Running ARIMA forecast...")

arima_model = train_arima(tsla)

arima_forecast = forecast_arima_with_intervals(
    arima_model,
    steps=252
)

arima_forecast.to_csv(
    "data/processed/tsla_arima_forecast_with_intervals.csv",
    index=False
)

print("ARIMA forecast saved")


# =========================
# LSTM Forecast
# =========================

print("Running LSTM bootstrap forecast...")

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    tsla.values.reshape(-1,1)
).flatten()

lstm_forecast = bootstrap_lstm_forecast(
    model_path="models/lstm_tsla_model.h5",
    scaled_data=scaled_data,
    scaler=scaler,
    steps=126,
    n_simulations=20
)

lstm_forecast.to_csv(
    "data/processed/tsla_lstm_forecast_with_intervals.csv",
    index=False
)

print("LSTM forecast saved")

print("Forecasting complete.")
