import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def create_sequences(data, window_size=60):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
    return np.array(X)


def bootstrap_lstm_forecast(
    model_path,
    scaled_data,
    scaler,
    steps=252,
    window_size=60,
    n_simulations=100
):
    """
    Bootstrap forecast intervals for LSTM

    n_simulations = number of simulated forecasts
    """

    model = load_model(model_path)

    simulations = []

    for sim in range(n_simulations):

        input_seq = scaled_data[-window_size:].copy()
        sim_forecast = []

        for _ in range(steps):

            pred = model.predict(
                input_seq.reshape(1, window_size, 1),
                verbose=0
            )[0, 0]

            # Add small noise for bootstrap variation
            noise = np.random.normal(0, 0.01)
            pred = pred + noise

            sim_forecast.append(pred)

            input_seq = np.append(input_seq[1:], pred)

        simulations.append(sim_forecast)

    simulations = np.array(simulations)

    mean_forecast = simulations.mean(axis=0)
    lower_bound = np.percentile(simulations, 2.5, axis=0)
    upper_bound = np.percentile(simulations, 97.5, axis=0)

    forecast_df = pd.DataFrame({
        "forecast": scaler.inverse_transform(
            mean_forecast.reshape(-1,1)
        ).flatten(),

        "lower_bound": scaler.inverse_transform(
            lower_bound.reshape(-1,1)
        ).flatten(),

        "upper_bound": scaler.inverse_transform(
            upper_bound.reshape(-1,1)
        ).flatten()
    })

    return forecast_df
