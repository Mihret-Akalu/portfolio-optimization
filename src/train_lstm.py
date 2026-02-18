import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler

import joblib


def prepare_data(series, window=60):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X = []
    y = []

    for i in range(window, len(scaled)):

        X.append(scaled[i-window:i])

        y.append(scaled[i])

    return np.array(X), np.array(y), scaler


def build_model():

    model = Sequential()

    model.add(LSTM(50))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def save_model_and_info(model, scaler, window_size, filepath="models/lstm_model"):
    """Save model and metadata"""
    model.save(f"{filepath}.h5")
    joblib.dump(scaler, f"{filepath}_scaler.pkl")
    
    # Save metadata
    info = {'window_size': window_size}
    joblib.dump(info, f"{filepath}_info.pkl")