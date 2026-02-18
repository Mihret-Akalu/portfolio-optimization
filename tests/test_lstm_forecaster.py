import numpy as np
from unittest.mock import MagicMock, patch

from src.lstm_forecaster import (
    create_sequences,
    bootstrap_lstm_forecast
)


def test_create_sequences():

    data = np.arange(100)

    sequences = create_sequences(data, window_size=10)

    assert sequences.shape[0] == 90
    assert sequences.shape[1] == 10


def test_bootstrap_lstm_forecast():

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.5]])

    scaled_data = np.random.rand(100)

    class DummyScaler:
        def inverse_transform(self, x):
            return x

    scaler = DummyScaler()

    with patch(
        "src.lstm_forecaster.load_model",
        return_value=mock_model
    ):

        forecast_df = bootstrap_lstm_forecast(
            model_path="dummy",
            scaled_data=scaled_data,
            scaler=scaler,
            steps=5,
            window_size=10,
            n_simulations=5
        )

    assert len(forecast_df) == 5
    assert "forecast" in forecast_df.columns
