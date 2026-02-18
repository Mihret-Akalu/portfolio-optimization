import numpy as np
import pandas as pd

from src.train_lstm import prepare_data, build_model


def test_lstm_training_step():

    series = pd.Series(np.random.randn(300))

    X, y, scaler = prepare_data(series, window=10)

    model = build_model()

    model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    pred = model.predict(X[:5])

    assert pred.shape[0] == 5
