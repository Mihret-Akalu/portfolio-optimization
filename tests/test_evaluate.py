import numpy as np

from src.evaluate import evaluate_model


def test_evaluate_model_outputs():

    true = np.array([100, 105, 110])
    pred = np.array([102, 104, 108])

    metrics = evaluate_model(true, pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
