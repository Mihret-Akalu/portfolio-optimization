from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(true, pred):

    mae = mean_absolute_error(true, pred)

    rmse = np.sqrt(mean_squared_error(true, pred))

    mape = np.mean(
        np.abs((true - pred) / true)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
