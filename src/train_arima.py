from statsmodels.tsa.arima.model import ARIMA 
import pickle

def train_arima(train): 
    model = ARIMA(train, order=(5,1,0)) 
    fitted = model.fit() 
    return fitted 

def save_arima(model): 
    with open("models/arima_model.pkl", "wb") as f:
       pickle.dump(model, f)