from src.data_loader import load_data, get_tsla_close

from src.preprocessing import split_time_series

from src.train_arima import train_arima, save_arima

from src.train_lstm import prepare_data, build_model, save_scaler


print("Loading data...")

df = load_data()

tsla = get_tsla_close(df)

train, test = split_time_series(tsla)


print("Training ARIMA...")

arima = train_arima(train)

save_arima(arima)


print("Training LSTM...")

X, y, scaler = prepare_data(train)

save_scaler(scaler)

lstm = build_model()

lstm.fit(X, y, epochs=10)

lstm.save("models/lstm_model.h5")


print("Training complete.")
