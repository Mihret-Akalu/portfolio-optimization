import pandas as pd

def load_data():
    df = pd.read_csv("data/processed/historical_prices.csv")
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Set Date as index (important for time series models)
    df.set_index("Date", inplace=True)
    
    return df


def get_tsla_close(df):
    return df["TSLA"]


def get_spy_close(df):
    return df["SPY"]


def get_bnd_close(df):
    return df["BND"]
