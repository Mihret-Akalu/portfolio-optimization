import pandas as pd
import yfinance as yf


# =========================
# DOWNLOAD DATA
# =========================

def download_price_data(tickers, start, end):

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # CRITICAL FIX
        progress=False
    )

    # With auto_adjust=True, we use Close directly
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]]

    df = df.dropna()

    return df

# =========================
# LOAD SAVED DATA
# =========================

def load_data():

    
    df = pd.read_csv("data/processed/historical_prices.csv", 
        index_col=0, 
        parse_dates=True, infer_datetime_format=True)
    

    # Force numeric (CRITICAL FIX)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    return df


def get_tsla_close(df):
    return df["TSLA"].astype(float)


def get_spy_close(df):
    return df["SPY"].astype(float)


def get_bnd_close(df):
    return df["BND"].astype(float)