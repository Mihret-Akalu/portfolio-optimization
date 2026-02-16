import pandas as pd


def clean_price_data(df):

    df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    return df.dropna()


def compute_returns(df):

    returns = df.pct_change().dropna()

    return returns


def split_time_series(series):

    train = series.loc[: "2024-12-31"]
    test = series.loc["2025-01-01":]

    return train, test
